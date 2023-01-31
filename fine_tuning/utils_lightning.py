import pytorch_lightning as pl

import torch
from discriminator.utils_general import get_config

adam_beta1, adam_beta2, adam_epsilon = .9, .999, 1e-08

get_train_ppl = False
get_val_ppl = True
use_torch_metric = False

if use_torch_metric:
    from fine_tuning.utils_metrics import PerplexityMetric as Perplexity
else:
    from fine_tuning.utils_metrics import PerplexityOrig as Perplexity



class LightningSteps(pl.LightningModule):
    """Mixin to handle Lightning hooks and metrics"""

    def __init__(self, *args, **kwargs):
        super().__init__()

        #####
        # metrics
        if get_train_ppl:
            self.training_perplexity = Perplexity(device=self.device)
        if get_val_ppl:
            self.validation_perplexity = Perplexity(device=self.device)

    def training_step(self, batch, batch_idx):
        loss = self.hf_model.forward(**batch)[0]
        self.log('Training Loss', loss)
        return {'loss': loss, 'input_ids': batch['input_ids']}

    def validation_step(self, batch, batch_idx):
        loss = self.hf_model.forward(**batch)[0]
        self.log('Validation loss', loss)
        if get_val_ppl:
            self.validation_perplexity(batch['input_ids'], self)
            self.get_ppl('validation')
        return {'loss': loss, 'input_ids': batch['input_ids']}

    def training_step_end(self, batch_parts):
        # if run on multi-GPUs, this is a list(?)
        if isinstance(batch_parts, list):
            for batch in batch_parts:
                if get_train_ppl:
                    self.training_perplexity(batch['input_ids'], self)
            return sum(map(lambda x: x['loss'], batch_parts))
        # otherwise, it's a float(?)
        else:
            if get_train_ppl:
                self.training_perplexity(batch_parts['input_ids'], self)
            return batch_parts['loss']

    def validation_step_end(self, batch_parts):
        if isinstance(batch_parts, list):
            return sum(map(lambda x: x['loss'], batch_parts))
        else:
            return batch_parts['loss']

    def training_epoch_end(self, outputs):
        if get_train_ppl:
            self.get_ppl('training')

    def validation_epoch_end(self, outputs):
        pass
        # if get_val_ppl:
        #     self.get_ppl('validation')

    def get_ppl(self, run):
        if run == 'validation':
            ppl = self.validation_perplexity.compute()
            self.log('Validation Perplexity', ppl)
            self.validation_perplexity.reset()
        else:
            ppl = self.training_perplexity.compute()
            self.log('Training Perplexity', ppl)
            self.training_perplexity.reset()


class LightningOptimizer(pl.LightningModule):
    """
    Contains logic for optimization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = get_config(kwargs=kwargs)
        #
        self.lr = self.config.learning_rate
        self.num_warmup_steps = self.config.num_warmup_steps
        self.dataset_size = self.config.num_steps_per_epoch

    # optimization
    def _lr_lambda(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1.0, self.num_warmup_steps))
        return 1.0

    def _lr_lambda_linear(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        num = self.num_training_steps - current_step
        denom = self.num_training_steps - self.num_warmup_steps
        num = float(max(0, num))
        denom = float(max(1, denom))
        return num / denom

    def configure_optimizers(self):
        self.num_training_steps = self.dataset_size * self.trainer.max_epochs
        optimizer_kwargs = {
            "betas": (adam_beta1, adam_beta2),
            "eps": adam_epsilon,
        }
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, **optimizer_kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, self._lr_lambda_linear),
                'interval': 'step',
            }
        }


