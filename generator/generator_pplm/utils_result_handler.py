import generator.generator_pplm.utils_methods as um
import generator.utils_methods
import generator.utils_result_handler_base as ugen_base
import torch

des_keys = [
    'per-step ppl',
    'combined_loss',
    'unpert_discrim_loss',
    'avg(\delta H) layer 22',
    'sd(\delta H) layer 22',
    'avg(\delta H) layer 23',
    'sd(\delta H) layer 23',
]

class Word(ugen_base.WordBase):
    def __init__(self, s_idx, w_idx, token_str, label, tok, *args, **kwargs):
        self.s_idx = s_idx
        self.w_idx = w_idx
        self.token_str = token_str
        self.label = label
        # other attributes
        self.combined_loss = kwargs.get('combined_loss')
        self.unpert_discrim_loss = kwargs.get('unpert_discrim_loss')
        self.word_level_nll = kwargs.get('word_level_nll')
        self.per_step_ppl = kwargs.get('per_step_ppl')
        self.tok = ugen_base.tensor_to_int(tok)
        super().__init__(s_idx, w_idx, token_str, label, tok, *args, **kwargs)


class GeneratorResultHandler(ugen_base.GeneratorResultHandlerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # helper methods
    def _get_word_level_metrics(
            self,
            last,
            sentence_var=None,
            loss_this_iter=None,
            unpert_discrim_loss=None,
            pert_h=None,
            unpert_h=None
    ):

        if sentence_var is None:
            sentence_var = self.curr_sentence_variant
        w_idx, s_idx = sentence_var.w_idx, sentence_var.s_idx
        token_str = self._decode(last)
        word_metrics = Word(s_idx=s_idx, w_idx=w_idx, token_str=token_str, label=self.current_class_label, tok=last)

        # process discrim, combined loss and hidden states
        global_step = w_idx + (s_idx * self.max_sentence_length)
        word_metrics.combined_loss = self._process_combined_loss(loss_this_iter, global_step)
        word_metrics.unpert_discrim_loss = self._process_unpert_discrim_loss(unpert_discrim_loss, global_step)
        word_level_metrics = self._process_hidden_layers(word_metrics, pert_h, unpert_h, global_step)

        # get log likelihoods
        all_tokens_tensor = self.document.get_all_tokens_tensor(sentence_var)
        curr_ll = generator.utils_methods.get_word_ll(all_tokens_tensor, self.lm)

        word_level_metrics.word_level_nll = curr_ll
        word_level_metrics.per_step_ppl = self._calculate_ppl(curr_ll, global_step, sentence_var)
        return word_level_metrics

    def _process_combined_loss(self, loss_this_iter, global_step):
        if loss_this_iter is not None:
            loss = ugen_base.tensor_to_float(loss_this_iter)
            if self.tb_logger is not None:
                self.tb_logger.add_scalar('discrim + \\alpha KL loss', loss, global_step=global_step)
            return loss

    def _process_unpert_discrim_loss(self, unpert_discrim_loss, global_step):
        if unpert_discrim_loss is not None:
            unpert_discrim_loss = ugen_base.tensor_to_float(unpert_discrim_loss)
            if self.tb_logger is not None:
                self.tb_logger.add_scalar('unpert_discrim_loss', unpert_discrim_loss, global_step=global_step)
            return unpert_discrim_loss

    def _process_hidden_layers(self, word_metric, pert_h, unpert_h, global_step):
        if pert_h is not None and unpert_h is not None:
            # hidden layers
            for layer, (pert_h_i, unpert_h_i) in enumerate(zip(pert_h, unpert_h)):
                # last input_id
                pert_last = pert_h_i.squeeze()
                unpert_last = unpert_h_i.squeeze()[-1]
                avg_tag = 'avg(\delta H) layer %s' % layer
                avg_delt = ugen_base.tensor_to_float(torch.mean(torch.abs(pert_last - unpert_last)))
                sd_tag = 'sd(\delta H) layer %s' % layer
                sd_delt = ugen_base.tensor_to_float(torch.std(torch.abs(pert_last - unpert_last)))

                # store metrics
                word_metric[avg_tag] = avg_delt
                word_metric[sd_tag] = sd_delt
                if self.tb_logger is not None:
                    self.tb_logger.add_scalar(avg_tag, avg_delt, global_step=global_step)
                    self.tb_logger.add_scalar(sd_tag, sd_delt, global_step=global_step)
        return word_metric