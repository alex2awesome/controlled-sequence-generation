import torch
from torchmetrics import Metric
from torch import nn
from fairscale.nn import checkpoint_wrapper, auto_wrap, wrap
import transformers
from packaging import version

def _get_words(word_ids, start_idx, end_idx):
    if len(word_ids.size()) > 1:
        return word_ids[:, start_idx:end_idx]
    return word_ids[start_idx:end_idx].clone()


def _set_words(word_ids, start_idx, end_idx, val):
    if len(word_ids.size()) > 1:
        word_ids[:, start_idx:end_idx] = val
    word_ids[start_idx:end_idx] = val
    return word_ids.clone()


class PerplexityBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def calculate_ppl_for_sequence(self, input_ids, labels, n_words_in_input, model):
        start_idx, end_idx = 0, 0
        for word_idx in range(0, n_words_in_input, self.stride):
            start_idx = max(word_idx + self.stride - self.max_context_len, 0)
            end_idx = min(word_idx + self.stride, n_words_in_input)
            target_len = end_idx - word_idx

            input_t = _get_words(input_ids, start_idx, end_idx)
            output_t = _get_words(labels, start_idx, end_idx)
            output_t = _set_words(output_t, 0, -target_len, -100)

            with torch.no_grad(): # to make sure there's no gradient computation
                if version.parse(transformers.__version__) > version.parse('4.0.0'):
                    loss = model.forward(input_ids=input_t, labels=output_t, return_dict=False)[0]
                else:
                    loss = model.forward(input_ids=input_t, labels=output_t)[0]
                loss = loss * target_len
                loss = loss.to(self.device)

            self.lls += loss
        #
        self.count += torch.tensor(end_idx, device=self.device)

    def update(self, input_ids, model):
        labels = input_ids.clone()
        if False and len(labels.size()) > 1: # then, first dim is batch (disabling this for now... for RoBERTa, we actually need a 2-d input)
            for input_i, labels_i in zip(input_ids, labels):
                n_words_in_input = input_i.size()[0]
                self.calculate_ppl_for_sequence(input_i, labels_i, n_words_in_input, model)
        else:
            n_words_in_input = labels.size()[0]
            self.calculate_ppl_for_sequence(input_ids, labels, n_words_in_input, model)

    def reset(self):
        self.lls = torch.tensor(0.0, device=self.device)
        self.count = torch.tensor(0.0, device=self.device)

    def compute(self):
        return torch.exp(self.lls / self.count)

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            self.update(*args, **kwargs)


class PerplexityMetric(Metric, PerplexityBase):
    def __init__(self, device, stride=128, max_context_len=2048, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.device = device
        self.add_state('lls', default=torch.tensor(0.0, device=self.device), dist_reduce_fx="sum")
        self.add_state('count', default=torch.tensor(0.0, device=self.device), dist_reduce_fx='sum')
        self.stride = stride
        self.max_context_len = max_context_len


class PerplexityOrig(PerplexityBase):
    def __init__(self, device, stride=128, max_context_len=2048):
        super().__init__()
        self.device = device
        self.lls = torch.tensor(0.0, device=self.device)
        self.count = torch.tensor(0.0, device=self.device)
        self.stride = stride
        self.max_context_len = max_context_len
