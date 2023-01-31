from discriminator.layers_classification import MultiClassMixin
from discriminator.models_full import BaseDiscriminator, SentenceEmbeddingsLayer
import transformers
if transformers.__version__ == '3.0.2':
    from discriminator.utils_futures import GPT2ForSequenceClassification
else: # transformers: version 4.0
    from transformers.models.gpt2.modeling_gpt2 import GPT2ForSequenceClassification
from discriminator.utils_general import get_config
from discriminator.utils_lightning import LightningMixin
from transformers import AutoConfig
from torch import nn
import torch

class BaselineHead(nn.Module):
    def __init__(self, *args, **kwargs):
        """Dummy so that MulticlassMixin doesn't have to pass the config to nn.Module"""
        super().__init__()


class BaselineHeadLayer(MultiClassMixin, BaselineHead):
    def __init__(self, *args, **kwargs):
        self.config = get_config(kwargs=kwargs)
        super().__init__(*args, **kwargs)

        self.w = nn.Linear(self.config.embedding_dim, self.config.hidden_dim)

    def forward(self, sent_embs, labels=None):
        hidden = self.w(torch.tanh(sent_embs))
        return self.classification(hidden, labels)


class BaselineDiscriminator(LightningMixin, BaseDiscriminator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transformer = SentenceEmbeddingsLayer(*args, **kwargs)
        self.head = BaselineHeadLayer(*args, **kwargs)

    def vec_or_nones(vec, output_len):
        return vec if vec is not None else [None] * output_len

    def predict_one_doc(self, input_ids, labels, attention_mask=None, *args, **kwargs):
        """Really, this can be predict_one_batch. The sentences need not belong to one document.
        Only named this way for consistency with methods in the other classes.
        """
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(dim=0)
        if attention_mask is not None and len(attention_mask.shape) == 1:
            attention_mask = attention_mask.unsqueeze(dim=0)
        if labels is not None and len(labels.shape) == 0:
            labels = labels.unsqueeze(dim=0)
        sent_embs = self.transformer.get_sentence_embedding(input_ids, attention_mask)
        loss, tag_preds = self.head(sent_embs, labels)
        return loss, tag_preds


class DiscriminatorGPT2Baseline(LightningMixin, BaseDiscriminator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        t_config = AutoConfig.from_pretrained(self.config.pretrained_cache_dir)
        setattr(t_config, 'pad_token_id', self.config.pad_id)
        setattr(t_config, 'num_labels', self.config.num_output_tags)
        self.classifier = GPT2ForSequenceClassification(config=t_config)

    def predict_one_doc(self, input_ids, labels, attention_mask=None):
        loss, tag_preds, _ = self.classifier(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        return loss, tag_preds