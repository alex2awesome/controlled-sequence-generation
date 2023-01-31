from allennlp.predictors.predictor import Predictor
from allennlp.data import Instance
from allennlp.data.fields import LabelField, TensorField
from allennlp.data.vocabulary import Vocabulary
import torch
import types
from allennlp.nn import util
from allennlp.common.checks import ConfigurationError


def forward_on_instance(self, instance):
    """Take in an instance and calculate probabilities """
    input_ids = instance.fields['input_ids'].tensor
    input_ids = input_ids.to(self.device)
    loss, preds, _ = self.predict_one_doc(
        input_ids = input_ids,
        # add_features='main',
        # label_idx=instance.fields.get('label_idx'),
        # labels=instance.fields.get('class_ids'),
        # generate=True
    )
    input_ids = input_ids.detach().cpu()
    return {'logits': preds[0]}

def make_output_human_readable(self, outputs):
    """Outputs for the model's native `forward` function. In this case, just the logits will be returned"""
    return {'logits': outputs}

def _get_prediction_device(self) -> int:
    """
    This method checks the device of the model parameters to determine the cuda_device
    this model should be run on for predictions.  If there are no parameters, it returns -1.

    # Returns

    The cuda device this model should run on for predictions.
    """
    devices = {util.get_device_of(param) for param in self.parameters()}

    if len(devices) > 1:
        devices_string = ", ".join(str(x) for x in devices)
        raise ConfigurationError(f"Parameters have mismatching cuda_devices: {devices_string}")
    elif len(devices) == 1:
        return devices.pop()
    else:
        return -1

def encode(self, text_input):
    tokens = self.tokenize(text_input)
    return list(map(lambda x: x.text_id, tokens))


class TorchGPT2Predictor(Predictor):
    def __init__(self, huggingface_dir=None, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if huggingface_dir is not None:
            self._model.vocab = Vocabulary.from_pretrained_transformer(model_name=huggingface_dir)
        self._dataset_reader._tokenizer._max_length = 1024
        self._dataset_reader._tokenizer.single_sequence_start_tokens = ['<|endoftext|>']
        self._dataset_reader._tokenizer.single_sequence_end_tokens = ['<|endoftext|>']
        self._dataset_reader._tokenizer.encode = types.MethodType(encode, self._dataset_reader._tokenizer)
        if device is None:
            from editing.src.utils import get_device
            device = get_device()
        self.device = device
        self._model.forward_on_instance = types.MethodType(forward_on_instance, self._model)
        self._model.make_output_human_readable = types.MethodType(make_output_human_readable, self._model)
        self._model._get_prediction_device = types.MethodType(_get_prediction_device, self._model)
        self.task = 'news_discourse'

    @property
    def tokenizer(self):
        return self._dataset_reader._tokenizer

    def get_interpretable_text_field_embedder(self):
        return self._model.transformer.encoder_model.transformer.wte

    def predict(self, text_input):
        tokens = self.tokenizer.encode(text_input)
        tokens_tensor = torch.tensor(tokens).unsqueeze(dim=0)
        loss, tag_preds = self._model.predict_one_doc(tokens_tensor)

        return {
            'preds': tag_preds,
            'label': torch.argmax(tag_preds.squeeze())
        }

    def _json_to_instance(self, json_dict):
        """Example input:

            json_dict = {'sentence': 'MULTAN, Pakistan – An Indian man on death row in Pakistan for espionage has died
            after being attacked by fellow inmates, Pakistan’s state television said early Thursday.'}
        """
        sentence = json_dict['sentence']
        tokens = self._dataset_reader._tokenizer.encode(sentence)
        tokens_tensor = torch.tensor(tokens, device=self.device)
        tokens_field = TensorField(tensor=tokens_tensor, padding_value=0, dtype=int)
        instance = Instance(fields={
            'input_ids': tokens_field
        })
        return instance

    def predictions_to_labeled_instances(self, instance, outputs):
        """Outputs can be multiple, so they're expecting possibly multiple instances being returned,
        one with each output."""
        logits = outputs['logits']
        label = torch.argmax(logits)
        y_pred_ll = TensorField(tensor=logits)
        y_pred = LabelField(label=int(label), skip_indexing=True)
        instance.fields['labels'] = y_pred
        instance.fields['logits'] = y_pred_ll
        return [instance]
