from typing import Dict, List, Optional
import logging

from allennlp.data import Tokenizer
from overrides import overrides
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TensorField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
import pandas as pd
from util.utils_data_access import download_file_to_filepath
from util.utils_general import label_idx_to_str, label_str_to_idx
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
import spacy, torch

from editing.src.predictors.predictor_utils import clean_text
import unidecode
import re

TRAIN_VAL_SPLIT_RATIO = 0.9
        


stopwords = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
    'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
    'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'am', 'a', 'an', 'the', 'and', 'but', 'or', 'as', 'of', 'at', 'for',
    'with', 'about', 'against', 'into', 'to', 'up', 'down',
    'in', 'out', 'on', 'off', 'here', 'there', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 'n\'t', '\'s', 'can', 'will', 'just', 'should',
]


def tag_tokens_with_spacy(transformer_tokenized, nlp_tokenized):
    nlp_word_idx = 0
    tok_word_idx = 0
    while (tok_word_idx < len(transformer_tokenized)) and (nlp_word_idx < len(nlp_tokenized)):
        # words
        tokenized_word = transformer_tokenized[tok_word_idx]
        nlp_word = nlp_tokenized[nlp_word_idx]

        # start and end indices
        nlp_word_start_idx = nlp_word.idx
        nlp_word_end_idx = nlp_word_start_idx + len(nlp_word)
        tok_start_idx = tokenized_word.idx
        if tokenized_word.text.startswith('Ġ'):
            tok_start_idx += 1
        tok_end_idx = tokenized_word.idx_end

        # label
        is_stopword = nlp_word.text.lower() in stopwords
        if tok_start_idx >= nlp_word_start_idx and tok_end_idx <=nlp_word_end_idx:
            tokenized_word.pos_ = nlp_word.pos_
            tokenized_word.tag_ = is_stopword
            tokenized_word.ent_type_ = nlp_word.ent_type_

        # progress
        if tok_end_idx < nlp_word_end_idx:
            tok_word_idx += 1
        elif tok_end_idx == nlp_word_end_idx:
            nlp_word_idx += 1
            tok_word_idx += 1
        elif tok_end_idx > nlp_word_end_idx:
            nlp_word_idx += 1
    return transformer_tokenized

class PretrainedTransformerTokenizerWithSpacy(PretrainedTransformerTokenizer):
    def __init__(self, spacy_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nlp = spacy_model

    def tokenize(self, text):
        text = unidecode.unidecode(text)
        token_list = super().tokenize(text=text)
        nlp_token_list = self.nlp(text)
        tagged_tokens = tag_tokens_with_spacy(token_list, nlp_token_list)
        return tagged_tokens

@DatasetReader.register("news_discourse")
class NewsDiscourseDatasetReader(DatasetReader):

    DATA_URL = 'data/news-discourse-processed.tsv'

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Optional[Tokenizer] = None,
                 local=False,
                 data_file=None,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)

        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or \
                {"tokens": SingleIdTokenIndexer()}

        self.random_seed = 0 # numpy random seed
        self.local = local
        if data_file is not None:
            self.DATA_URL = data_file

    def get_inputs(
            self,
            return_labels=False,
            train=True,
            test=True,
            sample=False,
            include_errors=True,
    ):
        if self.local:
            local_path = '/Users/alex/Projects/usc-research/controlled-sequence-gen/' + self.DATA_URL
        else:
            local_path = download_file_to_filepath(self.DATA_URL)
        data = pd.read_csv(local_path, sep='\t', header=None)
        if include_errors == False:
            data = data.loc[lambda df: df[0] != 'Error']
        data[1] = data[1].apply(lambda x: clean_text(x, special_chars=["<br />", "\t"]))
        data[0] = data[0].apply(label_str_to_idx.get)
        data = data.loc[lambda df: df[1].apply(lambda x: (len(x) > 0) and (re.search('[a-zA-Z]', x) is not None))]
        train_data = data.loc[lambda df: df[2].str.contains('/train/|/validation/')]
        test_data = data.loc[lambda df: ~df[2].str.contains('/train/|/validation/')]

        train_text = train_data[1].tolist()
        train_labels = train_data[0].tolist()
        test_text = test_data[1].tolist()
        test_labels = test_data[0].tolist()
        if sample:
            train_text=train_text[:2]
            train_labels=train_labels[:2]
            test_text=test_text[:2]
            test_labels=test_labels[:2]
        output = []
        if train:
            output.append(train_text)
            if return_labels:
                output.append(train_labels)
        if test:
            output.append(test_text)
            if return_labels:
                output.append(test_labels)
        return output

    @overrides
    def _read(self, file_path):
        local_path = download_file_to_filepath(self.DATA_URL)
        data = pd.read_csv(local_path, sep='\t', header=None)
        for label, text in data[[0, 1]].iteritems():
            label_idx = label_idx_to_str[label]
            yield self.text_to_instance(clean_text(text, special_chars=["<br />", "\t"]), label_idx)

    def text_to_instance(self, string: str, label:str = None, device=None) -> Optional[Instance]:
        tokens = self._tokenizer.encode(string)
        tokens_tensor = torch.tensor(tokens, device=device)
        tokens_field = TensorField(tensor=tokens_tensor, padding_value=0, dtype=int)
        fields = {'input_ids': tokens_field}
        if label is not None:
            fields["label"] = LabelField(label)
        return Instance(fields)
