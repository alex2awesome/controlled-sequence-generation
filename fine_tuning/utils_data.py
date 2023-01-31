import csv
import torch
import torch.optim
import torch.utils.data as data
import pytorch_lightning as pl
import os
from transformers import BertTokenizer, GPT2Tokenizer, RobertaTokenizer
from util.utils_general import (
    reformat_model_path,
    get_idx2class,
    format_local_vars,
    transpose_dict,
    _get_attention_mask
)
from torch.nn.utils.rnn import pad_sequence
import itertools

from util.utils_prompting import PromptGenerator

try: # version 3.0.2
    from transformers.tokenization_gpt2 import AddedToken
except: # version > 3.0.2
    pass


max_num_tokens_in_doc = 2045

class Dataset(data.Dataset):
    def __init__(self, X, y=None, split=None):
        """Reads source and target sequences from txt files."""
        self.X = X
        self.y = y
        self.split = split

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        data = {}
        data["input_ids"] = self.X[index]
        if self.y is not None:
            data['class_labels'] = self.y[index]
        return data


def _get_split(row):
    if '/test/' in row:
        return 'test'
    elif '/train/' in row:
        return 'train'
    elif '/validation/' in row:
        return 'val'


class BaseDiscourseDataModule(pl.LightningDataModule):
    def __init__(self, config=None, *args, **kwargs):
        super().__init__()
        self.config = config
        self.data_fp = kwargs.get('data_fp')
        self.add_eos_token = (kwargs.get('model_type') == "gpt2")
        self.max_length_seq = kwargs.get('max_length_seq')
        self.max_num_sentences = kwargs.get('max_num_sentences', 100)
        self.batch_size = kwargs.get('batch_size')
        self.num_cpus = kwargs.get('num_cpus')
        self.split_type = kwargs.get('split_type')
        self.split_perc = kwargs.get('split_perc', .9)
        self.load_tokenizer(
            model_type=kwargs.get('model_type'), pretrained_model_path=kwargs.get('pretrained_model_path')
        )

    def load_tokenizer(self, model_type, pretrained_model_path):
        if model_type == "gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained(reformat_model_path(pretrained_model_path))
            # self.tokenizer.add_special_tokens({
            #     'sep_token': AddedToken('<|sep|>', rstrip=False, lstrip=False, single_word=True, normalized=True)
            # })

        elif model_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(reformat_model_path(pretrained_model_path))
        elif model_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(reformat_model_path(pretrained_model_path))
        else:
            print('Model path not in {bert, roberta, gpt2}.')

    def prepare_data(self):
        """
        Checks if the data path exists.

        Occurs only on the master GPU.
        """
        if not os.path.exists(self.data_fp):
            raise FileNotFoundError('Data files... make sure to download them from S3!')

    def process_row(self, text):
        seq = self.tokenizer.encode(text, return_tensors='pt').squeeze(dim=0)
        seq = seq[:self.max_length_seq]

        #
        sep = torch.tensor([self.tokenizer.eos_token_id])
        seq = torch.hstack((seq, sep))
        return seq

    def setup(self, stage=None):
        """
            Download and split the dataset before training/testing.
            For Nonsequential datasets, this just splits on the sentences.
            For Sequential datasets (which are nested lists of sentences), this splits on the documents.

            Occurs on every GPU.
        """
        if stage in ('fit', None):
            d = self.get_dataset()
            # split randomly
            if self.split_type == 'random':
                train_size = int(self.split_perc * len(d))
                test_size = len(d) - train_size
                self.train_dataset, self.test_dataset = torch.utils.data.random_split(d, [train_size, test_size])
            # split by filename
            elif self.split_type == 'key':
                zipped = [d.X, d.split]
                if d.y is not None:
                    zipped.append(d.y)
                else:
                    zipped.append([None] * len(d.X))
                train_dataset = list(filter(lambda x: x[1] in ['train', 'val'], zip(*zipped)))
                train_X, _, train_y = zip(*train_dataset)
                self.train_dataset = Dataset(X=train_X, y=train_y)
                test_dataset = list(filter(lambda x: x[1] in ['test'], zip(*zipped)))
                test_X, _, test_y = zip(*test_dataset)
                self.test_dataset = Dataset(X=test_X, y=test_y)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_cpus
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_cpus
        )


class DiscourseFinetuningDataset(BaseDiscourseDataModule):
    def __init__(
            self, data_fp, model_type, max_length_seq,
            batch_size, pretrained_model_path, num_cpus=10,
            split_type='random', split_perc=.9, **kwargs,
    ):
        super().__init__(**format_local_vars(locals()))

    def get_dataset(self):
        """
        Read in dataset as a list of "label \t text" entries.
        Output flat lists of X = [sent_1, sent_2, ...], y = [label_1, label_2]
        """
        split, X, y = [], [], []
        with open(self.data_fp) as f:
            csv_reader = csv.reader(f, delimiter="\t")
            csv_data = list(csv_reader)

        for doc_idx, doc in itertools.groupby(csv_data, key=lambda x: x[2]):  # group by doc_id
            sorted_doc = sorted(doc, key=lambda x: int(x[3]))                 # sort by sent_id
            if len(sorted_doc) > self.max_num_sentences:
                continue
            doc_seqs, doc_labels = [], []
            for sentence in sorted_doc:
                text = sentence[1]
                processed_text = self.process_row(text)
                doc_seqs.append(processed_text)

            if len(torch.cat(doc_seqs)) > max_num_tokens_in_doc:
                continue

            # append processed data
            X.append(doc_seqs)

            # record dataset built-in splits
            split.append(_get_split(doc_idx))
        return Dataset(X, split=split)

    def collate_fn(self, dataset):
        """
        Takes in an instance of Torch Dataset (or a subclassed instance).
        Expects dataset[i]['X'] to be a list of list of sentences
            and dataset[i]['y'] to be a list of list of labels.
        Returns dict with key:
             input_ids: list of tensors of len # docs where each tensor is an entire document.
             labels: same as `input_ids`, for language modeling.
        """
        columns = transpose_dict(dataset)
        X_batch = list(map(lambda sents: torch.cat(sents), columns["input_ids"]))
        return {"input_ids": torch.cat(X_batch).unsqueeze(dim=0), 'labels': torch.cat(X_batch).unsqueeze(dim=0)}


class BaselineOneDataset(BaseDiscourseDataModule):
    def __init__(
            self, data_fp, model_type, max_length_seq,
            batch_size, pretrained_model_path, num_cpus=10,
            split_type='random', split_perc=.9, **kwargs,
    ):
        super().__init__(**format_local_vars(locals()))

    def setup(self, stage=None):
        self.idx2class, self.class2idx = get_idx2class(self.data_fp, use_headline=True)
        super().setup(stage=stage)

    def get_dataset(self):
        """
        Read in dataset as a list of "label \t text" entries.
        Output flat lists of X = [sent_1, sent_2, ...], y = [label_1, label_2]
        """
        split, X, y = [], [], []
        with open(self.data_fp) as f:
            csv_reader = csv.reader(f, delimiter="\t")
            csv_data = list(csv_reader)

        for doc_idx, doc in itertools.groupby(csv_data, key=lambda x: x[2]):  # group by doc_id
            sorted_doc = sorted(doc, key=lambda x: int(x[3]))                 # sort by sent_id
            if len(sorted_doc) > self.max_num_sentences:
                continue
            doc_seqs, doc_labels = [], []
            headline = sorted_doc[0][4]
            processed_headline = self.process_row(headline)
            doc_seqs.append(processed_headline)
            doc_labels.append(self.class2idx['headline'])
            for sentence in sorted_doc:
                text, label = sentence[1], sentence[0]

                processed_text = self.process_row(text)
                doc_seqs.append(processed_text)
                doc_labels.append(self.class2idx[label])

            if len(torch.cat(doc_seqs)) > max_num_tokens_in_doc:
                continue

            # append processed data
            X.append(doc_seqs)
            y.append(doc_labels)

            # record dataset built-in splits
            split.append(_get_split(doc_idx))
        return Dataset(X, y=y, split=split)

    def collate_fn(self, dataset):
        """
        Takes in an instance of Torch Dataset (or a subclassed instance).
        Expects dataset[i]['X'] to be a list of list of sentences
            and dataset[i]['y'] to be a list of list of labels.
        Returns dict with key:
             input_ids: list of tensors of len # docs where each tensor is an entire document.
             labels: same as `input_ids`, for language modeling.
             class_labels: a vector of labels for each sentence in the document.
        """
        columns = transpose_dict(dataset)
        X_batch = list(map(lambda sents: torch.cat(sents).to(int), columns["input_ids"]))
        sent_lens = list(map(lambda x: torch.tensor(list(map(len, x))).to(int), columns['input_ids']))
        class_labels = list(map(lambda x: torch.tensor(x).to(int), columns['class_labels']))
        return {
            "input_ids": X_batch,
            'labels': X_batch,
            "class_labels": class_labels,
            'sentence_lens': sent_lens
        }


class BaselineTwoDatasetBase(BaseDiscourseDataModule):
    def __init__(self, *args, **kwargs):
        self.prompt_gen = PromptGenerator()
        super().__init__(*args, **kwargs)

    def setup(self, stage=None):
        self.idx2class, self.class2idx = get_idx2class(self.data_fp, use_headline=True)
        self.tokenizer = self.prompt_gen.resize_tokenizer(self.tokenizer)
        super().setup(stage=stage)

    def get_dataset(self):
        """
        Read in dataset as a list of "label \t text" entries.
        Output flat lists of X = [sent_1, sent_2, ...], y = [label_1, label_2]
        """
        split, X, y = [], [], []
        with open(self.data_fp) as f:
            csv_reader = csv.reader(f, delimiter="\t")
            csv_data = list(csv_reader)

        for idx, (doc_idx, doc) in enumerate(itertools.groupby(csv_data, key=lambda x: x[2])):  # group by doc_id
            if self.config.env == 'local':
                if idx > 400:
                    break
            sorted_doc = sorted(doc, key=lambda x: int(x[3]))                 # sort by sent_id
            if len(sorted_doc) > self.max_num_sentences:
                continue
            headline = sorted_doc[0][4]
            for idx in range(len(sorted_doc)):
                prompt = self.generate_prompt(headline, sorted_doc, idx)
                processed_prompt = self.process_row(prompt)
                processed_text = self.process_row(sorted_doc[idx][1])

                # append processed data
                X.append(processed_prompt)
                y.append(processed_text)

                # record dataset built-in splits
                split.append(_get_split(doc_idx))
        return Dataset(X, y=y, split=split)

    def collate_fn(self, dataset):
        columns = transpose_dict(dataset)
        prompt_batch = pad_sequence(columns['input_ids'], batch_first=True)
        label_batch = pad_sequence(columns['class_labels'], batch_first=True)
        prompt_attention = _get_attention_mask(prompt_batch)
        label_attention  = _get_attention_mask(label_batch)
        label_batch = label_batch + ((label_attention - 1) * 100)
        #
        orig_prompt_shape = prompt_batch.shape
        prompt_batch = torch.hstack((prompt_batch, label_batch))
        label_batch = torch.hstack((torch.ones(orig_prompt_shape) * -100, label_batch))
        prompt_attention = torch.hstack((prompt_attention, label_attention))
        return {
            'input_ids': prompt_batch.to(int),
            'labels': label_batch.to(int),
            'attention_mask': prompt_attention.to(int)
        }


class BaselineTwoDatasetBaseline(BaselineTwoDatasetBase):
    def generate_prompt(self, headline, sorted_doc, s_idx):
        sentences = list(map(lambda x: x[1], sorted_doc))
        labels = list(map(lambda x: x[0], sorted_doc))
        labels = self.prompt_gen.baseline(labels, s_idx)
        return self.prompt_gen.generate_prompt(headline=headline, sentences=sentences, labels=labels, s_idx=s_idx)


class BaselineTwoDatasetPast(BaselineTwoDatasetBase):
    def generate_prompt(self, headline, sorted_doc, s_idx):
        sentences = list(map(lambda x: x[1], sorted_doc))
        labels = list(map(lambda x: x[0], sorted_doc))
        labels = self.prompt_gen.past_aware(labels, s_idx)
        return self.prompt_gen.generate_prompt(headline=headline, sentences=sentences, labels=labels, s_idx=s_idx)


class BaselineTwoDatasetFuture(BaselineTwoDatasetBase):
    def generate_prompt(self, headline, sorted_doc, s_idx):
        sentences = list(map(lambda x: x[1], sorted_doc))
        labels = list(map(lambda x: x[0], sorted_doc))
        labels = self.prompt_gen.future_aware(labels, s_idx)
        return self.prompt_gen.generate_prompt(headline=headline, sentences=sentences, labels=labels, s_idx=s_idx)


