# pretty print table
import textwrap
import logging

import numpy as np
import torch
import sys, os
from types import ModuleType, FunctionType
from gc import get_referents
# from generator.utils_result_handler_base import des_keys
BLACKLIST = type, ModuleType, FunctionType
from util.utils_general import CLASS_MAPPING, label_idx_to_str

class PrettyTable:
    def __init__(
            self,
            contents,
            col_widths,
            col_delim = "|",
            row_delim = "-"
    ):
        # make sure we convert all ints to strings
        self.contents = list(map(lambda row: list(map(lambda cell: str(cell), row)), contents))
        self.col_widths = col_widths
        self.col_delim = col_delim

        # Extra row_delim characters where col_delim characters are
        p = len(self.col_delim) * (len(self.contents[0]) - 1)

        # Line gets too long for one concatenation
        self.row_delim = self.col_delim
        self.row_delim += row_delim * (sum(self.col_widths) + p)
        self.row_delim += self.col_delim + "\n"

    def withTextWrap(self):
        string = self.row_delim
        # Restructure to get textwrap.wrap output for each cell
        l = []
        for row in self.contents:
            row_line = [textwrap.wrap(col, self.col_widths[col_idx]) for col_idx, col in enumerate(row)]
            l.append(row_line)

        for row in l:
            for n in range(max(map(len, row))):
                string += self.col_delim
                for col_idx, col in enumerate(row):
                    if n < len(col):
                        string += col[n].ljust(self.col_widths[col_idx])
                    else:
                        string += " " * self.col_widths[col_idx]
                    string += self.col_delim
                string += "\n"
            string += self.row_delim
        return string

    def __str__(self):
        return self.withTextWrap()


class PrettyDocumentTable():
    def __init__(self, kind):
        self.kind = kind
        #
        if kind == 'perturbed':
            self.pretty_table_output = [['S_idx', 'Label', 'Sentence']]
            self.sentence_level_colwidths = [15, 15, 100]
            self.word_level_colwidths = [5, 10, 10, 10, 10, 10, 10, 10, 10, 10]
            self.first_cols = ['s_idx', 'label', 'tok']
        else:
            self.pretty_table_output = [['S_idx', 'Sentence']]
            self.sentence_level_colwidths = [15, 115]
            self.word_level_colwidths = [5, 10, 10, 10, 10, 10, 10, 10, 10]
            self.first_cols = ['s_idx', 'tok']

    #
    # formatting table
    def append_to_table(self, s_idx, formatted_text, class_label=None):
        if class_label is None:
            self.pretty_table_output.append([s_idx, formatted_text])
        else:
            self.pretty_table_output.append([s_idx, class_label, formatted_text])

    def set_all_word_level_metrics(self, all_word_level_metrics):
        self.all_word_level_metrics = all_word_level_metrics

    def print_tables(self, print_sentence_table, print_word_table):
        if print_sentence_table:
            sentence_level_table = PrettyTable(self.pretty_table_output, col_widths=self.sentence_level_colwidths)
            sentence_level_table_output = str(sentence_level_table)
            logging.info(sentence_level_table_output)
            # self.tb_logger.add_text('%s text' % self.kind, table_output)

        if print_word_table:
            # word-level-table
            to_format = [self.first_cols + des_keys]
            for word_m in self.all_word_level_metrics:
                row = self._get_table_row(word_m)
                for k in des_keys:
                    row.append(word_m[k])
                to_format.append(row)
            word_level_table = PrettyTable(to_format, col_widths=self.word_level_colwidths)
            logging.info(str(word_level_table))

    def _get_table_row(self, word_m):
        if self.kind == 'perturbed':
            return [word_m['s_idx'], word_m['label'], word_m['de-tok']]
        else:
            return [word_m['s_idx'], word_m['de-tok']]

    # factory method
    def from_document(self, d):
        for idx, (formatted_sent, chosen_sent) in enumerate(zip(d.dictionary_formatted_output, d.all_sentences)):
            self.append_to_table(idx, formatted_sent['new_text'], chosen_sent.class_label)
        self.set_all_word_level_metrics(d.get_flat_word_metrics())
        return self


def fix_floats(row_dict):
    output_dict = {}
    for key, val in row_dict.items():
        if not isinstance(val, (str, type(None))):
            output_dict[key] = float(val)
        else:
            output_dict[key] = val
    return output_dict


def get_tb_logger(env):
    import os
    from torch.utils.tensorboard import SummaryWriter
    tb_env_var = 'TENSORBOARD_LOGDIR'
    if (tb_env_var in os.environ) and (env == 'bb'):
        import subprocess
        tb_dir = os.environ['TENSORBOARD_LOGDIR']
        cmd = 'katie hdfs --identity ai-clf-dob2-gen --namespace s-ai-classification ls %s' % (tb_dir.replace('hdfs://', ''))
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out, err = process.communicate()
        lines = out.decode('utf-8').strip().split('\n')
        lines = list(filter(lambda x: 'trial_run' in x, lines))
        tb_dir = os.path.join(tb_dir, 'trial_run_%s' % len(lines))
        tb_logger = SummaryWriter(log_dir=tb_dir)
    else:
        if env != 'local':
            tb_logger = SummaryWriter() # if no logdir, we just write to ./runs/
        else:
            tb_logger = None

    return tb_logger


def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size


def get_class_str(class_id, class_mapping=CLASS_MAPPING):
    if isinstance(class_id, (int, float, np.int64)):
        class_id = int(class_id)
    elif class_id is None:
        return class_id

    row = list(filter(lambda x: x['class_label'] == class_id, class_mapping))
    return row[0]['event_tag']


def get_class_id(class_label, class_mapping=CLASS_MAPPING):
    if isinstance(class_label, (int, float, np.int64)):
        return int(class_label)
    elif isinstance(class_label, (str)):
        row = list(filter(lambda x: x['event_tag'] == class_label, class_mapping))
        return int(row[0]['class_label'])
    elif class_label is None:
        return None
    else:
        raise ValueError('class_label is unhandled type %s' % str(type(class_label)))


def _get_class_label(class_id, class_mapping=None):
    if isinstance(class_id, (int, float, np.int64)):
        row = list(filter(lambda x: x['class_label'] == class_id, class_mapping))
        return int(row[0]['event_tag'])
    elif isinstance(class_id, (str)):
        return class_id
    else:
        raise ValueError('class_id is unhandled type %s' % str(type(class_id)))


def reformat_model_path(x, env=None):
    fp_marker = './'
    if os.environ.get('env') == 'bb' and (not x.startswith(fp_marker)):
        return os.path.join(fp_marker, x)
    if env is not None and env == 'local':
        return os.path.join('/Users/alex/.cache/torch/transformers/named-models', fp_marker)
    else:
        return x


def process_text(text):
    """Light processing script to process text:
        1. before concatenating to prompt, for the next sentence.
        2. using SpaCy sentence splitting.
    """
    return ' '.join(text.split())


def _get_sentence_length(class_label, class_mapping, max_len, mult_factor=1):
    if class_label:
        return min(class_mapping[class_label]['avg_sentence_len'] * mult_factor, max_len)
    return max_len


def tokens_to_str(tokens, tokenizer, remove_eos=True):
    if remove_eos:
        tokens = list(filter(lambda x: x != 50256, tokens))
    return tokenizer.decode(tokens)


def format_context(context, device):
    """Take the prompt input (primer text that's just been encoded by the Tokenizer)
        and format it in the shape that we expect.
    """
    output_so_far = None
    if context:
        context_t = torch.tensor(context, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)
        output_so_far = context_t
    return output_so_far


import string
def count_punct(text):
    punct = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    n = len(text)
    p = 0
    for c in text:
        if c in punct:
            p += 1
    return p

import unidecode
def starting_punct(text):
    punct = '!#$%&\)*+,-./:;<=>?@[\\]^_`{|}~'
    if unidecode.unidecode(text[0]) in punct:
        return True
    return False

acceptable_punctuation = ['.', '?', '!', '...', ')']
import re
def clean_punct_text(text):
    checked = 0
    text = text.replace('»', '')
    text = text.replace('<unk>', '')
    while True:
        if len(text) == 0:
            return ''
        w = int(len(text) * .2)
        while text[-1] not in acceptable_punctuation:
            text = text[:-1]
            checked = 0
        p = count_punct(text[-w:])
        n = len(text[-w:])
        if ((p / n) > .1) and not p <= 3:
            text = text[:-1]
            checked = 0
        if text.endswith('..'):
            text = text[:-1]
            checked = 0
        if text.endswith('. .'):
            text = text[:-1]
            checked = 0
        if starting_punct(text[0]):
            text = text[1:]
        checked += 1
        if checked > 5:
            break
    return text