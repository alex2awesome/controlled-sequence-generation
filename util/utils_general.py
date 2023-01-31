import csv
import gzip
import os

import torch

CLASS_MAPPING = [
    {'class_label': 0,  'event_tag': 'Cause_General',                       'avg_sentence_len': 23},
    {'class_label': 1,  'event_tag': 'Cause_Specific',                      'avg_sentence_len': 26},
    {'class_label': 2,  'event_tag': 'Distant_Anecdotal',                   'avg_sentence_len': 19},
    {'class_label': 3,  'event_tag': 'Distant_Evaluation',                  'avg_sentence_len': 22},
    {'class_label': 4,  'event_tag': 'Distant_Expectations_Consequences',   'avg_sentence_len': 25},
    {'class_label': 5,  'event_tag': 'Distant_Historical',                  'avg_sentence_len': 22},
    {'class_label': 6,  'event_tag': 'Main',                                'avg_sentence_len': 30},
    {'class_label': 7,  'event_tag': 'Main_Consequence',                    'avg_sentence_len': 22},
    {'class_label': 8,  'event_tag': 'Error',                               'avg_sentence_len': 22},
    {'class_label': 9,  'event_tag': 'headline',                            'avg_sentence_len': 22},
    {'class_label': 10,  'event_tag': '<start>',                            'avg_sentence_len': 22},
    {'class_label': 11,  'event_tag': '<end>',                              'avg_sentence_len': 22},
]
label_idx_to_str = {x['class_label']: x['event_tag'] for x in CLASS_MAPPING}
label_str_to_idx = {x['event_tag']: x['class_label'] for x in CLASS_MAPPING}


def reformat_model_path(x, args=None):
    fp_marker = './'
    if (
            (os.environ.get('env') == 'bb' or (args is not None and getattr(args, 'env', 'local') == 'bb'))
            and (not x.startswith(fp_marker))
    ):
        return os.path.join(fp_marker, x)
    else:
        return x


def get_idx2class(dataset_fp, config=None, use_headline=False):
    if (config is not None) and config.do_multitask:
        tasks_idx2class = []
        with get_fh(dataset_fp) as f:
            csv_reader = csv.reader(f, delimiter="\t")
            for row_idx, row in enumerate(csv_reader):
                if row:
                    ls = row[0].split('|||')
                    if row_idx == 0:
                        for _ in ls:
                            tasks_idx2class.append(set())
                    for l_idx, l in enumerate(ls):
                        tasks_idx2class[l_idx].add(l)
        tasks_class2idx = list(map(lambda idx2class: {v:k for k,v in enumerate(idx2class)}, tasks_idx2class))
        return tasks_idx2class, tasks_class2idx
    else:
        classes = set()
        with get_fh(dataset_fp) as f:
            csv_reader = csv.reader(f, delimiter="\t")
            for row in csv_reader:
                if row:
                    classes.add(row[0])
        if use_headline or ((config is not None) and config.use_headline):
            classes.add('headline')
        idx2class = sorted(classes)
        class2idx = {v:k for k,v in enumerate(idx2class)}
        return idx2class, class2idx


def _get_attention_mask(x, max_length_seq=float('inf')):
    max_len = max(map(lambda y: _get_len(y), x))
    max_len = min(max_len, max_length_seq)
    attention_masks = []
    for x_i in x:
        input_len = _get_len(x_i)
        if input_len < max_length_seq:
            mask = torch.cat((torch.ones(input_len), torch.zeros(max_len - input_len)))
        else:
            mask = torch.ones(max_length_seq)
        attention_masks.append(mask)
    return torch.stack(attention_masks)


def format_local_vars(locals):
    locals.pop('self', '')
    locals.pop('__class__', '')
    for k, v in locals.get('kwargs', {}).items():
        locals[k] = v
    return locals


def transpose_dict(dicts):
    """Take a dictionary in record-format and translate it into a columnar dict.

    [{'a': 1, 'b':2}, {'a':2, 'b':3}] -> {'a': [1,2], 'b': [2, 3]}
    """
    columns = {}
    for key in dicts[0].keys():
        columns[key] = list(map(lambda d: d[key], dicts))
    return columns


def _get_len(x):
    # x is already a length
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, list):
        return len(x)
    else:
        return x.shape.numel()


def get_fh(fp):
    if '.gz' in fp:
        fh = gzip.open(fp, 'rt')
    else:
        fh = open(fp)
    return fh