import numpy as np
import csv
import random
import os

def process_line(line, range_arr, topics):
    line = line.split('\t')
    if not (len(line) == 3):
        print("skipping " + str(line))
        return None, None

    ## label
    label = line[0]
    if label[0] == "\"":
        label = label[1:-1]
    label = int(label) - 1

    ## text
    text = line[2]
    if text[0] == "\"": ## remove quotes
        text = text[1:-1]
    if text[0] == " ": ## remote whitespace
        text = text.strip()

    ## sample positive and negative for output
    choice_array = range_arr[:label] + range_arr[label + 1:]
    not_label = random.choice(choice_array)
    true_ex = topics[label] + text
    false_ex = topics[not_label] + text
    return true_ex, false_ex

def proc_and_binarize(dir):
    fid = open(os.path.join(dir, "train.tsv"))
    train = fid.read().strip().split("\n")

    fid = open(os.path.join(dir, "dev.tsv"))
    test = fid.read().strip().split("\n")

    topics = open(os.path.join(dir, 'labels.txt'))
    topics = topics.read().strip().split('\n')

    true_test = []
    false_test = []

    true_train = []
    false_train = []

    range_arr = list(range(0, len(topics)))
    for line in test:
        true_ex, false_ex = process_line(line, range_arr, topics)
        if true_ex is not None and false_ex is not None:
            true_test.append(true_ex)
            false_test.append(false_ex)

    for line in train:
        true_ex, false_ex = process_line(line, range_arr, topics)
        if true_ex is not None and false_ex is not None:
            true_train.append(true_ex)
            false_train.append(false_ex)

    return true_train, false_train, true_test, false_test


def process_batch(true_batch, false_batch, batch_type='train'):
    true_lines, false_lines = [], []
    random.shuffle(true_batch)
    random.shuffle(false_batch)

    true_n = len(true_batch) if batch_type == 'train' else len(false_batch)
    false_n = len(false_batch)

    for i in range(0, false_n):
        false_lines.append(false_batch[i] + "\t0" + "\n")
    for i in range(0, true_n):
        true_lines.append(true_batch[i] + "\t1" + "\n")

    all_lines = true_lines + false_lines
    random.shuffle(all_lines)
    return all_lines


def main(args):
    import os
    here = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.join(here, '../..')


    train_file_csv = os.path.join(root_dir, args.train_file)
    train_file_tsv = train_file_csv.replace('.csv', '.tsv')
    csv.writer(open(train_file_tsv, 'w+'), delimiter='\t').writerows(csv.reader(open(train_file_csv)))
    test_file_csv = os.path.join(root_dir, args.test_file)
    test_file_tsv = test_file_csv.replace('.csv', '.tsv').replace('test', 'dev')
    csv.writer(open(test_file_tsv, 'w+'), delimiter='\t').writerows(csv.reader(open(test_file_csv)))

    data_dir = os.path.dirname(train_file_csv)
    true_train, false_train, true_test, false_test = proc_and_binarize(data_dir)

    test_lines = process_batch(true_test, false_test, batch_type='test')
    train_lines = process_batch(true_train, false_train, batch_type='train')

    train_split_all = "\n" + "".join(train_lines)
    test_split_all  = "\n" + "".join(test_lines)

    fid = open(train_file_tsv, 'w')
    fid.write(train_split_all)
    fid.close()

    fid = open(test_file_tsv, 'w')
    fid.write(test_split_all)
    fid.close()


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_file',
        type=str,
        default='baselines/GeDi/data/AG-news/train.csv',
        help='''
            location of the training data file (relative to root). 
            File format: Expects a CSV file with three columns:
                label, headline, text.
            Headline will be skipped.
            No headers.
            Labels should be categorical, with a labels file containing the necessary line-indicated mapping.
        '''
    )
    parser.add_argument(
        '--test_file',
        type=str,
        default='baselines/GeDi/data/AG-news/test.csv',
        help='location of the testing data file (relative to root)'
    )
    parser.add_argument(
        '--label_file',
        type=str,
        default='baselines/GeDi/data/AG-news/labels.txt',
        help='location of the labels file. labels will be in a line-separated list.'
    )
    args = parser.parse_args()

    main(args)
