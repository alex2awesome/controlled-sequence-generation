#! /usr/bin/env python3
# coding=utf-8

# This code is licensed under a non-commercial license.
import argparse
import csv
import json
import math
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
import torch.optim
import torch.optim as optim
import torch.utils.data as data
from nltk.tokenize.treebank import TreebankWordDetokenizer
from torchtext import data as torchtext_data
from torchtext import datasets
from tqdm import tqdm, trange
from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from baselines.PPLM.pplm_classification_head import ClassificationHead
from baselines.PPLM.util import attach_model_args_pplm_discrim_train, reformat_model_path
from sklearn.metrics import f1_score, classification_report

torch.manual_seed(0)
np.random.seed(0)
EPSILON = 1e-10
example_sentence = "This is incredible! I love it, this is the best chicken I have ever had."
max_length_seq = 100


class Discriminator(torch.nn.Module):
    """Transformer encoder followed by a Classification Head"""

    def __init__(
            self,
            class_size=None,
            pretrained_model="gpt2-medium",
            model_type='gpt2',
            classifier_head=None,
            cached_mode=False,
            device='cpu',
            **kwargs
    ):
        super(Discriminator, self).__init__()
        if model_type == "gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained(reformat_model_path(pretrained_model))
            self.encoder = GPT2LMHeadModel.from_pretrained(reformat_model_path(pretrained_model))
            self.embed_size = self.encoder.transformer.config.hidden_size
        elif model_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(reformat_model_path(pretrained_model))
            self.encoder = BertModel.from_pretrained(reformat_model_path(pretrained_model))
            self.embed_size = self.encoder.config.hidden_size
        else:
            raise ValueError(
                "{} model not yet supported".format(model_type)
            )
        if classifier_head:
            self.classifier_head = classifier_head
        else:
            if not class_size:
                raise ValueError("must specify class_size")
            self.classifier_head = ClassificationHead(
                class_size=class_size,
                embed_size=self.embed_size
            )
        self.cached_mode = cached_mode
        self.device = device

        ## freeze layers
        self.model_type = model_type
        self.freeze_transformer = kwargs.get('freeze_transformer')
        self.freeze_embedding_layer = kwargs.get('freeze_embedding_layer')
        self.freeze_positional_layer = kwargs.get('freeze_positional_layer')
        self.freeze_encoder_layers = kwargs.get('freeze_encoder_layers')

    def get_classifier(self):
        return self.classifier_head

    def train_custom(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier_head.train()

    def avg_representation(self, x):
        mask = (x.ne(0)
                .unsqueeze(2)
                .repeat(1, 1, self.embed_size)
                .float()
                .to(self.device)
                .detach()
                )
        if hasattr(self.encoder, 'transformer'):
            # for gpt2
            hidden, _ = self.encoder.transformer(x)
        else:
            # for bert
            hidden, _ = self.encoder(x)
        masked_hidden = hidden * mask
        avg_hidden = torch.sum(masked_hidden, dim=1) / (torch.sum(mask, dim=1).detach() + EPSILON)
        return avg_hidden

    def forward(self, x):
        if self.cached_mode:
            avg_hidden = x.to(self.device)
        else:
            avg_hidden = self.avg_representation(x.to(self.device))

        logits = self.classifier_head(avg_hidden)
        probs = F.log_softmax(logits, dim=-1)

        return probs

    def predict(self, input_sentence):
        input_t = self.tokenizer.encode(input_sentence)
        input_t = torch.tensor([input_t], dtype=torch.long, device=self.device)
        if self.cached_mode:
            input_t = self.avg_representation(input_t)

        log_probs = self(input_t).data.cpu().numpy().flatten().tolist()
        prob = [math.exp(log_prob) for log_prob in log_probs]
        return prob

    def freeze_encoder_layers(self):
        # freeze whole transformer
        if self.freeze_transformer:
            for p in self.encoder.params():
                p.requires_grad = False
            return

        # freeze embedding layer
        if self.freeze_embedding_layer:
            if self.model_type == 'gpt2':
                for p in self.encoder.wte.parameters():
                    p.requires_grad = False
            else:
                for p in self.encoder.embeddings.parameters():
                    p.requires_grad = False

        # freeze encoding layers
        if self.freeze_encoder_layers:
            for layer in self.config.freeze_encoder_layers:
                if self.model_type == 'gpt2':
                    for p in self.encoder.h[int(layer)].parameters():
                        p.requires_grad = False
                else:
                    for p in self.encoder.encoder.layer[int(layer)].parameters():
                        p.requires_grad = False


class Dataset(data.Dataset):
    def __init__(self, X, y):
        """Reads source and target sequences from txt files."""
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        data = {}
        data["X"] = self.X[index]
        data["y"] = self.y[index]
        return data


def collate_fn(data):
    def pad_sequences(sequences):
        lengths = [len(seq) for seq in sequences]

        padded_sequences = torch.zeros(
            len(sequences),
            max(lengths)
        ).long()  # padding value = 0

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_sequences[i, :end] = seq[:end]

        return padded_sequences, lengths

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    x_batch, _ = pad_sequences(item_info["X"])
    y_batch = torch.tensor(item_info["y"], dtype=torch.long)

    return x_batch, y_batch


def cached_collate_fn(data):
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    x_batch = torch.cat(item_info["X"], 0)
    y_batch = torch.tensor(item_info["y"], dtype=torch.long)

    return x_batch, y_batch


def train_epoch(data_loader, discriminator, optimizer, epoch=0, log_interval=10, device='cpu'):
    samples_so_far = 0
    discriminator.train_custom()
    for batch_idx, (input_t, target_t) in enumerate(data_loader):
        input_t, target_t = input_t.to(device), target_t.to(device)

        optimizer.zero_grad()

        output_t = discriminator(input_t)
        loss = F.nll_loss(output_t, target_t)
        loss.backward(retain_graph=True)
        optimizer.step()

        samples_so_far += len(input_t)

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch + 1,
                    samples_so_far, len(data_loader.dataset),
                    100 * samples_so_far / len(data_loader.dataset), loss.item()
                )
            )


def evaluate_performance(data_loader, discriminator, device='cpu'):
    discriminator.eval()
    test_loss = 0
    correct = 0

    y_true, y_pred = [], []
    with torch.no_grad():
        for input_t, target_t in data_loader:
            input_t, target_t = input_t.to(device), target_t.to(device)
            output_t = discriminator(input_t)
            # sum up batch loss
            test_loss += F.nll_loss(output_t, target_t, reduction="sum").item()
            # get the index of the max log-probability
            pred_t = output_t.argmax(dim=1, keepdim=True)
            correct += pred_t.eq(target_t.view_as(pred_t)).sum().item()

            y_pred += pred_t.detach().cpu().numpy().flatten().tolist()
            y_true += target_t.detach().cpu().numpy().flatten().tolist()

    test_loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)

    print(
        "Performance on test set: "
        "Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss, correct, len(data_loader.dataset),
            100. * accuracy
        )
    )
    print('F1 Score Weighted: %s' % f1_score(y_true=y_true, y_pred=y_pred, average='weighted'))
    print('F1 Score Macro: %s' % f1_score(y_true=y_true, y_pred=y_pred, average='macro'))
    print('Classification Report:')
    print(classification_report(y_true=y_true, y_pred=y_pred))

    return test_loss, accuracy


def predict(input_sentence, model, classes, cached=False, device='cpu'):
    input_t = model.tokenizer.encode(input_sentence)
    input_t = torch.tensor([input_t], dtype=torch.long, device=device)
    if cached:
        input_t = model.avg_representation(input_t)

    log_probs = model(input_t).data.cpu().numpy().flatten().tolist()
    print("Input sentence:", input_sentence)
    print("Predictions:", ", ".join(
        "{}: {:.4f}".format(c, math.exp(log_prob)) for c, log_prob in
        zip(classes, log_probs)
    ))


def get_cached_data_loader(dataset, batch_size, discriminator, shuffle=False, device='cpu'):
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn)

    xs = []
    ys = []
    for batch_idx, (x, y) in enumerate(tqdm(data_loader, ascii=True)):
        with torch.no_grad():
            x = x.to(device)
            avg_rep = discriminator.avg_representation(x).cpu().detach()
            avg_rep_list = torch.unbind(avg_rep.unsqueeze(1))
            xs += avg_rep_list
            ys += y.cpu().numpy().tolist()

    data_loader = torch.utils.data.DataLoader(
        dataset=Dataset(xs, ys),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=cached_collate_fn)

    return data_loader


def get_idx2class(dataset_fp):
    classes = set()
    with open(dataset_fp) as f:
        csv_reader = csv.reader(f, delimiter="\t")
        for row in tqdm(csv_reader, ascii=True):
            if row:
                classes.add(row[0])
    return sorted(classes)


def get_generic_dataset(dataset_fp, tokenizer, device, idx2class=None, add_eos_token=False):
    if not idx2class:
        idx2class = get_idx2class(dataset_fp)
    class2idx = {c: i for i, c in enumerate(idx2class)}

    x, y = [], []
    with open(dataset_fp) as f:
        csv_reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(tqdm(csv_reader, ascii=True)):
            if row:
                label = row[0]
                text = row[1]

                try:
                    seq = tokenizer.encode(text)
                    # if (len(seq) < max_length_seq):
                    if add_eos_token:
                        seq = [50256] + seq
                    seq = torch.tensor(seq, device=device, dtype=torch.long)
                    seq = seq[:max_length_seq]

                    # else:
                    #     print("Line {} is longer than maximum length {}".format(i, max_length_seq))
                    #     continue

                    x.append(seq)
                    y.append(class2idx[label])

                except:
                    print("Error tokenizing line {}, skipping it".format(i))
                    pass

    return Dataset(x, y)


def train_discriminator(
        dataset,
        dataset_fp=None,
        model_type='gpt2',
        pretrained_model="gpt2-medium",
        epochs=10,
        learning_rate=0.0001,
        batch_size=64,
        log_interval=10,
        save_model=False,
        cached=False,
        no_cuda=False,
        output_fp='.',
        env='local',
        **kwargs
):
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
    add_eos_token = model_type == "gpt2"

    if save_model:
        if not os.path.exists(output_fp):
            os.makedirs(output_fp)

    # output file-paths
    classifier_head_meta_fp = os.path.join(
        output_fp, "{}_classifier_head_meta.json".format(dataset)
    )
    classifier_head_fp_pattern = os.path.join(
        output_fp, "{}_classifier_head_epoch".format(dataset) + "_{}.pt"
    )

    print("Preprocessing {} dataset...".format(dataset))
    start = time.time()

    if dataset == "SST":
        idx2class = ["positive", "negative", "very positive", "very negative", "neutral"]
        class2idx = {c: i for i, c in enumerate(idx2class)}

        discriminator = Discriminator(
            class_size=len(idx2class),
            model_type=model_type,
            pretrained_model=pretrained_model,
            cached_mode=cached,
            device=device
        ).to(device)

        text = torchtext_data.Field()
        label = torchtext_data.Field(sequential=False)
        train_data, val_data, test_data = datasets.SST.splits(
            text,
            label,
            fine_grained=True,
            train_subtrees=True,
        )

        x = []
        y = []
        for i in trange(len(train_data), ascii=True):
            seq = TreebankWordDetokenizer().detokenize(
                vars(train_data[i])["text"]
            )
            seq = discriminator.tokenizer.encode(seq)
            if add_eos_token:
                seq = [50256] + seq
            seq = torch.tensor(seq, device=device, dtype=torch.long)
            x.append(seq)
            y.append(class2idx[vars(train_data[i])["label"]])
        train_dataset = Dataset(x, y)

        test_x = []
        test_y = []
        for i in trange(len(test_data), ascii=True):
            seq = TreebankWordDetokenizer().detokenize(
                vars(test_data[i])["text"]
            )
            seq = discriminator.tokenizer.encode(seq)
            if add_eos_token:
                seq = [50256] + seq
            seq = torch.tensor(seq, device=device, dtype=torch.long)
            test_x.append(seq)
            test_y.append(class2idx[vars(test_data[i])["label"]])
        test_dataset = Dataset(test_x, test_y)

        discriminator_meta = {
            "class_size": len(idx2class),
            "embed_size": discriminator.embed_size,
            'model_type': model_type,
            "pretrained_model": pretrained_model,
            "class_vocab": class2idx,
            "default_class": 2,
        }

    elif dataset == "clickbait":
        idx2class = ["non_clickbait", "clickbait"]
        class2idx = {c: i for i, c in enumerate(idx2class)}

        discriminator = Discriminator(
            class_size=len(idx2class),
            model_type=model_type,
            pretrained_model=pretrained_model,
            cached_mode=cached,
            device=device
        ).to(device)

        with open("datasets/clickbait/clickbait.txt") as f:
            data = []
            for i, line in enumerate(f):
                try:
                    data.append(eval(line))
                except:
                    print("Error evaluating line {}: {}".format(
                        i, line
                    ))
                    continue
        x = []
        y = []
        with open("datasets/clickbait/clickbait.txt") as f:
            for i, line in enumerate(tqdm(f, ascii=True)):
                try:
                    d = eval(line)
                    seq = discriminator.tokenizer.encode(d["text"])

                    if len(seq) < max_length_seq:
                        if add_eos_token:
                            seq = [50256] + seq
                        seq = torch.tensor(
                            seq, device=device, dtype=torch.long
                        )
                    else:
                        print("Line {} is longer than maximum length {}".format(
                            i, max_length_seq
                        ))
                        continue
                    x.append(seq)
                    y.append(d["label"])
                except:
                    print("Error evaluating / tokenizing"
                          " line {}, skipping it".format(i))
                    pass

        full_dataset = Dataset(x, y)
        train_size = int(0.9 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size]
        )

        discriminator_meta = {
            "class_size": len(idx2class),
            "embed_size": discriminator.embed_size,
            'model_type': model_type,
            "pretrained_model": pretrained_model,
            "class_vocab": class2idx,
            "default_class": 1,
        }

    elif dataset == "toxic":
        idx2class = ["non_toxic", "toxic"]
        class2idx = {c: i for i, c in enumerate(idx2class)}

        discriminator = Discriminator(
            class_size=len(idx2class),
            model_type=model_type,
            pretrained_model=pretrained_model,
            cached_mode=cached,
            device=device
        ).to(device)

        x = []
        y = []
        with open("datasets/toxic/toxic_train.txt") as f:
            for i, line in enumerate(tqdm(f, ascii=True)):
                try:
                    d = eval(line)
                    seq = discriminator.tokenizer.encode(d["text"])

                    if len(seq) < max_length_seq:
                        if add_eos_token:
                            seq = [50256] + seq
                        seq = torch.tensor(
                            seq, device=device, dtype=torch.long
                        )
                    else:
                        print("Line {} is longer than maximum length {}".format(
                            i, max_length_seq
                        ))
                        continue
                    x.append(seq)
                    y.append(int(np.sum(d["label"]) > 0))
                except:
                    print("Error evaluating / tokenizing"
                          " line {}, skipping it".format(i))
                    pass

        full_dataset = Dataset(x, y)
        train_size = int(0.9 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size]
        )

        discriminator_meta = {
            "class_size": len(idx2class),
            "embed_size": discriminator.embed_size,
            'model_type': model_type,
            "pretrained_model": pretrained_model,
            "class_vocab": class2idx,
            "default_class": 0,
        }

    else:  # if dataset == "generic":
        # This assumes the input dataset is a TSV with the following structure:
        # class \t text

        if dataset_fp is None:
            raise ValueError("When generic dataset is selected, "
                             "dataset_fp needs to be specified as well.")

        idx2class = get_idx2class(dataset_fp)

        discriminator = Discriminator(
            class_size=len(idx2class),
            model_type=model_type,
            pretrained_model=pretrained_model,
            cached_mode=cached,
            device=device,
            # freeze model layers
            freeze_transformer=kwargs.get('freeze_transformer'),
            freeze_embedding_layer=kwargs.get('freeze_embedding_layer'),
            freeze_positional_layer=kwargs.get('freeze_positional_layer'),
            freeze_encoder_layers=kwargs.get('freeze_encoder_layers'),
        ).to(device)

        full_dataset = get_generic_dataset(
            dataset_fp, discriminator.tokenizer, device,
            idx2class=idx2class, add_eos_token=add_eos_token
        )
        train_size = int(0.9 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, test_size]
        )

        discriminator_meta = {
            "class_size": len(idx2class),
            "embed_size": discriminator.embed_size,
            'model_type': model_type,
            "pretrained_model": pretrained_model,
            "class_vocab": {c: i for i, c in enumerate(idx2class)},
            "default_class": 0,
        }

    end = time.time()
    print("Preprocessed {} data points".format(len(train_dataset) + len(test_dataset)))
    print("Data preprocessing took: {:.3f}s".format(end - start))

    if cached:
        print("Building representation cache...")

        start = time.time()

        train_loader = get_cached_data_loader(
            train_dataset, batch_size, discriminator,
            shuffle=True, device=device
        )

        test_loader = get_cached_data_loader(
            test_dataset, batch_size, discriminator, device=device
        )

        end = time.time()
        print("Building representation cache took: {:.3f}s".format(end - start))

    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn)

    if save_model:
        with open(classifier_head_meta_fp, "w") as meta_file:
            json.dump(discriminator_meta, meta_file)
        if env == 'bb':
            fs = get_fs()
            remote_path = os.path.join('aspangher', 'controlled-sequence-gen', classifier_head_meta_fp)
            print('uploading model config file to: %s...' % classifier_head_meta_fp)
            fs.put(classifier_head_meta_fp, remote_path)


    optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

    test_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        start = time.time()
        print("\nEpoch", epoch + 1)

        train_epoch(
            discriminator=discriminator,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            log_interval=log_interval,
            device=device
        )
        test_loss, test_accuracy = evaluate_performance(
            data_loader=test_loader,
            discriminator=discriminator,
            device=device
        )

        end = time.time()
        print("Epoch took: {:.3f}s".format(end - start))

        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print("\nExample prediction")
        predict(example_sentence, discriminator, idx2class,
                cached=cached, device=device)

        if save_model:
            classifier_head_fp = classifier_head_fp_pattern.format(epoch + 1)
            torch.save(
                discriminator.get_classifier().state_dict(),
                classifier_head_fp
            )
            if env == 'bb':
                fs = get_fs()
                print('uploading model file to: %s...' % classifier_head_fp)
                remote_path = os.path.join('aspangher', 'controlled-sequence-gen', classifier_head_fp)
                fs.put(classifier_head_fp, remote_path)

    min_loss = float("inf")
    min_loss_epoch = 0
    max_acc = 0.0
    max_acc_epoch = 0
    print("Test performance per epoch")
    print("epoch\tloss\tacc")
    for e, (loss, acc) in enumerate(zip(test_losses, test_accuracies)):
        print("{}\t{}\t{}".format(e + 1, loss, acc))
        if loss < min_loss:
            min_loss = loss
            min_loss_epoch = e + 1
        if acc > max_acc:
            max_acc = acc
            max_acc_epoch = e + 1
    print("Min loss: {} - Epoch: {}".format(min_loss, min_loss_epoch))
    print("Max acc: {} - Epoch: {}".format(max_acc, max_acc_epoch))

    return discriminator, discriminator_meta


def load_classifier_head(weights_path, meta_path, device='cpu'):
    with open(meta_path, 'r', encoding="utf8") as f:
        meta_params = json.load(f)
    classifier_head = ClassificationHead(
        class_size=meta_params['class_size'],
        embed_size=meta_params['embed_size']
    ).to(device)
    classifier_head.load_state_dict(
        torch.load(weights_path, map_location=device))
    classifier_head.eval()
    return classifier_head, meta_params


def load_discriminator(weights_path, meta_path, device='cpu'):
    classifier_head, meta_param = load_classifier_head(
        weights_path, meta_path, device
    )
    discriminator = Discriminator(
        model_type=meta_param['model_type'],
        pretrained_model=meta_param['pretrained_model'],
        classifier_head=classifier_head,
        cached_mode=False,
        device=device
    )
    return discriminator, meta_param


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a discriminator on top of GPT-2 representations")
    parser = attach_model_args_pplm_discrim_train(parser)
    args = parser.parse_args()
    # download model
    if args.env == 'bb':
        os.environ['env'] = 'bb'
        from util.util_data_access import get_fs, download_model_files_bb, download_file_to_filepath
        print('downloading model files from s3...')
        download_model_files_bb(remote_model=args.pretrained_model)
        print('downloading data files from s3...')
        download_file_to_filepath(args.dataset_fp)

    train_discriminator(**(vars(args)))
