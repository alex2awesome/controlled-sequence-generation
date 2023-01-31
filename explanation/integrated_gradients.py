from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization
import torch
from transformers import AutoTokenizer
from discriminator.models_full import Discriminator as SequentialLSTMDiscriminator
from discriminator.utils_oldmethods import BaselineDiscriminator
from util.utils_general import label_str_to_idx, label_idx_to_str

import numpy as np
def get_model_prediction(input, model, label):
    output = model.forward(input, return_lls=True)
    if len(output) == 1:
        y_pred_lls = output
        y_pred_lls = y_pred_lls.detach().cpu().numpy()
        pred = np.argmax(y_pred_lls)
        return y_pred_lls, pred
    else:
        loss, pred, y_pred_lls = output
        return y_pred_lls.detach().cpu().numpy()[:, label], pred.detach().cpu().numpy().tolist()[0]


def interpret_sentence(model, sentence, tokenizer, lig, device, token_reference, min_len=7, label=0):
    text = tokenizer.encode(sentence)
    if len(text) < min_len:
        text += [0] * (min_len - len(text))
    input_indices = torch.tensor(text, device=device)
    input_indices = input_indices.unsqueeze(0)

    # generate reference indices for each sample
    seq_length = len(text)
    reference_indices = token_reference.generate_reference(seq_length, device=device).unsqueeze(0)

    # predict
    model.zero_grad()
    pred, pred_ind = get_model_prediction(input_indices, model, label)

    # compute attributions and approximation delta using layer integrated gradients
    attributions_ig, delta = lig.attribute(
        input_indices,
        reference_indices,
        n_steps=500,
        target=label,
        return_convergence_delta=True
    )

    if len(pred.shape) == 2:
        pred = pred.squeeze()

    print('pred: ', label_idx_to_str[pred_ind], '(', '%.2f' % pred, ')', ', delta: ', abs(delta))
    add_attributions_to_visualizer(attributions_ig, text, pred, pred_ind, label, delta, vis_data_records_ig)


def add_attributions_to_visualizer(attributions, text, pred, pred_ind, label, delta, vis_data_records):
    attributions = attributions.sum(dim=2).squeeze(0) # sum across embedding dimension.
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()

    # storing couple samples in an array for visualization purposes
    vis_data_records.append(visualization.VisualizationDataRecord(
        attributions,
        pred,
        label_idx_to_str[pred_ind],
        label_idx_to_str[label],
        label_idx_to_str[label],
        attributions.sum(),
        text,
        delta))


if __name__ == "__main__":
    import os
    import argparse
    from explanation.utils_argparse import attach_arguments
    parser = argparse.ArgumentParser()
    parser = attach_arguments(parser)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not args.local:
        from util.utils_data_access import download_all_necessary_files
        download_all_necessary_files(args)

    if args.config_path is not None:
        with open(args.config_path, 'r') as f:
            import json
            config = json.load(f)
    else:
        config = {}

    if args.model_type == 'gpt2':
        model_cls = SequentialLSTMDiscriminator
    else:
        model_cls = BaselineDiscriminator

    full_discriminator = (model_cls
                          .load_from_checkpoint(
                                checkpoint_path=args.discriminator_path,
                                loading_from_checkpoint=True,
                                pretrained_cache_dir=args.pretrained_model_path,
                                **config
                            )
                          )
    full_discriminator = full_discriminator.to(device)

    if 'roberta' in args.pretrained_model_path:
        embeddings = full_discriminator.transformer.encoder_model.embeddings
    else:
        embeddings = full_discriminator.transformer.encoder_model.transformer.wte

    lig = LayerIntegratedGradients(full_discriminator, embeddings)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    token_reference = TokenReferenceBase(reference_token_idx=0)

    for doc_idx in args.use_real_doc_idxes:
        import pandas as pd
        data_file = pd.read_csv(args.real_data_file, sep='\t', header=None)
        # get indexed document, headline and tags
        one_doc = data_file.iloc[int(doc_idx)]
        # input

    # accumalate couple samples in this array for visualization purposes
    vis_data_records_ig = []
    interpret_sentence(
        model=full_discriminator,
        sentence='This is a test',
        tokenizer=tokenizer,
        lig=lig,
        device=device,
        token_reference=token_reference
    )


#         # GPT2 sequential
#     # pretrained_model_name = 'gpt2-medium-expanded-embeddings'
#     # checkpoint_path = 'experiments/output_dir/trial-sequential_flattened_sentences__epoch=09-f1_macro=0.51.ckpt'
#
#     # RoBERTa
#     # pretrained_model_name = 'roberta-base'
#     # checkpoint_path = 'experiments/output_dir/roberta-nonsequential__balanced-training-data__epoch=02-f1_macro=0.56.ckpt'
#     # config_path = 'experiments/output_dir/config__baseline-discriminator-with-augmented-balanced-training-data.json'
