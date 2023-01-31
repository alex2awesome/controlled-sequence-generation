import sys, numpy as np, argparse, random
sys.path.append('../')

from tqdm import tqdm

import torch
from explanation.directed_integrated_gradients.dig import DiscretetizedIntegratedGradients
from explanation.directed_integrated_gradients.attributions import run_dig_explanation, make_visualization
from explanation.directed_integrated_gradients.metrics import eval_log_odds, eval_comprehensiveness, eval_sufficiency
import explanation.directed_integrated_gradients.monotonic_paths as monotonic_paths
import pickle
from util.utils_general import label_idx_to_str, label_str_to_idx

all_outputs = []

def calculate_attributions(inputs, device, args, attr_func, base_token_emb, nn_forward_func, label, type_embed=None):
    # computes the attributions for given input

    # move inputs to main device
    inp = [x.to(device) if x is not None else None for x in inputs]

    # compute attribution
    scaled_features, input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, attention_mask = inp
    attr, delta = run_dig_explanation(
        dig_func=attr_func,
        all_input_embed=scaled_features,
        position_embed=position_embed,
        attention_mask=attention_mask,
        steps=(2**args.factor)*(args.steps+1)+1,
        label=label
    )

    if args.compute_metrics:
        # compute metrics
        log_odd, pred	= eval_log_odds(nn_forward_func, input_embed, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=args.topk)
        comp			= eval_comprehensiveness(nn_forward_func, input_embed, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=args.topk)
        suff			= eval_sufficiency(nn_forward_func, input_embed, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=args.topk)

        return attr, delta, log_odd, comp, suff
    else:
        return attr, delta, None, None, None


def main(args, data, auxiliary_data):
    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # neural network specific imports
    if args.nn == 'distilbert':
        from explanation.directed_integrated_gradients.distilbert_helper import nn_forward_func, nn_init, get_inputs, get_base_token_emb
    elif args.nn == 'roberta':
        from explanation.directed_integrated_gradients.roberta_helper import nn_forward_func, nn_init, get_inputs, get_base_token_emb
    elif args.nn == 'bert':
        from explanation.directed_integrated_gradients.bert_helper import nn_forward_func, nn_init, get_inputs, get_base_token_emb
    elif args.nn == 'gpt2':
        from explanation.directed_integrated_gradients.gpt2_helper import nn_forward_func, nn_init, get_inputs, get_base_token_emb, get_model_prediction
    else:
        raise NotImplementedError

    # Fix the gpu to use
    device		= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('running on %s' % str(device))

    # init model and tokenizer in cpu first
    nn_init(args, device)

    # Define the Attribution function
    attr_func = DiscretetizedIntegratedGradients(nn_forward_func)

    # get ref token embedding
    base_token_emb = get_base_token_emb(device)

    # compute the DIG attributions for all the inputs
    print('Starting attribution computation...')
    log_odds, comps, suffs, count = 0, 0, 0, 0
    print_step = 2
    attributions = []
    for row in tqdm(data.itertuples(index=False)):
        text = row[0]
        if len(row) > 1:
            label = row[1]
            if isinstance(label, str):
                label = label_str_to_idx[label]
        else:
            label = None

        #
        (
            input_ids,
            ref_input_ids,
            input_embed,
            ref_input_embed,
            position_embed,
            ref_position_embed,
            attention_mask
        ) = get_inputs(text, device)
        #
        scaled_features 		= monotonic_paths.scale_inputs(
            input_ids.squeeze().tolist(),
            ref_input_ids.squeeze().tolist(),
            device,
            auxiliary_data,
            steps=args.steps,
            factor=args.factor,
            strategy=args.strategy
        )
        inputs					= [scaled_features, input_ids, ref_input_ids, input_embed, ref_input_embed,
                                     position_embed, ref_position_embed, attention_mask]
        ##
        #
        output    = calculate_attributions(
            inputs, device, args, attr_func, base_token_emb, nn_forward_func, label
        )

        if args.compute_metrics:
            attribution, delta, log_odd, comp, suff = output
            log_odds	+= log_odd
            comps		+= comp
            suffs 		+= suff
            count		+= 1

            # print the metrics
            if count % print_step == 0:
                print('Log-odds: ', np.round(log_odds / count, 4), 'Comprehensiveness: ', np.round(comps / count, 4), 'Sufficiency: ', np.round(suffs / count, 4))
        else:
            attribution, delta, _, _, _ = output

        if args.visualize:
            pred_ll, pred_ind = get_model_prediction(input_ids, label)
            attribution = make_visualization(
                attribution,
                delta,
                prediction=pred_ll,
                pred_label=pred_ind,
                true_label=label,
                target_label=label,
                label_idx_mapper=label_idx_to_str,
                text=text
            )

        attributions.append(attribution)

    if args.compute_metrics:
        print('Log-odds: ', np.round(log_odds / count, 4), 'Comprehensiveness: ', np.round(comps / count, 4), 'Sufficiency: ', np.round(suffs / count, 4))

    return attributions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IG Path')
    parser.add_argument('--dataset',    default='sst2', 		    choices=['sst2', 'imdb', 'ag', 'rotten', 'sst2_epoch'])
    parser.add_argument('--nn', 	    default='distilbert',    choices=['distilbert', 'roberta', 'lstm', 'bert', 'albert', 'gpt2'])
    parser.add_argument('--strategy',   default='greedy', 	    choices=['greedy', 'maxcount'], help='The algorithm to find the next anchor point')
    parser.add_argument('--steps',      default=30,              type=int)	# m
    parser.add_argument('--topk',  	    default=20,              type=int)	# k
    parser.add_argument('--factor',     default=0,               type=int)	# f
    parser.add_argument('--knn_nbrs',   default=500,             type=int)	# KNN
    parser.add_argument('--seed', 	    default=42,              type=int)
    parser.add_argument('--local',		action='store_true')
    parser.add_argument('--pretrained_lm_model_path',      type=str, default=None)
    parser.add_argument('--pretrained_discriminator_path', type=str, default=None)
    parser.add_argument('--real_data_file',                type=str)
    parser.add_argument('--auxiliary_data_file',           type=str)
    parser.add_argument('--config_path',                   type=str, default=None)
    parser.add_argument('--compute_metrics', action='store_true')
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()

    # get data
    if not args.local:
        from util.utils_data_access import download_all_necessary_files
        download_all_necessary_files(args)

    # load data
    import pandas as pd
    if 'tsv' in args.real_data_file:
        data = (
            pd.read_csv(args.real_data_file, sep='\t', header=None)[[1, 0]]
            .loc[lambda df: df[1] != 'Error']
        )
    else:
        data = (pd.read_csv(args.real_data_file)[['sentence', 'event_tag']]
            .loc[lambda df: df['event_tag'] != 'Error']
        )


    with open(args.auxiliary_data_file, 'rb') as f:
        auxiliary_data = pickle.load(f)

    main(args, data=data, auxiliary_data=auxiliary_data)
