import os, sys, numpy as np, pickle, argparse
from sklearn.neighbors import kneighbors_graph

import torch


def main(args):
	device = torch.device("cpu")

	if args.nn == 'distilbert':
		from explanation.directed_integrated_gradients.distilbert_helper import nn_init, get_word_embeddings
	elif args.nn == 'roberta':
		from explanation.directed_integrated_gradients.roberta_helper import nn_init, get_word_embeddings
	elif args.nn == 'gpt2':
		from explanation.directed_integrated_gradients.gpt2_helper import nn_init, get_word_embeddings
	elif args.nn == 'bert':
		from explanation.directed_integrated_gradients.bert_helper import nn_init, get_word_embeddings

	print(f'Starting KNN computation..')

	model, tokenizer	= nn_init(device, args.dataset, returns=True)
	word_features		= get_word_embeddings().cpu().detach().numpy()
	word_idx_map		= tokenizer.get_vocab()
	A					= kneighbors_graph(word_features, args.nbrs, mode='distance', n_jobs=args.procs)

	with open(args.output_filename, 'wb') as f:
		pickle.dump([word_idx_map, word_features, A], f)

	print(f'Written KNN data at {args.output_filename}')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='knn')
	parser.add_argument('--nn',    	 default='distilbert', choices=['distilbert', 'roberta', 'bert'])
	parser.add_argument('--dataset', default='sst2', choices=['sst2', 'imdb', 'rotten'])
	parser.add_argument('--procs',	 default=40, type=int)
	parser.add_argument('--nbrs',  	 default=500, type=int)
	parser.add_argument('--output_filename', type=str)

	args = parser.parse_args()

	main(args)
