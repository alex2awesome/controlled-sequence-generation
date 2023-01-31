"""
Example command with bag of words:
python examples/generator_module.py -B space --cond_text "The president" --length 100 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.01 --window_length 5 --kl_scale 0.01 --gm_scale 0.95

Example command with discriminator:
python examples/generator_module.py -D sentiment --class_label 3 --cond_text "The lake" --length 10 --gamma 1.0 --num_iterations 30 --num_samples 10 --stepsize 0.01 --kl_scale 0.01 --gm_scale 0.95
"""

import argparse
import logging
import numpy as np
import torch

import generator.generator_pplm.utils_generator as ugen_pplm
import generator.generator_fudge.utils_generator as ugen_fudge
import generator.generator_baselines.utils_generator as ugen_baseline
import generator.utils_general as ug
import util.utils_data_access

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s(%(lineno)s): %(message)s",
)

# set seeds
torch.manual_seed(1234)
np.random.seed(1234)

experiments = {
    'perturbed_pplm': ugen_pplm.GenerationPPLM,
    'perturbed_fudge': ugen_fudge.GenerationFUDGE,
    'unperturbed': ugen_baseline.GenerationUnperturbed,
    'baselinetwo-baseline': ugen_baseline.GenerationBaselineTwo,
    'baselinetwo-past': ugen_baseline.GenerationBaselineTwo,
    'baselinetwo-future': ugen_baseline.GenerationBaselineTwo,
    'human': ugen_baseline.GenerationHuman,
}


if __name__ == '__main__':
    from generator.utils_argparse import attach_model_args_generator
    parser = argparse.ArgumentParser()
    parser = attach_model_args_generator(parser)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('SYSTEM DEVICE: %s' % device)
    args = parser.parse_args()
    logging.info(f"Args: {args}")
    if args.remote_log_path is not None:
        from bloomberg.ai.logtools.dsp_logger import setup_logger
        setup_logger(hdfs_location=args.remote_log_path, level=logging.INFO)

    if args.notes is not None:
        logging.info('------------NOTES--------------')
        logging.info(args.notes)

    # Download files
    args = util.utils_data_access.download_all_necessary_files(args)

    # Instantiate Tensorboard
    tb_logger = ug.get_tb_logger(env=args.env)

    if args.config_path is not None:
        from discriminator.config_helper import TransformersConfig
        config = TransformersConfig.from_json_file(args.config_path)
        config.generate = True
    else:
        config = None

    # Instantiate Generator
    generator_class = experiments[args.generator_type]
    generator = generator_class(device=device, tb_logger=tb_logger, discrim_config=config, **vars(args))

    # Choose the document index
    all_documents = []
    if args.use_real_doc_idxes is not None:
        import pandas as pd
        # read data and group documents
        data_file = pd.read_csv(args.real_data_file)
        file_idxs = list(
            data_file
                 .loc[lambda df: df['file'].str.contains('/test/')]
                 .assign(s_id=lambda df: df['s_id'].str.replace('S', '').astype(int))
                 .sort_values('s_id')
                 .groupby('file')
                 .groups.values()
         )

        for doc_idx in args.use_real_doc_idxes:
            # get indexed document, headline and tags
            one_doc = data_file.loc[file_idxs[int(doc_idx)]]
            # input
            args.cond_text = one_doc['headline'].drop_duplicates().iloc[0]
            one_doc = one_doc.loc[lambda df: df['event_tag'].notnull()]
            args.cond_text = args.cond_text if args.cond_text.endswith('.<|endoftext|>') else args.cond_text + '.<|endoftext|>'
            args.class_labels = one_doc['event_tag'].tolist()
            sentences = one_doc['sentence'].tolist()
            args.sentences = list(map(lambda s: s if s.endswith('<|endoftext|>') else s + '<|endoftext|>', sentences))

            # run generation
            results = generator.generate_document(**vars(args))
            document = results.get_final_document()
            all_documents.append(document)
            logging.info('FINAL DOCUMENT: %s' % str(document))
            # results.print_tables(print_sentence_table=True, print_word_table=False)
            logging.info('WORD METRICS: %s' % str(results.get_final_word_metrics()))
            # print metrics
            # logging.info('ALL OUTPUT: %s' % str(metrics.dictionary_formatted_output))
            # all_unperturbed_metrics = list(map(fix_floats, all_unperturbed_metrics))
            # all_perturbed_metrics = list(map(fix_floats, all_perturbed_metrics))
            # logging.info('ALL UNPERT METRICS: %s' % str(all_unperturbed_metrics))
            # logging.info('ALL PERT METRICS: %s' % str(all_perturbed_metrics))

    else:
        args.class_labels = list(map(int, args.class_labels))
        metrics = generator.generate_document(**vars(args))

    if args.output_filename is not None:
        import json, os
        fn = os.path.basename(args.output_filename)
        with open(fn, 'w') as f:
            json.dump(all_documents, f)
        if args.env == 'bb':
            util.utils_data_access.upload_file_to_filepath(fn, args.output_filename)




# todo: how to handle generating multiple samples - is it useful?
# A, A1, A2, A3
#     |   |   |
# B, B1, B2, B3
# at each sentence, you have multiple options for the conditioned text
# if we generate multiple sentences, we can possibly compare and take the one we like the most.
# with multiple samples, we can see how PPLM diversifies the generation.
# We can do some beam search - only keep the top 5 most likely samples, so we don't do exponential computation.
# is there a lot of variance?
