import torch
from fine_tuning.utils_data import (
    DiscourseFinetuningDataset,
    BaselineOneDataset,
    BaselineTwoDatasetBaseline,
    BaselineTwoDatasetPast,
    BaselineTwoDatasetFuture
)
from fine_tuning.language_models import LMModel, GPT2Wrapper, GPT2LMHeadModel, RobertaForMaskedLM
from fine_tuning.utils_config import get_config
from util.utils_general import reformat_model_path
from util.utils_data_access import get_fs, download_model_files_bb, download_file_to_filepath
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers
import os, glob

experiment_dict = {
    'discourse-roberta': ('roberta', DiscourseFinetuningDataset, LMModel),
    'discourse-gpt2': ('gpt2', DiscourseFinetuningDataset, LMModel),
    'baseline-one': ('gpt2', BaselineOneDataset, LMModel),
    'baseline-two-baseline': ('gpt2', BaselineTwoDatasetBaseline, LMModel),
    'baseline-two-past': ('gpt2', BaselineTwoDatasetPast, LMModel),
    'baseline-two-future': ('gpt2', BaselineTwoDatasetFuture, LMModel),
}
local_output_fp = './runs/'

def main(args, output_fp='.'):
    accelerator = 'dp'
    accelerator = accelerator if ((args.num_nodes > 1) or (args.num_gpus > 1)) else None

    if not os.path.exists(output_fp):
        os.makedirs(output_fp)

    lm_type, datasetclass, lm_class = experiment_dict[args.experiment]
    #
    config = get_config(args.pretrained_model_path, args)
    config.pretrained_cache_dir = args.pretrained_model_path
    config.use_cache = False

    #########
    # load dataset and model classes
    dataset = datasetclass(
        config=config,
        data_fp=config.dataset,
        pretrained_model_path=config.pretrained_cache_dir,
        num_cpus=config.num_dataloader_cpus,
        split_type=args.split_type,
        split_perc=.95,
        model_type=lm_type,
        batch_size=args.batch_size,
        max_length_seq=args.max_length_seq,
    )
    dataset.prepare_data()
    dataset.setup(stage='fit')
    config.num_steps_per_epoch = len(dataset.train_dataset)

    if lm_type == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained(config.pretrained_cache_dir, config=config)
    else:
        print(config)
        model = RobertaForMaskedLM.from_pretrained(config.pretrained_cache_dir, config=config)
    #
    model.resize_token_embeddings(len(dataset.tokenizer))
    lm = lm_class(config=config, model=model)  # our experimental setup


    #########
    # get TB logger
    if os.environ.get('TENSORBOARD_LOGDIR'):
        tb_logger = loggers.TensorBoardLogger(save_dir=os.environ['TENSORBOARD_LOGDIR'])
        tb_logger.log_hyperparams({
            'notes': args.notes,
            'embedding_model_type': config.model_type,
            'dataset_size': len(dataset.train_dataset),
            'experiment': args.experiment,
            # trainer params
            'batch_size': config.batch_size,
            'num_warmup_steps': config.num_warmup_steps,
            'learning_rate': config.learning_rate,
            'gradient_accumulation': config.accumulate_grad_batches,
        })
    else:
        tb_logger = None

    #################
    #  Train model
    checkpoint_callback = ModelCheckpoint(
        monitor='Validation Perplexity',
        dirpath=local_output_fp,
        filename='trial-%s__epoch={epoch:02d}-perplexity={Validation Perplexity:.2f}' % args.notes,
        save_top_k=1,
        mode='min',
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        gpus=args.num_gpus,
        num_nodes=args.num_nodes,
        accelerator=accelerator if not args.use_deepspeed else None,
        max_epochs=10,
        logger=tb_logger,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.max_grad_norm,
        # plugins='ddp_sharded',
        plugins='deepspeed_stage_2' if args.use_deepspeed else None,
        precision=16 if args.use_deepspeed else 32
    )
    print('NUM GPUs USING:')
    print(trainer.gpus)
    # if args.num_gpus > 1:
    #     lm.parallelize()

    trainer.fit(lm, datamodule=dataset)

    # cache
    # upload best model
    best_model_path = checkpoint_callback.best_model_path
    if args.env == 'bb':
        fs = get_fs()
        fname = os.path.basename(best_model_path)
        remote_path = os.path.join('aspangher', 'controlled-sequence-gen', output_fp, fname)
        print('uploading model file at %s to: %s...' % (best_model_path, remote_path))
        fs.put(best_model_path, remote_path)
    # log best metric score
    best_metric = checkpoint_callback.best_model_score
    print('BEST MODEL SCORE: %s' % best_metric)


if __name__ == '__main__':
    import argparse
    from fine_tuning.utils_parser import attach_model_args
    parser = argparse.ArgumentParser()
    parser = attach_model_args(parser)
    args = parser.parse_args()

    # load data
    here = os.path.dirname(os.path.realpath(__file__))
    if args.env == 'local':
        # train and eval files
        args.dataset = os.path.join(here, '..', args.dataset)
    else:
        # train (and eval df)
        print('Downloading data...')
        data_fp = os.path.join(here, 'input_data.csv')
        download_file_to_filepath(remote_file_name=args.dataset, local_path=data_fp)
        args.dataset = data_fp

    # download model files
    if args.env == 'local':
        pretrained_path = args.pretrained_model_path
    else:
        if '/' not in args.pretrained_model_path:
            download_model_files_bb(remote_model=args.pretrained_model_path, local_path=here)
        else:
            download_file_to_filepath(remote_file_name=args.pretrained_model_path)
        output_path = os.path.join(here, args.pretrained_model_path, '*')
        print('files in: %s' % output_path)
        print(glob.glob(output_path))
        args.pretrained_path = reformat_model_path(os.path.join(here, args.pretrained_model_path), args)

    if args.experiment is None:
        if 'gpt2' in args.pretrained_model_path:
            args.experiment = 'discourse-gpt2'
        elif 'roberta' in args.pretrained_model_path:
            args.experiment = 'discourse-roberta'
        else:
            print('No args.experiment set, or can\'t infer it from args.pretrained_model_path!!!')

    # run fine-tuning
    main(args)
