def attach_model_arguments(parser):
    ## Required parameters
    ## data files
    parser.add_argument("--train_data_file", type=str)
    parser.add_argument("--transformer_model", default=None, type=str)
    parser.add_argument("--spacy_model", default=None, type=str)
    ##
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument('--do_train', action="store_true")
    parser.add_argument('--do_eval', action="store_true")
    parser.add_argument('--notes', type=str, default='')
    parser.add_argument('--random_split', action='store_true')
    parser.add_argument('--test_split', type=float, default=.2) ## size of test set
    parser.add_argument('--train_perc', type=float, default=1.0) ## whether to include more data in training

    # other training arguments
    parser.add_argument('--output_checkpoint_dir', type=str, default=None)
    parser.add_argument('--num_train_epochs', type=int, default=None)
    parser.add_argument('--training_batch_size', type=int, default=20)
    parser.add_argument('--eval_batch_size', type=int, default=20)

    ## model params
    #### general model params
    parser.add_argument('--freeze_transformer', action='store_true')
    parser.add_argument('--freeze_embedding_layer', action='store_true')
    parser.add_argument('--freeze_encoder_layers', nargs="*", default=[])
    parser.add_argument('--freeze_pooling_layer', action='store_true')
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=.2)
    parser.add_argument('--transformer_hidden_dropout_prob', type=float, default=.1)
    parser.add_argument('--transformer_attention_probs_dropout_prob', type=float, default=.1)

    #### classifier head differences
    parser.add_argument('--use_crf', action='store_true')
    parser.add_argument(
        '--subword_treatment', type=str,
        default='label_first', help='Tests how we parse and interpret subwords. Either `label_all` or `label_first`.'
    ) ##

    ## if running on DSP or not.
    parser.add_argument('--dsp', action='store_true')

    return parser


def formulate_module_args(args):
    """
    Helper script to automatically convert param_args into a list, to pass into a SLURM runner.

    params:
        * args: ArgParse arguments.
    """
    top_level_arguments = [
        'job_script_module',
        'package_uri',
        'job_size',
        'branch',
        'git_identity_id',
        'hadoop_identity_id',
        'gen_name',
        'n_gpus'
    ]

    module_args = []
    arg_vars = vars(args)
    for param, value in arg_vars.items():
        if param in top_level_arguments:
            continue

        ## exceptions
        if param == 'notes':
            module_args.extend(['--notes', '_'.join(value.split())])

        ## general rules
        elif isinstance(value, str):
            module_args.extend(["--%s" % param, value])
        elif isinstance(value, bool):
            if value:
                module_args.append('--%s' % param)
        elif isinstance(value, int) or isinstance(value, float):
            module_args.extend(["--%s" % param, str(value)])
        elif isinstance(value, list):
            module_args.extend(["--%s" % param, ' '.join(value)])

    return module_args