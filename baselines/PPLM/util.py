import os

def reformat_model_path(x, env=None):
    fp_marker = './'
    if os.environ.get('env') == 'bb' and (not x.startswith(fp_marker)):
        return os.path.join(fp_marker, x)
    if env is not None and env == 'local':
        return os.path.join('/Users/alex/.cache/torch/transformers/named-models', fp_marker)
    else:
        return x

def attach_model_args_pplm_discrim_train(parser):
    parser.add_argument('--env', type=str, default='local', help='Specify whether we\'re running '
                                                                 'locally or on BB compute.')
    parser.add_argument("--dataset", type=str, default="SST",
                        choices=("SST", "clickbait", "toxic", "generic"),
                        help="dataset to train the discriminator on."
                             "In case of generic, the dataset is expected"
                             "to be a TSV file with structure: class \\t text")
    parser.add_argument("--dataset_fp", type=str, default="",
                        help="File path of the dataset to use. "
                             "Needed only in case of generic datadset")
    parser.add_argument("--model_type", type=str, default="gpt2", help="Class of pretrained model.")
    parser.add_argument("--pretrained_model", type=str, default="gpt2-medium",
                        help="Pretrained model to use as encoder")
    parser.add_argument("--epochs", type=int, default=10, metavar="N",
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learnign rate")
    parser.add_argument("--batch_size", type=int, default=64, metavar="N",
                        help="input batch size for training (default: 64)")
    parser.add_argument("--log_interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--save_model", action="store_true",
                        help="whether to save the model")
    parser.add_argument("--cached", action="store_true",
                        help="whether to cache the input representations")
    parser.add_argument("--no_cuda", action="store_true",
                        help="use to turn off cuda")
    parser.add_argument("--output_fp", default=".",
                        help="path to save the output to")

    parser.add_argument('--freeze_transformer', action='store_true')
    parser.add_argument('--freeze_embedding_layer', action='store_true')
    parser.add_argument('--freeze_positional_layer', action='store_true')
    parser.add_argument('--freeze_encoder_layers', nargs="*", default=[])

    return parser


def attach_model_args_pplm(parser):
    parser.add_argument('--env', type=str, default='local', help='Whether to download from BB or not.')
    parser.add_argument(
        "--pretrained_model", "-M",
        type=str,
        default="gpt2-medium",
        help="pretrained model name or path to local checkpoint",
    )
    parser.add_argument(
        "--cond_text", type=str, default="The lake",
        help="Prefix texts to condition on"
    )
    parser.add_argument(
        "--uncond",
        action="store_true",
        help="Generate from end-of-text as prefix"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate from the modified latents",
    )
    parser.add_argument(
        "--bag_of_words",
        "-B",
        type=str,
        default=None,
        help="Bags of words used for PPLM-BoW. "
             "Either a BOW id (see list in code) or a filepath. "
             "Multiple BoWs separated by ;",
    )
    parser.add_argument(
        "--discrim", "-D", type=str, default=None,
        choices=("clickbait", "sentiment", "toxicity", "generic"),
        help="Discriminator to use",
    )
    parser.add_argument('--discrim_weights', type=str, default=None, help='Weights for the generic discriminator')
    parser.add_argument('--discrim_meta', type=str, default=None, help='Meta information for the generic discriminator')
    parser.add_argument(
        "--class_label",
        type=int,
        default=-1,
        help="Class label used for the discriminator",
    )
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--stepsize", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--sample", action="store_true",
        help="Generate from end-of-text as prefix"
    )
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--grad_length", type=int, default=10000)
    parser.add_argument(
        "--window_length",
        type=int,
        default=0,
        help="Length of past which is being optimized; "
             "0 corresponds to infinite window length",
    )
    parser.add_argument(
        "--horizon_length",
        type=int,
        default=1,
        help="Length of future to optimize over",
    )
    parser.add_argument("--decay", action="store_true",
                        help="whether to decay or not")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--gm_scale", type=float, default=0.9)
    parser.add_argument("--kl_scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    parser.add_argument("--colorama", action="store_true", help="colors keywords")
    parser.add_argument("--verbosity", type=str, default="very_verbose",
                        choices=("quiet", "regular", "verbose", "very_verbose"),
                        help="verbosiry level")
    return parser
