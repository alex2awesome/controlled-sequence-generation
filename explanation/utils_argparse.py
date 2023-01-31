
def attach_arguments(parser):
    parser.add_argument("--local", action='store_true')
    parser.add_argument('--model_type', type=str, default=None)
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--discriminator_path', type=str, default=None)
    parser.add_argument('--spacy_model_file', type=str, default=None)
    parser.add_argument('--real_data_file', type=str, default=None)
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--use_real_doc_idxes', nargs='+', default=None, help="Whether to use a real document or not. -1 = no.")

    return parser