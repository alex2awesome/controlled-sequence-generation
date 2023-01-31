# Local imports
from editing.src.stage_one import run_train_editor
from editing.src.utils import load_predictor, get_dataset_reader

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results", help='Results dir. Where to store results.')
    parser.add_argument("--mask_type", default="grad", choices=["grad", "random"])
    parser.add_argument("--grad_type", default="normal_l1", choices=["integrated_l1", "integrated_signed", "normal_l1", "normal_signed", "normal_l2", "integrated_l2"],
                             help="Which gradient method to use for grad-based masking. l1/signed/l2 determine how to aggregate over the emb dim.")
    parser.add_argument("--model_max_length", default=700, help="Maximum number of tokens that Editor model can take")
    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--data_split_ratio", default=0.75, type=float)
    parser.add_argument("--target_label", default="gold", choices=["gold", "pred"], help="Which label to use as the target during Editor training")
    parser.add_argument("--task", default="news_discourse", type=str)
    parser.add_argument("--stage1_exp", default="nd_classification", type=str)
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--use_heuristic_masks', action='store_true', help='Whether to further limit masks based on POS and stopwords.')

    # model paths
    parser.add_argument('--pretrained_discrim_path', default=None, type=str)
    parser.add_argument('--pretrained_discrim_config', default=None, type=str)
    parser.add_argument('--pretrained_lm_model_path', default=None, type=str)
    parser.add_argument('--pretrained_editor_lm_path', default='t5-base', type=str)
    parser.add_argument('--spacy_model_file', default=None, type=str)
    parser.add_argument('--real_data_file', default=None, type=str)
    parser.add_argument('--remote_editor_upload', default=None, type=str)

    args = parser.parse_args()
    if args.local == False:
        from util.utils_data_access import download_all_necessary_files
        download_all_necessary_files(args)
    else:
        import spacy
        args.spacy_model = spacy.load(args.spacy_model_file)

    predictor = load_predictor(args)
    dr = get_dataset_reader(args.task, predictor)
    best_path = run_train_editor(predictor, dr, args)

    if args.local == False:
        if args.remote_editor_upload is not None:
            from util.utils_data_access import get_fs
            fs = get_fs()
            fs.upload(best_path, args.remote_editor_upload)
        else:
            raise ValueError('YOU FORGOT TO SET AN UPLOAD PATH!!!')


# pretrained_discrim_path = '/Users/alex/Projects/usc-research/controlled-sequence-gen/experiments/output_dir/trial-Sequential, flattened sentences, large-corpus Fine-tuned LM__epoch=07-f1_macro=0.58.ckpt'
# pretrained_lm_model_path = '/Users/alex/.cache/torch/transformers/named-models/gpt2-medium-expanded-embeddings'
# pretrained_editor_lm_path = '/Users/alex/.cache/torch/transformers/named-models/t5-base'
