# Local imports
from editing.src.stage_two import *

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results", help='Results dir. Where to store results.')
    parser.add_argument("--mask_type", default="grad", choices=["grad", "random"])
    parser.add_argument(
        "--grad_type", default="normal_l1",
        choices=["integrated_l1", "integrated_signed", "normal_l1", "normal_signed", "normal_l2", "integrated_l2"],
        help="Which gradient method to use for grad-based masking. l1/signed/l2 determine how to aggregate over the emb dim."
    )
    parser.add_argument(
        '--grad_pred', default="original", choices=["original", "contrast"],
        help="Whether to take gradient with respect to the contrast or original prediction"
    )
    parser.add_argument("--model_max_length", default=700, help="Maximum number of tokens that Editor model can take")
    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--data_split_ratio", default=0.75, type=float)
    parser.add_argument("--target_label", default="gold", choices=["gold", "pred"], help="Which label to use as the target during Editor training")
    parser.add_argument("--task", default="news_discourse", type=str)
    parser.add_argument("--stage2_exp", default="nd_classification", type=str)
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--use_heuristic_masks', action='store_true', help='Whether to further limit masks based on POS and stopwords.')
    parser.add_argument('--use_gold_labels', action='store_true',
                        help='Whether to edit towards a gold label or not.')

    # search
    parser.add_argument('--search_beam_width', default=3, help="Beam width for beam search over edits.")
    parser.add_argument('--search_max_mask_frac', default=0.55, help="Maximum mask fraction")
    parser.add_argument('--search_search_method', default="binary", choices=["binary", "linear"], help="Which kind of search method to use: binary or linear.")
    parser.add_argument('--search_max_search_levels', default=4, help="Maximum number of search levels")
    parser.add_argument('--search_max_edit_rounds', default=3, help="Maximum number of edit rounds")

    # generation
    parser.add_argument('--num_generations', default=15)
    parser.add_argument('--generation_num_beams', default=15)
    parser.add_argument('--generate_type', default="sample", choices=['beam', 'sample'])
    parser.add_argument('--no_repeat_ngram_size', default=2)
    parser.add_argument('--generation_top_p', default=.9)
    parser.add_argument('--generation_top_k', default=200)
    parser.add_argument('--generation_length_penalty')

    # model paths
    parser.add_argument('--pretrained_discrim_path', default=None, type=str, help="Trained discriminator/Predictor.")
    parser.add_argument('--pretrained_lm_model_path', default=None, type=str, help="Base language model for the Predictor.")
    parser.add_argument('--pretrained_editor_lm_path', default='t5-base', type=str, help="Base language model for the Editor")
    parser.add_argument('--spacy_model_file', default=None, type=str)
    parser.add_argument('--real_data_file', default=None, type=str)
    parser.add_argument('--finetuned_editor_lm_path', default=None, type=str)
    parser.add_argument('--edit_output_file', type=str, default=None)

    args = parser.parse_args()
    if args.local == False:
        from util.utils_data_access import download_all_necessary_files
        print(args.finetuned_editor_lm_path)
        download_all_necessary_files(args)
    else:
        import spacy
        args.spacy_model = spacy.load(args.spacy_model_file)

    output_file = run_edit_test(args)
    if args.local == False:
        if args.edit_output_file is not None:
            from util.utils_data_access import get_fs
            fs = get_fs()
            fs.upload(output_file, args.edit_output_file)
        else:
            raise ValueError('YOU FORGOT TO SET AN UPLOAD PATH!!!')

