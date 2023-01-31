def attach_model_args_generator(parser):
    parser.add_argument('--env', type=str, default='local', help='Whether to download from BB or not.')
    parser.add_argument("--lm_model_type", type=str, default='gpt2', help="Non fine-tuned LM to load.")
    parser.add_argument("--pretrained_model_path", "-M", type=str, default="gpt2-medium", help="pretrained model name or path to local checkpoint.")
    parser.add_argument("--pretrained_lm_model_path", type=str, default=None, help="Path to local LM checkpoint.")
    parser.add_argument("--discriminator_path", "-D", type=str, default=None, help="Discriminator to use.")
    parser.add_argument('--real_data_file', type=str, default=None, help="Data file to use.")
    parser.add_argument('--spacy_model_file', type=str, default=None, help='Location of Spacy file. If None, don\'t retrieve any file.')
    parser.add_argument("--experiment", type=str, default=None, help="Which experiment we're running.")
    parser.add_argument('--remote_log_path', type=str, default=None, help='Where to remotely store logs.')
    parser.add_argument('--notes', type=str, default=None)
    parser.add_argument('--tag_each_sentence_variant', action='store_true')
    parser.add_argument('--output_filename', type=str, default=None)
    parser.add_argument('--config_path', type=str, default=None)
    #
    parser.add_argument("--cond_text", type=str, default="The lake", help="Prefix texts to condition on")
    parser.add_argument('--use_real_doc_idxes', nargs='+', default=None, help="Whether to use a real document or not. -1 = no.")
    parser.add_argument("--uncond", action="store_true", help="Generate from end-of-text as prefix")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate from the modified latents",)
    parser.add_argument("--class_labels", nargs='+', default=[4, 5, 2], help="Class labels used for the discriminator")
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--generator_type", type=str, default='perturbed', help="Get `perturbed` or `unperturbed` generation.")
    parser.add_argument('--run_tagger', action='store_true', help="Whether to output predicted tags or not.")
    parser.add_argument('--print_tables', action='store_true', help='Whether to print output tables or not.')
    parser.add_argument('--sentence_splitting_method', type=str, default='eos', help='Whether to split on EOS or using spacy sentence-boundary detection.')

    # runner arguments
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    parser.add_argument("--verbosity", type=str, default="very_verbose", choices=("quiet", "regular", "verbose", "very_verbose"), help="verbosity level")

    # discriminator arguments
    # parser.add_argument('--label_context_back', type=int, default=3, help='How many labels beforehand to include in prediction.')
    # parser.add_argument('--label_context_forward', type=int, default=3, help='How many labels beforehand to include in prediction.')
    # parser.add_argument('--num_labels_pred_window', type=int, default=None, help='Whether to do a multi-step prediction problem.')

    # model arguments
    parser.add_argument("--stepsize", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument('--top_k_fudge', type=int, default=200)
    parser.add_argument("--top_p", type=float, default=.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--force_eos_token", type=bool, default=True, help="whether to force eos token at maximum sentence length.")
    parser.add_argument("--sample", action="store_true", help="Generate from end-of-text as prefix")
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--num_sentences_to_generate", type=int, default=3)
    parser.add_argument("--grad_length", type=int, default=10000)
    parser.add_argument("--window_length", type=int, default=0, help="Length of past which is being optimized; 0 corresponds to infinite window length")
    parser.add_argument("--horizon_length", type=int, default=1, help="Length of future to optimize over")
    parser.add_argument("--decay", action="store_true", help="whether to decay or not")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--gm_scale", type=float, default=0.9, help="How to fuse the unperturbed probabilities with the perturbed probabilities.")
    parser.add_argument("--kl_scale", type=float, default=0.01, help="How to combine the discriminator loss and the kl divergence loss.")
    parser.add_argument('--max_sentence_length', type=int, default=100)
    parser.add_argument('--doc_sent_length_cutoff', type=int, default=None, help="How many sentences to generate.")
    parser.add_argument('--sentence_min_len', type=int, default=5)
    parser.add_argument('--heads_exp_backoff_left', type=float, default=1, help="The base of the exponent used to average the heads on the left of main together.")
    parser.add_argument('--heads_exp_backoff_right', type=float, default=1, help="The base of the exponent used to average the heads together on the right of main together.")

    # editor arguments
    parser.add_argument('--perform_edits', action='store_true')
    parser.add_argument('--edit_discriminator_path', type=str, default=None)
    parser.add_argument('--edit_discrim_config_path', type=str, default=None)
    parser.add_argument('--editor_search_beam_width', type=int, default=3, help="Beam width for beam search over edits.")
    parser.add_argument('--editor_search_max_mask_frac', type=float, default=0.55, help="Maximum mask fraction")
    parser.add_argument(
        '--editor_search_search_method',
        default="binary",
        choices=["binary", "linear"],
        help="Which kind of search method to use: binary or linear."
    )
    parser.add_argument('--editor_search_max_search_levels', type=int, default=4, help="Maximum number of search levels")
    parser.add_argument('--editor_search_max_edit_rounds', type=int, default=3, help="Maximum number of edit rounds")
    #
    parser.add_argument('--editor_num_generations', type=int, default=15)
    parser.add_argument('--editor_generation_num_beams', type=int, default=15)
    parser.add_argument('--editor_generate_type', default="sample", choices=['beam', 'sample'])
    parser.add_argument('--editor_no_repeat_ngram_size', type=int, default=2)
    parser.add_argument('--editor_generation_top_p', type=float, default=.9)
    parser.add_argument('--editor_generation_top_k', type=int, default=200)
    parser.add_argument('--editor_generation_length_penalty', type=float, default=1)
    parser.add_argument(
        '--editor_use_heuristic_masks',
        action='store_false',
        help='Whether to further limit masks based on POS and stopwords.'
    )
    #
    parser.add_argument('--editor_pretrained_editor_lm_path', default=None, type=str)
    parser.add_argument('--editor_finetuned_editor_lm_path', default=None, type=str)
    parser.add_argument(
        "--editor_grad_type",
        default="normal_l1",
        choices=["integrated_l1", "integrated_signed", "normal_l1", "normal_signed", "normal_l2", "integrated_l2"],
        help="Which gradient method to use for grad-based masking. l1/signed/l2 determine how to aggregate over the emb dim."
    )
    parser.add_argument(
        "--editor_model_max_length", default=700, help="Maximum number of tokens that Editor model can take"
    )
    parser.add_argument('--editor_grad_pred', default="original", choices=["original", "contrast"],
        help="Whether to take gradient with respect to the contrast or original prediction"
    )

    return parser
