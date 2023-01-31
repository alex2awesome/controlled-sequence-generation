from allennlp.data.tokenizers import PretrainedTransformerTokenizer
import os
import csv
from tqdm import tqdm
import time

# Local imports
from editing.src.masker import GradientMasker
from editing.src.utils import *
from editing.src.edit_finder import EditFinder, EditEvaluator
from editing.src.editor import Editor
from editing.src.utils import get_grad_sign_direction

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def load_editor_weights(editor_model, editor_path):
    """ Loads Editor weights from editor_path """

    if os.path.isdir(editor_path):
        editor_path = os.path.join(editor_path, "best.pth")
        if not os.path.exists(editor_path):
            raise NotImplementedError(f"If directory given for editor_path, \
                    it must contain a 'best.pth' file but found none in given \
                    dir. Please give direct path to file containing weights.")
    logging.info(f"Loading Editor weights from: {editor_path}")
    editor_model.load_state_dict(torch.load(editor_path, map_location=get_device()))
    return editor_model


def load_models(args, preloaded_hf_predictor=None):
    """ Loads Predictor and Editor by task and other args """
    logging.info("Loading models...")
    device = get_device()
    predictor = load_predictor(args, preloaded_hf_predictor=preloaded_hf_predictor)
    editor_tokenizer_wrapper = PretrainedTransformerTokenizer(
        model_name=args.pretrained_editor_lm_path,
        max_length=args.model_max_length
    )
    editor_tokenizer, editor_model = load_base_t5(
        dir_or_name=args.pretrained_editor_lm_path,
        max_length=args.model_max_length
    )
    editor_model = load_editor_weights(editor_model, args.finetuned_editor_lm_path)
    editor_model = editor_model.to(device)
    logging.info('Moving editor_model to %s' % str(device))
    logging.info('predictor._model on %s' % str(predictor._model.device))

    sign_direction = get_grad_sign_direction(
        args.grad_type, args.grad_pred
    )

    masker = GradientMasker(
        args.search_max_mask_frac,
        editor_tokenizer_wrapper,
        predictor,
        args.model_max_length,
        grad_type=args.grad_type,
        sign_direction=sign_direction,
        use_heuristic_masks=args.use_heuristic_masks
    )

    editor = Editor(
        editor_tokenizer_wrapper,
        editor_tokenizer,
        editor_model, masker,
        num_gens=args.num_generations,
        num_beams=args.generation_num_beams,
        grad_pred=args.grad_pred,
        generate_type=args.generate_type,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        top_p=args.generation_top_p,
        top_k=args.generation_top_k,
        length_penalty=args.generation_length_penalty,
        verbose=False
    )
    logging.info("Done loading models.")
    return editor, predictor


def run_edit_test(args):
    """ Runs Stage 2 on test inputs by task. """

    task_dir = os.path.join(args.results_dir, args.task)
    stage_two_dir = os.path.join(task_dir, f"edits/{args.stage2_exp}")

    if not os.path.exists(stage_two_dir):
        os.makedirs(stage_two_dir)

    logging.info(f"Task dir: {task_dir}")
    logging.info(f"Stage two dir: {stage_two_dir}")

    out_file = os.path.join(stage_two_dir, "edits.csv")
    meta_log_file = os.path.join(stage_two_dir, "meta_log.txt")

    meta_f = open(meta_log_file, 'w', 1)

    # Load models and Edit objects
    editor, predictor = load_models(args)
    dr = get_dataset_reader(args.task, predictor)
    edit_evaluator = EditEvaluator(
        fluency_model_name=args.pretrained_editor_lm_path,
        spacy_dir=args.spacy_model_file
    )
    edit_finder = EditFinder(
        predictor,
        editor,
        edit_evaluator=edit_evaluator,
        beam_width=args.search_beam_width,
        max_mask_frac=args.search_max_mask_frac,
        search_method=args.search_search_method,
        max_search_levels=args.search_max_search_levels
    )

    # Get inputs
    test_inputs, test_labels = dr.get_inputs(
        return_labels=True, sample=args.local,
        train=False, test=True, include_errors=False
    )

    input_indices = np.array(range(len(test_inputs)))
    np.random.shuffle(input_indices)

    # Find edits and write to file
    with open(out_file, "w") as csv_file:
        fieldnames = ["data_idx", "sorted_idx", "orig_pred", "new_pred",
                      "contrast_pred", "orig_contrast_prob_pred",
                      "new_contrast_prob_pred", "orig_input", "edited_input",
                      "orig_editable_seg", "edited_editable_seg",
                      "minimality", "num_edit_rounds", "mask_frac",
                      "duration", "error"]
        writer = csv.writer(csv_file, delimiter="\t")
        writer.writerow(fieldnames)

        for idx, i in tqdm(enumerate(input_indices), total=len(input_indices)):
            inp = test_inputs[i]
            logging.info(wrap_text(f"ORIGINAL INSTANCE ({i}): {inp}"))

            if args.use_gold_labels:
                label_idx = test_labels[i]
                label = get_ints_to_labels(label_idx)
                logging.info(wrap_text(f"GOLD LABEL ({i}): {label}"))
            else:
                label_idx = None

            start_time = time.time()
            error = False
            # try:
            edited_list = edit_finder.minimally_edit(
                inp,
                max_edit_rounds=args.search_max_edit_rounds,
                gold_label_idx=label_idx
            )

            torch.cuda.empty_cache()
            sorted_list = edited_list.get_sorted_edits()

            # except Exception as e:
            #     logger.info("ERROR: ", e)
            #     error = True
            #     sorted_list = []

            end_time = time.time()

            duration = end_time - start_time
            for s_idx, s in enumerate(sorted_list):
                writer.writerow([
                    i, s_idx, edited_list.orig_label,
                    s['edited_label'], edited_list.contrast_label,
                    edited_list.orig_contrast_prob, s['edited_contrast_prob'],
                    edited_list.orig_input, s['edited_input'],
                    edited_list.orig_editable_seg,
                    s['edited_editable_seg'], s['minimality'],
                    s['num_edit_rounds'], s['mask_frac'], duration, error
                ])
                csv_file.flush()
            if sorted_list == []:
                writer.writerow([i, 0, edited_list.orig_label,
                                 None, edited_list.contrast_label,
                                 edited_list.orig_contrast_prob, None,
                                 edited_list.orig_input, None,
                                 edited_list.orig_editable_seg,
                                 None, None, None, None, duration, error])
                csv_file.flush()
                meta_f.flush()

    csv_file.close()
    meta_f.close()
    return out_file