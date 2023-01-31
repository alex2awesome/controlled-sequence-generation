from transformers import T5ForConditionalGeneration, T5Config, T5TokenizerFast
import numpy as np
import torch
import textwrap
import logging
import json
import difflib
from discriminator.models_full import Discriminator as SequentialLSTMDiscriminator
from editing.src.predictors.news_discourse.news_discourse_dataset_reader import PretrainedTransformerTokenizerWithSpacy
from allennlp.nn.util import move_to_device

# local imports
from editing.src.predictors.news_discourse.news_discourse_dataset_reader import NewsDiscourseDatasetReader
from editing.predictor import TorchGPT2Predictor
from util.utils_general import label_idx_to_str, label_str_to_idx
import spacy
import logging


def write_args(args_path, args):
    """ Helper function to write args
    Args:
        args: list[Dict]
        args_path: str
    """
    logging.info("Writing args to: " + args_path)
    for name, sub_args in args.items():
        logging.info(f"{name} args: {sub_args}")
    f = open(args_path, "w")
    f.write(json.dumps(args, indent=4))
    f.close()


####################################################################
####################### Task Specific Utils ########################
####################################################################
def get_dataset_reader(task, predictor):
    return predictor._dataset_reader


def format_classif_input(inp, label):
    return "label: " + label + ". input: " + inp 


def format_multiple_choice_input(context, question, options, answer_idx):
    formatted_str = f"question: {question} answer: choice {answer_idx}:" + \
            f"{options[answer_idx]} context: {context}"
    for option_idx, option in enumerate(options):
        formatted_str += " choice" + str(option_idx) + ": " + option
    return formatted_str


def load_predictor(args, preloaded_hf_predictor=None, predictor_folder="trained_predictors/"):
    device = get_device()

    # when Editor is called in the larger Generation pipeline, we don't want to have to load the discriminator twice.
    if preloaded_hf_predictor is None:
        logging.info(f"Loading Predictor...")
        if args.pretrained_discrim_config is not None:
            from discriminator.config_helper import TransformersConfig
            config = TransformersConfig.from_json_file(args.pretrained_discrim_config)
        else:
            config = None

        discriminator = (SequentialLSTMDiscriminator
                              .load_from_checkpoint(checkpoint_path=args.pretrained_discrim_path,
                                                    loading_from_checkpoint=True,
                                                    pretrained_cache_dir=args.pretrained_lm_model_path,
                                                    config=config
                                                    )
                              )
    else:
        discriminator = preloaded_hf_predictor

    tokenizer = PretrainedTransformerTokenizerWithSpacy(
        model_name=args.pretrained_lm_model_path,
        spacy_model=args.spacy_model
    )
    dr = NewsDiscourseDatasetReader(tokenizer=tokenizer, local=args.local, data_file=args.real_data_file)
    predictor = TorchGPT2Predictor(
        model=discriminator,
        dataset_reader=dr,
        huggingface_dir=args.pretrained_lm_model_path,
        device=device
    )
    predictor = move_to_device(predictor, device)
    return predictor


####################################################################
########################### Model Utils ############################
####################################################################
def load_base_t5(max_length=700, dir_or_name='t5-base'):
    t5_config = T5Config.from_pretrained(dir_or_name, n_positions=max_length)
    model = T5ForConditionalGeneration.from_pretrained(dir_or_name, config=t5_config)
    tokenizer = T5TokenizerFast.from_pretrained(dir_or_name, truncation=True)
    return tokenizer, model


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_prob_pred(pred, label_idx):
    """ Given a prediction, gets predicted probability of label_idx. """
    for idx, prob in enumerate(pred['probs']):
        if idx == label_idx:
            return prob


def get_ints_to_labels(label=None):
    if label is not None:
        return label_idx_to_str.get(label)
    else:
        return label_idx_to_str


def get_labels_to_ints(label=None):
    if label is not None:
        return label_str_to_idx.get(label)
    else:
        return label_str_to_idx

def get_predictor_tokenized(predictor, string):
    return predictor._dataset_reader._tokenizer.tokenize(string)


def add_probs(pred):
    """ Computes predicted probs from logits. """
    if 'probs' not in pred:
        if isinstance(pred['logits'], torch.Tensor):
            pred['probs'] = torch.nn.functional.softmax(pred['logits'])
        else:
            pred['probs'] = np.exp(pred['logits'])/sum(np.exp(pred['logits'])) 
    return pred


####################################################################
########################### Other Utils ############################
####################################################################
def wrap_text(text, num_indents=6, width=100):
    """ Util for pretty printing. """

    indent = "".join(['\t' for _ in range(num_indents)])
    return textwrap.fill(text, subsequent_indent = indent, width=width)


def html_highlight_diffs(orig, edited, tokenizer_wrapper):
    """ Given an orig and edited inputs, mark up differences in HTML. """
    
    orig = orig.replace("<br ", "<-br ").replace(" .", ".")
    edited = edited.replace("<br ", "<-br ").replace(" .", ".")

    orig_tok = tokenizer_wrapper.tokenize(orig)
    edited_tok = tokenizer_wrapper.tokenize(edited)

    orig_text_tok = [t.text for t in orig_tok]
    edited_text_tok = [t.text for t in edited_tok]

    edited_mark_indices, num_add, num_del = get_marked_indices(orig_text_tok, edited_text_tok, "+")
    orig_mark_indices, num_add_2, num_del_2 = get_marked_indices(orig_text_tok, edited_text_tok, "-")

    marked_original = orig 
    for idx in reversed(orig_mark_indices):
        token = orig_tok[idx]
        start, end = token.idx, token.idx_end
        if start == None or end == None:
            logging.info(token, start, end)
        marked_original = marked_original[:start] + "<b>" + \
                marked_original[start:end] + "</b>" + marked_original[end:]
    
    marked_edited = edited.replace("<br />", "<-br />") 
    for idx in reversed(edited_mark_indices):
        token = edited_tok[idx]
        start, end = token.idx, token.idx_end
        if start == None or end == None:
            logging.info(token, start, end)
        marked_edited = marked_edited[:start] + "<b>" + \
                marked_edited[start:end] + "</b>" + marked_edited[end:]
    return marked_original, marked_edited


def get_marked_indices(orig_tokinal, tokenized_contrast, symbol):
    """ Helper function for html_highlight_diffs. 
    Will only return indices of words deleted or replaced (not inserted). """

    index_offset = 0
    d = difflib.Differ()
    diff = d.compare(orig_tokinal, tokenized_contrast)
    list_diff = list(diff)
    tokens, modified_tokens, indices = [], [], []
    counter = 0
    additions, deletions = 0, 0

    for token_idx, token in enumerate(list_diff):
        marker = token[0]
        word = token[2:]
        if marker == symbol:        
            tokens.append(word)
            indices.append(counter)
            counter += 1
        elif marker == " ":
            modified_tokens.append(word)
            counter += 1

        if marker == "+":
            additions += 1
        if marker == "-":
            deletions += 1
            
    return indices, additions, deletions


def get_grad_sign_direction(grad_type, grad_pred):
    """ Helper function to get sign direction. When grad_type is signed,
    determines whether to get most negative or positive gradient values.
    This should depend on grad_pred, i.e. what label is being used to
    compute gradients for masking.

    During Stage Two, we want to mask tokens that push *away* from the contrast
    label or *towards* the original label.

    Sign direction plays no role if only gradient *magnitudes* are used
        (i.e. if grad_type is not signed, but involves taking the l1/l2 norm.)
    """
    assert grad_pred in ["contrast", "original"]
    assert grad_type in ["integrated_l1", "integrated_signed", "normal_l1", "normal_signed", "normal_l2",
                         "integrated_l2"]
    if "signed" in grad_type and "contrast" in grad_pred:
        sign_direction = -1
    elif "signed" in grad_type and "original" in grad_pred:
        sign_direction = 1
    else:
        sign_direction = None
    return sign_direction