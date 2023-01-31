import sys
import allennlp
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.data.batch import Batch
import nltk
import time
import heapq

from transformers import T5Tokenizer

from editing.src.masker import RandomMasker, GradientMasker
from editing.src.utils import *

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

####################################################################
############################## Utils ###############################
####################################################################

def get_max_instance(instance_candidates, contrast_pred_idx):
    """ Returns candidate with highest predicted prob of contrast_pred_idx. """

    batched_preds = predictor.predict_batch_instance(instance_candidates)
    batched_preds = [add_prob(pred) for pred in batched_preds]
    max_idx = max(range(len(batched_preds)), key=lambda index: \
            batched_preds[index]['probs'][contrast_pred_idx])
    max_candidate = instance_candidates[max_idx]
    max_prob = batched_preds[max_idx]['probs'][contrast_pred_idx]

    return max_candidate, max_prob

class EditEvaluator():
    def __init__(
            self,
            args,
            fluency_model_name = "t5-base",
            fluency_masker = 'random',
            spacy_dir = 'en_core_web_sm',
            predictor = None,
            tokenizer_wrapper = None

    ):
        self.spacy_dir = spacy_dir
        if fluency_masker == 'random':
            fluency_masker = RandomMasker(None, SpacyTokenizer(language=spacy_dir), 512)
        elif fluency_masker == 'gradient':
            sign_direction = get_grad_sign_direction(
                args.grad_type, args.grad_pred
            )
            fluency_masker = GradientMasker(
                editor_tok_wrapper=tokenizer_wrapper,
                predictor=predictor,
                use_heuristic_masks=args.use_heuristic_masks,
                grad_type=args.grad_type,
                max_tokens=args.model_max_length,
                sign_direction=sign_direction
            )

        self.device = get_device()
        self.fluency_model = T5ForConditionalGeneration.from_pretrained(fluency_model_name).to(self.device)
        self.fluency_tokenizer = T5Tokenizer.from_pretrained(fluency_model_name)
        self.fluency_masker = fluency_masker 

    def score_fluency(self, sent, previous_text=None):
        temp_losses = []
        masked_strings, span_labels = self.fluency_masker.get_all_masked_strings(sent, previous_text=previous_text)
        for masked, label in zip(masked_strings, span_labels):
            input_ids = self.fluency_tokenizer.encode(
                masked,
                truncation="longest_first",
                max_length=600,
                return_tensors="pt"
            )
            input_ids = input_ids.to(self.device)
            labels = self.fluency_tokenizer.encode(
                label,
                truncation="longest_first",
                max_length=600,
                return_tensors="pt"
            )
            labels = labels.to(self.device)
            outputs = self.fluency_model(input_ids=input_ids, labels=labels)
            loss = outputs[0] 
            temp_losses.append(loss.item())
            del input_ids
            del labels
            del loss
            torch.cuda.empty_cache()
        avg_loss = sum(temp_losses)/len(temp_losses)
        return avg_loss
    
    def score_minimality(self, orig_sent, edited_sent, normalized=True):
        spacy = SpacyTokenizer(language=self.spacy_dir)
        tokenized_original = [t.text for t in spacy.tokenize(orig_sent)]
        tokenized_edited = [t.text for t in spacy.tokenize(edited_sent)]
        lev = nltk.edit_distance(tokenized_original, tokenized_edited)
        if normalized: 
            return lev/len(tokenized_original)
        else:
            return lev

def sort_instances_by_score(scores, *args):
    """ Sorts *args in order of decreasing scores """
    zipped = list(zip(scores, *args))
    zipped.sort(reverse = True)
    return list(zipped)

def get_scores(predictor, instance_candidates, contrast_pred_idx, device, k=None):
    """ Gets (top k) predicted probs of contrast_pred_idx on candidates. """

    # Get predictions
    with torch.no_grad():
        cuda_device = predictor._model._get_prediction_device()
        dataset = Batch(instance_candidates)
        dataset.index_instances(predictor._model.vocab)
        model_input = allennlp.nn.util.move_to_device(dataset.as_tensor_dict(), cuda_device)
        outputs = predictor._model.make_output_human_readable(predictor._model(**model_input))
        outputs = add_probs(outputs)
        probs = outputs['probs'].to(device)

    if k != None:
        pred_indices = torch.argmax(probs, dim=1).to(device)

        # Compute this only for remaining
        contrast_pred_tensor = torch.tensor([contrast_pred_idx]).to(device)
        bool_equal = (pred_indices == contrast_pred_tensor)
        pred_is_contrast_indices = bool_equal.reshape(-1).nonzero().reshape(-1)
        
        num_to_return = max(k, len(pred_is_contrast_indices))

        contrast_probs = probs[:, contrast_pred_idx]
        sorted_contrast_probs = torch.argsort(contrast_probs, descending=True)
        highest_indices = sorted_contrast_probs[:num_to_return]
        cpu_contrast_probs = torch.index_select(contrast_probs, 0, highest_indices).cpu().numpy()
        selected_pred_indices = torch.index_select(pred_indices, 0, highest_indices)
        cpu_pred_indices = selected_pred_indices.cpu().numpy()
        highest_indices = highest_indices.cpu().numpy()
    else:
        cpu_pred_indices = torch.argmax(probs, dim=1).cpu().numpy()
        cpu_contrast_probs = probs[:, contrast_pred_idx].cpu().numpy()
        highest_indices = range(len(cpu_contrast_probs)) 

    assert cpu_pred_indices.shape == cpu_contrast_probs.shape
    del outputs
    del probs
    del contrast_probs
    del selected_pred_indices
    del contrast_pred_tensor
    del pred_is_contrast_indices
    return cpu_contrast_probs, cpu_pred_indices, highest_indices 

####################################################################
########################## Main Classes ############################
####################################################################

class EditFinder():
    """ Runs search algorithms to find edits. Edit is considered successful if:
     (a) success_criteria == 'change label':
        -> the edit flips the label from the original label to the contrasted label.
     (b) success_criteria == 'increase prob':
        -> the edit increase the probability of the desired label.
     """

    def __init__(
        self,
        predictor, 
        editor, 
        beam_width = 3, 
        search_method = "binary", 
        max_mask_frac = 0.5, 
        max_search_levels = 10, 
        verbose = True,
        edit_evaluator = None,
        success_criteria = 'change label',
    ):
        self.predictor = predictor
        self.editor = editor
        self.beam_width = beam_width
        self.ints_to_labels = self.editor.ints_to_labels
        self.search_method = search_method 
        self.max_search_levels = max_search_levels
        self.device = get_device()
        self.verbose = verbose
        self.max_mask_frac = max_mask_frac
        self.edit_evaluator = edit_evaluator
        assert success_criteria in ['change label', 'increase prob']
        self.success_criteria = success_criteria

    def is_success(self, edit_prob, orig_prob, edit_label, orig_label):
        if self.success_criteria == 'change label':
            return edit_label == orig_label
        else:
            return edit_prob > orig_prob

    def run_edit_round(
        self, 
        edit_list, 
        input_cand, 
        contrast_pred_idx, 
        num_rounds, 
        mask_frac, 
        sorted_token_indices=None,
        previous_text=None
    ):
        """previous_text is the previous sentences in the document, for fluency."""

        logging.info(wrap_text(f"Running candidate generation for mask frac: \
                {mask_frac}; max mask frac: {self.max_mask_frac}"))
        self.editor.masker.mask_frac = mask_frac

        candidates, masked_sentence = self.editor.get_candidates(
                targ_pred_label=edit_list.contrast_label,
                inp=input_cand,
                targ_pred_idx=contrast_pred_idx,
                orig_pred_idx=edit_list.orig_pred_idx,
                sorted_token_indices=sorted_token_indices
        )

        input_cands = [e['edited_input'] for e in candidates]
        editable_seg_cands = [e['edited_editable_seg'] for e in candidates]

        instance_cands = [
            self.editor.input_to_instance(inp, editable_seg=es)
            for inp, es in zip(input_cands, editable_seg_cands)
        ]
        
        # TODO: does this happen? might get [] if all generations are bad, but 
        # Should not happen (if no good generations, use original infillings)
        if len(input_cands) == 0:
            logging.info("no candidates returned...")
            return False 

        # score each candidate (converted to instance) by the classifier score.
        # vv returns:
        # cpu_contrast_probs, cpu_pred_indices, highest_indices
        probs, pred_indices, highest_indices = get_scores(
            predictor=self.predictor,
            instance_candidates=instance_cands,
            contrast_pred_idx=contrast_pred_idx,
            device=self.device,
            k=self.beam_width
        )

        #
        input_cands = [input_cands[idx] for idx in highest_indices]
        editable_seg_cands = [editable_seg_cands[idx] for idx in highest_indices]
        
        # Sort these by highest score for iteration
        sorted_cands = sort_instances_by_score(probs, pred_indices, input_cands, editable_seg_cands)
        found_cand = False
        beam_inputs = [s for _, _, s in edit_list.beam]
        iterator = enumerate(sorted_cands)
        for sort_idx, (prob, pred_idx, input_cand, editable_seg_cand) in iterator:
            if self.verbose and sort_idx == 0:
                logging.info(wrap_text(f"Post edit round, top contr prob: {prob}"))
                logging.info(wrap_text(f"Post edit round, top cand: {input_cand}"))
            pred_idx = int(pred_idx)
            edit_list.counter += 1
            if input_cand not in beam_inputs:
                heapq.heappush(edit_list.beam, (prob, edit_list.counter, input_cand))

            label = self.ints_to_labels[pred_idx]
            
            if self.is_success(
                edit_prob=prob,
                orig_prob=edit_list.orig_contrast_prob,
                edit_label=pred_idx,
                orig_label=contrast_pred_idx
            ):
                found_cand = True
                minimality, fluency = None, None
                # Score minimality because we order edits by minimality scores
                if self.edit_evaluator is not None:
                    fluency = self.edit_evaluator.score_fluency(
                        sent=editable_seg_cand,
                        previous_text=previous_text
                    )
                    minimality = self.edit_evaluator.score_minimality( # levensthein distance between new edit and original
                        orig_sent=edit_list.orig_editable_seg,
                        edited_sent=editable_seg_cand,
                        normalized=True
                    )
                edit = {
                    "edited_editable_seg": editable_seg_cand,
                    "edited_input": input_cand,
                    "orig_input": edit_list.orig_input,
                    "minimality": minimality,
                    "fluency": fluency,
                    "masked_sentence": masked_sentence,
                    "edited_prob": prob,
                    "orig_prob": edit_list.orig_contrast_prob,
                    "orig_label": edit_list.contrast_label,
                    "edited_label": label,
                    "mask_frac": mask_frac,
                    "num_edit_rounds": num_rounds
                }
                edit_list.add_edit(edit)

            if len(edit_list.beam) > self.beam_width:
                _ = heapq.heappop(edit_list.beam)

        del probs
        del pred_indices
        return found_cand

    def binary_search_edit(
            self, edit_list, input_cand, contrast_pred_idx, num_rounds, 
            min_mask_frac=0.0, max_mask_frac=0.5, num_levels=1, 
            max_levels=None, sorted_token_indices=None, previous_text=None):
        """ Runs binary search over masking percentages, starting at
        midpoint between min_mask_frac and max_mask_frac.
        Calls `run_edit_round` at each mask percentage. """
      
        if max_levels == None:
            max_levels = self.max_search_levels

        mid_mask_frac = (max_mask_frac + min_mask_frac) / 2

        if self.verbose:
            logging.info(wrap_text("binary search mid: " + str(mid_mask_frac)))
        found_cand = self.run_edit_round(
            edit_list,
            input_cand,
            contrast_pred_idx,
            num_rounds,
            mid_mask_frac,
            sorted_token_indices=sorted_token_indices,
            previous_text=previous_text
        )
        if self.verbose:
            logging.info(wrap_text("Binary search # levels: " + str(num_levels)))
            logging.info(wrap_text("Found cand: " + str(found_cand)))

        mid_mask_frac = (max_mask_frac + min_mask_frac) / 2
        if num_levels == max_levels: 
            return found_cand 

        elif num_levels < max_levels: 
            if found_cand:
                return self.binary_search_edit(
                        edit_list, input_cand, contrast_pred_idx, num_rounds, 
                        min_mask_frac=min_mask_frac, 
                        max_mask_frac=mid_mask_frac, 
                        num_levels=num_levels+1,
                        sorted_token_indices=sorted_token_indices,
                        previous_text=previous_text
                )
            else:
                return self.binary_search_edit(
                    edit_list, input_cand,
                    contrast_pred_idx, num_rounds,
                    min_mask_frac=mid_mask_frac,
                    max_mask_frac=max_mask_frac,
                    num_levels=num_levels+1,
                    sorted_token_indices=sorted_token_indices,
                    previous_text=previous_text
                )
        else:
            error_msg = "Reached > max binary search levels." + \
                    f"({num_levels} > {max_levels})"
            raise RuntimeError(error_msg)

    def linear_search_edit(
            self, edit_list, input_cand, contrast_pred_idx, num_rounds, 
            min_mask_frac=0.0, max_mask_frac=0.5, max_levels=None, 
            sorted_token_indices=None, previous_text=None):
        
        """ Runs linear search over masking percentages from min_mask_frac
        to max_mask_frac. Calls run_edit_round at each mask percentage. """
        if max_levels == None: max_levels = self.max_search_levels
        mask_frac_step = (max_mask_frac - min_mask_frac) / max_levels
        mask_frac_iterator = np.arange(min_mask_frac+mask_frac_step, 
                                max_mask_frac + mask_frac_step, 
                                mask_frac_step)
        for mask_frac in mask_frac_iterator: 
            found_cand = self.run_edit_round(
                    edit_list, input_cand, contrast_pred_idx, 
                    num_rounds, mask_frac,
                    sorted_token_indices=sorted_token_indices, previous_text=previous_text)
            logging.info(wrap_text("Linear search mask_frac: " + str(mask_frac)))
            logging.info(wrap_text("Found cand: " + str(found_cand)))
            if found_cand:
                return found_cand
        return found_cand

    def clean_text_with_tokenizer(self, text_str):
        return self.editor.tokenizer.decode(
            self.editor.tokenizer.encode(text_str),
            clean_up_tokenization_spaces=True
        ).replace("</s>", " ")

    def minimally_edit(
            self,
            orig_input,
            contrast_pred_idx = -2,
            gold_label_idx = None,
            max_edit_rounds = 10,
            previous_text = None,
            curr_tokens = None,
            previous_tokens = None,
            label_position_idx = None,
            label_sequence = None
    ):

        """ Gets minimal edits for given input. 
        Calls search algorithm (linear/binary) based on self.search_method.
        contrast_pred_idx specifies which label to use as the contrast.
            Defaults to -2, i.e. use label with 2nd highest pred prob.

        Returns EditList() object. """

        # Get truncated editable part of input
        editable_seg = self.editor.get_editable_seg_from_input(orig_input)
        editable_seg = self.editor.truncate_editable_segs([editable_seg], inp=orig_input)[0]

        orig_input = self.editor.get_input_from_editable_seg(orig_input, editable_seg)
        num_toks = len(get_predictor_tokenized(self.predictor, editable_seg))
        assert num_toks <= self.predictor._dataset_reader._tokenizer._max_length

        editable_seg = self.clean_text_with_tokenizer(text_str=editable_seg)
        start_time = time.time()

        instance = self.editor.input_to_instance(orig_input, editable_seg=editable_seg)
        if isinstance(label_sequence, list):
            label_sequence = torch.tensor(label_sequence, device=self.device)
        if label_sequence is not None:
            instance.fields['class_ids'] = label_sequence
        if label_position_idx is not None:
            instance.fields['label_idx'] = label_position_idx

        orig_pred = self.predictor.predict_instance(instance)
        orig_pred = add_probs(orig_pred)
        orig_probs = orig_pred['probs']
        orig_pred_idx = np.array(orig_probs).argsort()[-1]
        orig_pred_label = self.editor.ints_to_labels[orig_pred_idx] 

        assert orig_pred_label == str(orig_pred_label)

        if gold_label_idx is None:
            contrast_pred_idx = np.array(orig_probs).argsort()[contrast_pred_idx]
            use_gold_label = False
        else:
            contrast_pred_idx = gold_label_idx
            use_gold_label = True

        contrast_label = self.ints_to_labels[contrast_pred_idx]
        orig_contrast_prob = orig_pred['probs'][contrast_pred_idx]
        assert orig_contrast_prob < 1.0

        num_rounds = 0
        new_pred_label = orig_pred_label

        logging.info(f"Predicted label: {orig_pred_label}")
        logging.info(f"Contrast label: {contrast_label}")
        logging.info(f"Orig contrast prob: {round(orig_contrast_prob, 3)}")

        edit_list = EditList(
            orig_input,
            editable_seg,
            orig_contrast_prob,
            orig_pred_label,
            contrast_label,
            orig_pred_idx
        )

        while (new_pred_label != contrast_label) or use_gold_label:
            num_rounds += 1
            prev_beam = edit_list.beam.copy()

            # Iterate through in reversed order (highest probabilities first)
            iterator = enumerate(reversed(sorted(prev_beam)))
            for beam_elem_idx, (score, _, input_cand) in iterator:
                sys.stdout.flush()
                logging.info(wrap_text(f"Updating beam for: {input_cand}"))
                logging.info(wrap_text(f"Edit round: {num_rounds} (1-indexed)"))
                logging.info(wrap_text(f"Element {beam_elem_idx} of beam"))
                logging.info(wrap_text(f"Predicted label: {orig_pred_label}"))
                logging.info(wrap_text(f"Contrast label: {contrast_label}"))
                logging.info(wrap_text(f"Contrast prob: {round(score, 3)}"))
                logging.info(wrap_text(f"Generating candidates..."))

                if self.editor.grad_pred == "original":
                    pred_idx = orig_pred_idx
                elif self.editor.grad_pred == "contrast":
                    pred_idx = contrast_pred_idx

                sorted_token_indices = self.editor.get_sorted_token_indices(input_cand, pred_idx)
                if len(sorted_token_indices) == 0:
                    logging.info(f"No masking candidates found for {input_cand}")
                    continue

                if self.search_method == "binary":
                    self.binary_search_edit(
                        edit_list,
                        input_cand,
                        contrast_pred_idx,
                        num_rounds,
                        max_mask_frac=self.max_mask_frac,
                        num_levels=1,
                        sorted_token_indices=sorted_token_indices,
                        previous_text=previous_text
                    )

                elif self.search_method == "linear":
                    self.linear_search_edit(
                        edit_list,
                        input_cand,
                        contrast_pred_idx,
                        num_rounds,
                        max_mask_frac=self.max_mask_frac,
                        sorted_token_indices=sorted_token_indices,
                        previous_text=previous_text
                    )
                
                if (len(edit_list.successful_edits) != 0) and (not use_gold_label):
                    logging.info(f"Found edit at edit round: {num_rounds}")
                    return edit_list

                logging.info("CURRENT BEAM after considering candidates: ")
                for prob, _, input_cand in reversed(sorted(edit_list.beam)):
                    logging.info(wrap_text(f"({round(prob, 4)}) {input_cand}"))

            highest_beam_element = sorted(list(edit_list.beam))[-1]
            _, _, input_cand = highest_beam_element
            num_minutes = round((time.time() - start_time)/60, 3)

            # If we've reached max # edit rounds, return highest cand in beam
            if num_rounds >= max_edit_rounds:
                logging.info(wrap_text("Reached max substitutions!"))
                return edit_list 

            if edit_list.beam == prev_beam:
                logging.info(wrap_text("Beam unchanged after updating beam."))
                return edit_list 

        return edit_list

class EditList():
    """ Class for storing edits/beam for a particular input. """

    def __init__(
            self, orig_input, orig_editable_seg, orig_contrast_prob, 
            orig_label, contrast_label, orig_pred_idx):

        self.orig_input = orig_input
        self.orig_editable_seg = orig_editable_seg 
        self.successful_edits = []
        self.orig_contrast_prob = orig_contrast_prob
        self.orig_label = orig_label
        self.orig_pred_idx = orig_pred_idx 
        self.contrast_label = contrast_label
        self.counter = 0
        self.beam = [(orig_contrast_prob, self.counter, orig_input)]
        heapq.heapify(self.beam)

    def add_edit(self, edit): # edit should be a dict
        orig_len = len(self.successful_edits)
        self.successful_edits.append(edit)
        assert len(self.successful_edits) == orig_len + 1

    def get_sorted_edits(self, key='minimality'):
        """How to sort the edits.
            If 'prob+' in key (e.g. key="prob+minimality"), then take the top 35% by probability and sort them
            by the second key.
        """
        if 'prob+' in key:
            edits = sorted(self.successful_edits, key=lambda x: -x['edited_prob'])
            top_k = max(5, int(len(edits) * .35))
            edits = edits[:top_k]
            return sorted(edits, key=lambda x: x[key.split('+')[1]])

        return sorted(self.successful_edits, key=lambda k: k[key])
