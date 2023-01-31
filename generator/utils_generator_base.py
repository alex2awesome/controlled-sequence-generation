import numpy as np
from pprint import pformat

import generator.utils_general as ug
import generator.utils_methods as um
import util.utils_model_loader as uml
from editing.src.masker import MaskError
import pandas as pd
import logging
from util.utils_general import label_idx_to_str
from transformers.generation_logits_process import LogitsProcessor
from transformers.generation_utils import (
    LogitsProcessorList,
    ForcedEOSTokenLogitsProcessor,
    MinLengthLogitsProcessor,
    TemperatureLogitsWarper,
    InfNanRemoveLogitsProcessor,
    TopPLogitsWarper,
    TopKLogitsWarper,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor
)
import torch

VERBOSE_LOGGING = False
NUM_CLASSES = 9
MAX_DOC_LENGTH = 2048
PPRINT = False


class BanSpecialTokensLogitProcessor(LogitsProcessor):
    def __init__(self, tokenizer, device):
        self.added_tokens_keys = torch.tensor(list(tokenizer.added_tokens_decoder.keys()), device=device)
        self.cull = len(self.added_tokens_keys) > 0

    def __call__(self, input_ids, scores):
        if self.cull:
            scores[:, self.added_tokens_keys] = -float("inf")
        return scores


class GenerationBase(uml.ModelLoader):
    def __init__(
            self,
            stepsize=0.01,
            horizon_length=1,
            window_length=0,
            max_sentence_length=100,
            decay=False,
            gamma=1.5,
            kl_scale=0.01,
            gm_scale=.9,
            grad_length=1000,
            num_iterations=3,
            temperature=1.0,
            top_k=10,
            top_p=1.0,
            repetition_penalty=1.2,
            sample=True,
            #
            num_sentences_to_generate=1,
            #
            sentence_splitting_method="eos",
            #
            device='cpu',
            force_eos_token=True,
            **kwargs
    ):
        super().__init__(device=device, **kwargs)
        self.num_iterations = num_iterations
        self.max_sentence_length = max_sentence_length
        self.device = device

        # model params
        self.step_size = stepsize
        self.horizon_length = horizon_length # for hidden states
        self.window_length = window_length   # for gradient mask
        self.decay = decay
        self.gamma = gamma
        self.kl_scale = kl_scale
        self.gm_scale = gm_scale
        self.grad_length = grad_length
        self.temperature = temperature
        self.sentence_len_min_cutoff = kwargs.get('sentence_min_len', 0)

        # sampling
        self.top_k = top_k
        self.top_p = top_p
        self.sample = sample
        self.repetition_penalty = repetition_penalty
        self.sentence_splitting_method = sentence_splitting_method
        self.num_sents_to_generate = num_sentences_to_generate
        self.force_eos_token = force_eos_token

        # spacy model
        self.spacy_nlp = kwargs.get('spacy_model')
        self.tb_logger = kwargs.get('tb_logger')
        self.run_tagger = kwargs.get('run_tagger')
        self.tag_each_sentence_variant = kwargs.get('tag_each_sentence_variant')

        # logits processor
        self._get_logits_processor()
        self.verbosity = kwargs.get('verbosity', 'regular')

    def _log_top_words(self, word_probs):
        vocab_dict = self.tokenizer.get_vocab()
        vocab_sorted = list(map(lambda x: x[0], sorted(vocab_dict.items(), key=lambda x: x[1])))
        word_probs = word_probs.squeeze().cpu().detach().numpy()
        vocab_by_prob = pd.Series(word_probs, index=vocab_sorted)
        top_vocab = vocab_by_prob.sort_values(ascending=False)[:200]
        logging.info('TOP VOCAB: %s' % str(top_vocab.to_dict()))

    def _get_logits_processor(self):
        logits_processor = LogitsProcessorList()
        logits_processor.append(TemperatureLogitsWarper(temperature=self.temperature))
        if self.force_eos_token:
            logits_processor.append(ForcedEOSTokenLogitsProcessor(max_length=self.max_sentence_length, eos_token_id=self.tokenizer.eos_token_id))
        logits_processor.append(InfNanRemoveLogitsProcessor())
        logits_processor.append(MinLengthLogitsProcessor(min_length=self.sentence_len_min_cutoff, eos_token_id=self.tokenizer.eos_token_id))
        logits_processor.append(NoRepeatNGramLogitsProcessor(ngram_size=3))
        logits_processor.append(BanSpecialTokensLogitProcessor(tokenizer=self.tokenizer, device=self.device))
        if abs(self.repetition_penalty - 1.0) > 1e-6:
            logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=self.repetition_penalty))

        logits_warper = LogitsProcessorList()
        logits_warper.append(TopKLogitsWarper(top_k=self.top_k))
        if self.top_p < 1 - 1e-6:
            logits_warper.append(TopPLogitsWarper(top_p=self.top_p))
        self.logits_processor = logits_processor
        self.logits_warper = logits_warper


    def edit_sentence(self, sent_ppls=None, curr_best_sentence=None, class_ids=None, label_idx=None):
        if sent_ppls is not None:
            curr_best_sent_idx = np.argmin(sent_ppls)
            curr_best_sentence = self.results.sentence_variants[curr_best_sent_idx]
        curr_best_sentence_text = curr_best_sentence.get_detokenized_text(remove_eos_token=True)
        # sometimes if the edit has length 0, we cannot edit.
        try:
            previous_text = self.results.document.get_current_detokenized_text(include_prompt=True, n_back=10)
            previous_tokens = self.results.document.get_all_tokens_tensor(include_prompt=True, n_back=10)
            edited_sentence_list = self.edit_finder.minimally_edit(
                orig_input=curr_best_sentence_text,
                gold_label_idx=class_ids[label_idx],
                previous_text=previous_text,
                previous_tokens=previous_tokens,
                label_position_idx=label_idx,
                label_sequence=class_ids
            )
            successful_edits = edited_sentence_list.successful_edits
        except (AssertionError, MaskError):
            successful_edits = []
            logging.info('BUG: Assertion Error in edit_sentence')

        found = False
        if len(successful_edits) > 0:
            candidates = edited_sentence_list.get_sorted_edits(key='prob+fluency')
            for idx, c in enumerate(candidates):
                try:
                    o = ug.clean_punct_text(c['edited_input'])
                except:
                    o = ''
                if o != '':
                    found = True
                    break
        if found:
            logging.info('EDITED SENTENCES: %s' % str(edited_sentence_list.successful_edits))
            best_edited_sentence = edited_sentence_list.successful_edits[idx]
            best_edited_sentence['edited_input'] = o
            sentence = self.results.make_sentence_from_text(best_edited_sentence['edited_input'], s_idx=label_idx)
            edited = True
            # convert from numpy to float
            best_edited_sentence['orig_prob'] = best_edited_sentence['orig_prob'].item()
            best_edited_sentence['edited_prob'] = best_edited_sentence['edited_prob'].item()
            sentence.edited = best_edited_sentence
        else:
            logging.info('NO SUCCESSFUL EDIT FOUND')
            sentence = curr_best_sentence
            edited = False
        return sentence, edited

    def generate_sentence(self, tokenized_cond_text, label_idx, class_labels=[-1], **kwargs):
        # start new sentence
        class_ids = list(map(lambda x: ug.get_class_id(x), class_labels))
        class_label = class_labels[label_idx]
        self.results.start_new_sentence(class_label=class_label, label_idx=label_idx)

        # generate text
        for _ in range(self.num_sents_to_generate):
            self.results.start_new_sentence_variant()
            if self.verbosity == 'verbose':
                logging.info('Starting new sentence variant....')
            self.generate_text(
                context=tokenized_cond_text,
                class_ids=class_ids,
                label_idx=label_idx,
            ) # this method updates self.results
            if self.tag_each_sentence_variant:
                self.results.finish_sentence_variant(full_discriminator=self.full_discriminator)
            else:
                self.results.finish_sentence_variant()

        # choose best sentence
        # self.results.reject_sentence_variants(min_len_cutoff=self.sentence_len_min_cutoff)
        sent_ppls = self.results._get_current_sent_variants_ppls()
        logging.info('SENTENCE PPLS: %s. Choosing: %s' % (str(sent_ppls), np.argmin(sent_ppls)))
        sentence_variants = self.results.get_all_decoded_sentence_variants()
        if PPRINT:
            logging.info('SENTENCE VARIANTS: %s' % pformat(sentence_variants))
        else:
            logging.info('SENTENCE VARIANTS: %s' % sentence_variants)
        if self.tag_each_sentence_variant:
            logging.info('SENTENCE VARIANT CLASS PREDS: %s' % str(self.results._get_class_predictions()))
        if self.perform_edits:
            sentence, edited = self.edit_sentence(sent_ppls, class_ids, label_idx)
            sentence = self.results.finish_sentence(chosen_sentence=sentence, full_discriminator=self.full_discriminator)
        else:
            edited = False
            sentence = self.results.finish_sentence(chosen_variant_idx=np.argmin(sent_ppls), full_discriminator=self.full_discriminator)

        logging.info('FINAL SENTENCE: %s' % str({
            'edited': edited,
            'sentence text': sentence.detokenized_text,
            'sentence ppl': sentence.sentence_ppl,
            'gold label': class_label,
            'predicted labels': sentence.class_predictions
        }))


    def generate_document(self, cond_text='', uncond=False, num_sentences_to_gen=3, class_labels=None, **kwargs):
        # figure out conditioning text
        cond_text, tokenized_cond_text = um._encode_text(self.tokenizer, cond_text, uncond, use_bos_token=False)

        # generate random sequence of class labels
        if class_labels is None:
            class_labels = np.random.choice(range(NUM_CLASSES), num_sentences_to_gen)

        if self.config.use_headline:
            class_labels = ['headline'] + class_labels
            s = 1
        else:
            s = 0

        # clear metrics state
        self.results.start_new_document(cond_text=cond_text, tokenized_cond_text=tokenized_cond_text, class_labels=class_labels)

        if self.config.doc_sent_length_cutoff is not None:
            n = min(len(class_labels), self.config.doc_sent_length_cutoff)
        else:
            n = len(class_labels)

        for label_idx in range(s, n):
            self.generate_sentence(
                tokenized_cond_text=tokenized_cond_text,
                class_labels=class_labels,
                label_idx=label_idx
            )
            # todo: move this text-handling into the results class
            sentence = self.results._get_recently_added_text()
            cond_text = ' '.join([cond_text, sentence]).strip()
            if self.results.document.total_word_counter > MAX_DOC_LENGTH:
                print('HITTING MAX DOC LENGTH...')
                break

            # prepare text for next iteration.
            cond_text, tokenized_cond_text = um._encode_text(self.tokenizer, cond_text, uncond, use_bos_token=False)

        # sentence-level metrics
        if self.run_tagger:
            label_ids, tags = um.tag_document_discriminator(
                self.results.document, self.full_discriminator, label_idx_to_str, self.device, self.config
            )
            logging.info('SENTENCE PREDICTIONS: %s' % str(tags))

        return self.results


