import torch

from generator import utils_result_handler_base as ur, utils_general as ug, utils_methods as um
from generator.utils_generator_base import GenerationBase, MAX_DOC_LENGTH
from util.utils_prompting import PromptGenerator
import logging
from util.utils_general import label_idx_to_str
import numpy as np
from editing.src.masker import MaskError

class GenerationUnperturbed(GenerationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # results
        self.results = ur.GeneratorResultHandlerBase(
            self.tokenizer, self.tb_logger, kwargs.get('print_tables', False),
            self.max_sentence_length, self.lm, sent_split_method=self.sentence_splitting_method,
            kind='unperturbed', device=self.device, config=self.config
        )

    def generate_text(self, context=None, class_ids=[-1], label_idx=0):
        # generate context
        output_so_far = ug.format_context(context, self.device)  # 2D tensor of token ids.
        #
        starting_word_idx = len(context)
        for word_idx in range(self.max_sentence_length + 1): # for i in length of sentence to generate
            # break if over max sentence length
            if starting_word_idx + word_idx > MAX_DOC_LENGTH:
                break

            # unpert logits: word distribution over the current output
            logits, past, all_hidden = self.lm.get_lmhead_logits_and_past_and_hidden(input_ids=output_so_far)
            all_hidden = tuple(list(map(lambda x: x[:, [-1]], all_hidden)))
            last_hidden = all_hidden[-1]
            logits = logits[:, -1, :]
            sentence_so_far = output_so_far[:, starting_word_idx:]
            logits = self.logits_processor(sentence_so_far, logits)
            logits = self.logits_warper(sentence_so_far, logits)

            probs = um._logits_to_probs(logits)
            discrim_loss, _, _ = self.full_discriminator.predict_one_doc(
                labels=class_ids,
                inputs_embeds=torch.mean(last_hidden, dim=1),
                label_idx=label_idx,
                generate=True,
                add_features='main',
            )

            # Get the next word. (Either by sampling or greedy selection).
            last = um.get_next_word(probs, self.sample)

            output_so_far, log_this_word, to_break, redo_word = um.check_sentence_end(
                last, output_so_far, context, starting_word_idx, self.sentence_len_min_cutoff,
                self.sentence_splitting_method, self.tokenizer, self.spacy_nlp
            )
            if redo_word:
                continue
            if log_this_word:
                # log
                self.results.add_word(last_token=last, discrim_loss=discrim_loss)

            if to_break:
                break


class GenerationBaselineTwo(GenerationUnperturbed):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # results
        self.results = ur.GeneratorResultHandlerBase(
            self.tokenizer, self.tb_logger, kwargs.get('print_tables', False),
            self.max_sentence_length, self.lm, sent_split_method=self.sentence_splitting_method,
            kind='unperturbed', device=self.device, config=self.config
        )
        gen_type = self.kwargs['generator_type'].split('-')[1]
        assert gen_type in ['future', 'baseline', 'past']
        self.prompt_gen = PromptGenerator(prompt_type=gen_type)

    def generate_text(self, context=None, class_ids=[-1], label_idx=0):
        # generate context
        headline = ug.tokens_to_str(self.results.document.tokenized_prompt, tokenizer=self.tokenizer, remove_eos=True)
        prior_sentences = self.results.document.sentences
        prior_sentences = list(map(lambda x: x.get_detokenized_text(remove_eos_token=True), prior_sentences))
        labels = list(map(ug.get_class_str, class_ids))
        prompt = self.prompt_gen.generate_prompt_full(
            headline=headline,
            sentences=prior_sentences,
            labels=labels,
            s_idx=label_idx,
        )
        tok_prompt = self.tokenizer.encode(prompt)
        super().generate_text(context=tok_prompt, class_ids=class_ids, label_idx=label_idx)


class GenerationHuman(GenerationUnperturbed):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # results
        self.results = ur.GeneratorResultHandlerBase(
            self.tokenizer, self.tb_logger, kwargs.get('print_tables', False),
            self.max_sentence_length, self.lm, sent_split_method=self.sentence_splitting_method,
            kind='human', device=self.device, config=self.config
        )

    def generate_sentence(self, tokenized_cond_text, label_idx, class_labels=[-1], **kwargs):
        # start new sentence
        class_ids = list(map(lambda x: ug.get_class_id(x), class_labels))
        class_label = class_labels[label_idx]
        self.results.start_new_sentence(class_label=class_label, label_idx=label_idx)
        sentence = self.results.make_sentence_from_text(kwargs.get('sentence_text'), s_idx=label_idx)
        if self.perform_edits:
            sentence, edited = self.edit_sentence(curr_best_sentence=sentence, class_ids=class_ids, label_idx=label_idx)
            sentence = self.results.finish_sentence(chosen_sentence=sentence, full_discriminator=self.full_discriminator)
        else:
            edited = False
            sentence = self.results.finish_sentence(chosen_sentence=sentence, full_discriminator=self.full_discriminator)
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
        sentences = kwargs.get('sentences')
        if self.config.use_headline:
            class_labels = ['headline'] + class_labels
            sentences = [cond_text] + sentences
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
                label_idx=label_idx,
                sentence_text=sentences[label_idx]
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
