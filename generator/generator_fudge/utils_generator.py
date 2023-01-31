from operator import add

import numpy as np
import torch
from torch.autograd import Variable

import generator.utils_general as ug
import generator.utils_methods as um
import generator.generator_fudge.utils_result_handler as ur

# constants
import generator.utils_generator_base as ugb
import logging
import math
from transformers.generation_utils import (
    LogitsProcessorList,
    TopPLogitsWarper,
    TopKLogitsWarper,
)


LEN_PER_SPLIT = 2000

class GenerationFUDGE(ugb.GenerationBase):
    def __init__(self, device, *args, **kwargs):
        self.top_k_fudge = kwargs.get('top_k_fudge', None)
        super().__init__(device=device, *args, **kwargs)
        # results
        self.results = ur.GeneratorResultHandler(
            self.tokenizer, self.tb_logger, kwargs.get('print_tables', False),
            self.max_sentence_length, self.lm, sent_split_method=self.sentence_splitting_method,
            kind='perturbed', device=self.device, config=self.config
        )

    def _get_logits_processor(self):
        super()._get_logits_processor()
        if self.top_k_fudge is not None:
            self.logits_warper_pre = LogitsProcessorList()
            self.logits_warper_pre.append(TopKLogitsWarper(top_k=self.top_k_fudge))
            if self.top_p < 1 - 1e-6:
                self.logits_warper_pre.append(TopPLogitsWarper(top_p=self.top_p))
        else:
            self.logits_warper_pre = self.logits_warper

    def get_candidate_logits(self, logits, curr_sent_variant, labels, label_idx):
        # get candidate next words
        next_tok_cands = torch.where(~torch.isinf(logits))[1].unsqueeze(dim=1)
        cand_sents = self.results.document.get_all_tokens_tensor(
            current_sentence=curr_sent_variant, candidate_tensor=next_tok_cands, device='cpu', include_prompt=self.config.use_headline
        )  # start on CPU, then move each batch to GPU
        sent_lens = self.results.document.get_sentence_lens_tensor(
            current_sentence=curr_sent_variant, candidate_token=1, include_prompt=self.config.use_headline
        )

        # split data
        n_rows, n_tokens_per_row = cand_sents.shape
        total_num_tokens = n_rows * n_tokens_per_row
        num_splits = math.ceil(total_num_tokens / LEN_PER_SPLIT)
        n_rows_per_split = int(n_rows / num_splits)

        all_tag_logits_by_word = []
        for start_idx in range(0, n_rows, n_rows_per_split):
            end_idx = start_idx + n_rows_per_split
            cand_sents_batch = cand_sents[start_idx: end_idx].to(self.device)
            batch_tag_logits_by_word = self.full_discriminator.predict_candidate_batches(
                input_ids=cand_sents_batch, sequence_lens=sent_lens, labels=labels, label_idx=label_idx
            )
            batch_tag_logits_by_word = batch_tag_logits_by_word.detach().cpu()
            all_tag_logits_by_word.append(batch_tag_logits_by_word)
        return torch.vstack(all_tag_logits_by_word)

    def generate_text(self, context=None, class_ids=None, label_idx=None):
        # generate context
        class_id = class_ids[label_idx]
        output_so_far = ug.format_context(context, 'cpu')  # 2D tensor of token ids.

        #
        starting_word_idx = len(context)
        for word_idx in range(self.max_sentence_length + 1): # for i in length of sentence to generate
            # break if over max sentence length
            if starting_word_idx + word_idx > ugb.MAX_DOC_LENGTH:
                break

            # unpert logits: word distribution over the current output
            output_so_far = output_so_far.to(self.device)
            logits, past, all_hidden = self.lm.get_lmhead_logits_and_past_and_hidden(input_ids=output_so_far)
            logits = logits[:, -1, :]
            sentence_so_far = output_so_far[:, starting_word_idx:]
            logits = self.logits_processor(sentence_so_far, logits)
            logits = self.logits_warper_pre(sentence_so_far, logits)  # top k for selecting FUDGE
            logits = logits.detach().cpu()
            output_so_far = output_so_far.detach().cpu()

            all_tag_logits_by_word = self.get_candidate_logits(
                logits, self.results.curr_sentence_variant,
                labels=class_ids, label_idx=label_idx
            )
            #
            top_k_words = torch.where(~torch.isinf(logits))[1]
            logits = (
                 (all_tag_logits_by_word[:, class_id] * self.gm_scale) +
                 (logits[:, top_k_words] * (1 - self.gm_scale))
            )
            logits = self.logits_warper(sentence_so_far, logits)  # overall top-k
            probs = um._logits_to_probs(logits)

            if self.verbosity == 'verbose':
                self._log_top_words(word_probs=probs, cand_tokens=top_k_words)

            # Get the next word. (Either by sampling or greedy selection).
            cand_token = um.get_next_word(probs, self.sample)
            cand_token = top_k_words[cand_token]
            output_so_far, log_this_word, to_break, redo_word = um.check_sentence_end(
                cand_token, output_so_far, context, starting_word_idx, self.sentence_len_min_cutoff,
                self.sentence_splitting_method, self.tokenizer, self.spacy_nlp
            )

            if redo_word:
                continue

            if log_this_word:
                # log
                discrim_loss, predicted_tags = um.discrim_loss_on_tags(
                    document=self.results.document,
                    class_labels=class_ids,
                    current_sentence=self.results.curr_sentence_variant,
                    next_token=cand_token,
                    discriminator=self.full_discriminator,
                    idx_to_label=ugb.label_idx_to_str,
                    label_idx=label_idx,
                    device=self.device,
                    head='main',
                    config=self.config
                )
                self.results.add_word(last_token=cand_token, discrim_loss=discrim_loss)

            if to_break:
                break

    def _log_top_words(self, word_probs, cand_tokens):
        import pandas as pd
        vocab_dict = self.tokenizer.get_vocab()
        vocab_pairs = list(filter(lambda x: x[1] in cand_tokens, vocab_dict.items()))
        vocab_pairs = sorted(vocab_pairs, key=lambda x: x[1])
        vocab_sorted = list(map(lambda x: x[0], vocab_pairs))
        word_probs = word_probs.squeeze().cpu().detach().numpy()
        vocab_by_prob = pd.Series(word_probs, index=vocab_sorted)
        top_vocab = vocab_by_prob.sort_values(ascending=False)[:200]
        logging.info('TOP VOCAB: %s' % str(top_vocab.to_dict()))