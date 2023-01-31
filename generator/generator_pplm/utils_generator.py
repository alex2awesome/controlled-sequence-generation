from operator import add

import numpy as np
import torch
from torch.autograd import Variable

import generator.utils_general
import generator.utils_methods
from generator.generator_pplm import utils_methods as um
import generator.generator_pplm.utils_result_handler as ur
import generator.utils_generator_base as ugb
import logging
import torch.nn.functional as F


class GenerationPPLM(ugb.GenerationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # results
        self.results = ur.GeneratorResultHandler(
            self.tokenizer, self.tb_logger, kwargs.get('print_tables', False),
            self.max_sentence_length, self.lm, sent_split_method=self.sentence_splitting_method,
            kind='perturbed', device=self.device, config=self.config
        )

    def perturb_past(
            self, past, last, word_idx, unpert_past=None, unpert_logits=None, accumulated_hidden=None,
            class_labels=None, label_idx=None
    ):
        if self.num_iterations == 0 or past is None:
            return past, None

        # Generate initial perturbed past.a
        grad_accumulator = [np.zeros(p.shape).astype("float32") for p in past]
        accumulated_hidden = 0 if accumulated_hidden is None else accumulated_hidden

        # Get window mask and stepsize.
        window_mask = um._get_gradient_mask(past=past, window_length=self.window_length, decay=self.decay)
        current_stepsize = um._get_current_step_size(word_idx, self.step_size, self.grad_length)

        # Accumulate perturbations for num_iterations.
        loss_per_iter = torch.zeros(self.num_iterations)
        for i in range(self.num_iterations):
            curr_perturbation = [
                Variable(torch.from_numpy(p_).to(self.device), requires_grad=True) for p_ in grad_accumulator
            ]

            # 1. Compute hidden using perturbed past.
            probs, new_accumulated_hidden = um._compute_hidden_states(
                past, last, unpert_past, curr_perturbation,
                accumulated_hidden, self.lm, self.horizon_length
            )

            # 2. Get classifier predictions using perturbed hidden state.
            discrim_loss, prediction, _ = self.full_discriminator.predict_one_doc(
                inputs_embeds=new_accumulated_hidden,
                labels=class_labels,
                label_idx=label_idx,
                generate=True
            )

            # 3. Calculate the loss (loss = Discriminator loss + \alpha KL Loss)
            discrim_loss, kl_loss, combined_loss = um._calculate_loss(discrim_loss, unpert_logits, probs, self.kl_scale)
            loss_per_iter[i] = combined_loss.clone()

            # 4. Compute gradients based on the loss, store them
            combined_loss.backward()
            # discrim_loss.backward()
            grads = um._norm_gradients(window_mask, curr_perturbation, current_stepsize, self.gamma)
            grad_accumulator = list(map(add, grads, grad_accumulator))

            # 5. Clean up.
            past = um.clean_up(curr_perturbation, past)

        # apply the accumulated perturbations to the past
        grad_accumulator = [
            Variable(torch.from_numpy(p_).to(self.device), requires_grad=True) for p_ in grad_accumulator
        ]
        pert_past = list(map(add, past, grad_accumulator))
        return pert_past, torch.mean(loss_per_iter)

    def generate_text(self, context=None, class_ids=None, label_idx=None):
        # generate context
        class_id = class_ids[label_idx]
        output_so_far = generator.utils_general.format_context(context, self.device)  # 2D tensor of token ids.
        #
        past, last = None, None
        starting_word_idx = len(context)
        for word_idx in range(self.max_sentence_length + 1): # for i in length of sentence to generate
            # break if over max sentence length
            if ((starting_word_idx + word_idx) > ugb.MAX_DOC_LENGTH) or (output_so_far.shape[1] > ugb.MAX_DOC_LENGTH):
                logging.info('HITTING MAX_DOC_LEN: %s ' % starting_word_idx + word_idx)
                break

            # Get past/probs for current output, except for last word. Run model forward to obtain unperturbed
            # (GPT takes 2 inputs: past + current_token.)
            if output_so_far is not None:
                last = output_so_far[:, -1:]
                if output_so_far.shape[1] > 1:
                    _, past, _ = self.lm.get_lmhead_logits_and_past_and_hidden(input_ids=output_so_far[:, :-1])

            sentence_so_far = output_so_far[:, starting_word_idx:]

            # unpert logits: word distribution over the current output
            unpert_logits, unpert_past, unpert_all_hidden = self.lm.get_lmhead_logits_and_past_and_hidden(
                input_ids=output_so_far,
                past_key_values=past
            )
            unpert_last_hidden = unpert_all_hidden[-1]

            unpert_probs = F.softmax(unpert_logits, dim=-1)
            if unpert_probs[0, -1, 50256].item() <= 0.5: # prob of the EOS token
                # modify the past if to_perturb == True. (pert_past = past and loss_this_iter = None if to_perturb == False)
                accumulated_hidden = torch.sum(unpert_last_hidden[:, :-1, :], dim=1)
                pert_past, loss_this_iter = self.perturb_past(
                    past, last, unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    word_idx=word_idx, label_idx=label_idx,
                    accumulated_hidden=accumulated_hidden, class_labels=class_ids,
                )

                # using perturbed past, generate new logits.
                pert_logits, past, pert_all_hidden = self.lm.get_lmhead_logits_and_past_and_hidden(input_ids=last, past_key_values=pert_past)

                logits = self.gm_scale * pert_logits[:, -1, :] + (1 - self.gm_scale) * unpert_logits[:, -1, :]
                logits = self.logits_processor(sentence_so_far, logits)
                logits = self.logits_warper(sentence_so_far, logits)
                probs = F.softmax(logits, dim=-1)
            else:
                probs = unpert_probs[:, -1, :]
                loss_this_iter = None
                pert_all_hidden = None

            # rescale
            if torch.sum(probs) <= 1:
                probs = probs / torch.sum(probs)

            # fuse predictions made by perturbed past with unpeturbed past.
            unpert_discrim_loss, _, _ = self.full_discriminator.predict_one_doc(
                labels=class_ids, label_idx=label_idx, inputs_embeds=torch.mean(unpert_last_hidden, dim=1),
                generate=True
            )

            if self.verbosity == 'verbose':
                self._log_top_words(word_probs=probs)

            # Get the next word. (Either by sampling or greedy selection).
            last = generator.utils_methods.get_next_word(probs, self.sample)

            if self.verbosity == 'verbose':
                logging.info('CHOSEN WORD: %s' % self.tokenizer.decode(last.squeeze()))

            output_so_far, log_this_word, to_break, redo_word = generator.utils_methods.check_sentence_end(
                last, output_so_far, context, starting_word_idx, self.sentence_len_min_cutoff,
                self.sentence_splitting_method, self.tokenizer, self.spacy_nlp
            )
            if redo_word: # if we want to skip a token for some reason
                continue

            if log_this_word:
                # log
                self.results.add_word(
                    last_token=last,
                    loss_this_iter=loss_this_iter,
                    unpert_discrim_loss=unpert_discrim_loss,
                    pert_h=pert_all_hidden,
                    unpert_h=unpert_all_hidden
                )

            if to_break:
                break
