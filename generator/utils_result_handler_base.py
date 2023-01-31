import generator.utils_general as ug
import generator.utils_methods as um
import torch
import transformers
from packaging import version
import numpy as np
import logging

des_keys = [
    'per-step ppl',
]


def tensor_to_float(x):
    if isinstance(x, float):
        return x
    return float(x.detach().cpu())


def tensor_to_int(x):
    if isinstance(x, int):
        return x
    return int(x.detach().cpu())


class WordBase():
    def __init__(self, s_idx, w_idx, token_str, label, tok, *args, **kwargs):
        self.s_idx = s_idx
        self.w_idx = w_idx
        self.token_str = token_str
        self.label = label
        # other attributes
        self.word_level_nll = kwargs.get('word_level_nll')
        self.per_step_ppl = kwargs.get('per_step_ppl')
        self.tok = tensor_to_int(tok)

    def __getattr__(self, item):
        if item == 'de-tok':
            return self.token_str
        return self.__dict__[item]

    def to_dict(self):
        return self.__dict__

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, item):
        return self.__dict__[item]


class Sentence():
    def __init__(self, kind, class_label=None, device='cpu', tokenizer=None, s_idx=None):
        self.word_metrics = []
        self._tokens_so_far = []    # list of ints (OLD: each of last token ( ex. tensor([[318]]) ).)
                                    # NOTE: this does not capture the prompt, just generated text.
        self._tokens_so_far_tensor = None
        self.w_idx = 0
        self.s_idx = s_idx
        self.class_label = class_label
        self.kind = kind
        self._detokenized_text = None
        self.device = device
        self.class_predictions = None
        self.tokenizer = tokenizer
        self.ppl_method = 'mean'
        self.edited = None

    @property
    def tokens_so_far_tensor(self):
        if self._tokens_so_far_tensor is not None:
            return self._tokens_so_far_tensor
        return torch.tensor(self._tokens_so_far, device=self.device)

    @property
    def tokens_so_far_list(self):
        return self._tokens_so_far

    @property
    def word_lls(self):
        word_lls = list(map(lambda x: x.word_level_nll, self.word_metrics))
        if np.isnan(word_lls).any():
            logging.warning('You have some words that have NaN for their word_level_nll. This will cause sentence-level ppl to be NaN. Dropping for now, but you better double-check. Most likely has to do with no context during LM evaluation...')
            word_lls = list(filter(lambda x: ~np.isnan(x), word_lls))
        return word_lls

    @property
    def num_tokens(self):
        return len(self.tokens_so_far_list)

    @property
    def per_step_ppls(self):
        return list(map(lambda x: x.per_step_ppl, self.word_metrics))

    @property
    def sentence_ppl(self):
        if self.ppl_method == 'mean':
            return sum(self.per_step_ppls) / len(self.per_step_ppls)
        else:
            return np.exp(sum(self.word_lls) / len(self.word_lls))

    @property
    def detokenized_text(self):
        if self._detokenized_text is None:
            self._detokenized_text = self.get_detokenized_text(append_eos_token=True)
        return self._detokenized_text

    def _replace_tokens(self, tokens, tokens_to_replace):
        """
        `tokens` is the input text.
        `tokens_to_replace` is a tuple of (old, new).
        """
        output_list = []
        for tok in tokens:
            replaced = False
            for old, new in tokens_to_replace:
                if tok == old:
                    output_list.append(new)
                    replaced = True
                    break
            if not replaced:
                output_list.append(tok)
        return output_list

    def post_process_sentence(self):
        """Replace all newlines with spaces and add a period if the sentence doesn't end with
        a period, exclamation mark or question mark.
        """
        print('POST-PROCESSING SENTENCE')
        space_token = self.tokenizer.encode('I ')[-1] # for some reason, space by itself doesn't encode to anything.
        newline_token = self.tokenizer.encode('I\n')[-1]
        question_mark = self.tokenizer.encode('? ?')
        exclamation_mark = self.tokenizer.encode('! !')
        period = self.tokenizer.encode('. .')

        comma = self.tokenizer.encode(', ,')
        colon = self.tokenizer.encode(': :')
        semi_colon = self.tokenizer.encode('; ;')
        # replace
        replacement_pairs = [(newline_token, space_token)]
        self._tokens_so_far = self._replace_tokens(self._tokens_so_far, replacement_pairs)
        # chop off
        non_sent_enders = comma + colon + semi_colon
        if self._tokens_so_far[-1] in non_sent_enders:
            self._tokens_so_far.pop(-1)
        # add
        sentence_enders = period + question_mark + exclamation_mark
        if self._tokens_so_far[-1] not in sentence_enders:
            self._tokens_so_far.append(period[0])

        self._tokens_so_far_tensor = torch.tensor(self._tokens_so_far, device=self.device)

    def get_detokenized_text(self, remove_eos_token=False, append_eos_token=False):
        tokens = self.tokens_so_far_list
        if remove_eos_token:
            tokens = list(filter(lambda x: x != self.tokenizer.eos_token_id, tokens))
            return self.tokenizer.decode(tokens)
        if append_eos_token:
            if self.tokenizer.eos_token_id not in tokens:
                tokens.append(self.tokenizer.eos_token_id)
        return self.tokenizer.decode(tokens)

    def add_word_metric(self, metric):
        self.word_metrics.append(metric)
        self.w_idx += 1

    def add_token(self, token):
        token = tensor_to_int(token)
        self._tokens_so_far.append(token)

    def __len__(self):
        return len(self._tokens_so_far)


class Document():
    def __init__(self, kind, tokenized_prompt=None, device='cpu', tokenizer=None, class_labels=None):
        self.kind = kind
        self.s_idx = 0
        self.total_word_counter = 0
        self.sentences = []
        self.device = device
        self.tokenized_prompt = tokenized_prompt  # assumes tokenized prompt is a 1-D LIST of ints
        self.tokenizer = tokenizer
        self.class_labels = class_labels
        self._class_ids = None

    @property
    def class_ids(self):
        if self._class_ids is None:
            self._class_ids = list(map(ug.get_class_id, self.class_labels))
        return self._class_ids

    @property
    def prompt_as_tensor(self):
        return torch.tensor(self.tokenized_prompt, device=self.device)

    def add_sentence(self, chosen_sentence):
        self.sentences.append(chosen_sentence)
        self.total_word_counter += len(chosen_sentence.detokenized_text.split())
        self.s_idx += 1

    def get_sentence_lens_list(self, curr_sentence=None, include_prompt=False):
        lengths = []
        if include_prompt:
            lengths.append(len(self.tokenized_prompt))
        lengths = list(map(lambda x: x.num_tokens, self.sentences))
        if curr_sentence is not None:
            lengths.append(curr_sentence.num_tokens)
        return lengths

    def get_sentence_lens_tensor(self, current_sentence=None, include_prompt=False):
        lens_list = self.get_sentence_lens_list(current_sentence, include_prompt)
        return torch.tensor(lens_list, device=self.device)

    def get_current_detokenized_text(self, n_back=-1, include_prompt=True, remove_eos_token=True, append_eos_token=False):
        output_text = []
        if include_prompt:
            tokens = self.tokenized_prompt
            if remove_eos_token:
                tokens = list(filter(lambda x: x != self.tokenizer.eos_token_id, tokens))
            output_text.append(self.tokenizer.decode(tokens))
        n_back = None if n_back == -1 else -abs(n_back)
        for s in self.sentences[:n_back]:
            output_text.append(s.get_detokenized_text(remove_eos_token=remove_eos_token, append_eos_token=append_eos_token))
        return ' '.join(output_text)

    def get_all_tokens_list(self, curr_sentence=None, n_back=-1, include_prompt=None):
        """
            Returns a list of prompt.tokens + previous sentences.tokens + curr_sentence.tokens
            return a torch.tensor([[ <tokens> ]])
        """
        output_tokens = []
        if n_back == -1:
            n_back = len(self.sentences) + 1

        # prompt
        if n_back > len(self.sentences):
            output_tokens += self.tokenized_prompt

        # previous sentences
        for s in self.sentences[-n_back:]:
            output_tokens += s.tokens_so_far_list

        # current sentence
        if curr_sentence is not None:
            output_tokens += curr_sentence.tokens_so_far_list

        return output_tokens

    def get_all_tokens_tensor(self, current_sentence=None, n_back=-1, unsqueeze=True, include_prompt=True):
        # output_tokens_list = self.get_all_tokens_list(current_sentence, n_back, include_prompt=include_prompt)
        # output_tokens_tensor = torch.tensor(output_tokens_list, device=self.device)
        sentence_tensors = list(map(lambda x: x.tokens_so_far_tensor, self.sentences))
        if include_prompt:
            sentence_tensors = [self.prompt_as_tensor] + sentence_tensors
        if current_sentence is not None:
            sentence_tensors.append(current_sentence.tokens_so_far_tensor)
        output_tokens_tensor = torch.cat(sentence_tensors)

        if unsqueeze:
            return output_tokens_tensor.unsqueeze(dim=0)
        else:
            return output_tokens_tensor

    def get_flat_word_lls(self, sentence):
        word_level_lls = [l for s in self.sentences for l in s.word_lls]
        word_level_lls += sentence.word_lls
        return word_level_lls

    def get_flat_word_metrics(self):
        return [l for s in self.sentences for l in s.word_metrics]

    def get_new_text(self):
        return self.sentences[-1].detokenized_text


class GeneratorResultHandlerBase():
    def __init__(self, tokenizer, tb_logger, print_tables,
                 max_sentence_length, lm,
                 sent_split_method,
                 kind='perturbed', device='cpu', config=None):
        self.print_tables = print_tables
        self.tokenizer = tokenizer
        self.tb_logger = tb_logger
        self.sent_split_method = sent_split_method
        self.max_sentence_length = max_sentence_length
        self.lm = lm
        self.kind = kind
        self.device = device
        self.config = config

        # self.start_new_document(cond_text=None)
        logging.info('instantializing GeneratorResultHandler on device: %s' % self.device)

    def start_new_document(self, cond_text, tokenized_cond_text=None, class_labels=None):
        if cond_text is not None:
            logging.info('------- PROMPT ---------------')
            logging.info(cond_text)
            logging.info('------------------------------')
            if self.tb_logger is not None:
                self.tb_logger.add_text('prompt', cond_text)
        self.document = Document(
            kind=self.kind, class_labels =class_labels,
            tokenized_prompt=tokenized_cond_text, device=self.device,
            tokenizer=self.tokenizer,
        )

    def start_new_sentence(self, class_label=None, label_idx=None):
        self.sentence_variants = []
        self.current_class_label = class_label
        self.current_s_idx = label_idx

    def start_new_sentence_variant(self):
        self.curr_sentence_variant = Sentence(
            kind=self.kind,
            class_label=self.current_class_label,
            device=self.device,
            tokenizer=self.tokenizer,
            s_idx=self.current_s_idx
        )

    def finish_sentence(self, chosen_variant_idx=0, chosen_sentence=None, full_discriminator=None):
        """Choose the sentence variant we want, delete the list of sentence variants and
        append the sentence to the document.

        Optionally tag the sentence with class predictions and return the sentence so we can examine it.
        """
        if chosen_sentence is None:
            chosen_sentence = self.sentence_variants[chosen_variant_idx]

        if full_discriminator is not None:
            predictions = self._calculate_class_predictions(full_discriminator, chosen_sentence)
            chosen_sentence.class_predictions = predictions

        # memory management
        del self.sentence_variants
        torch.cuda.empty_cache()
        #
        self.sentence_variants = []
        self.document.add_sentence(chosen_sentence)
        return chosen_sentence

    def _calculate_class_predictions(self, full_discriminator, sentence=None):
        if sentence is None:
            sentence = self.curr_sentence_variant

        toks = self.document.get_all_tokens_tensor(current_sentence=sentence, include_prompt=self.config.use_headline)
        lens = self.document.get_sentence_lens_tensor(current_sentence=sentence, include_prompt=self.config.use_headline)
        #
        _, predictions, _ = full_discriminator.predict_one_doc(
            input_ids=toks,
            sequence_lens=lens,
            add_features='main',
            labels=self.document.class_ids,
            label_idx=sentence.s_idx,
            generate=True
        )
        predictions = predictions.squeeze().detach().cpu().tolist()
        return predictions

    def finish_sentence_variant(self, full_discriminator=None):
        if full_discriminator is not None:
            predictions = self._calculate_class_predictions(full_discriminator)
            self.curr_sentence_variant.class_predictions = predictions
        if self.sent_split_method == 'spacy':
            self.curr_sentence_variant.post_process_sentence()
        self.sentence_variants.append(self.curr_sentence_variant)
        return self.curr_sentence_variant

    def make_sentence_from_text(self, input_text, full_discriminator=None, s_idx=None):
        """Used when we want to edit a sentence."""
        sentence = Sentence(
            kind=self.kind,
            class_label=self.current_class_label,
            device=self.device,
            tokenizer=self.tokenizer,
            s_idx=s_idx
        )
        tokens = self.tokenizer.encode(input_text)
        for tok in tokens:
            sentence = self.add_word(tok, sentence=sentence)
        if full_discriminator is not None:
            preds = self._calculate_class_predictions(full_discriminator, sentence)
            sentence.class_predictions = preds
        return sentence

    def add_word(self, last_token, sentence=None, *args, **kwargs):
        curr_sentence_var = self.curr_sentence_variant if sentence is None else sentence
        curr_sentence_var.add_token(last_token)
        word_metrics = self._get_word_level_metrics(last_token, sentence_var=curr_sentence_var,  *args, **kwargs)
        curr_sentence_var.add_word_metric(word_metrics)
        return curr_sentence_var

    def reject_sentence_variants(self, min_len_cutoff=None):
        """Reject sentence variants that don't meet a certain criteria.

        * `min_len_cutoff`: number of tokens that sentence needs to be longer than.
        """
        self.sentence_variants = list(filter(lambda x: len(x) > min_len_cutoff, self.sentence_variants))

    # helper methods
    def _get_current_sent_variants_ppls(self):
        return list(map(lambda x: x.sentence_ppl, self.sentence_variants))

    def _get_class_predictions(self):
        return list(map(lambda x: x.class_predictions, self.sentence_variants))

    def _get_recently_added_text(self):
        return self.document.get_new_text()

    # helper methods
    def _get_word_level_metrics(self, last_token, sentence_var=None, discrim_loss=None):
        if sentence_var is None:
            sentence_var = self.curr_sentence_variant

        w_idx, s_idx = sentence_var.w_idx, sentence_var.s_idx
        token_str = self._decode(last_token)
        word_metrics = WordBase(s_idx=s_idx, w_idx=w_idx, token_str=token_str,
                                label=self.current_class_label, tok=last_token)

        # process discrim, combined loss and hidden states
        global_step = w_idx + (s_idx * self.max_sentence_length)
        word_metrics.discrim_loss = self._process_discrim_loss(discrim_loss, global_step)

        # get log likelihoods
        all_tokens_tensor = self.document.get_all_tokens_tensor(
            sentence_var, unsqueeze=True, include_prompt=True
        )
        curr_ll = um.get_word_ll(all_tokens_tensor, self.lm)

        word_metrics.word_level_nll = curr_ll
        word_metrics.per_step_ppl = self._calculate_ppl(curr_ll, global_step, sentence_var)
        return word_metrics

    def _process_discrim_loss(self, discrim_loss, global_step):
        if discrim_loss is not None:
            discrim_loss = tensor_to_float(discrim_loss)
            if self.tb_logger is not None:
                self.tb_logger.add_scalar('discrim_loss', discrim_loss, global_step=global_step)
            return discrim_loss

    def _decode(self, last_token):
        if version.parse(transformers.__version__) > version.parse('4.0.0'):
            if isinstance(last_token, int):
                return self.tokenizer.decode(last_token)
            return self.tokenizer.decode(last_token[0])
        else:
            return self.tokenizer.decode(last_token)

    def _calculate_ppl(self, curr_ll, global_step, sentence_var):
        word_lls = [curr_ll] + self.document.get_flat_word_lls(sentence_var)  # ex. word_lls = [1.6538]
        ppl = np.exp(sum(word_lls) / len(word_lls))
        if self.tb_logger is not None:
            self.tb_logger.add_scalar('per-step PPL', ppl, global_step=global_step)
        return ppl

    # prettify output methods
    def print_tables(self, print_sentence_table, print_word_table):
        table = ug.PrettyDocumentTable(kind=self.kind).from_document(d=self.document)
        table.print_tables(print_sentence_table, print_word_table)

    def get_all_decoded_sentence_variants(self):
        """
        Get all decoded sentences as dicts of {'sentence_text': '...',  'ppl': float} measurements.
        """
        output_sentences = []
        for sentence in self.sentence_variants:
            output_block = {
                'sentence_text': sentence.detokenized_text,
                'ppl': sentence.sentence_ppl,
                'token_len': len(sentence)
            }
            if sentence.class_predictions is not None:
                output_block['class_predictions'] = sentence.class_predictions
            output_sentences.append(output_block)
        return output_sentences

    def get_final_document(self):
        output_sentences = []
        for sentence in self.document.sentences:
            output_sentences.append({
                'sentence_text': sentence.detokenized_text,
                'ppl': sentence.sentence_ppl,
                'class_label': sentence.class_label,
                'class_predictions': sentence.class_predictions,
                'edits': sentence.edited
            })
        return output_sentences

    def get_final_word_metrics(self):
        output_words = []
        for sentence in self.document.sentences:
            for word in sentence.word_metrics:
                output_words.append(word.to_dict())
        return output_words