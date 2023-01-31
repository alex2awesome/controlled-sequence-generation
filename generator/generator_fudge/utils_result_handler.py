import generator.utils_result_handler_base as urb
import torch
import logging

des_keys = [
    'per-step ppl',
    'combined_loss',
    'unpert_discrim_loss',
]

class Word(urb.WordBase):
    def __init__(self, s_idx, w_idx, token_str, label, tok, *args, **kwargs):
        super().__init__(s_idx, w_idx, token_str, label, tok, *args, **kwargs)

class Document(urb.Document):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_sentence_lens_list(self, curr_sentence=None, candidate_token=None, include_prompt=False):
        prompt_len = [len(self.tokenized_prompt)] if include_prompt else []
        lengths = prompt_len + list(map(lambda x: x.num_tokens, self.sentences))
        if curr_sentence is not None:
            curr_sent_n_toks = curr_sentence.num_tokens
            if candidate_token is not None:
                curr_sent_n_toks += 1
            lengths.append(curr_sent_n_toks)
        return lengths

    def get_sentence_lens_tensor(self, current_sentence=None, candidate_token=None, include_prompt=False):
        lens_list = self.get_sentence_lens_list(current_sentence, candidate_token, include_prompt)
        return torch.tensor(lens_list, device=self.device)

    def get_all_tokens_list(self, curr_sentence=None, n_back=-1, include_prompt=False):
        """
            Returns a list of prompt.tokens + previous sentences.tokens + curr_sentence.tokens
            return a torch.tensor([[ <tokens> ]])
        """
        output_tokens = []
        if n_back == -1:
            n_back = len(self.sentences)

        # prompt
        if (n_back > len(self.sentences)) or (include_prompt):
            output_tokens += self.tokenized_prompt

        # previous sentences
        for s in self.sentences[-n_back:]:
            output_tokens += s.tokens_so_far_list

        # current sentence
        if curr_sentence is not None:
            output_tokens += curr_sentence.tokens_so_far_list

        return output_tokens

    def get_all_tokens_tensor(
            self, current_sentence=None, n_back=-1,
            candidate_tensor=None, next_tensor=None,
            unsqueeze=False, include_prompt=False, device=None
    ):
        # this is to allow us to start on CPU if we want to chunk up the inputs.
        if device is None:
            device = self.device

        output_tokens_list = self.get_all_tokens_list(current_sentence, n_back, include_prompt=include_prompt)
        output_tokens_tensor = torch.tensor(output_tokens_list, device=device)
        # sentence_tensors = list(map(lambda x: x.tokens_so_far_tensor, self.sentences))
        # output_tokens_tensor = torch.cat([self.prompt_as_tensor] + sentence_tensors)

        if candidate_tensor is not None:
            candidate_tensor = candidate_tensor.to(device)
            previous_tokens = output_tokens_tensor.repeat((len(candidate_tensor), 1))
            output_tokens_tensor = torch.hstack((previous_tokens, candidate_tensor))
        if next_tensor is not None:
            if len(next_tensor.shape) == 2:
                next_tensor = next_tensor.squeeze()
            next_tensor = next_tensor.to(device)
            output_tokens_tensor = torch.hstack((output_tokens_tensor, next_tensor))

        if unsqueeze:
            return output_tokens_tensor.unsqueeze(dim=0)

        return output_tokens_tensor.to(int)


class GeneratorResultHandler(urb.GeneratorResultHandlerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def start_new_document(self, cond_text, tokenized_cond_text=None, class_labels=None):
        if cond_text is not None:
            logging.info('------- PROMPT ---------------')
            logging.info(cond_text)
            logging.info('------------------------------')
            self.tb_logger.add_text('prompt', cond_text)
        self.document = Document(kind=self.kind, tokenized_prompt=tokenized_cond_text,
                                 device=self.device, tokenizer=self.tokenizer, class_labels=class_labels)
