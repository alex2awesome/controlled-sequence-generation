import torch
import transformers
from packaging import version
from torch.nn import functional as F
import util.utils_model_loader as uml
import generator.utils_general as ug

SMALL_CONST = 1e-15
BIG_CONST = 1e10


def get_num_sentences_in_tok_list(tokens, tokenizer, spacy_nlp=None):
    """Attempts to split an input sentence to see how many sentences it contains.
    We use this to determine whether a generated sentence is complete or not."""
    spacy_nlp = spacy_nlp or uml.get_spacy_nlp()
    tokens_slim = tokens.squeeze() # tokens.shape: 1 x num_words_generated
    tok_text = tokenizer.decode(tokens_slim)
    processed_tok_text = ug.process_text(tok_text)
    for p in '#$%&()*+,-/:<=>?@[\\]^_`{|}~':
        processed_tok_text = processed_tok_text.replace(p, '')
    spacy_sent_gen = spacy_nlp(processed_tok_text).sents
    return len(list(spacy_sent_gen))


def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins, torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -BIG_CONST, logits)


def get_next_word(pert_probs, sample):
    if sample:
        # todo: control the randomness of the sampling -- sample from the top 10words, for example
        last = torch.multinomial(pert_probs, num_samples=1)
    else:
        _, last = torch.topk(pert_probs, k=1, dim=-1)
    return last


def _get_encoder_model(lm):
    if hasattr(lm, 'encoder_model'):
        return lm.encoder_model
    if hasattr(lm, 'hf_model'):
        return lm.hf_model


def get_word_ll(output_so_far, lm):
    encoder_model = _get_encoder_model(lm)
    if output_so_far.device != encoder_model.device:
        output_so_far = output_so_far.to(encoder_model.device)

    # per-step perplexity
    target_ids = output_so_far.clone()
    target_ids[:, :-1] = -100
    if version.parse(transformers.__version__) > version.parse('4.0.0'):
        log_likelihood = encoder_model(output_so_far, labels=target_ids, return_dict=False)[0]
    else:
        log_likelihood = encoder_model(output_so_far, labels=target_ids)[0]
    ll = float(log_likelihood.detach().cpu())
    return ll


def check_sentence_end(
        last, output_so_far, context, starting_word_idx,
        min_sent_len, sentence_splitting_method, tokenizer, spacy_nlp
):
    """
    Returns (output_so_far, log_this_word, to_break, to_continue)

    `to_break` and `to_continue` both refer to the word-gen loop
        `to_break` is if we're at a sentence end.
        `to_continue` is if we're going to reject the word for some reason.

    `spacy` also looks for EOS first.
    """
    if len(last.shape) != len(output_so_far.shape):
        last = last.unsqueeze(dim=0)

    output_so_far = last if output_so_far is None else torch.cat((output_so_far, last), dim=1)
    sentence_so_far = output_so_far[:, starting_word_idx:]

    if last == tokenizer.eos_token_id:
        return output_so_far, True, True, False

    if len(tokenizer.decode(sentence_so_far.squeeze()).split()) <= min_sent_len * 2:
        return output_so_far, True, False, False

    if sentence_splitting_method == 'spacy':
        # see if the next token makes us have 2 sentences. If so, break and return.
        num_sents = get_num_sentences_in_tok_list(sentence_so_far, tokenizer, spacy_nlp)
        if num_sents > 1:
            return output_so_far[:, :-1], False, True, False

    return output_so_far, True, False, False


def check_forbidden_list(last, tokenizer):
    if version.parse(transformers.__version__) > version.parse('4.0.0'):
        last = last[0]
    words = tokenizer.decode(last)
    if '�' in words:
        return True

    return False


def reshape_tensor(t):
    if len(t.shape) == 0:
        return t.unsqueeze(dim=0)
    return t


def discrim_loss_on_tags(document, class_labels,
                         current_sentence, next_token, discriminator, idx_to_label, device,
                         label_idx=None, head=None, config=None
                         ):
    if any(map(lambda x: isinstance(x, str), class_labels)):
        label_to_idx = {v:k for k,v in idx_to_label.items()}
        t = []
        for c in class_labels:
            if isinstance(c, str):
                t.append(label_to_idx[c])
            else:
                t.append(c)
        class_labels = t

    from util.utils_general import _get_attention_mask
    sent_lens = document.get_sentence_lens_list(
        curr_sentence=current_sentence, candidate_token=next_token, include_prompt=config.use_headline
    )
    attention_mask = _get_attention_mask(sent_lens, max_length_seq=512).to(device)
    all_toks = document.get_all_tokens_tensor(
        current_sentence=current_sentence, next_tensor=next_token, include_prompt=config.use_headline
    )
    sentence_lens = torch.tensor(sent_lens).to(device)
    loss, preds, _ = discriminator.predict_one_doc(
        input_ids=all_toks, labels=class_labels,
        attention_mask=attention_mask, sequence_lens=sentence_lens,
        label_idx=label_idx,
        generate=True,
        add_features=head
    )
    preds = torch.argmax(preds, axis=1)
    preds = preds.cpu().detach().numpy().tolist()
    tags = list(map(idx_to_label.get, preds))
    return loss, tags


def tag_document_discriminator(document, discriminator, idx_to_label, device, config):
    from util.utils_general import _get_attention_mask
    # attention_mask = _get_attention_mask(document.get_sentence_lens_list(), max_length_seq=512).to(device)
    # all_toks = document.get_all_tokens_tensor(unsqueeze=False, include_prompt=config.use_headline)
    # sentence_lens = document.get_sentence_lens_tensor(include_prompt=config.use_headline)
    # _, predicted_labels, _ = discriminator.predict_one_doc(
    #     input_ids=all_toks, attention_mask=attention_mask, sequence_lens=sentence_lens, add_features='main',
    # )
    # predicted_labels = predicted_labels.cpu().detach().numpy().tolist()
    import numpy as np
    sentence_probs = list(map(lambda x: x.class_predictions, document.sentences))
    sentence_tags = np.array(sentence_probs).argmax(axis=1).tolist()
    labels = list(map(idx_to_label.get, sentence_tags))
    return sentence_tags, labels


def _logits_to_probs(pert_logits, top_k=None):
    if top_k is not None:
        pert_logits = top_k_filter(pert_logits, k=top_k)  # + SMALL_CONST
    pert_probs = F.softmax(pert_logits, dim=-1)
    return pert_probs


def _get_discrim_loss_from_hidden_states(class_label, unpert_last_hidden, classifier, device):
    if class_label is not None:
        ce_loss = torch.nn.CrossEntropyLoss()
        _, prediction = classifier(torch.mean(unpert_last_hidden, dim=1))
        label = torch.tensor([class_label], device=device, dtype=torch.long)  ##
        return ce_loss(prediction, label)
    else:
        return None


def _encode_text(tokenizer, cond_text='', uncond=False, use_bos_token=False):
    if use_bos_token:
        cond_text = [tokenizer.bos_token] if uncond else tokenizer.bos_token + cond_text
    else:
        cond_text = [] if uncond else cond_text
    tokenized_cond_text = tokenizer.encode(cond_text, add_special_tokens=False)
    return cond_text, tokenized_cond_text