import numpy as np
import torch
from transformers import AutoTokenizer

full_discriminator = None
tokenizer = None


def nn_init(args, device):
    global full_discriminator, tokenizer
    from discriminator.models_full import Discriminator as SequentialLSTMDiscriminator
    full_discriminator = (SequentialLSTMDiscriminator
                          .load_from_checkpoint(checkpoint_path=args.pretrained_discriminator_path,
                                                loading_from_checkpoint=True,
                                                pretrained_cache_dir=args.pretrained_lm_model_path,
                                                )
                          )
    full_discriminator = full_discriminator.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_lm_model_path)


def nn_forward_func(input_embed, attention_mask=None, position_embed=None, type_embed=None, return_all_logits=False):
    return full_discriminator(inputs_embeds=input_embed, attention_mask=attention_mask)


def get_ref_token_id():
    return tokenizer.encode(' ')[0]


def get_base_token_emb(device):
    global full_discriminator
    input_tensor = torch.tensor([get_ref_token_id()], device=device)
    return construct_word_embedding(input_tensor)


def get_tokens(text_ids):
    global tokenizer
    return tokenizer.convert_ids_to_tokens(text_ids.squeeze())


def construct_input_ref_pair(tokenizer, text, ref_token_id, sep_token_id, cls_token_id, device):
    text_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        #         truncation=True,
        #         max_length=tokenizer.max_len_single_sentence
    )
    input_ids = [cls_token_id] + text_ids + [sep_token_id]  # construct input token ids
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]  # construct reference token ids
    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device)


# Handle GPT2
## get transformer from full_discriminator
def get_gpt2_transformer():
    global full_discriminator
    return full_discriminator.transformer.encoder_model.transformer


def construct_position_embedding(input_ids, device):
    seq_length = input_ids.size(1)
    position_ids = get_gpt2_transformer().wpe.weight.data[:, 0:seq_length].to(device)
    ref_position_ids = get_gpt2_transformer().wpe.weight.data[:, 0:seq_length].to(device)
    return position_ids, ref_position_ids


def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)


def construct_word_embedding(input_ids):
    return get_gpt2_transformer().wte(input_ids)


def construct_sub_embedding(input_ids, ref_input_ids, device):
    input_embeddings = construct_word_embedding(input_ids)
    ref_input_embeddings = construct_word_embedding(ref_input_ids)
    input_position_embeddings, ref_input_position_embeddings = construct_position_embedding(input_ids, device)

    return (input_embeddings, ref_input_embeddings), \
           (input_position_embeddings, ref_input_position_embeddings)


def get_inputs(text, device):
    global tokenizer
    ref_token_id = get_ref_token_id()
    sep_token_id = tokenizer.eos_token_id
    cls_token_id = tokenizer.bos_token_id

    input_ids, ref_input_ids = construct_input_ref_pair(
        tokenizer, text, ref_token_id, sep_token_id, cls_token_id, device
    )
    attention_mask = construct_attention_mask(input_ids)

    (input_embed, ref_input_embed), (position_embed, ref_position_embed) = construct_sub_embedding(
        input_ids, ref_input_ids, device
    )

    return [input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, attention_mask]


def get_model_prediction(input_ids, label):
    global full_discriminator
    output = full_discriminator.forward(input_ids, return_lls=True)
    if len(output) == 1:
        y_pred_lls = output
        y_pred_lls = y_pred_lls.detach().cpu().numpy()
        pred = np.argmax(y_pred_lls)
        return y_pred_lls, pred
    else:
        loss, pred, y_pred_lls = output
        return y_pred_lls.detach().cpu().numpy()[:, label][0], pred.detach().cpu().numpy().tolist()[0]