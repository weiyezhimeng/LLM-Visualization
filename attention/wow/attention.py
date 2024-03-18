
import torch



def aggregate_attention(attn, device):
    '''Extract average attention vector'''
    avged = []
    #do average to every layer and head
    for layer in attn:
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = torch.concat((
            # We zero the first entry because it's what's called
            # null attention (https://aclanthology.org/W19-4808.pdf)
            torch.tensor([0.]).to(device),
            # usually there's only one item in attns_per_head but
            # on the first generation, there's a row for each token
            # in the prompt as well, so take [-1]
            attns_per_head[0][1:]
        ))
        avged.append(vec / vec.sum())
    tensor_result = torch.stack(avged).mean(dim=0)
    list_result = tensor_result.tolist()
    return list_result


def handle_attention(attention):
    # attn_m[0] is the attention martix of prompt, so begin from 1
    attention = attention[1:]
    attention_list = [[t.squeeze(0).squeeze(1).tolist() for t in inner_tuple] for inner_tuple in attention]
    return attention_list


def decode(tokens, tokenizer):
    '''Turn tokens into text with mapping index'''
    full_text = ''
    chunks = []
    for i, token in enumerate(tokens):
        text = tokenizer.decode(token)
        text = " "+text
        chunks.append(text)
    return chunks

def get_completion(prompt, model, tokenizer, device):
    '''Get full text, token mapping, and attention matrix for a completion'''
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = tokens.shape[1]
    outputs = model.generate(
        tokens,
        max_new_tokens=50,
        output_attentions=True,
        return_dict_in_generate=True,
        early_stopping=True,
        length_penalty=-1
    )
    sequences = outputs.sequences
    attention = outputs.attentions
    attention = handle_attention(attention)
    chunks = decode(sequences[0], tokenizer)
    device_for_map = [device for i in range(len(outputs.attentions[1:]))]
    ave_layer_head_attention = list(map(aggregate_attention, outputs.attentions[1:], device_for_map))

    return chunks, attention, prompt_len, ave_layer_head_attention

def  get_head_and_layer_nums(model):
    if hasattr(model.config,"n_head"):
        return model.config.n_head, model.config.n_layer
    elif hasattr(model.config,"num_attention_heads"):
        return model.config.num_attention_heads, model.config.num_hidden_layers

