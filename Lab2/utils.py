import torch
import numpy as np


def find_span(tag_list: list, idx: int):
    """ trans label_sequence to standard form of output json file """
    span = []
    flag = False
    start_pos = -1
    for i in range(len(tag_list)):
        if tag_list[i] == idx and not flag:
            flag = True
            start_pos = str(i)
        if tag_list[i] != idx and flag:
            flag = False
            span.append(start_pos+";"+str(i))
    return span


def trans(data: dict, ENTITIES: list = None):
    """
    transform the data format
    input: {'id':...,'context':...,'entities':[...]}
    output:[id,context,label_sequence]
    """
    new_data = [data['id'], data['context'], np.zeros(len(data['context']),dtype=int)]
    for item in data['entities']:
        label = ENTITIES.index(item['label'])
        for pos_str in item['span']:
            mid_pos = pos_str.find(";")
            pos_start = int(pos_str[: mid_pos])
            pos_end = int(pos_str[mid_pos + 1:])
            new_data[2][pos_start: pos_end] = label
    return new_data


def to_id(batch, maps, word:bool = True):
    pad = maps['<pad>']
    unknow = maps['<unknow>']
    batch_size = len(batch)
    max_len = max([len(batch[i]) for i in range(batch_size)])
    batch_tensor = torch.ones(batch_size, max_len).long() * pad
    for i, l in enumerate(batch):
        for j, e in enumerate(l):
            if word:
                batch_tensor[i][j] = maps.get(e, unknow)
            else:
                batch_tensor[i][j] = e
    lengths = [len(l) for l in batch]
    return batch_tensor, lengths


def my_sort(word_lists, tag_lists):
    """sort by length for torch.nn.utils.rnn.pad_packed_sequence"""
    pairs = list(zip(word_lists, tag_lists))
    # indices[i]=j means original index is j and sorted index is i
    indices = sorted(range(len(pairs)), key=lambda k: len(pairs[k][0]), reverse=True)
    pairs = [pairs[i] for i in indices]
    word_lists, tag_lists = list(zip(*pairs))
    return word_lists, tag_lists, indices


def indexed(targets, tagset_size, start_id):
    """trans nums in targets to index"""
    _, max_len = targets.size()
    for col in range(max_len-1, 0, -1):
        targets[:, col] += (targets[:, col-1] * tagset_size)
    targets[:, 0] += (start_id * tagset_size)
    return targets