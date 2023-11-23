import json
import random
from utils import find_span, trans


def build_labels(entities):
    """ build tag2idx map """
    label_maps = {}
    label_maps['O'] = 0
    for i in range(1, len(entities)):
        label_maps[entities[i]] = i
    return label_maps


def build_corpus(data_set: list):
    print("Building corpus")
    context = [data_set[i][1] for i in range(len(data_set))]
    corpus = {}
    length = 0
    for text in context:
        words = list(text)
        words = set(words)
        for word in words:
            if not word in corpus.keys():
                corpus[word] = length
                length += 1
    print(f"Num of words in vocabulary: {len(corpus)}")
    return corpus


def load_and_process(data_path: str, mode: str = 'train', ENTITIES: list = None):
    """ load data from json file """
    f = open(data_path, 'r', encoding="utf-8")
    lines = f.readlines()
    data_set = []  # list of [id,context,label_sequence]
    for line in lines:
        data = json.loads(line)
        data = trans(data, ENTITIES)
        data_set.append(data)
    if mode == 'train':
        random.shuffle(data_set)
        dev_num = int(len(data_set)/5)
        dev_set = data_set[:dev_num]
        train_set = data_set[dev_num:]
        corpus = build_corpus(train_set)
        return train_set, dev_set, corpus
    else:
        return data_set  # test_set


def extend_maps(word2id: map, tag2id: map):
    """  add marks   """
    word2id['<unknow>'] = len(word2id)
    word2id['<pad>'] = len(word2id)
    word2id['<start>'] = len(word2id)
    word2id['<stop>'] = len(word2id)

    tag2id['<unknow>'] = len(tag2id)
    tag2id['<pad>'] = len(tag2id)
    tag2id['<start>'] = len(tag2id)
    tag2id['<stop>'] = len(tag2id)
    return word2id, tag2id


def extend_lists(word_lists: list, tag_lists: list, tag2id, train: bool = True):
    """ add marks """
    for i in range(len(word_lists)):
        word_lists[i].append('<stop>')
        if train:
            tag_lists[i].append(tag2id['<stop>'])
    return word_lists, tag_lists


def save_predict(test_output_path: str, test_id: list = None, pred_tag_lists: list = None, ENTITIES: list = None):
    with open(test_output_path, "w") as f:
        pass
    with open(test_output_path, "a", encoding="utf-8") as f:
        for i in range(len(pred_tag_lists)):
            predict = {}
            if test_id is not None:
                predict['id'] = test_id[i]
            else:
                predict['id'] = f'dev{i}'
            entities = []
            for j in range(1, len(ENTITIES)):
                dic = {}
                dic['label'] = ENTITIES[j]
                span = find_span(pred_tag_lists[i], j)
                dic['span'] = span
                entities.append(dic)
            predict['entities'] = entities
            json.dump(predict, f)
            f.write('\n')
        f.close()