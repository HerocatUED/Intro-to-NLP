import os
import logging
from collections import Counter
from parser_transitions import minibatch_parse

from tqdm import tqdm
import torch
import numpy as np

P_PREFIX = '<p>:'
L_PREFIX = '<l>:'
UNK = '<UNK>'
NULL = '<NULL>'
ROOT = '<ROOT>'
PUNCTS = ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]


class Config(object):
    with_punct = True
    unlabeled = False
    lowercase = True
    use_pos = True
    use_dep = True
    use_dep = use_dep and (not unlabeled)
    data_path = './data'
    train_file = 'train.conll'
    dev_file = 'dev.conll'
    test_file = 'test.conll'


class Parser(object):
    """Contains everything needed for transition-based dependency parsing except for the model"""

    def __init__(self, dataset):
        root_labels = list([l for ex in dataset
                           for (h, l) in zip(ex['head'], ex['label']) if h == 0])
        counter = Counter(root_labels)
        if len(counter) > 1:
            logging.info('Warning: more than one root label')
            logging.info(counter)
        self.root_label = counter.most_common()[0][0]
        deprel = [self.root_label] + list(set([w for ex in dataset
                                               for w in ex['label']
                                               if w != self.root_label]))
        tok2id = {L_PREFIX + l: i for (i, l) in enumerate(deprel)}
        tok2id[L_PREFIX + NULL] = self.L_NULL = len(tok2id)

        config = Config()
        self.unlabeled = config.unlabeled
        self.with_punct = config.with_punct
        self.use_pos = config.use_pos
        self.use_dep = config.use_dep

        if self.unlabeled:
            trans = ['L', 'R', 'S']
            self.n_deprel = 1
        else:
            trans = ['L-' + l for l in deprel] + \
                ['R-' + l for l in deprel] + ['S']
            self.n_deprel = len(deprel)

        self.n_trans = len(trans)
        self.tran2id = {t: i for (i, t) in enumerate(trans)}
        self.id2tran = {i: t for (i, t) in enumerate(trans)}

        # logging.info('Build dictionary for part-of-speech tags.')
        tok2id.update(build_dict([P_PREFIX + w for ex in dataset for w in ex['pos']],
                                 offset=len(tok2id)))
        tok2id[P_PREFIX + UNK] = self.P_UNK = len(tok2id)
        tok2id[P_PREFIX + NULL] = self.P_NULL = len(tok2id)
        tok2id[P_PREFIX + ROOT] = self.P_ROOT = len(tok2id)

        # logging.info('Build dictionary for words.')
        tok2id.update(build_dict([w for ex in dataset for w in ex['word']],
                                 offset=len(tok2id)))
        tok2id[UNK] = self.UNK = len(tok2id)
        tok2id[NULL] = self.NULL = len(tok2id)
        tok2id[ROOT] = self.ROOT = len(tok2id)

        self.tok2id = tok2id
        self.id2tok = {v: k for (k, v) in tok2id.items()}

        self.n_tokens = len(tok2id)
        self.model = None

    def vectorize(self, examples):
        vec_examples = []
        for ex in examples:
            word = [self.ROOT] + [self.tok2id[w] if w in self.tok2id
                                  else self.UNK for w in ex['word']]
            pos = [self.P_ROOT] + [self.tok2id[P_PREFIX + w] if P_PREFIX + w in self.tok2id
                                   else self.P_UNK for w in ex['pos']]
            head = [-1] + ex['head']
            label = [-1] + [self.tok2id[L_PREFIX + w] if L_PREFIX + w in self.tok2id
                            else -1 for w in ex['label']]
            vec_examples.append({'word': word, 'pos': pos,
                                 'head': head, 'label': label})
        return vec_examples

    def extract_features(self, stack, buf, arcs, ex):
        # TODO:
        # You should implement your feature extraction here.
        # Extract the features for one example, ex
        # The features could include the word itself, the part-of-speech and so on.
        # Every feature could be represented by a string,
        # and the string can be converted to an id(int), according to the self.tok2id
        # Return: A list of token_ids corresponding to tok2id
        if stack[0] == ROOT:
            stack[0] = 0
        Sw = [ex['word'][stack[-1-i]] if i < len(stack) else self.NULL for i in range(
            3)] + [ex['word'][buf[i]] if i < len(buf) else self.NULL for i in range(3)]
        for i in range(2):
            if i < len(stack):
                s = stack[-i - 1]
                lc = self.get_left_child(s, arcs)
                rc = self.get_right_child(s, arcs)
                llc = self.get_left_child(lc[0], arcs) if len(lc) > 0 else []
                rrc = self.get_right_child(rc[0], arcs) if len(rc) > 0 else []
                Sw.append(ex["word"][lc[0]] if len(lc) > 0 else self.NULL)
                Sw.append(ex["word"][rc[0]] if len(rc) > 0 else self.NULL)
                Sw.append(ex["word"][lc[1]] if len(lc) > 1 else self.NULL)
                Sw.append(ex["word"][rc[1]] if len(rc) > 1 else self.NULL)
                Sw.append(ex["word"][llc[0]] if len(llc) > 0 else self.NULL)
                Sw.append(ex["word"][rrc[0]] if len(rrc) > 0 else self.NULL)
            else:
                Sw.extend([self.NULL] * 6)
        idx = [ex['word'].index(w) if w in ex['word'] else 0.1 for w in Sw]
        # Sw = [self.tok2id[w] if w in self.tok2id else self.UNK for w in Sw]
        St = [ex['pos'][i] if i != 0.1 else self.P_NULL for i in idx]
        Sl = [ex['label'][i] if i != 0.1 else self.L_NULL for i in idx[6:]]
        features = np.array(Sw + St + Sl)
        return features

    def get_left_child(self, word, arcs):
        left_child = sorted(
            [arc[1] for arc in arcs if arc[0] == word and arc[1] < word])
        return left_child

    def get_right_child(self, word, arcs):
        right_child = sorted(
            [arc[1] for arc in arcs if arc[0] == word and arc[1] > word], reverse=True)
        return right_child

    def get_oracle(self, stack, buf, ex):
        # TODO: 根据当前状态，返回应该执行的操作编号（对应__init__中的trans），若无操作则返回None。
        if len(stack) < 2:
            return None if len(buf) == 0 else self.tran2id['S']
        s1, s2 = stack[-1], stack[-2]
        h1, h2 = ex['head'][s1], ex['head'][s2]
        if s2 > 0 and h2 == s1:
            label = self.id2tok[ex['label'][s2]][4:]
            return self.tran2id['L-'+label]
        elif s2 >= 0 and h1 == s2 and (not any([x for x in buf if ex['head'][x] == s1])):
            label = self.id2tok[ex['label'][s1]][4:]
            return self.tran2id['R-'+label]
        else:
            return None if len(buf) == 0 else self.tran2id['S']

    def create_instances(self, examples):
        all_instances = []
        succ = 0
        for id, ex in enumerate(examples):
            n_words = len(ex['word']) - 1

            # arcs = {(h, t, label)}
            stack = [0]
            buf = [i + 1 for i in range(n_words)]
            arcs = []
            instances = []
            for i in range(n_words * 2):
                gold_t = self.get_oracle(stack, buf, ex)
                if gold_t is None:
                    break
                legal_labels = self.legal_labels(stack, buf)
                assert legal_labels[gold_t] == 1
                instances.append((self.extract_features(
                    stack, buf, arcs, ex), legal_labels, gold_t))
                # TODO: 根据gold_t，更新stack, arcs, buf
                transition = self.id2tran[gold_t]
                if transition[0] == 'S':
                    stack.append(buf.pop(0))
                elif transition[0] == 'L':
                    h = stack[-1]
                    t = stack.pop(-2)
                    label = self.tok2id[L_PREFIX+transition[2:]]
                    arcs.append((h, t, label))
                else:
                    h = stack[-2]
                    t = stack.pop(-1)
                    label = self.tok2id[L_PREFIX+transition[2:]]
                    arcs.append((h, t, label))
            else:
                succ += 1
                all_instances += instances

        return all_instances

    def legal_labels(self, stack, buf):
        labels = ([1] if len(stack) > 2 else [0]) * self.n_deprel
        labels += ([1] if len(stack) >= 2 else [0]) * self.n_deprel
        labels += [1] if len(buf) > 0 else [0]
        return labels

    def parse(self, dataset, eval_batch_size=5000):
        sentences = []
        sentence_id_to_idx = {}
        for i, example in enumerate(dataset):
            n_words = len(example['word']) - 1
            sentence = [j + 1 for j in range(n_words)]
            sentences.append(sentence)
            sentence_id_to_idx[id(sentence)] = i

        model = ModelWrapper(self, dataset, sentence_id_to_idx)
        dependencies = minibatch_parse(sentences, model, eval_batch_size)

        LAS = all_tokens = 0.0
        with tqdm(total=len(dataset)) as prog:
            for i, ex in enumerate(dataset):
                head = [-1] * len(ex['word'])
                label = [-1] * len(ex['word'])
                for h, t, l in dependencies[i]:
                    head[t] = h
                    label[t] = l
                for pred_h, pred_l, gold_h, gold_l, pos in zip(head[1:], label[1:], ex['head'][1:], ex['label'][1:], ex['pos'][1:]):
                    assert self.id2tok[pos].startswith(P_PREFIX)
                    pos_str = self.id2tok[pos][len(P_PREFIX):]
                    if (self.with_punct) or (not (pos_str in PUNCTS)):
                        LAS += 1 if pred_h == gold_h and pred_l == gold_l else 0
                        all_tokens += 1
                prog.update(i + 1)
        LAS /= all_tokens
        return LAS, dependencies


class ModelWrapper(object):
    def __init__(self, parser, dataset, sentence_id_to_idx):
        self.parser = parser
        self.dataset = dataset
        self.sentence_id_to_idx = sentence_id_to_idx

    def predict(self, partial_parses):
        mb_x = [self.parser.extract_features(p.stack, p.buffer, p.dependencies, self.dataset[self.sentence_id_to_idx[id(p.sentence)]])
                for p in partial_parses]
        mb_x = np.array(mb_x).astype('int32')
        mb_x = torch.from_numpy(mb_x).long().to(torch.device("cuda"))
        mb_l = [self.parser.legal_labels(p.stack, p.buffer)
                for p in partial_parses]

        pred = self.parser.model(mb_x)
        pred = pred.detach().cpu().numpy()
        pred = np.argmax(pred + 10000 * np.array(mb_l).astype('float32'), 1)
        pred = [self.parser.id2tran[p] for p in pred]
        return pred


def read_conll(in_file, lowercase=False, max_example=None):
    examples = []
    with open(in_file) as f:
        word, pos, head, label = [], [], [], []
        for line in f.readlines():
            sp = line.strip().split('\t')
            if len(sp) == 10:
                if '-' not in sp[0]:
                    word.append(sp[1].lower() if lowercase else sp[1])
                    pos.append(sp[4])
                    head.append(int(sp[6]))
                    label.append(sp[7])
            elif len(word) > 0:
                examples.append({'word': word, 'pos': pos,
                                'head': head, 'label': label})
                word, pos, head, label = [], [], [], []
                if (max_example is not None) and (len(examples) == max_example):
                    break
        if len(word) > 0:
            examples.append({'word': word, 'pos': pos,
                            'head': head, 'label': label})
    return examples


def build_dict(keys, n_max=None, offset=0):
    count = Counter()
    for key in keys:
        count[key] += 1
    ls = count.most_common() if n_max is None \
        else count.most_common(n_max)

    return {w[0]: index + offset for (index, w) in enumerate(ls)}


def load_and_preprocess_data(reduced=True):
    config = Config()

    print("Loading data...",)
    train_set = read_conll(os.path.join(config.data_path, config.train_file),
                           lowercase=config.lowercase)
    dev_set = read_conll(os.path.join(config.data_path, config.dev_file),
                         lowercase=config.lowercase)
    test_set = read_conll(os.path.join(config.data_path, config.test_file),
                          lowercase=config.lowercase)
    if reduced:
        train_set = train_set[:1000]
        dev_set = dev_set[:500]
        test_set = test_set[:500]

    print("Building parser...",)
    parser = Parser(train_set)

    print("Vectorizing data...",)
    train_set = parser.vectorize(train_set)
    dev_set = parser.vectorize(dev_set)
    test_set = parser.vectorize(test_set)

    print("Preprocessing training data...",)
    train_examples = parser.create_instances(train_set)

    return parser, train_examples, dev_set, test_set,
