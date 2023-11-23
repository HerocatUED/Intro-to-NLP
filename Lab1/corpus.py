import math
import pandas as pd
from DataPreprocess import preprocess


class Corpus:
    def __init__(self):
        self.num_docs = 0
        self.vocabulary = {}
        self.voc_index = {}
        self.idf = []

    def add_corpus(self, corpus: str):
        self.num_docs += 1
        words = corpus.split()
        words = set(words)
        for word in words:
            self.vocabulary[word] = self.vocabulary.get(word, 0) + 1


def build_corpus():
    print("Building corpus using train.csv")
    my_corpus = Corpus()
    file = preprocess("train.csv")
    file['data'].apply(my_corpus.add_corpus)
    my_corpus.idf = [0]*len(my_corpus.vocabulary)
    index = 0
    new_vocabulary = {}
    for key in my_corpus.vocabulary.keys():
        if my_corpus.vocabulary[key] >= 2:
            new_vocabulary[key] = my_corpus.vocabulary[key]
            my_corpus.voc_index[key] = index
            my_corpus.idf[index] = math.log(
                my_corpus.num_docs/my_corpus.vocabulary[key])
            index += 1
    my_corpus.vocabulary = new_vocabulary
    print(f"Succeed, num of words in vocabulary: {len(my_corpus.vocabulary)}")
    return my_corpus


my_corpus = build_corpus()
