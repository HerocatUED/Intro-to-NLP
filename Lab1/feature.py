from corpus import my_corpus


def extract_feature(x: str, use_tfidf: bool = False):
    corpus = my_corpus
    feature = []
    words = x.split()
    num_words = len(words)
    words = set(words)
    # TF-IDF
    for word in words:
        if word in corpus.vocabulary.keys():
            cnt = x.count(word)
            index = corpus.voc_index[word]
            f = 0
            if use_tfidf:
                f = cnt / num_words * corpus.idf[index]
            else:
                f = 1
            feature.append((index, f))
    return feature
