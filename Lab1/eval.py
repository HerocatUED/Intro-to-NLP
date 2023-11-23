# import sklearn.metrics as sm


# def sm_out(classes_num, labels, predicts):
#     classes = [i for i in range(classes_num)]
#     f1 = sm.f1_score(labels, predicts, labels=classes, average='macro')
#     ac = sm.accuracy_score(labels, predicts)
#     print(f"Macro-F1:{f1}, accuracy:{ac}")


def macro_f1(classes_num, labels, predicts):
    f1 = [0]*classes_num
    for i in range(classes_num):
        tp = 0
        for j in range(len(labels)):
            if labels[j] == i and predicts[j] == i:
                tp += 1
        p_label = labels.count(i)
        p_predict = predicts.count(i)
        if p_label == 0 or p_predict == 0 or tp == 0:
            f1[i] = 0
            continue
        precision = tp/p_predict
        recall = tp/p_label
        f1[i] = 2*precision*recall/(precision+recall)
    macro_f1 = sum(f1)/classes_num
    return macro_f1


def accuracy(labels, predicts):
    ac_num = 0
    total_num = len(labels)
    for i in range(total_num):
        if labels[i] == predicts[i]:
            ac_num += 1
    ac = ac_num/total_num
    return ac


def evaluation(model, data):
    data_len = len(data)
    labels = [data[i][1] for i in range(data_len)]
    predicts = [model.predict(data[i][0]) for i in range(data_len)]
    f1 = macro_f1(model.class_num, labels, predicts)
    ac = accuracy(labels, predicts)
    return f1, ac
