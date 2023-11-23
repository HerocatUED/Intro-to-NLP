import json
import torch
from utils import evaluate
from trainer import get_gold_dependencies
from parser_utils import read_conll


def predict(model_path, test_path):
    print("Loading model...")
    parser = torch.load(model_path)
    print("Loading data...")
    test_set = read_conll(test_path, lowercase=True)
    test_data = parser.vectorize(test_set)
    print("Predicting...",)
    parser.model.eval()
    _, dependencies = parser.parse(test_data)
    for i in range(len(dependencies)):
        dependencies[i] = [(dp[0], dp[2]) for dp in dependencies[i]]
    gold_dependencies = get_gold_dependencies(test_data)
    uas, las = evaluate(dependencies, gold_dependencies)
    print("- test UAS: {:.2f}".format(uas * 100.0),
          "- test las: {:.2f}".format(las * 100.0))
    print("Saving prediction...")
    save_predict(dependencies, gold_dependencies)
    print("Done!")


def save_predict(pred_dependencies, gold_dependencies):
    with open('./prediction.txt', 'w') as f:
        for pred_dependent, gold_dependent in zip(pred_dependencies, gold_dependencies):
            for i, (pred_dp, gold_dp) in enumerate(zip(pred_dependent, gold_dependent)):
                content = f'word_id: {"%02d"%(i+1)} \t' + \
                    f'head: {"%02d"%pred_dp[0]} // {"%02d"%gold_dp[0]} \t' + \
                    f'label:{"%02d"%pred_dp[1]} // {"%02d"%gold_dp[1]} \n'
                f.write(content)
            f.write('\n')


if __name__ == '__main__':
    model_path = 'results/model.pt'
    test_path = './data/real_test.conll'
    predict(model_path, test_path)
