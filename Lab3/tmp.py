import json
import torch
from parser_utils import load_and_preprocess_data
from utils import evaluate
from trainer import get_gold_dependencies
from parsing_model import ParsingModel
from parser_utils import read_conll, Parser


def predict(model_path, test_path):
    parser = torch.load(model_path)
    test_data = load_and_preprocess_data(test_path, parser)
    print("Final evaluation on test set",)
    parser.model.eval()
    _, dependencies = parser.parse(test_data)
    for i in range(len(dependencies)):
        dependencies[i] = [(dp[0], dp[2]) for dp in dependencies[i]]
    with open('./prediction.json', 'w') as fh:
        json.dump(dependencies, fh)
    gold_dependencies = get_gold_dependencies(test_data)
    uas,las = evaluate(dependencies, gold_dependencies)
    print("- test UAS: {:.2f}".format(uas * 100.0), "- test las: {:.2f}".format(las * 100.0))
    print("Done!")


def load_and_preprocess_data(test_path, parser):
    print("Loading data...",)
    test_set = read_conll(test_path, lowercase=True)
    print("Vectorizing data...",)
    test_set = parser.vectorize(test_set)
    return test_set


if __name__ == '__main__':
    model_path = 'results/test/model.weights'
    train_path = './data/train.conll'
    test_path = './data/test.conll'
    predict(model_path, test_path)