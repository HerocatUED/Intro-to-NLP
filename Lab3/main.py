from datetime import datetime
import os

import torch
from torch import optim
from torch import nn

from trainer import ParserTrainer
from parsing_model import ParsingModel
from parser_utils import load_and_preprocess_data
from utils import evaluate
from trainer import get_gold_dependencies
import json


if __name__ == "__main__":
    # Note: Set debug to False, when training on entire corpus
    # debug = True
    debug = False

    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    parser, train_data, dev_data, test_data = load_and_preprocess_data(debug)

    # You can add more arguments, depending on how you designed your parsing model
    parser.model = ParsingModel(len(parser.tok2id), len(parser.tran2id))
    parser.model.to(torch.device("cuda"))
    print(80 * "=")
    print("TRAINING")
    print(80 * "=")
    output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
    output_path = output_dir + "model.pt"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # TODO:
    # 1. Call an optimizer (no need to specify parameters yet, which will be implemented during training)
    # 2. Construct the Cross Entropy Loss Function in variable `loss_func`
    optimizer = optim.Adam
    loss_func = nn.CrossEntropyLoss()
    trainer = ParserTrainer(
        train_data=train_data,
        dev_data=dev_data,
        optimizer=optimizer,
        loss_func=loss_func,
        output_path=output_path,
        batch_size=1024,
        n_epochs=60,
        lr=5e-5,
    )
    trainer.train(parser, )

    if not debug:
        print(80 * "=")
        print("TESTING")
        print(80 * "=")
        print("Restoring the best model weights found on the dev set")
        # parser.model.load_state_dict(torch.load(output_path))
        parser = torch.load(output_path)
        print("Final evaluation on test set",)
        parser.model.eval()
        _, dependencies = parser.parse(test_data)
        for i in range(len(dependencies)):
            dependencies[i] = [(dp[0], dp[2]) for dp in dependencies[i]]
        with open('./prediction.json', 'w') as fh:
            json.dump(dependencies, fh)
        gold_dependencies = get_gold_dependencies(test_data)
        # To check the format of the input, please refer to the utils.py
        uas, las = evaluate(dependencies, gold_dependencies)
        print("- test UAS: {:.2f}".format(uas * 100.0),
              "- test las: {:.2f}".format(las * 100.0))
        print("Done!")
