import torch
import random
import numpy as np
from utils import evaluate


class ParserTrainer():

    def __init__(
        self,
        train_data,
        dev_data,
        optimizer,
        loss_func,
        output_path,
        batch_size=1024,
        n_epochs=10,
        lr=0.0005,
    ):  # You can add more arguments
        """
        Initialize the trainer.

        Inputs:
            - train_data: Packed train data
            - dev_data: Packed dev data
            - optimizer: The optimizer used to optimize the parsing model
            - loss_func: The cross entropy function to calculate loss, initialized beforehand
            - output_path (str): Path to which model weights and results are written
            - batch_size (int): Number of examples in a single batch
            - n_epochs (int): Number of training epochs
            - lr (float): Learning rate
        """
        self.train_data = train_data
        self.dev_data = dev_data
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.output_path = output_path
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        # TODO: You can add more initializations here

    def train(self, parser, ):  # You can add more arguments as you need
        """
        Given packed train_data, train the neural dependency parser (including optimization),
        save checkpoints, print loss, log the best epoch, and run tests on packed dev_data.

        Inputs:
            - parser (Parser): Neural Dependency Parser
        """
        best_dev_UAS = 0
        best_dev_LAS = 0

        # TODO: Initialize `self.optimizer`, i.e., specify parameters to optimize
        self.optimizer = self.optimizer(
            parser.model.parameters(), lr=self.lr, weight_decay=1e-8)
        print("start train.....")
        for epoch in range(self.n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, self.n_epochs))
            dev_UAS, dev_LAS = self._train_for_epoch(parser,)
            # TODO: you can change this part, to use either uas or las to select best model
            if dev_LAS > best_dev_LAS:
                best_dev_LAS = dev_LAS
                best_dev_UAS = dev_UAS
                print("New best dev LAS! Saving model.")
                # torch.save(parser.model.state_dict(), self.output_path)
                torch.save(parser, self.output_path)
            print("")
        print("best model: UAS = {:.2f}".format(best_dev_UAS*100), "LAS = {:.2f}".format(best_dev_LAS*100))

    def _train_for_epoch(self, parser, ):  # You can add more arguments as you need
        """ 
        Train the neural dependency parser for single epoch.

        Inputs:
            - parser (Parser): Neural Dependency Parser
        Return:
            - dev_UAS (float): Unlabeled Attachment Score (UAS) for dev data
        """
        parser.model.train()  # Places model in "train" mode, e.g., apply dropout layer, etc.
        # TODO: Train all batches of train_data in an epoch.
        # Remember to shuffle before training the first batch (You can use Dataloader of PyTorch)

        random.shuffle(self.train_data)

        count = 0
        total_loss = 0
        B = len(self.train_data)//self.batch_size + 1
        cnt = 0
        for i in range(B):
            batch_instances = self.train_data[cnt: cnt + self.batch_size] if count < B-1 else self.train_data[cnt:]
            cnt += self.batch_size
            self.optimizer.zero_grad()
            features = np.array([x[0] for x in batch_instances])
            gold_t = np.array([x[2] for x in batch_instances])
            x = torch.tensor(features).to(torch.device("cuda"))
            y = torch.tensor(gold_t, dtype=torch.long).to(torch.device("cuda"))
            y_predict = parser.model(x)
            loss = self.loss_func(y_predict, y) 
            loss += parser.model.reg() * 5e-9
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            count += 1
        print("-----------------------------------------------")
        print("avg loss : %.5f" % (total_loss / count))
        print("-----------------------------------------------")

        print("Evaluating on dev set",)
        # Places model in "eval" mode, e.g., don't apply dropout layer, etc.
        parser.model.eval()
        _, dependencies = parser.parse(self.dev_data)
        for i in range(len(dependencies)):
            dependencies[i] = [(dp[0], dp[2]) for dp in dependencies[i]]
        gold_dependencies = get_gold_dependencies(self.dev_data)
        # To check the format of the input, please refer to the utils.py
        uas, las = evaluate(dependencies, gold_dependencies)
        print("- dev UAS: {:.2f}".format(uas * 100.0),
              "- dev LAS: {:.2f}".format(las * 100.0))
        return uas, las


def get_gold_dependencies(data):
    gold_dependencies = []
    for ex in data:
        sent_dependencies = [(ex['head'][i], ex['label'][i])
                             for i in range(1, len(ex['word']))]
        gold_dependencies.append(sent_dependencies)
    return gold_dependencies
