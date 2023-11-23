import torch
import torch.optim as optim
import random
import time
from copy import deepcopy
from evaluate import get_score
from DataProcess import save_predict
from utils import to_id, my_sort


class Trainer(object):
    def __init__(self, model, batch_size:int = 128, epoches:int = 50, lr:float = 0.001):
        # model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        # traning parameters
        self.batch_size = batch_size
        self.epoches = epoches
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        # best model
        self.best_f1 = 0.
        self.best_model = self.model

    def train(self, word_lists, tag_lists, dev_id, dev_word_lists, dev_tag_lists, word2id, tag2id, ENTITIES):
        t1 = time.time()
        print("Training...")
        B = self.batch_size
        for e in range(1, self.epoches + 1):
            # randomly shuffle the training data
            pairs = list(zip(word_lists, tag_lists))
            random.shuffle(pairs)
            word_lists, tag_lists = list(zip(*pairs))
            for index in range(0, len(word_lists), B):
                # sample and prepare data
                batch_sentences = deepcopy(word_lists[index:index+B])
                batch_tags = deepcopy(tag_lists[index:index+B])
                batch_sentences, batch_tags, _ = my_sort(batch_sentences, batch_tags)
                sentences, lengths = to_id(batch_sentences, word2id)
                sentences = sentences.to(self.device)
                targets, _ = to_id(batch_tags, tag2id, False)
                targets = targets.to(self.device)
                # forward
                scores = self.model(sentences, lengths)
                # loss and update
                self.optimizer.zero_grad()
                loss = self.model.loss(scores, targets, tag2id).to(self.device)
                loss.backward()
                self.optimizer.step()
            # validate in dev_set
            val_score = self.validate(dev_id, dev_word_lists, dev_tag_lists, word2id, tag2id, ENTITIES)
            print(f"Epoch {e}/{self.epoches}: current_f1 = {val_score},  best_f1 = {self.best_f1}")
        torch.save(self.best_model,'model.pt')
        t2 = time.time()
        print("Done, model saved to model.pt")
        print(f"Time cost: {int((t2-t1)/60)}min{int((t2-t1)%60)}s")

    def validate(self, dev_id, dev_word_lists, dev_tag_lists, word2id, tag2id, ENTITIES):
        with torch.no_grad():
            pre_tag_list = self.test(dev_word_lists, dev_tag_lists, word2id, tag2id, dev=True)
            truth_dev_path = './input/dev_data_truth.json'
            pre_dev_path = './input/dev_pre_data.json'
            save_predict(truth_dev_path, dev_id, dev_tag_lists, ENTITIES)
            save_predict(pre_dev_path, dev_id, pre_tag_list, ENTITIES)
            val_score = get_score(truth_dev_path, pre_dev_path)['f']
            # preserve the best one
            if val_score > self.best_f1:
                self.best_model = deepcopy(self.model)
                self.best_f1 = val_score
            return val_score

    def test(self, word_lists, tag_lists, word2id, tag2id, dev:bool = False):
        word_lists, _, indices = my_sort(word_lists, tag_lists)
        sentences, lengths = to_id(word_lists, word2id)
        sentences = sentences.to(self.device)
        with torch.no_grad():
            if dev:
                predicted_tagids = self.model.test(sentences, lengths, tag2id)
            else :
                predicted_tagids = self.best_model.test(sentences, lengths, tag2id)
        pred_tag_lists = []
        for i, ids in enumerate(predicted_tagids):
            tag_list = []
            for j in range(lengths[i] - 1):  # the last tag is <stop>
                tag_list.append(ids[j].item())
            pred_tag_lists.append(tag_list)
        # Revert to the original order with indices
        ind_maps = sorted(list(enumerate(indices)), key=lambda x: x[1])
        indices, _ = list(zip(*ind_maps))
        pred_tag_lists = [pred_tag_lists[i] for i in indices]
        return pred_tag_lists
