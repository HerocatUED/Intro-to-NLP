import argparse
import torch
from model import BiLSTM_CRF
from trainer import Trainer
from DataProcess import *
from evaluate import get_score


# ENTITIES = ['O', 'NHCS', 'NHVI', 'NCSM', 'NCGV', 'NCSP', 'NASI', 'NATS', 'NT', 'NS', 'NO']
ENTITIES = ['O', 'NHCS', 'NHVI', 'NCSM', 'NCGV', 'NASI', 'NT', 'NS', 'NO', 'NATS', 'NCSP']

def load_data(train_data_path: str, test_data_path: str):
    """
    Load, split and pack your data.
    Input: The paths to the files of your original data
    Output: Packed data, e.g., a list like [train_data, dev_data, test_data]
    """
    tag2idx = build_labels(ENTITIES)
    train_set, dev_set, corpus = load_and_process(train_data_path, 'train', ENTITIES)
    test_set = load_and_process(test_data_path, 'test', ENTITIES)
    extend_maps(corpus, tag2idx)

    train_word_lists = [list(train_set[i][1]) for i in range(len(train_set))]
    train_tag_lists = [list(train_set[i][2]) for i in range(len(train_set))]
    train_data = extend_lists(train_word_lists, train_tag_lists, tag2idx, train=True)

    dev_id = [dev_set[i][0] for i in range(len(dev_set))]
    dev_word_lists = [list(dev_set[i][1]) for i in range(len(dev_set))]
    dev_tag_lists = [list(dev_set[i][2]) for i in range(len(dev_set))]
    dev_data = extend_lists(dev_word_lists, dev_tag_lists, tag2idx, train=True)
    dev_data = [dev_id, dev_word_lists, dev_tag_lists] 

    test_id = [test_set[i][0] for i in range(len(test_set))]
    test_word_lists = [list(test_set[i][1]) for i in range(len(test_set))]
    test_tag_lists = [list(test_set[i][2]) for i in range(len(test_set))]
    test_word_lists, test_tag_lists = extend_lists(test_word_lists, test_tag_lists, tag2idx, train=False)
    test_data = [test_id, test_word_lists, test_tag_lists] 

    return train_data, dev_data, test_data, corpus, tag2idx


def main(args):
    # NOTE: You can use variables in args as further arguments of the following functions
    train_data_path = './input/train_data.json'
    test_data_path = './input/test_data.json'
    test_output_path = './output/output.json'

    train_data, dev_data, test_data, corpus, tag2idx = load_data(train_data_path, test_data_path)
    train_word_lists, train_tag_lists = train_data
    dev_id, dev_word_lists, dev_tag_lists = dev_data
    test_id, test_word_lists, test_tag_lists = test_data
    vocab_size = len(corpus)
    out_size = len(tag2idx)

    model = None
    trainer = None
    if args.load:
        model = torch.load(args.model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        trainer = Trainer(model=model, batch_size=args.batch_size, epoches=args.epochs, lr=args.lr)
        pre_dev_tag_list = trainer.test(dev_word_lists, dev_tag_lists, model.corpus, tag2idx, dev=True)
        truth_dev_path = './input/dev_data_truth.json'
        pre_dev_path = './input/dev_pre_data.json'
        save_predict(truth_dev_path, dev_id, dev_tag_lists, ENTITIES)
        save_predict(pre_dev_path, dev_id, pre_dev_tag_list, ENTITIES)
        val_score = get_score(truth_dev_path, pre_dev_path)['f']
        print(f"f1 on devset{val_score}")
    else:
        model = BiLSTM_CRF(vocab_size=vocab_size, out_size=out_size, corpus=corpus)
        trainer = Trainer(model=model, batch_size=args.batch_size, epoches=args.epochs, lr=args.lr)
        trainer.train(train_word_lists, train_tag_lists, dev_id, dev_word_lists, dev_tag_lists, corpus, tag2idx, ENTITIES)
    pred_tag_lists = trainer.test(test_word_lists, test_tag_lists, model.corpus, tag2idx, dev = False)
    save_predict(test_output_path, test_id, pred_tag_lists, ENTITIES)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="arguments")
    # You can add more arguments as you want
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch Size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning Rate"
    )
    parser.add_argument(
        "--load",
        type=bool,
        default=False,
        help="True: Load a model; False: Train a model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='./model.pt',
        help="Path of loaded model"
    )
    args = parser.parse_args()

    main(args)