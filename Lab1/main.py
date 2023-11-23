import pandas as pd
from DataPreprocess import preprocess
from corpus import my_corpus
from model import LogLinear
from eval import evaluation
from LogFile import save_log


def train(my_model, filename: str):
    file = preprocess(filename)
    cnt = 0
    sample_num = 200
    print(f"Begin training, batch_size = {sample_num}")
    while cnt+sample_num <= 8*file.shape[0]:
        # 采样、转化成列表输入
        print(f"epoch {int(cnt/sample_num)}")
        epoch=random_sample(file,sample_num)
        epoch = epoch.values.tolist()
        my_model.update(epoch=epoch, lr=1e-2, alpha=1e-3, max_loop=2)
        # 评估模型的效果
        f1, ac = evaluation(my_model, epoch)
        print(f"macro-F1:{f1}, accuracy:{ac}")
        save_log(train=True, cnt=int(cnt/sample_num), f1=f1, ac=ac)
        cnt += sample_num

def random_sample(source:pd.DataFrame,sample_num:int=200):
    num=int(sample_num/20)
    epoch=pd.DataFrame()
    for i in range(20):
        sample_range=source[source.target==i]
        sample_data=sample_range.sample(n=num)
        epoch=pd.concat([epoch,sample_data])
    return epoch

def test(my_model, filename: str):
    print("Testing")
    test_data = preprocess(filename)
    test_data = test_data.values.tolist()
    f1, ac = evaluation(my_model, test_data)
    print(f"macro-F1:{f1}, accuracy:{ac}")
    save_log(train=False, cnt=0, f1=f1, ac=ac)


def main():
    my_model = LogLinear(len(my_corpus.vocabulary), 20)
    train(my_model, "train.csv")
    test(my_model, "test.csv")


if __name__ == "__main__":
    main()
