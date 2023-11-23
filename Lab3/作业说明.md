# 作业四

依存句法分析（dependency parsing）是自然语言处理句法分析方向的一个重要分支。不同于上下文无关文法（CFG），依存句法分析旨在分析一句话中单词之间的依赖关系，识别每个单词的头部单词（head），并将它们之间的依赖关系表示为图形结构。这种分析可以帮助我们理解一句话的逻辑结构和语义含义，以及单词之间的关系。在本次作业中，同学们将在英文语料上实现依存句法分析。最终须完成一份书面技术报告。

## 建议
在开始本次作业之前，建议仔细阅读理解课件`lec8_dp.pdf`中关于依存句法分析的相关知识。

## 代码补全说明
本次作业的形式是代码补全，下面为作业文件夹的代码结构。
```
hw4_dep_parsing
├─作业说明.md
├─main.py
├─parser_transitions.py
├─parser_utils.py
├─parsing_model.py
├─trainer.py
├─data
|  ├─dev.conll
|  ├─test.conll
|  └train.conll
```

请注意，本次评测有两项指标，UAS（Unlabeled Attachment Score）和LAS（Labeled Attachement Score）。前者只关注每个word预测的head是否准确，而后者则进一步要求该word和head之间的dependency relation也预测准确。关于这两个指标的具体定义，请参考[https://web.stanford.edu/~jurafsky/slp3/18.pdf](https://web.stanford.edu/~jurafsky/slp3/18.pdf)的evaluation小节。

### 请将下列代码补全：
#### 1. parser_transitions.py
请按照文件中的注释填写代码，完成parser的基本功能。如果对此概念不明，请参考课件`lec8_dp.pdf`中关于transition-based-parsing的介绍。

#### 2. parser_utils.py
请补全文件中的`extract_features`函数。该函数抽取一个特定的状态ex所对应的特征，例如单词本身（word）、词性标签（POS tag）、当前stack、当前buffer、当前已得到的依存边（arc）等。函数返回该实例对应的特征列表，每个元素是一个特征经过`self.tok2id`转化过后，得到的id (int变量)。请查看`Parser`类的初始化方法`__init__`及`vectorize`函数。
关于如何进行特征抽取，请参阅这篇文章：[https://aclanthology.org/D14-1082/](https://aclanthology.org/D14-1082/)。当然，你可以减少特征抽取量，或者还可以抽取其他特征。
补全``create_instance``和``get_oracle``这两个函数，其作用是生成transition-based method训练所需的 action 序列。

#### 3. parsing_model.py
请按照文件中的提示填写代码，完成parser的预测模型。
在模型中，你可以选择用之前所学的特征工程方法（参考文章[https://aclanthology.org/D08-1059.pdf](https://aclanthology.org/D08-1059.pdf)）；可以在抽取特征基础上，加入预训练的词向量word2vec、GloVe等（参考[https://aclanthology.org/D14-1082/](https://aclanthology.org/D14-1082/)）；也可以利用预训练模型，例如BERT等，来进一步增强特征（请注意，如果你使用BERT，模型的大小不能超过bert-base）

#### 4. trainer.py和main.py
请按照文件中的提示填写代码，完成训练代码和主函数。

### 请实现依存边的关系分类和LAS功能：
我们给出的代码中，只包含对每个word对应的head的预测，并最终用UAS进行评价。你需要在已有代码基础上，进一步修改实现对dependency relation的预测，并用LAS进行评价。（提示：对于dependency relation的预测，可将`parser_utils.py`中`Config.unlabeled`参数设置为`False`，但这并不能直接得到正确的dependency relation预测和LAS评价。请在通读并正确理解代码基础上，进一步修改以完成该功能。）

## 评分标准
1. 训练数据生成（30%）：
	目的：生成 transition-based method 所需的 action 序列，并生成每个状态对应的feature。
	代码：`parser_utils.py`；请详细阅读这部分代码，并将每个函数所完成的任务，记录在最终的实验报告中。
2. 动作预测（10%）：
	目的：给定实例，预测下一步的 action。
	代码：`parsing_model.py`
3. 整合成完整的parser（10%）：
	目的：调用parsing_model，完成 action 序列预测，并根据预测出的 action 序列，生成dependencies
	代码：`parser_transitions.py`
4. 训练代码补充（5%）：
	代码：`train.py` 和 `main.py`
5. 对依存边的关系分类和LAS对应代码修改（15%）
6. 模型效果（20%）：
	UAS 和 LAS 单独评分，各占10%。
6. 技术报告（10%）：
	至少需要包括`parser_utils.py`中代码的分析，自己实现代码的解释，实验结果，1-2个句子的parsing结果展示。

## 作业提交说明：
1. 完整的代码
2. 训练好的模型
3. 额外的数据和资源（如果你使用了的话）
4. `predicte.py`：从`./data/real_test.conll`文件读取数据，load训练好的模型，打印 uas 和 las。`real_test.conll` 和 `train.conll` 格式完全一致。
5. `requirement.txt`：调用`predict.py`所需要安装的包。可以用`pip freeze > requirement.txt`导出
6. 书面的技术报告
