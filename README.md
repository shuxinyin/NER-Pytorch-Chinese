# NER系列-中文实体识别模型实践

## Introduction
本项目的目的在于验证常见的NER范式模型在不同中文NER数据集上(Flat、Nested、Discontinuous)的表现
NER系列模型实践，包括如下：
1. Bert-Softmax、Bert-Crf、Bert-BiLSTM-Softmax、Bert-BiLSTM-Crf
2. 词汇增强模型：FlatNER、[LEBERT](https://arxiv.org/abs/2105.07148)
3. Calscal Rel (To do)
4. 机器阅读理解式 MRC (To do)
5. span-based NER (To do)

### Dataset Introduction

中文NER数据集：

- Flat NER Datasets: Ontonote4、Msra
- Nested NER Datasets：ACE 2004、 ACE 2005
- Discontinuous NER Datasets： CADEC

本项目将数据集统一处理成相同的数据格式，每一行表示一条数据。
格式如下：
```
{'text': ['吴', '重', '阳', '，', '中', '国', '国', '籍', ','], 
'label': ['B-NAME', 'I-NAME', 'I-NAME', 'O', 'B-CONT', 'I-CONT', 'I-CONT', 'I-CONT', 'O',]}
```

## Environment
python==3.8、transformers==4.12.3、torch==1.8.0
Or run the shell
```
pip install -r requirements.txt
```

## Project Structure

- datasets：NER dataset
- losses:损失函数
- metrics:评价指标
- models:存放自己实现的BERT模型代码
- output:输出目录,存放模型、训练日志
- processors:数据管道
- script：脚本
- utils: 工具类
- train.py: 主函数

## Usage

### Quick Start

you can start training model by run the shell
```
bash script/train.sh
```


### Results

F1 socre of results：

| model/f1_score            | Msra       | Ontonote   |
|---------------------------|------------|------------|
| BERT-Sotfmax              | 0.9553     | 0.8181     |
| BERT-LSTM-Sotfmax         | __0.9566__ | 0.8177     |
| BERT-LSTM-LabelSmooth     | 0.9549     | 0.8215     |
| BERT+Crf                  | 0.9562     | 0.8218     |
| BERT-LSTM-Crf             | 0.9561     | __0.8227__ |
| BERT-LSTM-Crf-LabelSmooth | 0.9547     | 0.8216     |

you also can see the indicator between training by running the shell below：
```
tensorboard --logdir ./output 
```


## Reference

- https://ai.tencent.com/ailab/nlp/en/embedding.html
- https://github.com/yangjianxin1/LEBERT-NER-Chinese
- https://arxiv.org/abs/2105.07148
- https://github.com/lonePatient/BERT-NER-Pytorch








