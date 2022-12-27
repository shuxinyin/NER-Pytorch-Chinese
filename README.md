# NER系列-中文实体识别模型实践

## Introduction

本项目主要基于Pytorch, 验证常见的NER范式模型在不同中文NER数据集上(Flat、Nested、Discontinuous)的表现
NER系列模型实践，包括如下：

1. Bert-Softmax、Bert-Crf、Bert-BiLSTM-Softmax、Bert-BiLSTM-Crf
2. Word-Feature Model(词汇增强模型)：FlatNER、[LEBERT](https://arxiv.org/abs/2105.07148)
3. PointerNET (To do)
4. MRC(Machine Reading Comprehension, MRC)
5. span-based NER (To do)

### Dataset Introduction

mainly tested on ner dataset as below:  
中文NER数据集：

- **Flat** NER Datasets: Ontonote4、Msra
- **Nested** NER Datasets：ACE 2004、 ACE 2005
- **Discontinuous** NER Datasets： CADEC

关于一般NER数据处理成以下格式:

```yaml
{
  "text": ["吴", "重", "阳", "，", "中", "国", "国", "籍",","],
  "label": ["B-NAME", "I-NAME", "I-NAME", "O", "B-CONT", "I-CONT", "I-CONT", "I-CONT", "O"]
}
```

阅读理解-NER（MRC-NER）处理成以下格式:

```yaml
{
  "context": "图 为 马 拉 维 首 都 利 隆 圭 政 府 办 公 大 楼 。 （ 本 报 记 者 温 宪 摄 ）",
  "end_position": [4,15],
  "entity_label": "NS",
  "impossible": false,
  "qas_id": "3820.1",
  "query": "按照地理位置划分的国家,城市,乡镇,大洲",
  "span_position": ["2;4", "7;15"],
  "start_position": [2, 7]
}
```

## Environment

python==3.8、transformers==4.12.3、torch==1.8.0
Or run the shell

```
pip install -r requirements.txt
```

## Project Structure

- config：some model parameters define
- datasets：数据管道
- losses:损失函数
- metrics:评价指标
- models:存放自己实现的BERT模型代码
- output:输出目录,存放模型、训练日志
- processors:数据处理
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

top F1 score of results on test：

| model/f1_score              | Msra       | Ontonote   |
|-----------------------------|------------|------------|
| BERT-Sotfmax                | 0.9553     | 0.8181     |
| BERT-BiLSTM-Sotfmax         | __0.9566__ | 0.8177     |
| BERT-BiLSTM-LabelSmooth     | 0.9549     | 0.8215     |
| BERT-Crf                    | 0.9562     | 0.8218     |
| BERT-BiLSTM-Crf             | 0.9561     | __0.8227__ |
| BERT-BiLSTM-Crf-LabelSmooth | 0.9547     | 0.8216     |
| BERT-BiLSTM-Crf-LEBERT      | 0.9518     | 0.8094     |
| BERT-BiLSTM-Sotfmax-LEBERT  | 0.9544     | 0.8196     |
| MRC                         | 0.942      | 0.812      |

#### Speed

GPU: 3060TI 8G  
在速度上，以Msra数据集为例，train数据量41728， 完成训练花费时间大概是如下，总体来说CRF要慢不少。

| model               | time      | batch_size |
|---------------------|-----------|------------|
| BERT-Sotfmax        | 6min 14s  | 24         |
| BERT-BiLSTM-Sotfmax | 6min 46s  | 24         |
| BERT+Crf            | 8min 06s  | 24         |
| BERT-BiLSTM-Crf     | 8min 20s  | 24         |
| MRC                 | 50min 10s | 4          |

## Paper & Refer

- [A Unified MRC Framework for Named Entity Recognition](https://arxiv.org/abs/1910.11476)
- [Lexicon Enhanced Chinese Sequence Labeling Using BERT Adapter](https://arxiv.org/abs/2105.07148)
- [tencent-ailab-embedding](https://ai.tencent.com/ailab/nlp/en/embedding.html)
- https://github.com/yangjianxin1/LEBERT-NER-Chinese
- https://github.com/lonePatient/BERT-NER-Pytorch








