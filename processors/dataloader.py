import sys
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append("..")
from utils.utils import load_lines, write_lines


class NERDataset(Dataset):
    def __init__(self, file):
        self.lines = load_lines(file)
        self.data = [json.loads(line) for line in lines]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feature = self.data[index]

        return feature


class BatchTextCall(object):
    """call function for tokenizing and getting batch text
    """

    def __init__(self, tokenizer, max_len=312):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def text2id(self, batch_text):
        return self.tokenizer(batch_text, max_length=self.max_len,
                              truncation=True, padding='max_length', return_tensors='pt')

    def __call__(self, batch):
        batch_text = [item[0]['text'] for item in batch]
        batch_label = [item[1]['label'] for item in batch]

        source = self.text2id(batch_text)
        token = source.get('input_ids').squeeze(1)
        mask = source.get('attention_mask').squeeze(1)
        segment = source.get('token_type_ids').squeeze(1)
        label = torch.tensor(batch_label)

        return token, segment, mask, label


if __name__ == '__main__':
    import json
    import argparse
    from transformers import BertTokenizer

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.train_file = '../data/ner_data_put_here/ner_data/resume/train.json'
    args.dev_file = '../data/ner_data_put_here/ner_data/resume/dev.json'
    args.test_file = '../data/ner_data_put_here/ner_data/resume/test.json'
    args.max_seq_len = 150
    args.max_word_num = 5
    args.pretrain_embed_path = '/data/Learn_Project/Backup_Data/tencent-ailab-embedding-zh-d200-v0.2.0-s/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt'
    args.output_path = '../output/cner'
    args.max_scan_num = 10000
    args.label_path = '../data/ner_data_put_here/ner_data/resume/labels.txt'
    args.data_path = '../data/ner_data_put_here/ner_data/resume'
    args.overwrite = True
    tokenizer = BertTokenizer.from_pretrained('/data/Learn_Project/Backup_Data/bert_chinese')

    # feature = {
    #     'text': text, 'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': token_type_ids,
    #     'label_ids': label_ids
    # }
    # features.append(feature)

    lines = load_lines(args.dev_file)
    for line in lines:
        data = json.loads(line)
        print(data)
        sys.exit()
