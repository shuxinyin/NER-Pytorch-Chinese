import sys

sys.path.append("..")

import torch
from transformers import BertModel, BertPreTrainedModel, BertConfig
from models.ner_model import BertSoftmaxForNer, BertLSTMSoftmaxForNer, BertCrfForNer, LEBertSoftmaxForNer


def test_bert():
    # pretrain_model_path = '../pretrain_model/test'
    pretrain_model_path = '/data/Learn_Project/Backup_Data/bert_chinese'
    input_ids = torch.randint(0, 100, (4, 10))
    token_type_ids = torch.randint(0, 1, (4, 10))
    attention_mask = torch.randint(1, 2, (4, 10))

    config = BertConfig.from_pretrained(pretrain_model_path, num_labels=20)
    config.word_embed_dim = 200
    config.num_labels = 20
    config.loss_type = 'ce'
    config.add_layer = 0

    labels = torch.randint(0, 3, (4, 10))

    model = BertLSTMSoftmaxForNer.from_pretrained(pretrain_model_path, config=config)
    # model = BertSoftmaxForNer.from_pretrained(pretrain_model_path, config=config)
    loss, logits = model(
        input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)

    print(loss)
    print(logits.shape)


def test_lebert():
    # pretrain_model_path = '../pretrain_model/test'
    pretrain_model_path = '/data/Learn_Project/Backup_Data/bert_chinese'
    input_ids = torch.randint(0, 100, (4, 10))
    token_type_ids = torch.randint(0, 1, (4, 10))
    attention_mask = torch.randint(1, 2, (4, 10))

    word_ids = torch.randn((4, 10, 5, 200))
    word_mask = torch.randint(0, 1, (4, 10, 5))
    config = BertConfig.from_pretrained(pretrain_model_path, num_labels=20)
    config.word_embed_dim = 200
    config.num_labels = 20
    config.loss_type = 'ce'
    config.add_layer = 0

    labels = torch.randint(0, 3, (4, 10))

    # LEBertSoftmaxForNer
    model = LEBertSoftmaxForNer.from_pretrained(pretrain_model_path, config=config)
    loss, logits = model(
        input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
        word_ids=word_ids, word_mask=word_mask, labels=labels)

    print(loss)
    print(logits.shape)


if __name__ == '__main__':
    test_bert()