import sys

sys.path.append('..')

import os
from os.path import join

import time
from tqdm import tqdm
import logging as logger

# from loguru import logger

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import transformers
from transformers import BertTokenizer, BertConfig

from model_class import MODEL_CLASS, PROCESSOR_CLASS
from metrics.ner_metrics import SeqEntityScore
from config.config import set_train_args, set_random_seed


def get_optimizer(model, args, warmup_steps, t_total):
    no_bert = ["word_embedding_adapter", "word_embeddings", "classifier", "crf"]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        # bert no_decay
        {
            "params": [p for n, p in model.named_parameters()
                       if (not any(nd in n for nd in no_bert) or n == 'bert.embeddings.word_embeddings.weight') and any(
                    nd in n for nd in no_decay)],
            "weight_decay": 0.0, 'lr': args.lr
        },
        # bert decay
        {
            "params": [p for n, p in model.named_parameters()
                       if (not any(
                    nd in n for nd in no_bert) or n == 'bert.embeddings.word_embeddings.weight') and not any(
                    nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay, 'lr': args.lr
        },
        # other no_decay
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_bert) and n != 'bert.embeddings.word_embeddings.weight' and any(
                    nd in n for nd in no_decay)],
            "weight_decay": 0.0, "lr": args.adapter_lr
        },
        # other decay
        {
            "params": [p for n, p in model.named_parameters() if
                       any(nd in n for nd in no_bert) and n != 'bert.embeddings.word_embeddings.weight' and not any(
                           nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay, "lr": args.adapter_lr
        }
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.eps)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )
    return optimizer, scheduler


def train(model, train_loader, dev_loader, test_loader, optimizer, scheduler, args):
    logger.info("start training")
    model.train()
    device = args.device
    best = 0
    dev = 0
    for epoch in range(args.epochs):
        logger.info('start {}-th epoch training'.format(epoch + 1))
        for batch_idx, data in enumerate(tqdm(train_loader)):
            step = epoch * len(train_loader) + batch_idx + 1
            input_ids = data['input_ids'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            label_ids = data['label_ids'].to(device)
            # 不同模型输入不同
            if args.model_class == 'bert-softmax' or args.model_class == 'bert-lstm-softmax':
                loss, logits = model(input_ids, attention_mask, token_type_ids, args.ignore_index, label_ids)
            elif args.model_class == 'bert-crf' or args.model_class == 'bert-lstm-crf':
                loss, logits = model(input_ids, attention_mask, token_type_ids, label_ids)
            elif args.model_class == 'lebert-softmax':
                word_ids = data['word_ids'].to(device)
                word_mask = data['word_mask'].to(device)
                loss, logits = model(input_ids, attention_mask, token_type_ids, word_ids, word_mask, args.ignore_index,
                                     label_ids)
            elif args.model_class == 'lebert-crf':
                word_ids = data['word_ids'].to(device)
                word_mask = data['word_mask'].to(device)
                loss, logits = model(input_ids, attention_mask, token_type_ids, word_ids, word_mask, label_ids)

            loss = loss.mean()  # 对多卡的loss取平均

            # 梯度累积
            loss = loss / args.grad_acc_step
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # 进行一定step的梯度累计之后，更新参数
            if step % args.grad_acc_step == 0:
                # 更新参数
                optimizer.step()
                # 更新学习率
                scheduler.step()
                # 清空梯度信息
                optimizer.zero_grad()

            # 评测验证集和测试集上的指标
            if step % args.eval_step == 0:
                logger.info('evaluate dev and test set')
                dev_result = evaluate(args, model, dev_loader)
                test_result = evaluate(args, model, test_loader)
                writer.add_scalar('dev loss', dev_result['loss'], step)
                writer.add_scalar('dev f1', dev_result['f1'], step)
                writer.add_scalar('dev precision', dev_result['acc'], step)
                writer.add_scalar('dev recall', dev_result['recall'], step)

                writer.add_scalar('test loss', test_result['loss'], step)
                writer.add_scalar('test f1', test_result['f1'], step)
                writer.add_scalar('test precision', test_result['acc'], step)
                writer.add_scalar('test recall', test_result['recall'], step)

                model.train()
                if best < test_result['f1']:
                    best = test_result['f1']
                    dev = dev_result['f1']
                    logger.info(
                        'higher f1 of Test is {}, dev is {} in step {} epoch {}'.format(best, dev, step, epoch + 1))
                    # save_pa th = join(args.output_path, 'checkpoint-{}'.format(step))
                    # model_to_save = model.module if hasattr(model, 'module') else model
                    # model_to_save.save_pretrained(args.output_path)
    logger.info('best f1 of test is {}, dev is {}'.format(best, dev))


def evaluate(args, model, dataloader):
    """
    计算数据集上的指标
    :param args:
    :param model:
    :param dataloader:
    :return:
    """
    model.eval()
    device = args.device
    metric = SeqEntityScore(args.id2label, markup=args.markup)

    # Eval!
    eval_loss = 0.0  #
    with torch.no_grad():
        for data in tqdm(dataloader):
            input_ids = data['input_ids'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            label_ids = data['label_ids'].to(device)
            # 不同模型输入不同
            if args.model_class == 'bert-softmax' or args.model_class == 'bert-lstm-softmax':
                loss, logits = model(input_ids, attention_mask, token_type_ids, args.ignore_index, label_ids)
            elif args.model_class == 'bert-crf' or args.model_class == 'bert-lstm-crf':
                loss, logits = model(input_ids, attention_mask, token_type_ids, label_ids)
            elif args.model_class == 'lebert-softmax':
                word_ids = data['word_ids'].to(device)
                word_mask = data['word_mask'].to(device)
                loss, logits = model(input_ids, attention_mask, token_type_ids, word_ids, word_mask, args.ignore_index,
                                     label_ids)
            elif args.model_class == 'lebert-crf':
                word_ids = data['word_ids'].to(device)
                word_mask = data['word_mask'].to(device)
                loss, logits = model(input_ids, attention_mask, token_type_ids, word_ids, word_mask, label_ids)
            loss = loss.mean()  # 对多卡的loss取平均
            eval_loss += loss

            input_lens = (torch.sum(input_ids != 0, dim=-1) - 2).tolist()  # 减去padding的[CLS]与[SEP]
            if args.model_class in ['lebert-crf', 'bert-crf']:
                preds = model.crf.decode(logits, attention_mask).squeeze(0)
                preds = preds[:, 1:].tolist()  # 减去padding的[CLS]
            else:
                preds = torch.argmax(logits, dim=2)[:, 1:].tolist()  # 减去padding的[CLS]
            label_ids = label_ids[:, 1:].tolist()  # 减去padding的[CLS]
            # preds = np.argmax(logits.cpu().numpy(), axis=2).tolist()
            # label_ids = label_ids.cpu().numpy().tolist()
            for i in range(len(label_ids)):
                input_len = input_lens[i]
                pred = preds[i][:input_len]
                label = label_ids[i][:input_len]
                metric.update(pred_paths=[pred], label_paths=[label])

    eval_loss = eval_loss / len(dataloader)
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss

    return results


def main(args):
    # 分词器
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path, do_lower_case=True)
    # 数据处理器
    processor = PROCESSOR_CLASS[args.model_class](args, tokenizer)
    args.id2label = processor.label_vocab.idx2token
    args.ignore_index = processor.label_vocab.convert_token_to_id('[PAD]')
    # 初始化模型配置
    config = BertConfig.from_pretrained(args.pretrain_model_path)
    config.num_labels = processor.label_vocab.size
    config.loss_type = args.loss_type
    if args.model_class in ['lebert-softmax', 'lebert-crf']:
        config.add_layer = args.add_layer
        config.word_vocab_size = processor.word_embedding.shape[0]
        config.word_embed_dim = processor.word_embedding.shape[1]
    # 初始化模型
    model = MODEL_CLASS[args.model_class].from_pretrained(args.pretrain_model_path, config=config).to(args.device)
    # 初始化模型的词向量
    if args.model_class in ['lebert-softmax', 'lebert-crf'] and args.load_word_embed:
        logger.info('initialize word_embeddings with pretrained embedding')
        model.word_embeddings.weight.data.copy_(torch.from_numpy(processor.word_embedding))

    # 训练
    if args.do_train:
        # 加载数据集
        train_dataset = processor.get_train_data()
        # train_dataset = train_dataset[:8]
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True,
                                      num_workers=args.num_workers)
        dev_dataset = processor.get_dev_data()
        # dev_dataset = dev_dataset[:4]
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size_eval, shuffle=False,
                                    num_workers=args.num_workers)
        test_dataset = processor.get_test_data()
        # test_dataset = test_dataset[:4]
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_eval, shuffle=False,
                                     num_workers=args.num_workers)

        t_total = len(train_dataloader) // args.grad_acc_step * args.epochs
        warmup_steps = int(t_total * args.warmup_proportion)
        optimizer, scheduler = get_optimizer(model, args, warmup_steps, t_total)
        train(model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, args)

    # 测试集上的指标
    if args.do_eval:
        # 加载验证集
        dev_dataset = processor.get_dev_data()
        # dev_dataset = dev_dataset[:4]
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size_eval, shuffle=False,
                                    num_workers=args.num_workers)
        # 加载测试集
        test_dataset = processor.get_test_data()
        # test_dataset = test_dataset[:4]
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_eval, shuffle=False,
                                     num_workers=args.num_workers)
        model = MODEL_CLASS[args.model_class].from_pretrained(args.output_path, config=config).to(args.device)
        model.eval()

        result = evaluate(args, model, dev_dataloader)
        logger.info(
            'Dev Set precision:{}, recall:{}, f1:{}, loss:{}'.format(result['acc'], result['recall'], result['f1'],
                                                                     result['loss'].item()))
        # 测试集上的指标
        result = evaluate(args, model, test_dataloader)
        logger.info(
            'Test Set precision:{}, recall:{}, f1:{}, loss:{}'.format(result['acc'], result['recall'], result['f1'],
                                                                      result['loss'].item()))


if __name__ == '__main__':
    # 设置参数
    args = set_train_args()
    set_random_seed(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else "cpu")

    args.output_path = join(args.output_path, args.dataset_name, args.model_class,
                            'load_word_embed' if args.load_word_embed else 'not_load_word_embed')
    args.train_file = join(args.data_path, 'train.json')
    args.dev_file = join(args.data_path, 'dev.json')
    args.test_file = join(args.data_path, 'test.json')
    args.label_path = join(args.data_path, 'labels.txt')
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.do_train:
        cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        format = '%(asctime)s - %(name)s - %(message)s'
        logger.basicConfig(format=format,
                           filename=join(args.output_path, f'train-{args.loss_type}-{cur_time}.log'),
                           level=logger.INFO)
        logger.info(args)
        writer = SummaryWriter(args.output_path)
    main(args)
