#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import logging
from collections import namedtuple
from typing import Dict

import torch
import torch.nn as nn
from tokenizers import BertWordPieceTokenizer
from torch import Tensor
from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from transformers import AdamW

from datasets.mrc_ner_dataset import MRCNERDataset
from datasets.truncate_dataset import TruncateDataset
from datasets.collate_functions import collate_to_max_length
from metrics.query_span_f1 import QuerySpanF1
from models.mrc_query_ner import BertQueryNER, BertQueryNerConfig
from config.get_parser import get_parser
from config.config import set_random_seed

set_random_seed(0)


class BertLabeling(nn.Module):
    def __init__(
            self, args: argparse.Namespace):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        format = '%(asctime)s - %(name)s - %(message)s'
        if isinstance(args, argparse.Namespace):
            # self.save_hyperparameters(args)
            self.args = args
            logging.basicConfig(format=format,
                                filename=os.path.join(self.args.default_root_dir, "eval_result_log.txt"),
                                level=logging.INFO)
        else:
            # eval mode
            TmpArgs = namedtuple("tmp_args", field_names=list(args.keys()))
            self.args = args = TmpArgs(**args)
            logging.basicConfig(format=format,
                                filename=os.path.join(self.args.default_root_dir, "eval_test.txt"),
                                level=logging.INFO)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bert_dir = args.bert_config_dir
        self.data_dir = self.args.data_dir

        bert_config = BertQueryNerConfig.from_pretrained(args.bert_config_dir,
                                                         hidden_dropout_prob=args.bert_dropout,
                                                         attention_probs_dropout_prob=args.bert_dropout,
                                                         mrc_dropout=args.mrc_dropout,
                                                         classifier_act_func=args.classifier_act_func,
                                                         classifier_intermediate_hidden_size=args.classifier_intermediate_hidden_size)

        self.model = BertQueryNER.from_pretrained(args.bert_config_dir,
                                                  config=bert_config).to(self.device)

        logging.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))
        self.result_logger = logging.getLogger(__name__)
        self.result_logger.setLevel(logging.INFO)
        self.result_logger.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))
        self.bce_loss = BCEWithLogitsLoss(reduction="none")

        weight_sum = args.weight_start + args.weight_end + args.weight_span
        self.weight_start = args.weight_start / weight_sum
        self.weight_end = args.weight_end / weight_sum
        self.weight_span = args.weight_span / weight_sum
        self.flat_ner = args.flat
        self.span_f1 = QuerySpanF1(flat=self.flat_ner)
        self.chinese = args.chinese
        self.optimizer = args.optimizer
        self.span_loss_candidates = args.span_loss_candidates

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--mrc_dropout", type=float, default=0.1,
                            help="mrc dropout rate")
        parser.add_argument("--bert_dropout", type=float, default=0.1,
                            help="bert dropout rate")
        parser.add_argument("--classifier_act_func", type=str, default="gelu")
        parser.add_argument("--classifier_intermediate_hidden_size", type=int, default=1024)
        parser.add_argument("--weight_start", type=float, default=1.0)
        parser.add_argument("--weight_end", type=float, default=1.0)
        parser.add_argument("--weight_span", type=float, default=1.0)
        parser.add_argument("--flat", action="store_true", help="is flat ner")
        parser.add_argument("--span_loss_candidates", choices=["all", "pred_and_gold", "pred_gold_random", "gold"],
                            default="all", help="Candidates used to compute span loss")
        parser.add_argument("--chinese", action="store_true",
                            help="is chinese dataset")
        parser.add_argument("--optimizer", choices=["adamw", "sgd", "torch.adam"], default="adamw",
                            help="loss type")
        parser.add_argument("--final_div_factor", type=float, default=1e4,
                            help="final div factor of linear decay scheduler")
        parser.add_argument("--lr_scheduler", type=str, default="onecycle", )
        parser.add_argument("--lr_mini", type=float, default=-1)
        parser.add_argument("--default_root_dir", type=str, default="../../output")
        parser.add_argument("--max_epochs", type=int, default=3)
        return parser

    def configure_optimizers(self, len_train_dataloader):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.98),  # according to RoBERTa paper
                              lr=self.args.lr,
                              eps=self.args.adam_epsilon, )

        t_total = len_train_dataloader * self.args.max_epochs
        if self.args.lr_scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.args.lr, pct_start=float(self.args.warmup_steps / t_total),
                final_div_factor=self.args.final_div_factor,
                total_steps=t_total, anneal_strategy='linear'
            )

        return optimizer, scheduler

    # def forward(self, input_ids, attention_mask, token_type_ids):
    #     return self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    def compute_loss(self, start_logits, end_logits, span_logits,
                     start_labels, end_labels, match_labels, start_label_mask, end_label_mask):
        batch_size, seq_len = start_logits.size()

        start_float_label_mask = start_label_mask.view(-1).float()
        end_float_label_mask = end_label_mask.view(-1).float()
        match_label_row_mask = start_label_mask.bool().unsqueeze(-1).expand(-1, -1, seq_len)
        match_label_col_mask = end_label_mask.bool().unsqueeze(-2).expand(-1, seq_len, -1)
        match_label_mask = match_label_row_mask & match_label_col_mask
        match_label_mask = torch.triu(match_label_mask, 0)  # start should be less equal to end

        if self.span_loss_candidates == "all":
            # naive mask
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()
        else:
            # use only pred or golden start/end to compute match loss
            start_preds = start_logits > 0
            end_preds = end_logits > 0
            if self.span_loss_candidates == "gold":
                match_candidates = ((start_labels.unsqueeze(-1).expand(-1, -1, seq_len) > 0)
                                    & (end_labels.unsqueeze(-2).expand(-1, seq_len, -1) > 0))
            elif self.span_loss_candidates == "pred_gold_random":
                gold_and_pred = torch.logical_or(
                    (start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                    (start_labels.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_labels.unsqueeze(-2).expand(-1, seq_len, -1))
                )
                data_generator = torch.Generator()
                data_generator.manual_seed(0)
                random_matrix = torch.empty(batch_size, seq_len, seq_len).uniform_(0, 1)
                random_matrix = torch.bernoulli(random_matrix, generator=data_generator).long()
                random_matrix = random_matrix.cuda()
                match_candidates = torch.logical_or(
                    gold_and_pred, random_matrix
                )
            else:
                match_candidates = torch.logical_or(
                    (start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                    (start_labels.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_labels.unsqueeze(-2).expand(-1, seq_len, -1))
                )
            match_label_mask = match_label_mask & match_candidates
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()

        start_loss = self.bce_loss(start_logits.view(-1), start_labels.view(-1).float())
        start_loss = (start_loss * start_float_label_mask).sum() / start_float_label_mask.sum()
        end_loss = self.bce_loss(end_logits.view(-1), end_labels.view(-1).float())
        end_loss = (end_loss * end_float_label_mask).sum() / end_float_label_mask.sum()
        match_loss = self.bce_loss(span_logits.view(batch_size, -1), match_labels.view(batch_size, -1).float())
        match_loss = match_loss * float_match_label_mask
        match_loss = match_loss.sum() / (float_match_label_mask.sum() + 1e-10)

        return start_loss, end_loss, match_loss

    def training_step(self, batch, batch_idx):
        # tf_board_logs = {
        #     "lr": self.trainer.optimizers[0].param_groups[0]['lr']
        # }
        tf_board_logs = {}
        tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx = batch

        # num_tasks * [bsz, length, num_labels]
        attention_mask = (tokens != 0).long()

        tokens, token_type_ids, attention_mask, start_labels, end_labels, start_label_mask, end_label_mask, match_labels = tokens.to(
            self.device), token_type_ids.to(self.device), attention_mask.to(self.device), start_labels.to(
            self.device), end_labels.to(self.device), start_label_mask.to(self.device), end_label_mask.to(
            self.device), match_labels.to(self.device)

        start_logits, end_logits, span_logits = self.model(tokens, attention_mask, token_type_ids)

        start_loss, end_loss, match_loss = self.compute_loss(start_logits=start_logits,
                                                             end_logits=end_logits,
                                                             span_logits=span_logits,
                                                             start_labels=start_labels,
                                                             end_labels=end_labels,
                                                             match_labels=match_labels,
                                                             start_label_mask=start_label_mask,
                                                             end_label_mask=end_label_mask
                                                             )

        total_loss = self.weight_start * start_loss + self.weight_end * end_loss + self.weight_span * match_loss

        tf_board_logs[f"train_loss"] = total_loss
        tf_board_logs[f"start_loss"] = start_loss
        tf_board_logs[f"end_loss"] = end_loss
        tf_board_logs[f"match_loss"] = match_loss

        return {'loss': total_loss, 'log': tf_board_logs}

    def validation_step(self, batch, batch_idx):
        output = {}
        tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx = batch

        attention_mask = (tokens != 0).long()
        start_logits, end_logits, span_logits = self(tokens, attention_mask, token_type_ids)

        start_loss, end_loss, match_loss = self.compute_loss(start_logits=start_logits,
                                                             end_logits=end_logits,
                                                             span_logits=span_logits,
                                                             start_labels=start_labels,
                                                             end_labels=end_labels,
                                                             match_labels=match_labels,
                                                             start_label_mask=start_label_mask,
                                                             end_label_mask=end_label_mask
                                                             )

        total_loss = self.weight_start * start_loss + self.weight_end * end_loss + self.weight_span * match_loss

        output[f"val_loss"] = total_loss
        output[f"start_loss"] = start_loss
        output[f"end_loss"] = end_loss
        output[f"match_loss"] = match_loss

        start_preds, end_preds = start_logits > 0, end_logits > 0
        span_f1_stats = self.span_f1(start_preds=start_preds, end_preds=end_preds, match_logits=span_logits,
                                     start_label_mask=start_label_mask, end_label_mask=end_label_mask,
                                     match_labels=match_labels)
        output["span_f1_stats"] = span_f1_stats

        return output

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}

        all_counts = torch.stack([x[f'span_f1_stats'] for x in outputs]).view(-1, 3).sum(0)
        span_tp, span_fp, span_fn = all_counts
        span_recall = span_tp / (span_tp + span_fn + 1e-10)
        span_precision = span_tp / (span_tp + span_fp + 1e-10)
        span_f1 = span_precision * span_recall * 2 / (span_recall + span_precision + 1e-10)
        tensorboard_logs[f"span_precision"] = span_precision
        tensorboard_logs[f"span_recall"] = span_recall
        tensorboard_logs[f"span_f1"] = span_f1
        self.result_logger.info(
            f"EVAL INFO -> current_epoch is: {self.trainer.current_epoch}, current_global_step is: {self.trainer.global_step} ")
        self.result_logger.info(
            f"EVAL INFO -> valid_f1 is: {span_f1}; precision: {span_precision}, recall: {span_recall}.")

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        """"""
        output = {}
        tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx = batch

        attention_mask = (tokens != 0).long()

        tokens, token_type_ids, attention_mask, start_labels, end_labels, start_label_mask, end_label_mask, match_labels = tokens.to(
            self.device), token_type_ids.to(self.device), attention_mask.to(self.device), start_labels.to(
            self.device), end_labels.to(self.device), start_label_mask.to(self.device), end_label_mask.to(
            self.device), match_labels.to(self.device)

        start_logits, end_logits, span_logits = self.model(tokens, attention_mask, token_type_ids)

        start_preds, end_preds = start_logits > 0, end_logits > 0
        span_f1_stats = self.span_f1(start_preds=start_preds, end_preds=end_preds, match_logits=span_logits,
                                     start_label_mask=start_label_mask, end_label_mask=end_label_mask,
                                     match_labels=match_labels)
        output["span_f1_stats"] = span_f1_stats

        return output

    def test_epoch_end(self, outputs) -> Dict[str, Dict[str, Tensor]]:
        tensorboard_logs = {}
        all_counts = torch.stack([x[f'span_f1_stats'] for x in outputs]).view(-1, 3).sum(0)
        span_tp, span_fp, span_fn = all_counts
        span_recall = span_tp / (span_tp + span_fn + 1e-10)
        span_precision = span_tp / (span_tp + span_fp + 1e-10)
        span_f1 = span_precision * span_recall * 2 / (span_recall + span_precision + 1e-10)
        print(f"TEST INFO -> test_f1 is: {span_f1} precision: {span_precision}, recall: {span_recall}")
        self.result_logger.info(
            f"TEST INFO -> test_f1 is: {span_f1} precision: {span_precision}, recall: {span_recall}")
        return {'log': tensorboard_logs}

    def get_dataloader(self, prefix="train", limit: int = None) -> DataLoader:
        """get training dataloader"""
        """
        load_mmap_dataset
        """
        json_path = os.path.join(self.data_dir, f"mrc-ner.{prefix}")
        vocab_path = os.path.join(self.bert_dir, "vocab.txt")
        dataset = MRCNERDataset(json_path=json_path,
                                tokenizer=BertWordPieceTokenizer(vocab_path),
                                max_length=self.args.max_length,
                                is_chinese=self.chinese,
                                pad_to_maxlen=False
                                )

        if limit is not None:
            dataset = TruncateDataset(dataset, limit)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size if prefix == "train" else self.args.batch_size // 2,
            num_workers=self.args.workers,
            shuffle=True if prefix == "train" else False,
            collate_fn=collate_to_max_length
        )

        return dataloader


def main():
    import sys
    import time
    from tqdm import tqdm
    parser = get_parser()
    # add model specific args
    parser = BertLabeling.add_model_specific_args(parser)
    args = parser.parse_args()
    print(args)

    Trainer = BertLabeling(args)

    train_dataloader = Trainer.get_dataloader(prefix="train")
    val_dataloader = Trainer.get_dataloader(prefix="dev")
    test_dataloader = Trainer.get_dataloader(prefix="test")

    optimizer, scheduler = Trainer.configure_optimizers(len(train_dataloader))

    for epoch in range(args.max_epochs):
        tqdm_bar = tqdm(train_dataloader, desc="Training epoch{epoch}".format(epoch=epoch))

        for batch_idx, batch in enumerate(train_dataloader):
            out_dic = Trainer.training_step(batch, batch_idx)
            total_loss = out_dic['loss']
            tqdm_bar.set_postfix(loss=total_loss.item())
            time.sleep(0.01)
            tqdm_bar.update(1)

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        # if len(train_dataloader) // 4 == batch_idx:
        print(f"------------VAL-{epoch}---------------------")
        out_list = []
        for batch_idx, batch in enumerate(test_dataloader):
            out_dic = Trainer.test_step(batch, batch_idx)
            out_list.append(out_dic)
        Trainer.test_epoch_end(out_list)


if __name__ == '__main__':
    main()
