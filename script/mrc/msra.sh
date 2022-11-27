#!/usr/bin/env bash
# -*- coding: utf-8 -*-
set -x

REPO_PATH=/data/GitProject/NER/NER-Pytorch
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
export TOKENIZERS_PARALLELISM=false

DATA_DIR=/data/GitProject/NER/NER-Pytorch/mrc_data/zh_msra
BERT_DIR=/data/Learn_Project/Backup_Data/bert_chinese
SPAN_WEIGHT=0.1
DROPOUT=0.2
LR=8e-6
MAXLEN=108
INTER_HIDDEN=1536

BATCH_SIZE=4
PREC=16
VAL_CKPT=0.25
ACC_GRAD=1
MAX_EPOCH=20
SPAN_CANDI=pred_and_gold
PROGRESS_BAR=1

OUTPUT_DIR=/data/GitProject/NER/NER-Pytorch/output/zh_msra${LR}20200913_dropout${DROPOUT}_maxlen${MAXLEN}

mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=0 python ${REPO_PATH}/train/mrc_ner_train.py \
--data_dir ${DATA_DIR} \
--bert_config_dir ${BERT_DIR} \
--max_length ${MAXLEN} \
--batch_size ${BATCH_SIZE} \
--lr ${LR} \
--mrc_dropout ${DROPOUT} \
--weight_span ${SPAN_WEIGHT} \
--span_loss_candidates ${SPAN_CANDI} \
--chinese \
--workers 6 \
--classifier_intermediate_hidden_size ${INTER_HIDDEN}

