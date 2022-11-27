set -x

DEVICE=0
DATA_SET='msra'
MODEL_CLASS='bert-softmax'
LR=1e-5
CRF_LR=1e-3
ADAPTER_LR=1e-3

REPO_PATH=/data/GitProject/NER/NER-Pytorch
DATA_DIR=/data/GitProject/NER/NER-Pytorch/data
OUTPUT_PATH=/data/GitProject/NER/NER-Pytorch/output

PRETRAIN_MODEL='/data/Learn_Project/Backup_Data/bert_chinese'
PRETRAIN_EMBED_PATH='/data/Learn_Project/Backup_Data/tencent-ailab-embedding-zh-d200-v0.2.0-s/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt'
#export CUDA_VISIBLE_DEVICES=${DEVICE}

for DATA_SET in 'msra' 'ontonote4'; do
#for MODEL_CLASS in 'bert-softmax' 'bert-crf' 'bert-lstm-softmax' 'bert-lstm-crf'; do
for MODEL_CLASS in 'bert-lstm-softmax' 'bert-lstm-crf'; do
  echo "----------------------------------------${DATA_SET}:${MODEL_CLASS}----------------------------------------"
  python ${REPO_PATH}/train/train.py \
      --device gpu \
      --output_path ${OUTPUT_PATH} \
      --add_layer 1 \
      --loss_type lsr \
      --lr ${LR} \
      --crf_lr ${CRF_LR} \
      --adapter_lr ${ADAPTER_LR} \
      --weight_decay 0.01 \
      --eps 1.0e-08 \
      --epochs 1 \
      --batch_size_train 24 \
      --batch_size_eval 256 \
      --num_workers 0 \
      --eval_step 100 \
      --max_seq_len 150 \
      --max_word_num  3 \
      --max_scan_num 3000000 \
      --data_path ${DATA_DIR}/${DATA_SET}/ \
      --dataset_name ${DATA_SET} \
      --model_class ${MODEL_CLASS} \
      --pretrain_model_path ${PRETRAIN_MODEL} \
      --pretrain_embed_path  ${PRETRAIN_EMBED_PATH} \
      --seed 42 \
      --markup bios \
      --grad_acc_step 1 \
      --max_grad_norm 1.0 \
      --num_workers 0 \
      --warmup_proportion 0.1 \
      --load_word_embed \
      --do_train

done
done