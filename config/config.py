import torch
import argparse
import random
import os
import numpy as np


def set_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='gpu', choices=['gpu', 'cpu'], help="gpu or cpu")
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')
    parser.add_argument("--output_path", type=str, default='output/', help='模型与预处理数据的存放位置')
    parser.add_argument("--pretrain_embed_path", type=str,
                        default='/data/Learn_Project/Backup_Data/tencent-ailab-embedding-zh-d200-v0.2.0-s/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt',
                        help='预训练词向量路径')

    parser.add_argument('--loss_type', default='ce', type=str, choices=['lsr', 'focal', 'ce'], help='损失函数类型')
    parser.add_argument('--add_layer', default=1, type=str, help='在bert的第几层后面融入词汇信息')
    parser.add_argument("--lr", type=float, default=1e-5, help='Bert的学习率')
    parser.add_argument("--crf_lr", default=1e-3, type=float, help="crf的学习率")
    parser.add_argument("--adapter_lr", default=1e-3, type=float, help="crf的学习率")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--eps', default=1.0e-08, type=float, required=False, help='AdamW优化器的衰减率')
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size_train", type=int, default=4)
    parser.add_argument("--batch_size_eval", type=int, default=4)
    parser.add_argument("--eval_step", type=int, default=2, help="训练多少步，查看验证集的指标")
    parser.add_argument("--max_seq_len", type=int, default=150, help="输入的最大长度")
    parser.add_argument("--max_word_num", type=int, default=3, help="每个汉字最多融合多少个词汇信息")
    parser.add_argument("--max_scan_num", type=int, default=10000, help="取预训练词向量的前max_scan_num个构造字典树")
    parser.add_argument("--data_path", type=str, default="data/resume/", help='数据集存放路径')
    parser.add_argument("--dataset_name", type=str, choices=['resume', "weibo", 'ontonote4', 'msra'], default='resume',
                        help='数据集名称')
    parser.add_argument("--model_class", type=str,
                        choices=['lebert-softmax', 'bert-softmax', 'bert-lstm-softmax', 'bert-lstm-crf', 'bert-crf',
                                 'lebert-crf'],
                        default='lebert-softmax', help='模型类别')
    parser.add_argument("--pretrain_model_path", type=str, default="/data/Learn_Project/Backup_Data/bert_chinese")
    parser.add_argument("--overwrite", action='store_true', default=True, help="覆盖数据处理的结果")
    parser.add_argument("--do_train", action='store_true', default=True)
    parser.add_argument("--do_eval", action='store_true', default=True)
    parser.add_argument("--load_word_embed", action='store_true', default=True, help='是否加载预训练的词向量')

    parser.add_argument('--markup', default='bios', type=str, choices=['bios', 'bio'], help='数据集的标注方式')
    parser.add_argument('--grad_acc_step', default=1, type=int, required=False, help='梯度积累的步数')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False, help='梯度裁剪阈值')
    parser.add_argument('--seed', type=int, default=42, help='设置随机种子')
    parser.add_argument('--num_workers', type=int, default=0, help="dataloader加载数据时使用的线程数量")
    # parser.add_argument('--patience', type=int, default=0, help="用于early stopping,设为0时,
    #                       不进行early stopping.early stop得到的模型的生成效果不一定会更好。")
    parser.add_argument('--warmup_proportion', type=float, default=0.1,
                        help='Proportion of training to perform linear learning rate warmup '
                             'for,E.g., 0.1 = 10% of training.')
    args = parser.parse_args()
    return args


def set_random_seed(seed=42):
    """set seeds for reproducibility
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



