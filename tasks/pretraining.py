import os
import sys

sys.path.append(os.getcwd())

import torch
import logging
from utils import logger_init
from model import BertConfig
from torch.utils.tensorboard import SummaryWriter


class ModelConfig:
    """基于BERT的MLM、NSP预训练模型的配置类"""

    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # # ========== wikitext2数据集相关配置 ==========
        # self.dataset_dir = os.path.join(
        #     self.project_dir, "data", "mlm_nsp_pretraining", "wikitext2"
        # )
        # self.pretrained_model_dir = os.path.join(
        #     self.project_dir, "archive", "bert_base_uncased_english"
        # )
        # self.train_filepath = os.path.join(self.dataset_dir, "wiki.train.tokens")
        # self.val_filepath = os.path.join(self.dataset_dir, "wiki.valid.tokens")
        # self.test_filepath = os.path.join(self.dataset_dir, "wiki.test.tokens")
        # self.dataset_name = "wikitext2"
        # self.split_sep = " . "

        # ========== songci数据集相关配置 ==========
        self.dataset_dir = os.path.join(
            self.project_dir, "data", "pretraining", "songci"
        )
        self.pretrained_model_dir = os.path.join(
            self.project_dir, "archive", "bert_base_chinese"
        )
        self.train_filepath = os.path.join(self.dataset_dir, "songci.train.txt")
        self.val_filepath = os.path.join(self.dataset_dir, "songci.valid.txt")
        self.test_filepath = os.path.join(self.dataset_dir, "songci.test.txt")
        self.dataset_name = "songci"
        self.split_sep = "。"

        self.vocab_path = os.path.join(self.pretrained_model_dir, "vocab.txt")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_save_dir = os.path.join(self.project_dir, "cache")
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        self.model_save_path = os.path.join(
            self.model_save_dir, f"{self.dataset_name}_pretraining_model.bin"
        )

        self.log_level = logging.INFO
        self.log_save_dir = os.path.join(self.project_dir, "logs")
        logger_init(
            log_filename=self.dataset_name + "_pretraining",
            log_level=self.log_level,
            log_dir=self.log_save_dir,
        )

        self.epochs = 200
        self.batch_size = 32
        self.is_sample_shuffle = True
        self.max_sen_len = None  # 填充模式
        self.eval_per_epoch = 1  # 验证模型的epoch数
        self.pad_index = 0
        self.random_state = 2023
        self.learning_rate = 4e-5
        self.weight_decay = 0.1
        # False表示使用自定义的多头注意力模块；True表示使用torch框架中实现的
        self.use_torch_multi_head = False

        self.masked_rate = 0.15  # 掩码比例
        self.masked_token_rate = 0.8  # 用[MASK]token替换的比例
        self.masked_token_unchanged_rate = 0.5  # 保持不变的token比例
        self.use_embedding_weight = True
        self.writer = SummaryWriter(f"runs/{self.dataset_name}_pretraining")

        # 导入BERT模型部分配置
        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value

        # 将当前配置打印到日志文件中
        logging.info("=" * 20)
        logging.info("### 将当前配置打印到日志文件中")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")
