import os
import sys

sys.path.append(os.getcwd())

import logging
import torch
from model.BERT.config import BertConfig
from utils.log_helper import logger_init


class ModelConfig:
    """基于BERT的文本对分类模型配置类"""

    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(
            self.project_dir, "data", "pair_sentence_classification"
        )
        self.pretrained_model_dir = os.path.join(
            self.project_dir, "archive", "bert_base_uncased_english"
        )
        self.vocab_path = os.path.join(self.pretrained_model_dir, "vocab.txt")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_filepath = os.path.join(self.dataset_dir, "train.txt")
        self.val_filepath = os.path.join(self.dataset_dir, "val.txt")
        self.test_filepath = os.path.join(self.dataset_dir, "test.txt")

        self.model_save_dir = os.path.join(self.project_dir, "cache")
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        self.log_save_dir = os.path.join(self.project_dir, "logs")
        logger_init(
            log_filename="pair_sen_cls",
            log_level=logging.INFO,
            log_dir=self.log_save_dir,
        )

        self.epochs = 5
        self.batch_size = 16
        self.num_labels = 3  # 分类标签0~2
        self.split_sep = "_!_"
        self.is_sample_shuffle = True
        self.max_sen_len = None  # 填充模式
        self.eval_per_epoch = 2
        self.learning_rate = 3.5e-5

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
