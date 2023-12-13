import os
import sys

sys.path.append(os.getcwd())

import time
import copy
import torch
import logging
from model.BERT.config import BertConfig
from utils.log_helper import logger_init
from transformers import BertTokenizer
from utils.data_helpers import LoadSenClsDataset
from model.downstream.bert_for_sen_cls import BertForSenCls


class ModelConfig:
    """基于BERT的文本分类模型的配置类"""

    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(
            self.project_dir, "data", "sentence_classification"
        )
        self.pretrained_model_dir = os.path.join(
            self.project_dir, "archive", "bert_base_chinese"
        )
        self.vocab_path = os.path.join(self.pretrained_model_dir, "vocab.txt")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_filepath = os.path.join(self.dataset_dir, "toutiao_train.txt")
        self.val_filepath = os.path.join(self.dataset_dir, "toutiao_val.txt")
        self.test_filepath = os.path.join(self.dataset_dir, "toutiao_test.txt")

        self.model_save_dir = os.path.join(self.project_dir, "cache")
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        self.log_save_dir = os.path.join(self.project_dir, "logs")

        self.epochs = 10
        self.batch_size = 64
        self.num_labels = 15
        self.split_sep = "_!_"
        self.is_sample_shuffle = True  # 是否打乱数据集
        self.max_sen_len = None  # 填充模式
        self.eval_per_epoch = 2  # 验证模型的epoch数

        logger_init(
            log_filename="sen_cls",
            log_level=logging.INFO,
            log_dir=self.log_save_dir,
        )

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


def evaluate(model, data_loader, pad_idx, device="cpu"):
    model.eval()
    with torch.no_grad():
        corrects, total = 0.0, 0
        for seqs, labels in data_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            padding_mask = (seqs == pad_idx).transpose(0, 1)
            logits = model(seqs, attention_mask=padding_mask)
            corrects += (logits.argmax(1) == labels).float().sum().item()
            total += len(labels)

    model.train()
    return corrects / total


def train(config: ModelConfig):
    """训练过程"""

    # 1.加载数据集并预处理
    # 借用transformers框架中的分词器
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    dataset = LoadSenClsDataset(
        vocab_path=config.vocab_path,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_sen_len=config.max_sen_len,
        split_sep=config.split_sep,
        max_position_embeddings=config.max_position_embeddings,
        pad_index=config.pad_token_id,
        is_sample_shuffle=config.is_sample_shuffle,
    )
    train_loader, test_loader, val_loader = dataset.data_loader(
        config.train_filepath, config.val_filepath, config.test_filepath
    )

    # 2.从本地BERT模型文件创建文本分类模型
    model = BertForSenCls(config, config.pretrained_model_dir)
    # 若不是第一次训练，则加载已有权重
    model_save_path = os.path.join(config.model_save_dir, "sen_cls_model.pt")
    if os.path.exists(model_save_path):
        loaded_params = torch.load(model_save_path)
        model.load_state_dict(loaded_params)
        logging.info("## 成功载入已有模型，继续训练......")
    model = model.to(config.device)

    # 3.定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    # 4.执行训练
    model.train()
    best_eval_acc = 0.0
    for epoch in range(config.epochs):
        losses = 0.0
        start_time = time.time()
        for idx, (seqs, labels) in enumerate(train_loader):
            seqs = seqs.to(config.device)  # #[src_len, batch_size]
            labels = labels.to(config.device)
            # #[batch_size, src_len]
            padding_mask = (seqs == dataset.PAD_IDX).transpose(0, 1)
            # feed forward
            loss, logits = model(
                input_ids=seqs,
                position_ids=None,
                token_type_ids=None,  # 输入的序列是单个文本序列
                attention_mask=padding_mask,
                labels=labels,
            )
            optimizer.zero_grad()  # 清除上一批次计算的权重梯度值
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新权重

            # # acc = (logits.argmax(1) == labels).float().mean()
            if (idx + 1) % 50 == 0:
                logging.info(
                    f"Epoch: [{epoch + 1}/{config.epochs}], Batch: [{idx + 1}/{len(train_loader)}], "
                    f"Batch loss: {loss.item():.3f}"
                )
            losses += loss.item()

        end_time = time.time()
        train_loss = losses / len(train_loader)
        logging.info(
            f"Epoch: [{epoch + 1}/{config.epochs}], Train loss: {train_loss:.3f}, Epoch time: {(end_time - start_time):.3f}s"
        )
        # 评估模型性能
        if (epoch + 1) % config.eval_per_epoch == 0:
            eval_acc = evaluate(model, val_loader, dataset.PAD_IDX, config.device)
            # 保存性能最好的模型权重
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                state_dict = copy.deepcopy(model.state_dict())
                torch.save(state_dict, model_save_path)
            logging.info(
                f"Epoch: [{epoch + 1}/{config.epochs}], Eval acc: {eval_acc:.3f}, Best eval acc: {best_eval_acc:.3f}"
            )


def inference(config: ModelConfig):
    """推理过程"""

    # 1.加载数据集并预处理
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    dataset = LoadSenClsDataset(
        vocab_path=config.vocab_path,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_sen_len=config.max_sen_len,
        split_sep=config.split_sep,
        max_position_embeddings=config.max_position_embeddings,
        pad_index=config.pad_token_id,
        is_sample_shuffle=config.is_sample_shuffle,
    )
    # 仅加载测试集即可
    test_loader = dataset.data_loader(
        test_filepath=config.test_filepath, only_test=True
    )

    # 2.创建模型并加载权重
    model = BertForSenCls(config, config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir, "sen_cls_model.pt")
    if os.path.exists(model_save_path):
        loaded_params = torch.load(model_save_path)
        model.load_state_dict(loaded_params)
        print("## 成功载入已有模型，执行推理......")
    else:
        raise FileNotFoundError("未找到模型权重文件，请先执行模型训练......")
    model = model.to(config.device)

    test_acc = evaluate(model, test_loader, dataset.PAD_IDX, config.device)
    return test_acc


if __name__ == "__main__":
    model_config = ModelConfig()
    # 训练
    train(model_config)
    # 推理
    infer_ct = 5
    for idx in range(infer_ct):
        start_time = time.time()
        infer_acc = inference(model_config)
        end_time = time.time()
        print(
            f"Infer number: [{idx + 1}/{infer_ct}], Acc: {infer_acc:.3f}, Cost time: {end_time - start_time:.4f}s"
        )
