import os
import random
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .data_helpers import Vocab
from .data_helpers import pad_sequence
from .data_helpers import process_cache


def format_wikitext2(filepath=None, sep=" . "):
    """
    格式化原始的wikitext2数据集
    :return: 返回一个二维list，外层list元素为一个文本段落；内层list元素为一个段落中的句子 `[[para1_sen1, para1_sen2, ...], [para2_sen1, para2_sen2, ...], ...]`
    """

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()  # 读取所有行，每一行为一个文本段落

    paragraphs = []
    for line in tqdm(lines, ncols=80, desc="## 正在读取wikitext2原始数据"):
        # 将段落转为小写，并按分隔符分为句子
        sentences = line.lower().split(sep)

        # 给每个句子加上分隔符，并且去除最后的空句
        tmp_sens = []
        for sen in sentences:
            sen = sen.strip()
            if len(sen) == 0:
                continue
            sen += sep
            tmp_sens.append(sen)

        # 若段落内少于两条句子，则舍弃，因为NSP任务的输入需要一对句子
        if len(tmp_sens) < 2:
            continue

        paragraphs.append(tmp_sens)

    random.shuffle(paragraphs)  # 将所有段落打乱
    return paragraphs


def format_songci(filepath=None, sep="。"):
    """格式化原始的宋词数据集"""

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()  # 一次读取所有行，每一行为一首词

    paragraphs = []
    for line in tqdm(lines, ncols=80, desc="## 正在读取宋词原始数据"):
        # 去除有乱码字符的段落
        if "□" in line or "……" in line:
            continue

        sentences = line.split(sep)

        # 给每个句子加上分隔符，并且去除最后的空句
        tmp_sens = []
        for sen in sentences:
            sen = sen.strip()
            if len(sen) == 0:
                continue
            sen += sep
            tmp_sens.append(sen)

        # 去除少于两个句子的段落
        if len(tmp_sens) < 2:
            continue

        paragraphs.append(tmp_sens)

    random.shuffle(paragraphs)  # 将所有段落打乱
    return paragraphs


def format_custom(filepath=None, sep=None):
    """格式化自定义的数据集"""

    raise NotImplementedError(
        "本函数未实现，请参照 `format_wikitext2()` 或 `format_songci()` 函数返回格式进行实现"
    )


class LoadPretrainingDataset(object):
    """加载预训练数据集"""

    def __init__(
        self,
        vocab_path="./vocab.txt",
        tokenizer=None,
        batch_size=32,
        max_sen_len=None,
        max_position_embeddings=512,
        split_sep="。",
        pad_index=0,
        is_sample_shuffle=True,
        dataset_name="wikitext2",
        masked_rate=0.15,
        masked_token_rate=0.8,
        masked_token_unchanged_rate=0.5,
        random_state=2023,
    ):
        self.vocab = Vocab(vocab_path)
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        # min(max_sen_len, max_position_embeddings)决定了输入序列长度
        if isinstance(max_sen_len, int) and max_sen_len > max_position_embeddings:
            max_sen_len = max_position_embeddings
        self.max_sen_len = max_sen_len
        self.max_position_embeddings = max_position_embeddings

        self.split_sep = split_sep
        self.PAD_IDX = pad_index
        self.CLS_IDX = self.vocab["[CLS]"]
        self.SEP_IDX = self.vocab["[SEP]"]
        self.MASK_IDX = self.vocab["[MASK]"]
        self.is_sample_shuffle = is_sample_shuffle

        self.dataset_name = dataset_name
        self.masked_rate = masked_rate
        self.masked_token_rate = masked_token_rate
        self.masked_token_unchanged_rate = masked_token_unchanged_rate
        self.random_state = random_state
        random.seed(random_state)  # 设置随机状态，用于复现结果

    def format_data(self, filepath):
        """
        将原始数据集格式化为标准形式
        :return: `[[para1_sen1, para1_sen2, ...], [para2_sen1, para2_sen2, ...], ...]`
        """

        # 依据数据集名称调用对应的格式化函数，注意：格式化函数返回格式需要保持一致
        # wikitext2数据集
        if self.dataset_name == "wikitext2":
            return format_wikitext2(filepath, self.split_sep)
        # 宋词数据集
        elif self.dataset_name == "songci":
            return format_songci(filepath, self.split_sep)
        # 其他自定义数据集
        elif self.dataset_name == "custom":
            return format_custom(filepath)
        else:
            raise ValueError(
                f"数据集 {self.dataset_name} 不存在对应的格式化函数，"
                f"请参考函数 `format_wikitext2()` 实现对应的格式化函数！"
            )

    @staticmethod
    def get_next_sentence_sample(sentence, next_sentence, paragraphs):
        """由给定的连续两个序列和所有文本段落，生成一个句子对样本"""

        # 正负样本数量相同
        # 传入的两个句子构成正样本，标签为True
        if random.random() < 0.5:
            is_next = True
        # 构造负样本，标签为False
        else:
            # 先随机选中一个段落，再随机选中一个句子
            new_next_sentence = next_sentence
            while next_sentence == new_next_sentence:  # 避免随机选中的下一个句子和传入的下一个句子相同
                new_next_sentence = random.choice(random.choice(paragraphs))
            next_sentence = new_next_sentence
            is_next = False

        return sentence, next_sentence, is_next

    def masking_tokens(self, token_ids, candidate_mask_positions, num_mask_ids):
        """依据需要mask的tokens数量和候选mask位置，对token_ids进行mask"""

        # MLM任务样本的标签——若token被mask，则对应标签为词表内的索引id；若没有被mask，则标签为PAD_IDX，在计算loss时会被忽略
        mlm_label = [self.PAD_IDX] * len(token_ids)
        mask_ct = 0  # 记录已被mask的tokens数量

        for mask_pos in candidate_mask_positions:
            # 被mask的tokens数量已达到要求
            if mask_ct >= num_mask_ids:
                break

            new_token_id = None  # 用于mask（即替换）的token id
            # 15%的tokens中的80%替换为[MASK] token
            if random.random() < self.masked_token_rate:
                new_token_id = self.MASK_IDX
            else:
                # 15%的tokens中的10%保持不变
                # 20% * 0.5 = 10%
                if random.random() < self.masked_token_unchanged_rate:
                    new_token_id = token_ids[mask_pos]
                # 15%的tokens中的最后10%随机替换为一个词表内的token
                else:
                    new_token_id = random.randint(0, len(self.vocab.itos) - 1)

            # 保存原索引并mask
            mlm_label[mask_pos] = token_ids[mask_pos]
            token_ids[mask_pos] = new_token_id
            mask_ct += 1

        return token_ids, mlm_label

    def get_masked_sample(self, token_ids):
        """
        对token_ids进行mask处理
        :param token_ids: e.g. `[101, 1031, 4895, 2243, 1033, 10029, 2000, 2624, 1031,....]`
        :return mlm_id_seq: `[101, 1031, 103, 2243, 1033, 10029, 2000, 103,  1031, ...]`
                mlm_label: `[ 0,   0,   4895,  0,    0,    0,    0,   2624,  0,...]`
        """
        candidate_mask_positions = []  # 候选mask位置
        for pos, id in enumerate(token_ids):
            # 在MLM任务中，不会mask特殊token
            if id in [self.CLS_IDX, self.SEP_IDX]:
                continue
            candidate_mask_positions.append(pos)  # 例如[2, 3, 4, 5, ....]
        random.shuffle(candidate_mask_positions)  # 打乱候选mask位置

        # 计算需要被mask的tokens数量，BERT模型中的默认mask比例是15%
        num_mask_ids = max(1, round(len(token_ids) * self.masked_rate))
        logging.debug(f"## 被Mask的tokens数量为：{num_mask_ids}")

        mlm_id_seq, mlm_label = self.masking_tokens(
            token_ids, candidate_mask_positions, num_mask_ids
        )

        return mlm_id_seq, mlm_label

    @process_cache(
        unique_keys=[
            "max_sen_len",
            "masked_rate",
            "masked_token_rate",
            "masked_token_unchanged_rate",
            "random_state",
        ]
    )
    def data_process(self, filepath=None):
        """构造NSP和MLM两个预训练任务接收格式的样本"""

        paragraphs = self.format_data(filepath)  # 格式化原始数据
        data = []  # 每个元素为一个样本，包括Masked索引序列、token_type_id序列、MLM、NSP任务的标签
        max_len = 0  # 保存最长序列长度

        desc = f"## 正在处理NSP和MLM预训练数据集 {filepath.split(os.sep)[-1]}"
        for para in tqdm(paragraphs, ncols=80, desc=desc):  # 遍历每个段落
            for i in range(len(para) - 1):  # 遍历每个句子
                # 生成一条句子对样本及标签
                sen, next_sen, is_next = self.get_next_sentence_sample(
                    para[i], para[i + 1], paragraphs
                )
                logging.debug(f"## 当前句子文本：{sen}")
                logging.debug(f"## 下一句文本：{next_sen}")
                logging.debug(f"## 句子对标签：{is_next}")
                # 下一句为空或者只有一个字符，舍弃
                if len(next_sen) < 2:
                    logging.warning(
                        f"句子 '{sen}' 的下一句 '{next_sen}' 为空，应舍弃，此时NSP标签为：{is_next}"
                    )
                    continue

                # 分词、转换为索引序列
                id_seq1 = [self.vocab[token] for token in self.tokenizer(sen)]
                id_seq2 = [self.vocab[token] for token in self.tokenizer(next_sen)]
                # 拼接两个句子的索引序列，并加上[CLS]、[SEP] token
                id_seq = [self.CLS_IDX] + id_seq1 + [self.SEP_IDX] + id_seq2

                # BERT模型最大支持512个token的序列，若超过，则截断
                if len(id_seq) > self.max_position_embeddings - 1:
                    id_seq = id_seq[: self.max_position_embeddings - 1]
                id_seq += [self.SEP_IDX]
                assert len(id_seq) <= self.max_position_embeddings

                # 创建token_type_id序列，用于表示token所在序列
                seg1 = [0] * (len(id_seq1) + 2)  # 起始[CLS]和中间的[SEP]两个token属于第一个序列
                seg2 = [1] * (len(id_seq) - len(seg1))  # 末尾的[SEP]token则属于第二个序列
                seg = seg1 + seg2
                assert len(seg) == len(id_seq)

                logging.debug(
                    f"## Mask之前tokens：{[self.vocab.itos[id] for id in id_seq]}"
                )
                logging.debug(f"## Mask之前token ids：{id_seq}")
                logging.debug(f"## segment ids：{seg}，序列长度：{len(seg)}")

                # 对token索引序列进行mask操作，生成Masked序列样本及标签
                mlm_id_seq, mlm_label = self.get_masked_sample(id_seq)
                logging.debug(
                    f"## Mask之后tokens：{[self.vocab.itos[id] for id in mlm_id_seq]}"
                )
                logging.debug(f"## Mask之后token ids：{mlm_id_seq}")
                logging.debug(f"## Mask之后labels：{mlm_label}")
                logging.debug("=" * 20)

                id_seq = torch.tensor(mlm_id_seq, dtype=torch.long)
                seg = torch.tensor(seg, dtype=torch.long)
                mlm_label = torch.tensor(mlm_label, dtype=torch.long)
                nsp_label = torch.tensor(int(is_next), dtype=torch.long)
                max_len = max(max_len, id_seq.size(0))
                data.append([id_seq, seg, mlm_label, nsp_label])

        return {"data": data, "max_len": max_len}

    def generate_batch(self, data_batch):
        """
        对每个批次中的样本进行处理的函数，将作为一个参数传入DataLoader的构造函数
        :param data_batch: 一个批次的数据
        """
        b_id_seqs, b_segs, b_mlm_labels, b_nsp_labels = [], [], [], []

        # 遍历一个批次内的样本，取出索引序列、token_type_id序列和MLM、NSP任务的样本标签
        for id_seq, seg, mlm_label, nsp_label in data_batch:
            b_id_seqs.append(id_seq)
            b_segs.append(seg)
            b_mlm_labels.append(mlm_label)
            b_nsp_labels.append(nsp_label)

        # 填充
        # #[max_sen_len, batch_size]
        b_id_seqs = pad_sequence(
            b_id_seqs,
            padding_value=self.PAD_IDX,
            max_len=self.max_sen_len,
            batch_first=False,
        )

        b_segs = pad_sequence(
            b_segs,
            padding_value=self.PAD_IDX,
            max_len=self.max_sen_len,
            batch_first=False,
        )

        b_mlm_labels = pad_sequence(
            b_mlm_labels,
            padding_value=self.PAD_IDX,
            max_len=self.max_sen_len,
            batch_first=False,
        )

        # 生成Padding mask
        # #[batch_size, max_sen_len]
        b_mask = (b_id_seqs == self.PAD_IDX).transpose(0, 1)

        # #[batch_size, ]
        b_nsp_labels = torch.tensor(b_nsp_labels, dtype=torch.long)

        return b_id_seqs, b_segs, b_mask, b_mlm_labels, b_nsp_labels

    def data_loader(
        self,
        train_filepath=None,
        val_filepath=None,
        test_filepath=None,
        only_test=False,
    ):
        """
        创建DataLoader
        :param only_test: 是否只返回测试集
        """

        test_data = self.data_process(filepath=test_filepath)["data"]
        test_loader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=False,  # 测试集不打乱
            collate_fn=self.generate_batch,
        )

        if only_test:
            logging.info(f"## 成功返回测试集，包含样本{len(test_loader.dataset)}个")
            return test_loader

        tmp_data = self.data_process(filepath=train_filepath)
        train_data, max_len = tmp_data["data"], tmp_data["max_len"]
        if self.max_sen_len == "same":
            self.max_sen_len = max_len

        train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=self.is_sample_shuffle,
            collate_fn=self.generate_batch,
        )

        val_data = self.data_process(filepath=val_filepath)["data"]
        val_loader = DataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=False,  # 验证集不打乱
            collate_fn=self.generate_batch,
        )
        logging.info(
            f"## 成功返回训练集样本{len(train_loader.dataset)}个，验证集样本{len(val_loader.dataset)}个，"
            f"测试集样本{len(test_loader.dataset)}个"
        )

        return train_loader, val_loader, test_loader

    def get_inference_samples(self, sentences=None, masked=False):
        """
        制作推理阶段输入模型的样本
        :param sentences: 列表，每个元素表示一个文本段落
        :param masked: 传入的句子是否已被mask
        """

        # sentences可能是单个文本段落（字符串），需要转换为列表
        if not isinstance(sentences, list):
            sentences = [sentences]

        mask_token = self.vocab.itos[self.MASK_IDX]  # [MASK]
        b_id_seq = []  # 保存所有样本句子的索引序列
        b_mask_pos = []  # 保存所有样本句子中的mask_token位置

        for sentence in sentences:
            # 分词，注意：推理阶段，只需把段落中所有token分开即可，不用考虑上下句关系
            token_seq = self.tokenizer(sentence)

            # 传入的句子没有被mask，则执行mask
            if not masked:
                # 候选的mask位置
                candidate_mask_positions = [pos for pos in range(len(token_seq))]
                random.shuffle(candidate_mask_positions)  # 打乱，以实现随机mask
                # 和训练时设置的mask比例15%保持一致
                num_mask_tokens = max(1, round(len(token_seq) * self.masked_rate))
                # 执行mask
                for pos in candidate_mask_positions[:num_mask_tokens]:
                    token_seq[pos] = mask_token

            # 转换为索引序列
            id_seq = [self.vocab[token] for token in token_seq]
            # 加上[CLS]和[SEP] tokens
            id_seq = [self.CLS_IDX] + id_seq + [self.SEP_IDX]

            # 得到被mask的token位置（包含[CLS]、[SEP] token在内的序列内位置）
            b_mask_pos.append(self.get_mask_pos(id_seq))

            b_id_seq.append(torch.tensor(id_seq, dtype=torch.long))

        # 填充，按一个批次内的最长序列长度填充
        b_id_seq = pad_sequence(
            b_id_seq,
            padding_value=self.PAD_IDX,
            max_len=None,
            batch_first=False,
        )

        b_mask = (b_id_seq == self.PAD_IDX).transpose(0, 1)

        return b_id_seq, b_mask_pos, b_mask

    def get_mask_pos(self, token_ids):
        """返回token_ids中[MASK] token所在的位置"""

        mask_positions = []
        for pos, id in enumerate(token_ids):
            if id == self.MASK_IDX:
                mask_positions.append(pos)
        return mask_positions
