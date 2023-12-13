import torch.nn as nn
from ..BERT.bert import BertModel


class BertForSenCls(nn.Module):
    """基于BERT的文本分类模型"""

    def __init__(self, config, pretrained_model_dir=None):
        """
        :param pretrained_model_dir: 预训练BERT模型文件所在目录
        """
        super().__init__()
        # 预训练模型文件不为空，则从该文件创建BERT模型，否则创建随机初始化权重的新BERT模型
        if pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(config, pretrained_model_dir)
        else:
            self.bert = BertModel(config)

        self.num_labels = config.num_labels  # 分类类别数
        # 在BERT之上添加的分类层
        self.classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, self.num_labels),
        )

    def forward(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
    ):
        """
        :param input_ids: `#[src_len, batch_size]`
        :param position_ids: `#[1, src_len]`
        :param token_type_ids: `#[src_len, batch_size]`，句子分类任务中，输入的token属于同一序列，所以该值置为None
        :param attention_mask: Padding mask `#[batch_size, src_len]`
        :param labels: 句子的真实标签，`#[batch_size,]`
        """
        # 取[CLS] token对应的embedding（或者所有token embedding的平均值）作为整个序列语义的表示
        # #[batch_size, hidden_size]
        pooled_out, _ = self.bert(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        logits = self.classifier(pooled_out)  # #[batch_size, num_label]

        # 若传入了真实标签，则计算loss
        if labels is not None:
            loss_fc = nn.CrossEntropyLoss()  # 交叉熵损失
            loss = loss_fc(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits
