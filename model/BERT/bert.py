import os
import logging
import torch
import torch.nn as nn
from copy import deepcopy
from torch.nn.init import normal_
from .config import BertConfig
from .embedding import BertEmbedding
from .transformer import MultiheadAttention


def get_activation(activation_string):
    """将字符串转换为激活函数"""
    activation = activation_string.lower()
    if activation == "linear":
        return None
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "tanh":
        return nn.Tanh()
    else:
        raise ValueError("Unsupported activation: %s" % activation)


class BertSelfAttention(nn.Module):
    """多头自注意力模块"""

    def __init__(self, config: BertConfig):
        super(BertSelfAttention, self).__init__()
        # 使用Pytorch中的多头注意力模块
        if "use_torch_multi_head" in config.__dict__ and config.use_torch_multi_head:
            MultiHeadAttention = nn.MultiheadAttention
        # 使用自实现的多头注意力模块
        else:
            MultiHeadAttention = MultiheadAttention

        self.multi_head_attention = MultiHeadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
        )

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        return self.multi_head_attention(
            query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )


class BertSelfOutput(nn.Module):
    """自注意力模块后的残差连接和标准化"""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """
        Args:
            hidden_states: 多头自注意力模块输出 `#[src_len, batch_size, hidden_size]`
            input_tensor: 多头自注意力模块输入 `#[src_len, batch_size, hidden_size]`
        """
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(input_tensor + hidden_states)
        return hidden_states


class BertAttention(nn.Module):
    """完整的自注意力模块"""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.self_attention = BertSelfAttention(config)
        self.self_output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: 自注意力模块输入 `#[src_len, batch_size, hidden_size]`
            attention_mask: Padding mask，需要被mask的token用`True`表示，否者用`False`表示 `#[batch_size, src_len]`
        """
        # self_attn返回编码结果和注意力权重矩阵
        attn_outputs = self.self_attention(
            hidden_states,
            hidden_states,
            hidden_states,
            attn_mask=None,
            key_padding_mask=attention_mask,  # 注意：attention_mask是填充掩码，而不是注意力掩码
        )
        # attn_outputs[0]: #[src_len, batch_size, hidden_size]
        output = self.self_output(attn_outputs[0], hidden_states)
        return output


class BertIntermediate(nn.Module):
    """
    自注意力模块后的线性层——即Transformer FFN中的第一个线性层
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        # 线性层
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 激活函数
        if isinstance(config.hidden_act, str):
            self.inter_activation = get_activation(config.hidden_act)
        else:
            self.inter_activation = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        if self.inter_activation is None:
            hidden_states = hidden_states
        else:
            hidden_states = self.inter_activation(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """
    第二个线性层及残差连接、标准化等模块——即Transformer FFN中的第二个线性层
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """
        Args:
            hidden_states: 第一个线性层输出 `#[src_len, batch_size, intermediate_size]`
            input_tensor: 第一个线性层输入 `#[src_len, batch_size, hidden_size]`
        Return:
            `#[src_len, batch_size, hidden_size]`
        """
        # #[src_len, batch_size, hidden_size]
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(input_tensor + hidden_states)
        return hidden_states


class BertLayer(nn.Module):
    """单个Encoder Layer"""

    def __init__(self, config):
        super().__init__()
        self.bert_attention = BertAttention(config)
        self.bert_intermediate = BertIntermediate(config)
        self.bert_output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: `#[src_len, batch_size, hidden_size]`
            attention_mask: padding mask `#[batch_size, src_len]`
        Return:
            `#[src_len, batch_size, hidden_size]`
        """
        # #[src_len, batch_size, hidden_size]
        attn_output = self.bert_attention(hidden_states, attention_mask)
        # #[src_len, batch_size, intermediate_size]
        inter_output = self.bert_intermediate(attn_output)
        # #[src_len, batch_size, hidden_size]
        output = self.bert_output(inter_output, attn_output)

        return output


class BertEncoder(nn.Module):
    """
    Encoder——由多个Encoder Layer堆叠而成
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        # 创建num_hidden_layers个Encoder Layer
        self.bert_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states, attention_mask=None):
        all_encoder_layers = []  # 保存所有Encoder Layer的输出
        output = hidden_states
        for _, layer in enumerate(self.bert_layers):
            output = layer(output, attention_mask)
            all_encoder_layers.append(output)

        return all_encoder_layers


class BertPooler(nn.Module):
    """
    用于获取整个句子的语义信息
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        if "pooler_type" not in self.config.__dict__:
            raise ValueError(
                "pooler_type must be in ['first_token_transform', 'all_token_average']"
                "请在配置文件config.json中添加一个pooler_type参数"
            )
        # 取第一个token，即[cls] token embedding
        if self.config.pooler_type == "first_token_transform":
            # #[batch_size, hidden_size]
            token_tensor = hidden_states[0, :].reshape(-1, self.config.hidden_size)
        # 取所有token embedding的平均值
        elif self.config.pooler_type == "all_token_average":
            token_tensor = torch.mean(hidden_states, dim=0)

        # #[batch_size, hidden_size]
        output = self.dense(token_tensor)
        output = self.activation(output)

        return output


def format_params_for_torch(loaded_params_names, loaded_params):
    """
    将加载的预训练模型参数格式化为符合torch(1.12.0)框架中MultiHeadAttention的形式——Q、K、V weight/bias放在一个tnesor中
    """
    qkv_weight_names = ["query.weight", "key.weight", "value.weight"]
    qkv_bias_names = ["query.bias", "key.bias", "value.bias"]
    qkv_weight, qkv_bias = [], []
    torch_params = []
    for i in range(len(loaded_params_names)):
        param_name_in_pretrained = loaded_params_names[i]
        param_name = ".".join(param_name_in_pretrained.split(".")[-2:])
        if param_name in qkv_weight_names:
            qkv_weight.append(loaded_params[param_name_in_pretrained])
        elif param_name in qkv_bias_names:
            qkv_bias.append(loaded_params[param_name_in_pretrained])
        else:
            torch_params.append(loaded_params[param_name_in_pretrained])
        if len(qkv_weight) == 3:
            torch_params.append(torch.cat(qkv_weight, dim=0))
            qkv_weight = []
        if len(qkv_bias) == 3:
            torch_params.append(torch.cat(qkv_bias, dim=0))
            qkv_bias = []

    return torch_params


def load_512_position(init_embedding, loaded_embedding):
    """
    预训练的BERT模型仅支持最大512个`position_ids`，而自定义的模型配置中
    `max_positional_embeddings`可能大于512，所以加载时用预训练模型的`positional embedding`矩阵替换随机初始化的`positional embedding`矩阵前512行\ 
    Args:
        init_embedding: 随机初始化的positional embedding矩阵，可能大于512行
        loaded_embedding: 加载的预训练模型的positional embedding矩阵，等于512行
    """
    logging.info(f"模型配置 max_positional_embeddings > 512")
    init_embedding[:512, :] = loaded_embedding[:512, :]
    return init_embedding


class BertModel(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.bert_embedding = BertEmbedding(config)
        self.bert_encoder = BertEncoder(config)
        self.bert_pooler = BertPooler(config)
        self._reset_parameters()

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
    ):
        """
        Args:
            input_ids: `#[src_len, batch_size]`
            position_ids: `#[1, src_len]`
            token_type_ids: `#[src_len, batch_size]`
            attention_mask: `#[batch_size, src_len]`
        Return:
            `#[src_len, batch_size, hidden_size]`
        """
        input_embed = self.bert_embedding(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        # encoder_outputs包含num_hidden_layers个encoder layers的输出
        encoder_outputs = self.bert_encoder(
            hidden_states=input_embed, attention_mask=attention_mask
        )
        # 取最后一层encoder layer的输出结果传入pooler获取整个句子的语义信息
        sequence_output = encoder_outputs[-1]  # #[src_len, batch_size, hidden_size]
        pooled_output = self.bert_pooler(sequence_output)  # #[batch_size, hidden_size]

        return pooled_output, encoder_outputs

    def _reset_parameters(self):
        """初始化参数"""
        for param in self.parameters():
            if param.dim() > 1:
                normal_(param, mean=0.0, std=self.config.initializer_range)

    @classmethod
    def from_pretrained(cls, config: BertConfig, pretrained_model_dir=None):
        """从预训练模型文件创建模型"""
        model = cls(config)  # 创建模型，cls表示类名BertModel
        # 加载预训练模型
        pretrained_model_path = os.path.join(pretrained_model_dir, "pytorch_model.bin")
        if not os.path.exists(pretrained_model_path):
            raise ValueError(
                f"<路径：{pretrained_model_path} 中的模型不存在，请仔细检查！>\n"
                f"中文模型下载地址：https://huggingface.co/bert-base-chinese/tree/main\n"
                f"英文模型下载地址：https://huggingface.co/bert-base-uncased/tree/main\n"
            )
        loaded_params = torch.load(pretrained_model_path)
        loaded_params_names = list(loaded_params.keys())[:-8]
        model_params = deepcopy(model.state_dict())
        model_params_names = list(model_params.keys())[1:]

        if "use_torch_multi_head" in config.__dict__ and config.use_torch_multi_head:
            logging.info(f"## 注意，正在使用torch框架中的MultiHeadAttention实现")

            torch_params = format_params_for_torch(loaded_params_names, loaded_params)
            for i in range(len(model_params_names)):
                logging.debug(
                    f"## 成功赋值参数 {model_params_names[i]} 参数形状为 {torch_params[i].size()}"
                )
                if "position_embedding" in model_params_names[i]:
                    if config.max_position_embeddings > 512:
                        new_embedding = load_512_position(
                            model_params[model_params_names[i]],
                            torch_params[i],
                        )
                        model_params[model_params_names[i]] = new_embedding
                        continue

                model_params[model_params_names[i]] = torch_params[i]
        else:
            logging.info(
                f"## 注意，正在使用本地transformer.py中的MultiheadAttention实现，"
                f"如需使用torch框架中的MultiHeadAttention模块，可设置config.__dict__['use_torch_multi_head'] = True实现"
            )

            for i in range(len(loaded_params_names)):
                logging.debug(
                    f"## 成功将参数 {loaded_params_names[i]} 赋值给 {model_params_names[i]} "
                    f"参数形状为 {loaded_params[loaded_params_names[i]].size()}"
                )
                if "position_embedding" in model_params_names[i]:
                    if config.max_position_embeddings > 512:
                        new_embedding = load_512_position(
                            model_params[model_params_names[i]],
                            loaded_params[loaded_params_names[i]],
                        )
                        model_params[model_params_names[i]] = new_embedding
                        continue
                # 把加载的预训练模型参数值赋给新创建模型
                model_params[model_params_names[i]] = loaded_params[
                    loaded_params_names[i]
                ]

        model.load_state_dict(model_params)
        return model
