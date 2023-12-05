import json
import copy

# import six
import logging


class BertConfig(object):
    """BertModel的配置类"""

    def __init__(
        self,
        vocab_size=21128,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        pad_token_id=0,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
    ):
        """
        Args:
          vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
          hidden_size: Size of the encoder layers and the pooler layer.
          num_hidden_layers: Number of hidden layers in the Transformer encoder.
          num_attention_heads: Number of attention heads for each attention\
            layer in the Transformer encoder.
          intermediate_size: The size of the `intermediate` (i.e., feed-forward)\
            layer in the Transformer encoder.
          hidden_act: The non-linear activation function (function or string)\
            in the encoder and pooler.
          hidden_dropout_prob: The dropout probability for all fully connected\
            layers in the embeddings, encoder, and pooler.
          attention_probs_dropout_prob: The dropout ratio for the attention\
            probabilities.
          max_position_embeddings: The maximum sequence length that this model might\
            ever be used with. Typically set this to something large just in case\
            (e.g., 512 or 1024 or 2048).
          type_vocab_size: The vocabulary size of the `token_type_ids` passed into\
            `BertModel`.
          initializer_range: The stdev of the `truncated_normal_initializer` for\
            initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.pad_token_id = pad_token_id
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_json_file(cls, json_file):
        """从json格式配置文件读取配置信息"""
        with open(json_file, "r") as reader:
            json_obj = reader.read()

        logging.info(f"成功导入BERT配置文件 {json_file}")
        return cls.from_dict(json.loads(json_obj))

    @classmethod
    def from_dict(cls, dict_obj):
        """从Python字典中读取配置信息"""
        config = BertConfig(vocab_size=None)  # 创建Config对象
        for key, value in dict_obj.items():  # 从字典中读取配置信息
            config.__dict__[key] = value

        return config

    def to_json_str(self):
        """把对象转换为json格式字符串"""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_dict(self):
        """把对象转换为字典"""
        out = copy.deepcopy(self.__dict__)
        return out


if __name__ == "__main__":
    import sys, os

    sys.path.append(os.getcwd())

    json_file = "./archive/bert_base_chinese/config.json"
    config = BertConfig.from_json_file(json_file)

    for key, value in config.__dict__.items():
        print(f"{key} = {value}")

    print("=" * 20)
    print(config.to_json_str())