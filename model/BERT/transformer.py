import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_


class MultiheadAttention(nn.Module):
    """
    多头注意力机制
    """

    def __init__(self, embed_dim=512, num_heads=8, dropout=0.0, bias=True):
        """
        :param embed_dim:   词嵌入维度，即参数d_model
        :param num_heads:   多头注意力机制中头的数量，即参数nhead
        :param bias:        线性变换时，是否使用偏置
        """
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # head_dim是指单头注意力中变换矩阵的列数，也即q，k，v向量的维度
        self.kdim = self.head_dim
        self.vdim = self.head_dim
        self.dropout = dropout

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), "embed_dim除以num_heads必须为整数"
        # 原论文中的 d_k = d_v = d_model/nhead 限制条件

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        # 变换矩阵W_q，embed_dim = num_heads * kdim，kdim=qdim
        # 第二个维度之所以是embed_dim，因为这里同时初始化了num_heads个W_q，也就是num_heads个头，然后横向拼接
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        # W_k，embed_dim = num_heads * kdim
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        # W_v，embed_dim = num_heads * vdim

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        # 将多头注意力计算结果（横向拼接）再执行一次线性转换后输出

        self._reset_parameters()

    def forward(
        self,
        query,
        key,
        value,
        attn_mask=None,
        key_padding_mask=None,
        training=True,
        is_print_shape=False,
    ):
        """
        Encoder中，query、key、value都是源序列src seq\\
        Decoder中，query、key、value都是目标序列tgt seq\\
        Encoder和Decoder交互时，key、value指的是Encoder memory，query指的是tgt seq\\
        :param query:   # [tgt_len, batch_size, embed_dim]
        :param key:     # [src_len, batch_size, embed_dim]
        :param value:   # [src_len, batch_size, embed_dim]
        :param attn_mask: 注意力掩码矩阵 # [tgt_len, src_len] 或 [batch_size * num_heads, tgt_len, src_len]
        一般只在Decoder的Training中使用，因为训练时并行传入所有tgt tokens，需要掩盖当前时刻之后的tokens信息
        :param key_padding_mask: 对Padding tokens进行掩码 # [batch_size, src_len]
        :return:
        attn_output: 多头注意力计算结果 # [tgt_len, batch_size, embed_dim]
        attn_output_weights: 多头注意力平均权重矩阵 # [batch_size, tgt_len, src_len]
        """

        # 1.计算Q、K、V
        # 注意：query、key、value是没有经过线性变换前的序列，例如在Encoder中都是源序列src seq
        Q = self.q_proj(query)
        # [tgt_len, batch_size, embed_dim] x [embed_dim, num_heads * kdim] = [tgt_len, batch_size, num_heads * kdim]
        K = self.k_proj(key)
        # [src_len, batch_size, embed_dim] x [embed_dim, num_heads * kdim] = [src_len, batch_size, num_heads * kdim]
        V = self.v_proj(value)
        # [src_len, batch_size, embed_dim] x [embed_dim, num_heads * vdim] = [src_len, batch_size, num_heads * vdim]

        if is_print_shape:
            print("=" * 80)
            print("开始计算多头注意力：")
            print(
                f"\t 多头数num_heads = {self.num_heads}，d_model={query.size(-1)}，d_k = d_v = d_model/num_heads={query.size(-1) // self.num_heads}"
            )
            print(f"\t query的shape([tgt_len, batch_size, embed_dim])：{query.shape}")
            print(
                f"\t W_q的shape([embed_dim, num_heads * kdim])：{self.q_proj.weight.shape}"
            )
            print(f"\t Q的shape([tgt_len, batch_size, num_heads * kdim])：{Q.shape}")
            print("\t" + "-" * 70)

            print(f"\t key的shape([src_len, batch_size, embed_dim])：{key.shape}")
            print(
                f"\t W_k的shape([embed_dim, num_heads * kdim])：{self.k_proj.weight.shape}"
            )
            print(f"\t K的shape([src_len, batch_size, num_heads * kdim])：{K.shape}")
            print("\t" + "-" * 70)

            print(f"\t value的shape([src_len, batch_size, embed_dim])：{value.shape}")
            print(
                f"\t W_v的shape([embed_dim, num_heads * vdim])：{self.v_proj.weight.shape}"
            )
            print(f"\t V的shape([src_len, batch_size, num_heads * vdim])：{V.shape}")
            print("\t" + "-" * 70)
            print(
                "\t ***** 注意，这里的W_q、W_k、W_v是多头注意力变换矩阵拼接的，因此，Q、K、V也是多个q、k、v向量拼接的结果 *****"
            )

        # 2.缩放，并判断attn_mask维度是否正确
        scaling = float(self.head_dim) ** -0.5  # 缩放系数
        Q = Q * scaling
        # [query_len, batch_size, num_heads * kdim]，其中query_len就是tgt_len

        src_len = key.size(0)
        tgt_len, bsz, _ = query.size()  # [tgt_len, batch_size, embed_dim]

        if attn_mask is not None:
            # [tgt_len, src_len] 或 [batch_size * num_heads, tgt_len, src_len]
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)  # [1, tgt_len, src_len]
                if list(attn_mask.size()) != [1, tgt_len, src_len]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * self.num_heads,
                    tgt_len,
                    src_len,
                ]:
                    raise RuntimeError("The size of the 3D attn_mask is not correct.")
            # 此时atten_mask的维度变成了3

        # 3.计算注意力得分
        # 这里需要进行一下变形，以便后续执行bmm运算
        Q = (
            Q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.kdim)
            .transpose(0, 1)
        )
        # [batch_size * num_heads, tgt_len, kdim]
        K = (
            K.contiguous()
            .view(src_len, bsz * self.num_heads, self.kdim)
            .transpose(0, 1)
        )
        # [batch_size * num_heads, src_len, kdim]
        V = (
            V.contiguous()
            .view(src_len, bsz * self.num_heads, self.vdim)
            .transpose(0, 1)
        )
        # [batch_size * num_heads, src_len, vdim]

        attn_weights = torch.bmm(Q, K.transpose(1, 2))  # bmm用于三维tensor的矩阵运算
        # [batch_size * num_heads, tgt_len, kdim] x [batch_size * num_heads, kdim, src_len]
        # -> [batch_size * num_heads, tgt_len, src_len] 这是num_heads个Q、K相乘后的注意力矩阵

        # 4.进行掩码操作
        # Attention mask
        if attn_mask is not None:
            attn_weights += attn_mask
            # [batch_size * num_heads, tgt_len, src_len]

        # Padding mask（列Padding mask）
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            # [batch_size, num_heads, tgt_len, src_len]
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
            # key_padding_mask扩展维度 [batch_size, src_len] -> [batch_size, 1, 1, src_len]
            # 然后对attn_output_weights进行掩码，其中masked_fill会将值为True（或非0值）的列mask掉
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
            # [batch_size * num_heads, tgt_len, src_len]

        # 5.计算多头注意力输出
        # 计算注意力权重
        attn_weights = F.softmax(attn_weights, dim=-1)
        # [batch_size * num_heads, tgt_len, src_len]
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=training)

        # 计算MultiheadAttention(Q, K, V)
        attn_output = torch.bmm(attn_weights, V)
        # [batch_size * num_heads, tgt_len, src_len] x [batch_size * num_heads, src_len, vdim]
        # -> [batch_size * num_heads, tgt_len, vdim]

        # 最后执行一次线性变换，输出
        attn_output = (
            attn_output.transpose(0, 1)
            .contiguous()
            .view(tgt_len, bsz, self.num_heads * self.vdim)
        )
        # [tgt_len, batch_size, num_heads * vdim]
        Z = self.out_proj(attn_output)
        # [tgt_len, batch_size, embed_dim]

        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        # [batch_size, num_heads, tgt_len, src_len]

        if is_print_shape:
            print(
                f"\t 多头注意力计算结束后的形状（横向拼接）为([tgt_len, batch_size, num_heads * vdim])：{attn_output.shape}"
            )
            print(
                f"\t 对多头注意力计算结果进行线性变换的权重W_o形状为([num_heads * vdim, embed_dim])：{self.out_proj.weight.shape}"
            )
            print(f"\t 多头注意力计算结果线性变换后的形状为([tgt_len, batch_size, embed_dim])：{Z.shape}")

        return (
            Z,
            attn_weights.sum(dim=1) / self.num_heads,  # 返回多头注意力权重矩阵的平均值
        )

    def _reset_parameters(self):
        """
        初始化参数
        """
        for param in self.parameters():
            if param.dim() > 1:
                xavier_uniform_(param)


def _get_clones(module, N):
    """
    对module进行N次拷贝
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderLayer(nn.Module):
    """
    单个编码层
    """

    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        """
        :param d_model:         模型中向量维度，即词嵌入维度
        :param nhead:           多头注意力中的多头数量
        :param dim_feedforward: 全连接层的输出维度
        :param dropout:         丢弃率
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout
        )

        # 多头注意力输出后的Add&Norm
        self.dropout1 = nn.Dropout(dropout)
        # 注意：LayerNorm是沿着feature维度归一化；BatchNorm是沿着batch维度归一化
        self.norm1 = nn.LayerNorm(d_model)

        # Feed Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        :param src: # [src_len, batch_size, embed_dim]
        :param src_mask: None，Encoder中不需要Attention Mask
        :param src_key_padding_mask: # [batch_size, src_len]
        :return: # [src_len, batch_size, embed_dim] <==> [src_len, batch_size, num_heads * kdim]
        """
        # 计算多头注意力
        src1 = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        # [src_len, batch_size, embed_dim]，其中embed_dim = num_heads * kdim

        # 残差连接和LayerNorm
        src = src + self.dropout1(src1)
        src = self.norm1(src)

        # Feed Forward
        src1 = self.activation(self.linear1(src))
        # [src_len, batch_size, dim_feedforward]
        src1 = self.linear2(self.dropout2(src1))
        # [src_len, batch_size, embed_dim]

        # 残差连接和LayerNorm
        src = src + self.dropout3(src1)
        src = self.norm2(src)

        return src


class Encoder(nn.Module):
    """
    编码器，由多个编码层堆叠而成
    """

    def __init__(self, encoder_layer, num_layers=6, norm=None):
        """
        :param encoder_layer:   单个编码层
        :param num_layers:      编码层数
        :param norm:            归一化层
        """
        super(Encoder, self).__init__()
        # 拷贝多个编码层，得到编码层列表
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        :param src: # [src_len, batch_size, embed_dim]
        :param src_mask: None，Encoder中不需要Attention Mask
        :param src_key_padding_mask: # [batch_size, src_len]
        :return: # [src_len, batch_size, embed_dim] <==> [src_len, batch_size, num_heads * kdim]
        """
        output = src
        # 遍历每一个编码层，执行forward，并传递给下一层
        for layer in self.layers:
            output = layer(
                output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
            )
        # 对最后一层输出执行Norm操作
        if self.norm is not None:
            output = self.norm(output)

        return output


class DecoderLayer(nn.Module):
    """
    单个解码层
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        """
        :param d_model:         模型中向量维度，即词嵌入维度
        :param nhead:           多头注意力中的多头数量
        :param dim_feedforward: 全连接层的输出维度
        :param dropout:         丢弃率
        """
        super(DecoderLayer, self).__init__()
        # Masked多头注意力，对解码层输入序列进行计算
        self.self_attn = MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout
        )
        # 编码器输出（memory）和解码层交互的多头注意力
        self.multihead_attn = MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Feed Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        :param tgt: 解码层输入序列 # [tgt_len, batch_size, embed_dim]
        :param memory: 编码器输出（memory） # [src_len, batch_size, embed_dim]
        :param tgt_mask: 解码层多头注意力掩码 # [tgt_len, tgt_len]
        :param memory_mask: 编码器-解码器交互多头注意力掩码，一般为None
        :param tgt_key_padding_mask: 解码器输入序列的Padding情况 # [batch_size, tgt_len]
        :param memory_key_padding_mask: 编码器输入序列的Padding情况 # [batch_size, src_len]
        :return: # [tgt_len, batch_size, embed_dim] <==> [tgt_len, batch_size, num_heads * kdim]
        """
        # Masked多头注意力计算
        tgt1 = self.self_attn(
            tgt,
            tgt,
            tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]

        # 残差连接&LayerNorm
        tgt = tgt + self.dropout1(tgt1)
        tgt = self.norm1(tgt)

        # 编码器-解码器交互多头注意力计算
        tgt1 = self.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]

        # 残差连接&LayerNorm
        tgt = tgt + self.dropout2(tgt1)
        tgt = self.norm2(tgt)

        # Feed Forward
        tgt1 = self.activation(self.linear1(tgt))
        # [tgt_len, batch_size, dim_feedforward]
        tgt1 = self.linear2(self.dropout4(tgt1))
        # [tgt_len, batch_size, embed_dim]

        # 残差连接&LayerNorm
        tgt = tgt + self.dropout3(tgt1)
        tgt = self.norm3(tgt)

        return tgt


class Decoder(nn.Module):
    """
    解码器，由多个解码层堆叠而成
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        """
        :param decoder_layer:   单个解码层
        :param num_layers:      解码层数
        :param norm:            归一化层
        """
        super(Decoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        :param tgt: 解码层输入序列 # [tgt_len, batch_size, embed_dim]
        :param memory: 编码器输出（memory） # [src_len, batch_size, embed_dim]
        :param tgt_mask: 解码层多头注意力掩码 # [tgt_len, tgt_len]
        :param memory_mask: 编码器-解码器交互多头注意力掩码，一般为None
        :param tgt_key_padding_mask: 解码器输入序列的Padding情况 # [batch_size, tgt_len]
        :param memory_key_padding_mask: 编码器输入序列的Padding情况 # [batch_size, src_len]
        :return: # [tgt_len, batch_size, embed_dim] <==> [tgt_len, batch_size, num_heads * kdim]
        """
        output = tgt  # [tgt_len,batch_size, embed_dim]
        # 遍历每一个解码层，执行forward，并传递给下一层
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        # 对最后一层输出执行Norm操作
        if self.norm is not None:
            output = self.norm(output)

        return output


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        """
        :param d_model:             模型中向量维度，即词嵌入维度
        :param nhead:               多头注意力中的多头数量
        :param num_encoder_layers:  EncoderLayer堆叠的数量
        :param num_decoder_layers:  DecoderLayer堆叠的数量
        :param dim_feedforward:     全连接层的输出维度
        :param dropout:             丢弃率
        """
        super(Transformer, self).__init__()
        # Encoder
        encoder_layer = EncoderLayer(d_model, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = Encoder(encoder_layer, num_encoder_layers, encoder_norm)

        # Decoder
        decoder_layer = DecoderLayer(d_model, nhead, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = Decoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        memory_mask=None,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        :param src: # [src_len, batch_size, embed_dim]
        :param tgt: # [tgt_len, batch_size, embed_dim]
        :param src_mask:    None
        :param tgt_mask:    # [tgt_len, tgt_len]
        :param memory_mask: None
        :param src_key_padding_mask:    # [batch_size, src_len]
        :param tgt_key_padding_mask:    # [batch_size, tgt_len]
        :param memory_key_padding_mask: # [batch_size, src_len]
        :return: [tgt_len, batch_size, embed_dim] <==> [tgt_len, batch_size, num_heads * kdim]
        """

        # Encoding，生成memory
        memory = self.encoder(
            src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )

        # Decoding
        output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return output

    def _reset_parameters(self):
        """
        初始化参数
        """
        for param in self.parameters():
            if param.dim() > 1:
                xavier_uniform_(param)

    def generate_attn_mask(self, sz):
        """
        生成注意力掩码矩阵
        """
        mask = torch.tril(torch.ones(sz, sz))  # tril取矩阵下三角（包括对角线）
        mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(
            mask == 1, float(0.0)
        )
        return mask
