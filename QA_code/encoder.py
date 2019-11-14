import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from args import get_train_args
#
# config = get_train_args()

D = 64  # config.connector_dim. Dimension of connectors of each layer. hidden_size
Nh = 1  # config.num_heads. Dimension of connectors of each layer
# Dword = config.glove_dim  # get_setup_args
# Dchar = config.char_dim  # get_setup_args
batch_size = 64  # config.batch_size  # add_train_test_args
dropout = 0.2  # get_train_args
dropout_char = dropout / 2.0  # config.dropout_char

Lc = 400  # config.para_limit
Lq = 50  # config.ques_limit


def mask_logits(inputs, mask):
    mask = mask.type(torch.float32)
    return inputs + (-1e30) * (1 - mask)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                        padding=k // 2, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.mem_conv = Initialized_Conv1d(D, D * 2, kernel_size=1, relu=False, bias=False)
        self.query_conv = Initialized_Conv1d(D, D, kernel_size=1, relu=False, bias=False)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, queries, mask):
        memory = queries

        memory = self.mem_conv(memory)
        query = self.query_conv(queries)
        memory = memory.transpose(1, 2)
        query = query.transpose(1, 2)
        Q = self.split_last_dim(query, Nh)
        K, V = [self.split_last_dim(tensor, Nh) for tensor in torch.split(memory, D, dim=2)]

        key_depth_per_head = D // Nh
        Q *= key_depth_per_head ** -0.5
        x = self.dot_product_attention(Q, K, V, mask=mask)
        return self.combine_last_two_dim(x.permute(0, 2, 1, 3)).transpose(1, 2)

    def dot_product_attention(self, q, k, v, bias=False, mask=None):
        """dot-product attention.
        Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        is_training: a bool of training
        scope: an optional string
        Returns:
        A Tensor.
        """
        logits = torch.matmul(q, k.permute(0, 1, 3, 2))
        if bias:
            logits += self.bias
        if mask is not None:
            shapes = [x if x != None else -1 for x in list(logits.size())]
            mask = mask.view(shapes[0], 1, 1, shapes[-1])
            logits = mask_logits(logits, mask)
        weights = F.softmax(logits, dim=-1)
        # dropping out the attention links for each of the heads
        weights = F.dropout(weights, p=dropout, training=self.training)
        return torch.matmul(weights, v)

    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [..., m]
        n: an integer.
        Returns:
        a Tensor with shape [..., n, m/n]
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)

    def combine_last_two_dim(self, x):
        """Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret


class Initialized_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, relu=False, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.out = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups,
                             bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu == True:
            return F.relu(self.out(x))
        else:
            return self.out(x)


def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    x = x.transpose(1, 2)
    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return (x + signal.cuda()).transpose(1, 2)


def get_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal


class EncoderBlock(nn.Module):
    def __init__(self, conv_num: int, ch_num: int, k: int):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(ch_num, ch_num, k) for _ in range(conv_num)])
        self.self_att = SelfAttention()
        self.FFN_1 = Initialized_Conv1d(ch_num, ch_num, relu=True, bias=True)
        self.FFN_2 = Initialized_Conv1d(ch_num, ch_num, bias=True)
        self.norm_C = nn.ModuleList([nn.LayerNorm(D) for _ in range(conv_num)])
        self.norm_1 = nn.LayerNorm(D)
        self.norm_2 = nn.LayerNorm(D)
        self.conv_num = conv_num

    def forward(self, x, mask, l, blks):
        total_layers = (self.conv_num + 1) * blks
        out = PosEncoder(x)
        # print("out: ", out.size())
        for i, conv in enumerate(self.convs):
            res = out
            # print("transpose: ", out.size(), out.transpose(1, 2).size())
            out = self.norm_C[i](out.transpose(1, 2)).transpose(1, 2)
            if (i) % 2 == 0:
                out = F.dropout(out, p=dropout, training=self.training)
            out = conv(out)
            out = self.layer_dropout(out, res, dropout * float(l) / total_layers)
            l += 1
        res = out
        out = self.norm_1(out.transpose(1, 2)).transpose(1, 2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.self_att(out, mask)
        out = self.layer_dropout(out, res, dropout * float(l) / total_layers)
        l += 1
        res = out

        out = self.norm_2(out.transpose(1, 2)).transpose(1, 2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.FFN_1(out)
        out = self.FFN_2(out)
        out = self.layer_dropout(out, res, dropout * float(l) / total_layers)
        return out

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual
