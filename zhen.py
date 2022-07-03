import unittest
from enum import Enum
from typing import List, Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
import alpa.torch.optim as torchoptim
from alpa.torch.trainer import train_torch_module
import alpa
from alpa.device_mesh import get_global_cluster
from alpa import PipeshardParallel


# Copied from timm
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(
            0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return torch.nn.functional.relu
    elif activation == "gelu":
        return torch.nn.functional.gelu

    raise RuntimeError(
        "activation should be relu/gelu, not {}".format(activation))


# Adapted from torch/nn/modules/transformer.py
class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = "relu",
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = False,
                 norm_first: bool = False,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = Attention(d_model, num_heads=nhead, attn_drop=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask,
                                   src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x +
                           self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor],
                  key_padding_mask: Optional[Tensor]) -> Tensor:
        # x = self.self_attn(x, x, x,
        #                    attn_mask=attn_mask,
        #                    key_padding_mask=key_padding_mask,
        #                    need_weights=False)[0]
        # TODO: add support for `attn_mask` / `key_padding_mask` if needed.
        x = self.self_attn(x)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TokenMixer(Enum):
    DOT = 1
    LINEAR = 2
    ATTENTION = 3
    CONVOLUTION = 4


# util for generating a weight and a bias based on a size, and initializing them
def construct_w_b_pair(
        shape: List[int],
        uniform_const: float) -> Tuple[nn.Parameter, nn.Parameter]:
    assert len(shape) == 2
    w = nn.Parameter(
        torch.empty(shape).uniform_(-1 * uniform_const, uniform_const))
    b = nn.Parameter(
        torch.empty([shape[0]]).uniform_(-1 * uniform_const,
                                         uniform_const))  # UniformFillß

    return w, b


# this is a single ZHEN layer. It:#
# - receives an input from the previous layer, or the embedding (first layer)
# - receives the skip connection, which is the input to the previous layer (or nothing, in the case of first ZHEN layer)
# - adds input and skip connection together, and treat it as the new input
# and runs the new input through the different modules in token_mixer_list one by one, and concat them together as the ensemble.
# It outputs the ensemble result and the new input
# see https://bit.ly/3wNuqfz for a visualization.
class ZHENLayer(nn.Module):

    def __init__(
        self,
        layer_index: int,
        emb_dim: int,
        token_mixer_list: List[
            TokenMixer],  # determines this layer's output features
        previous_n_embs:
        int = 369,  # previous layer's output dim, may not be inferrable if token_mixer is different per layer. If 0th layer, this is original_n_embs.
        previous_input_embs:
        int = 369,  # skip connection's num embs. This is previous layer's input num embs.
        output_embs_per_mixer: int = 50,  # each module outputs 50 embeddings
        original_n_embs:
        int = 369,  # whatever overarch gives us for the 0th zhen layer . the rest, is whatever output previous layer is
    ):
        super().__init__()
        print("ZHENLayer params:", previous_n_embs, previous_input_embs, output_embs_per_mixer, original_n_embs)

        self.layer_index = layer_index
        self.emb_dim = emb_dim
        self.token_mixer_list = token_mixer_list
        self.mismatched_skip_and_input_shape = previous_n_embs != previous_input_embs
        if token_mixer_list is not None:
            self.token_mixer_list = token_mixer_list
        # self.sum_for_skip = sum_for_skip
        zhen_n_embs = len(token_mixer_list) * output_embs_per_mixer
        self.n_embs = zhen_n_embs
        if self.layer_index != 0:
            if self.mismatched_skip_and_input_shape:
                self.match_w, self.match_b = construct_w_b_pair(
                    [previous_n_embs, previous_input_embs], 0.0)

        self.layer_norm_w = nn.Parameter(torch.empty(
            [emb_dim]).fill_(0.0))  # ConstantFill
        self.layer_norm_b = nn.Parameter(torch.empty(
            [emb_dim]).fill_(0.0))  # ConstantFill
        for token_mixer in self.token_mixer_list:
            if token_mixer == TokenMixer.DOT:
                self.ffn_w, self.ffn_b = construct_w_b_pair(
                    [
                        512,
                        original_n_embs**2
                        if self.layer_index == 0 else previous_n_embs**2,
                    ],
                    0.03125,
                )
                self.pool_w, self.pool_b = construct_w_b_pair(
                    [
                        output_embs_per_mixer * emb_dim,
                        512,
                    ],
                    0.3125,
                )
            elif token_mixer == TokenMixer.LINEAR:  # n = 50
                self.w_linear, self.b_linear = construct_w_b_pair(
                    [output_embs_per_mixer, previous_n_embs], 0.0)

            elif token_mixer == TokenMixer.ATTENTION:  # n = 50
                self.encoder_layer = TransformerEncoderLayer(d_model=emb_dim,
                                                             nhead=1,
                                                             batch_first=True)

                self.w_attention, self.b_attention = construct_w_b_pair(
                    [output_embs_per_mixer, previous_n_embs], 0.0)

            elif token_mixer == TokenMixer.CONVOLUTION:
                self.conv = nn.Conv2d(1, 1, 5, stride=1, padding=(2, 2))
                self.w_conv, self.b_conv = construct_w_b_pair(
                    [
                        output_embs_per_mixer,
                        original_n_embs
                        if self.layer_index == 0 else previous_n_embs,
                    ],
                    0.0,
                )

    def get_dense_params(self) -> List[nn.Parameter]:
        # do not save because this may turn into FSDP
        return list(self.parameters())

    def forward(
        self,
        skip_connection: Optional[
            torch.Tensor],  # the skip connection, i.e., previous layer's input
        input: torch.Tensor,  # this is previous layer's ensemble output
        # embs_for_skip, # original emb
        # orig_embs_concat, # previous layer's ensemble
    ):
        B = input.shape[0]
        # process orig embs
        # token mixer not None
        if self.layer_index != 0:
            if self.mismatched_skip_and_input_shape:
                skip_connection = torch.nn.functional.linear(skip_connection,
                                                             self.match_w,
                                                             bias=self.match_b)
            input_feature = skip_connection + input
        else:
            # 0th layer, no skip
            input_feature = input

        output = []  # do not call cat N times. Call it once.
        for token_mixer in self.token_mixer_list:
            if token_mixer == TokenMixer.DOT:  # num_dot_emb = 50
                input_feature_t = input_feature.permute(0, 2, 1)
                dot_products = torch.bmm(input_feature_t, input_feature)
                flattened_dot_products = torch.flatten(dot_products,
                                                       start_dim=-2)  # Flatten
                r = torch.addmm(self.ffn_b, flattened_dot_products,
                                self.ffn_w.t())  # FC
                r_act = torch.relu(r)  # Relu
                r_pooled = torch.nn.functional.linear(
                    r_act,
                    self.pool_w,
                    bias=self.pool_b,
                )
                output.append(r_pooled)

            elif token_mixer == TokenMixer.LINEAR:
                linear_emb_list = torch.nn.functional.linear(input_feature,
                                                             self.w_linear,
                                                             bias=self.b_linear)
                flat_linear_emb_list = linear_emb_list.permute(0, 2, 1).reshape(
                    B, -1)  # (B, feature, dim)

                output.append(flat_linear_emb_list)

            elif token_mixer == TokenMixer.ATTENTION:
                compress_list = torch.nn.functional.linear(
                    input_feature, self.w_attention, bias=self.b_attention)
                compress_list_t = compress_list.permute(0, 2, 1)  # (b, 369, 64)
                attention_emb_list = self.encoder_layer(compress_list_t)

                flat_compress_list = attention_emb_list.permute(
                    0, 2, 1).reshape(B, -1)  # (B, feature, dim)
                output.append(flat_compress_list)

            elif token_mixer == TokenMixer.CONVOLUTION:
                reshape_input_feature = input_feature.reshape(
                    B, 1, self.emb_dim, -1)
                r_conv = self.conv(reshape_input_feature)
                reshape_r_conv = r_conv.reshape(B, self.emb_dim, -1)
                compress_list = torch.nn.functional.linear(reshape_r_conv,
                                                           self.w_conv,
                                                           bias=self.b_conv)
                flat_compress_list = compress_list.permute(0, 2, 1).reshape(
                    B, -1)  # (B, feature, dim)
                output.append(flat_compress_list)
            else:
                assert 0, f"unknown module: {token_mixer}"

        output = torch.cat(output, dim=1).view(B, self.emb_dim, -1)
        output_embs = torch.nn.functional.layer_norm(
            output,
            output.size()[2:],
            weight=self.layer_norm_w,
            bias=self.layer_norm_b,
        )
        return output_embs, input_feature


# ZHEN collection is different ZHEN layers
class ZHENCollection(nn.Module):

    def __init__(
        self,
        num_layers: int,
        emb_dim: int,
        token_mixer_list: Union[List[TokenMixer], List[List[TokenMixer]]],
        original_emb_num: int,
        output_emb_per_ensemble_module: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.token_mixer_list = token_mixer_list
        self.layers: nn.ModuleList = nn.ModuleList([])

        assert len(token_mixer_list) > 0
        if type(token_mixer_list[0]) == list:
            # this is a heterogeneous ZHEN
            assert num_layers == len(
                token_mixer_list
            ), "if token_mixer_list is a list of list of modules, ensure num_layers = len(token_mixer_list)"  # noqa
        else:
            # this is a homogeneous ZHEN. Convert it to heterogeneous ZHEN
            # pyre-ignore
            token_mixer_list = [token_mixer_list] * num_layers

        for i in range(num_layers):
            layer = ZHENLayer(
                layer_index=i,
                emb_dim=emb_dim,
                # pyre-ignore[6]
                token_mixer_list=token_mixer_list[i],
                previous_n_embs=(
                    original_emb_num if i == 0
                    # pyre-ignore[6]
                    else len(token_mixer_list[i - 1]) *
                    output_emb_per_ensemble_module),
                previous_input_embs=(
                    original_emb_num if i <= 1
                    # pyre-ignore[6]
                    else len(token_mixer_list[i - 2]) *
                    output_emb_per_ensemble_module),
                output_embs_per_mixer=output_emb_per_ensemble_module,
                original_n_embs=original_emb_num,
            )
            self.layers.append(layer)

    def forward(
        self,
        input: torch.Tensor,
        skip_connection: Optional[torch.Tensor] = None,
    ):
        skip_connection = None  # previous layer's input
        for layer in self.layers:
            input, skip_connection = layer(skip_connection, input)

        output = input.view(input.shape[0], -1)
        return output

    def get_dense_params(self) -> List[nn.Parameter]:
        return list(self.parameters())


def weight_init_func(pt_module, name_map, params, bufs):
    # for k, m in pt_module.named_modules():
    #     if isinstance(m, torch.nn.Linear):
    #         params[name_map[f"{k}.weight"]] = torch.nn.init.xavier_uniform(params[name_map[f"{k}.weight"]])
    #         params[name_map[f"{k}.bias"]] = torch.nn.init.normal(params[name_map[f"{k}.bias"]], std=1e-6)
    return params, bufs


def train_zhen_homogeneous(global_batch_size, num_iters, parallel_method, num_auto_layers):
    B = global_batch_size # 59  # made multiples of 8
    F = 512
    D = 128
    LAYERS = 4
    OUTPUT_PER_ENSEMBLE = 32  # 50  # made multiples of 8
    # OUTPUT_PER_ENSEMBLE = 48  # 50  # made multiples of 8
    # TOKENS = [
    #     TokenMixer.ATTENTION, TokenMixer.LINEAR, TokenMixer.ATTENTION,
    #     TokenMixer.CONVOLUTION, TokenMixer.DOT
    # ]
    TOKENS = [
        TokenMixer.ATTENTION, TokenMixer.LINEAR, TokenMixer.ATTENTION, TokenMixer.DOT
    ]

    pt_module_gen = lambda: ZHENCollection(LAYERS, D, TOKENS, F,
                                            OUTPUT_PER_ENSEMBLE)

    dataloader = [(torch.empty(
        B, D, F), torch.empty(B, D * LAYERS * OUTPUT_PER_ENSEMBLE))] * num_iters

    loss_func = lambda *args, **kwargs: torch.nn.functional.mse_loss(
        *args, **kwargs)
    optim_gen = torchoptim.adam(lr=1e-3)

    alpa.global_config.xla_client_mem_fraction = 0.7

    train_torch_module(pt_module_gen,
                       weight_init_func,
                       dataloader,
                       loss_func,
                       optim_gen,
                       parallel_method)


def precheck(pp: int, dp: int, op: int, num_micro_batches: int,
             num_auto_layers: int):
    assert num_auto_layers % pp == 0, f"{num_auto_layers} vs {pp}"

    alpa.init("ray")
    cluster = alpa.device_mesh.get_global_cluster()
    assert cluster.num_devices == pp * dp * op, f"{cluster.num_devices} vs {pp} * {dp} * {op}"


def data_parallel(pp: int, dp: int, op: int, num_micro_batches: int,
        num_auto_layers: int):
    precheck(pp, dp, op, num_micro_batches, num_auto_layers)

    cluster = get_global_cluster()
    return ManualPipeshardParallel(
        forward_stage_layer_ids=np.array_split(range(num_auto_layers), pp),
        submesh_physical_shapes=[cluster.get_virtual_physical_mesh().shape] * pp,
        submesh_logical_shapes=[(dp, op)] * pp,
        submesh_autosharding_option_dicts=[{ 'force_batch_dim_to_mesh_dim': 0 }] * pp,
        num_micro_batches=num_micro_batches,
        default_auto_sharding_option=alpa.AutoShardingOption(
            prefer_reduce_scatter=True))


PARALLEL_METHODS = [
    {
        # duration of each iteration avg: 2.120391 secs, median: 1.5998216139851138 secs, 90P: 6.935175623977557 secs, 99P: 6.935175623977557 secs
        "description": "data parallel",
        "(pp, dp, op)": (1, 16, 1),
        "physical_mesh_shape": (2, 8),
        "auto_sharding_option": {'force_batch_dim_to_mesh_dim': 0},
    },
    {
        # incompatible shapes before, removing conv2d bypasses this issue
        # duration of each iteration avg: 9.556799 secs, median: 9.174856601981446 secs, 90P: 13.079619469936006 secs, 99P: 13.079619469936006 secs
        "description": "two nodes, within each node running operator parallel",
        "(pp, dp, op)": (1, 2, 8),
        "physical_mesh_shape": (2, 8),
    },
    {
        # this plan looks suboptimal, low priority
        # failed: jax._src.traceback_util.UnfilteredStackTrace: TypeError: lax.dynamic_update_slice requires arguments to have the same dtypes, got bool, uint8.
        "description": "Example #1: two data pipelines (dp=2), each data pipeline has 4 pipeline stages (pp=4), within each pipeline stages, run tensor parallel on two GPUs.",
        "(pp, dp, op)": (4, 2, 2),
        "physical_mesh_shape": (1, 4),
    },
    {
        # failed: never finish the first iteration
        "description": "Example #2: intra-numa-node (4 gpus) operator parallel, cross-numa and cross-machine pipeline",
        "(pp, dp, op)": (4, 1, 4),
        "physical_mesh_shape": (1, 4),
    },
    {
        # This plan looks not bad
        # failed: assert required_num_hosts == 1
        "description": "Example #4: data-parallel across numa-nodes, tensor-parallel within a numa-node",
        "(pp, dp, op)": (1, 4, 4),
        "physical_mesh_shape": (2, 8), # (4, 4)
    },
    {
        # failed: assert required_num_hosts == 1
        # failed: never finish the first iteration
        "description": "Example #5.1: data parallel + cross-machine pipeline parallel",
        "(pp, dp, op)": (4, 4, 1),
        "physical_mesh_shape": (1, 4),
    },
    {
        # This plan looks not bad
        # failed: assert required_num_hosts == 1
        "description": "Example #5.2: data parallel + intra-machine pipeline parallel",
        "(pp, dp, op)": (4, 4, 1),
        "physical_mesh_shape": (2, 2), # (4, 1)
    },
    {
        # this looks suboptimal
        # failed: assert tuple(expected_microbatched_shape) == microbatch_shape, AssertionError: (1, 128) vs (128,)
        "description": "two gpus a group, each group in charge of a pipeline stage (8 stages)",
        "(pp, dp, op)": (8, 1, 2),
        "physical_mesh_shape": (1, 2),
    },
]

BENCHMARK_SUITES = {
    "global_batch_size": 1024 * 16,
    "num_micro_batches": 32,
    # actual_batch = global_batch_size / num_devices * num_micro_batches
    # actual_batch_per_device = actual_batch / num_devices
    "num_iters": 10,
    "parallel_methods": PARALLEL_METHODS[5:6],
}

def benchmark_zhen_homogeneous(benchmark_case):
    print(f"benchmarking: {benchmark_case}")

    c = benchmark_case
    global_batch_size = c["global_batch_size"]
    num_micro_batches = c["num_micro_batches"]
    num_iters = c["num_iters"]
    p = c["parallel_method"]

    # high-level parallelism specification
    pp, dp, op = p["(pp, dp, op)"]
    physical_mesh_shape = p["physical_mesh_shape"]
    auto_sharding_option = p.get("auto_sharding_option", {})

    # (failed)  # assert tuple(expected_microbatched_shape) == microbatch_shape
    # pp, dp, op = 8, 1, 2 
    # physical_mesh_shape = (1, 2)

    # pp = LAYERS (failed) same above
    # pp, dp, op = 4, 2, 2
    # physical_mesh_shape = (1, 4)

    # intra-node operator parallel, cross-machine pipeline
    # pp = LAYERS (never finish the first iteration)
    # pp, dp, op = 4, 1, 4
    # physical_mesh_shape = (1, 4)

    # data parallel + cross-machine pipeline parallel
    # Example #5.1 (never finish the first iteration)
    # pp, dp, op = 4, 4, 1
    # physical_mesh_shape = (1, 4)

    # data parallel + intra-machine pipeline parallel
    # Example #5.2
    # assert required_num_hosts == 1
    # pp, dp, op = 4, 4, 1
    # physical_mesh_shape = (4, 1)
    # auto_sharding_option = {'force_batch_dim_to_mesh_dim': 0}

    # two nodes, within each node running operator parallel
    # duration of each iteration avg: 9.556799 secs, median: 9.174856601981446 secs, 90P: 13.079619469936006 secs, 99P: 13.079619469936006 secs
    # pp, dp, op = 1, 2, 8
    # physical_mesh_shape = (2, 8)

    # data parallel
    # duration of each iteration avg: 2.120391 secs, median: 1.5998216139851138 secs, 90P: 6.935175623977557 secs, 99P: 6.935175623977557 secs
    # pp, dp, op = 1, 16, 1
    # physical_mesh_shape = (2, 8)
    # auto_sharding_option = {'force_batch_dim_to_mesh_dim': 0}

    # num_micro_batches = 32
    num_auto_layers = pp * 1

    # check
    precheck(pp, dp, op, num_micro_batches, num_auto_layers)

    # cluster = alpa.device_mesh.get_global_cluster()
    # virtual_mesh = cluster.get_virtual_physical_mesh()
    # logical_mesh_shape = (dp, op)
    # num_mesh_devices = np.prod(logical_mesh_shape)
    # num_devices_per_host = virtual_mesh.num_devices_per_host
    # if num_mesh_devices <= num_devices_per_host:
    #     physical_mesh_shape = (1, num_mesh_devices)
    # else:
    #     assert num_mesh_devices % num_devices_per_host == 0
    #     physical_mesh_shape = (num_mesh_devices // num_devices_per_host,
    #                            num_devices_per_host)

    # data parallel + cross-mesh pipeline parallel + intra-mesh tensor parallel
    parallel_method = PipeshardParallel(
        num_micro_batches=num_micro_batches,
        layer_option=alpa.AutoLayerOption(num_auto_layers),
        stage_option=alpa.ManualStageOption(
            forward_stage_layer_ids=np.array_split(range(num_auto_layers), pp),
            submesh_physical_shapes=[physical_mesh_shape] * pp,
            submesh_logical_shapes=[(dp, op)] * pp,
            submesh_autosharding_option_dicts=[auto_sharding_option] * pp),
        default_auto_sharding_option=alpa.AutoShardingOption(
            prefer_reduce_scatter=True))

    # cross-mesh pipeline parallel, intra-mesh data parallel
    # parallel_method = ManualPipeshardParallel(
    #     forward_stage_layer_ids=[[i] for i in range(pp)],
    #     submesh_physical_shapes=[(1, 8)] * pp,
    #     submesh_logical_shapes=[(dp, op)] * pp,
    #     submesh_autosharding_option_dicts=[{'force_batch_dim_to_mesh_dim': 0}, {'force_batch_dim_to_mesh_dim': 0}],
    #     num_micro_batches=num_micro_batches,
    #     default_auto_sharding_option=alpa.AutoShardingOption(
    #         prefer_reduce_scatter=True))

    # data parallel
    # duration of each iteration avg: 8.134366 secs, median: 6.646504681993974 secs, 90P: 21.352595987002132 secs, 99P: 21.352595987002132 secs
    # parallel_method = data_parallel(pp, dp, op, num_micro_batches,
    #                                 num_auto_layers)

    # Result forward_stage_layer_ids: [[0], [1]]
    # Result meshes: [(1, 8), (1, 8)]
    # Result logical_mesh_shapes: [(1, 8), (1, 8)]
    # Result autosharding_option_dicts: [{}, {}]
    # auto search
    # parallel_method = alpa.PipeshardParallel(
    #     stage_mode="auto", num_micro_batches=num_micro_batches)

    # faster auto search
    # parallel_method = alpa.PipeshardParallel(
    #     stage_mode="uniform",
    #     num_micro_batches=num_micro_batches,
    #     default_auto_sharding_option=alpa.AutoShardingOption(force_data_parallel=True))
    # auto_layer_con_func = alpa.automatic_layer_construction(
    #     layer_num=num_auto_layers, remat_layer=True)

    train_zhen_homogeneous(global_batch_size, num_iters, parallel_method, num_auto_layers)


def main():
    for parallel_method in BENCHMARK_SUITES["parallel_methods"]:
        case = BENCHMARK_SUITES.copy()
        del case["parallel_methods"]
        case["parallel_method"] = parallel_method
        benchmark_zhen_homogeneous(case)


if __name__ == '__main__':
    main()
