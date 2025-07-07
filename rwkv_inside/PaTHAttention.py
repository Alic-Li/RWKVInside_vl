# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


#Some modified(interface) for Attention Conversion
#cxa078 + PaTH Hybrid Attention
#2025 OpenMOSE

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from transformers.utils import logging

from fla.layers.utils import pad_input, unpad_input
from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2_norm
from fla.ops.attn.decoding import attn_decoding_one_step
from fla.ops.path_attn.parallel import parallel_path_attention


from loralinear import LoraLinear

if TYPE_CHECKING:
    from fla.models.utils import Cache

logger = logging.get_logger(__name__)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat KV heads along the head dimension (GQA).
    Input:  (B, T, H_kv, D)
    Output: (B, T, H_kv * n_rep, D)
    """
    B, T, H_kv, D = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # Expand head dim
    hidden_states = hidden_states[:, :, :, None, :]  # (B, T, H_kv, 1, D)
    hidden_states = hidden_states.expand(B, T, H_kv, n_rep, D)  # (B, T, H_kv, n_rep, D)
    return hidden_states.reshape(B, T, H_kv * n_rep, D).contiguous()  
class T5RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
class PaTHAttention(nn.Module):
    def __init__(
        # self,
        # hidden_size: int = 2048,
        # num_heads: int = 32,
        # num_kv_heads: Optional[int] = None,
        # use_forget_gate: bool = False,
        # use_qk_norm: bool = False,
        # use_w_shortconv: bool = True,
        # layer_idx: int = None,
        self, args, layer_id
    ):
        super().__init__()

        # self.hidden_size = hidden_size
        # self.num_heads = num_heads
        # if num_kv_heads is None:
        #     self.num_kv_heads = self.num_heads
        # else:
        #     self.num_kv_heads = num_kv_heads
        # self.head_dim = self.hidden_size // self.num_heads
        # self.kv_dim = self.num_kv_heads * self.head_dim

        # self.layer_idx = layer_idx



        self.training = True
        
        self.args = args
        self.layer_id = layer_id
        self.Attention = 1

        self.head_size = args.head_size_a // 2 #Currently can use only up to 64
        self.head_dim = self.head_size
        #self.scaling = self.head_size ** -0.5
        self.n_head = args.dim_att // self.head_size

        self.max_position_embeddings = args.config.max_position_embeddings
        self.num_attention_heads = args.num_attention_heads * 2
        self.num_key_value_heads = args.num_key_value_heads * 2
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.rms_norm_eps = args.rms_norm_eps
        self.rope_theta = args.config.rope_theta

        self.QKNormMode = True

        use_qk_norm = self.QKNormMode
        use_w_shortconv = True 
        use_forget_gate = True
        

        print(f'layer = {layer_id} head_size {self.head_size} n_head {self.n_head}')



        assert args.dim_att % self.n_head == 0
        H = self.num_attention_heads#self.n_head
        N = self.head_size

        C = H*N#args.n_embd
        Hidden_dim = args.n_embd

   

        if args.freeze_hybrid_attention:
                peftmode = 'full'
        else:
            peftmode = args.peftmode

        # self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        # self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)

        self.num_heads = self.num_attention_heads
        self.num_kv_heads =  self.num_key_value_heads
        self.att_dim = self.num_attention_heads * self.head_size
        self.kv_dim = self.num_key_value_heads * self.head_size
        self.layer_idx = layer_id
        self.hidden_size = Hidden_dim

        self.q_proj = LoraLinear(Hidden_dim, self.num_attention_heads * self.head_size, bias=self.args.is_attention_bias,peftmode=peftmode)
        self.k_proj = LoraLinear(Hidden_dim, self.num_key_value_heads * self.head_size, bias=self.args.is_attention_bias,peftmode=peftmode)
        self.v_proj = LoraLinear(Hidden_dim, self.num_key_value_heads * self.head_size, bias=self.args.is_attention_bias,peftmode=peftmode)
        
        # We use low-rank parameterization for the w_proj to reduce parameters in MHA settings.
        if self.num_heads == self.num_kv_heads:
            self.w_proj = nn.Sequential(
                nn.Linear(self.hidden_size, 32, bias=False),
                nn.Linear(32, self.kv_dim, bias=False)
            )
        # In MQA/GQA settings, key/value heads are shared, so we use a standard linear projection
        # which doesn't introduce too many parameters
        else:
            self.w_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)

        if use_qk_norm:
            self.q_norm = T5RMSNorm(self.head_size,eps=self.rms_norm_eps)
            self.k_norm = T5RMSNorm(self.head_size,eps=self.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        if use_w_shortconv:
            self.w_conv1d = ShortConvolution(self.kv_dim, 4)
        self.use_w_shortconv = use_w_shortconv
        self.bt_proj = nn.Linear(self.hidden_size, self.num_kv_heads, bias=True)
        self.use_forget_gate = use_forget_gate
        if use_forget_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
        #self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = LoraLinear(self.num_attention_heads * self.head_size, Hidden_dim, bias=self.args.is_attention_output_bias,peftmode=peftmode)

    #@torch.compile
    def forward(
        # self,
        # hidden_states: torch.Tensor,
        # attention_mask: Optional[torch.LongTensor] = None,
        # past_key_values: Optional[Cache] = None,
        # output_attentions: bool = False,
        # use_cache: bool = False,
        # **kwargs,
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        x_emb:torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
       # past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        use_cache = False # no need for training
        past_key_values = None
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_size)

        if use_cache:
            assert past_key_values is not None, "past_key_values must be provided when use_cache is True"
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )
        batch_size, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(hidden_shape)
        k = self.k_proj(hidden_states).view(hidden_shape)
        v = self.v_proj(hidden_states)
        w = self.w_proj(hidden_states)

        q, k = self.q_norm(q).view(batch_size,q_len,-1), self.k_norm(k).view(batch_size,q_len,-1)

        # k = k.view(batch_size, q_len, self.num_key_value_heads, self.head_size)
        # v = v.view(batch_size, q_len, self.num_key_value_heads, self.head_size)

        # k = repeat_kv(k, self.num_key_value_groups)
        # v = repeat_kv(v, self.num_key_value_groups)

        # k = k.view(batch_size, q_len, -1)
        # v = v.view(batch_size, q_len, -1)


        beta = self.bt_proj(hidden_states).sigmoid() * 2  # allowing negative eigenvalues
        g = F.logsigmoid(self.g_proj(hidden_states).float()) if self.use_forget_gate else None
        

        

        cu_seqlens = kwargs.get('cu_seqlens', None)
        assert not (cu_seqlens is not None and attention_mask is not None), (
            "cu_seqlens should not be provided when attention_mask is not None"
        )
        # Training
        if attention_mask is None:
            assert use_cache is False, "use_cache should be False in training"
            if self.use_w_shortconv:
                w, _ = self.w_conv1d(w, cache=None, output_final_state=False, cu_seqlens=cu_seqlens)
            q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim)
            k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
            v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)
            w = rearrange(w, '... (h d) -> ... h d', d=self.head_dim)
            w = l2_norm(w)
            o, _ = parallel_path_attention(q=q, k=k, v=v, w=w, beta=beta, g=g, cu_seqlens=cu_seqlens)

        # Prefilling or decoding
        # else:
        #     assert self.training is False, "attention mask is not supported in training. Please use variable length input."
        #     try:
        #         last_state = past_key_values[self.layer_idx]
        #     except KeyError:
        #         last_state = None
        #     # Decoding
        #     if last_state is not None:
        #         if g is not None:
        #             past_k, past_v, past_g = last_state['attn_state']
        #         else:
        #             past_k, past_v = last_state['attn_state']
        #         w_conv_state = last_state['conv_state']
        #         past_k = rearrange(past_k, '... (h d) -> ... h d', d=self.head_dim)
        #         if self.use_w_shortconv:
        #             w, w_conv_state = self.w_conv1d(w, cache=w_conv_state, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        #         w = rearrange(w, '... (h d) -> ... h d', d=self.head_dim)
        #         w = l2_norm(w)

        #         def rank_one_update(k, w, beta):
        #             original_dtype = k.dtype
        #             k = k.float()
        #             w = w.float()
        #             beta = beta.float()
        #             k = k - beta[..., None].float() * (k * w).sum(-1, keepdim=True) * w
        #             return k.to(original_dtype)

        #         past_k = rank_one_update(past_k, w, beta)
        #         past_k = rearrange(past_k, '... h d -> ... (h d)')
        #         k = torch.cat([past_k, k], dim=1)
        #         v = torch.cat([past_v, v], dim=1)
        #         g = torch.cat([past_g, g], dim=1) if g is not None else None
        #         past_key_values[self.layer_idx]['attn_state'] = (k, v, g) if g is not None else (k, v)
        #         past_key_values.update(
        #             conv_state=w_conv_state,
        #             layer_idx=self.layer_idx,
        #             offset=q_len
        #         )
        #         if g is not None:
        #             q, (k, v, g), indices_q, cu_seqlens, max_seq_lens = unpad_input(
        #                 q, (k, v, g), attention_mask, q_len, keepdim=True)
        #             max_seqlen_q, max_seqlen_k = max_seq_lens
        #         else:
        #             q, (k, v), indices_q, cu_seqlens, max_seq_lens = unpad_input(
        #                 q, (k, v), attention_mask, q_len, keepdim=True)
        #             max_seqlen_q, max_seqlen_k = max_seq_lens
        #         _, cu_seqlens = cu_seqlens
        #         q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim)
        #         k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
        #         v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)
        #         assert max_seqlen_q == 1, "only support q_len == 1 for decoding"
        #         o = attn_decoding_one_step(q, k, v, g, cu_seqlens=cu_seqlens, do_gate_scale=True)  # reduced to fox's decoding
        #     # Prefilling
        #     else:
        #         v_cache = v.clone()
        #         g_cache = g.clone() if g is not None else None
        #         if g is None:
        #             q, (k, v, w, beta), indices_q, cu_seqlens, max_seq_lens = unpad_input(
        #                 q, (k, v, w, beta), attention_mask, q_len, keepdim=True)
        #         else:
        #             q, (k, v, w, beta, g), indices_q, cu_seqlens, max_seq_lens = unpad_input(
        #                 q, (k, v, w, beta, g), attention_mask, q_len, keepdim=True)
        #         max_seqlen_q, max_seqlen_k = max_seq_lens
        #         assert max_seqlen_q == max_seqlen_k, "max_seqlen_q should be equal to max_seqlen_k in prefilling"
        #         _, cu_seqlens = cu_seqlens
        #         if self.use_w_shortconv:
        #             w, w_conv_state = self.w_conv1d(w, cache=None, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        #         q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim)
        #         k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
        #         v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)
        #         w = rearrange(w, '... (h d) -> ... h d', d=self.head_dim)
        #         w = l2_norm(w)
        #         o, k_cache = parallel_path_attention(q=q, k=k, v=v, w=w, beta=beta, g=g,
        #                                              cu_seqlens=cu_seqlens, use_cache=use_cache)
        #         if use_cache:
        #             k_cache = pad_input(k_cache.squeeze(0), indices_q, batch_size, q_len)
        #             k_cache = rearrange(k_cache, '... h d -> ... (h d)')
        #             past_key_values.update(
        #                 attn_state=(k_cache, v_cache, g_cache) if g_cache is not None else (k_cache, v_cache),
        #                 conv_state=w_conv_state,
        #                 layer_idx=self.layer_idx,
        #                 offset=q_len
        #             )
        #     o = pad_input(o.squeeze(0), indices_q, batch_size, q_len)
        o = rearrange(o, '... h d -> ... (h d)')
        o = self.o_proj(o)
        return o#, None, past_key_values