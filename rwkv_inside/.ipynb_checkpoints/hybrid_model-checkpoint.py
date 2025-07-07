import sys
import os
from typing import Optional, Tuple
RWKV_VERSION=os.environ.get('RWKV_VERSION','v7')
is_rwkv_7 = RWKV_VERSION == 'v7'
if is_rwkv_7 :
    from TimeMixer import RWKV_Tmix_x070_Mose_cxa075 as TimeMixer
    #from TimeMixer import RWKV_Tmix_x070_Mose_v2 as TimeMixer
else:
    from TimeMixer import RWKV_Tmix_x060 as TimeMixer
import torch
from torch.nn import functional as F
import torch.nn as nn

from transformers import AutoModelForCausalLM
import gc
import logging
import deepspeed
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

class VFirstHolder(nn.Module):
    
    def __init__(self, batch_size: int, seq_length: int, hidden_size: int,world_size: int,dtype=torch.bfloat16):
        super().__init__()
        self.shared_state = nn.Parameter(
            torch.zeros(
                (world_size,batch_size, seq_length, hidden_size),
                dtype=dtype
            ),
            requires_grad=False
        )
    


class AttentionWrapper(nn.Module):
    
    def __init__(self,student_attn,layer_idx,args):
        super(AttentionWrapper, self).__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.teacher_attn = None
        self.student_attn = student_attn
        if self.student_attn is not None:
            self.student_attn.requires_grad_(True)
            print(f'Layer:{layer_idx} is trained')
        else:
            print(f'Layer:{layer_idx} is not trained')
        self.add_module("student_attn", self.student_attn)
        self.v_first_state = None#v6 will benefit from v_first_state
        self.global_rank = None
        self.attention_mask = None

    def comprehensive_attention_mimicking_loss_old(self,teacher_hidden_states, student_hidden_states, layer_idx=0, n_layer=32, args=None):
        """
        AttentionモデルをRWKVに極限まで近づけるための包括的Loss関数
        """
        batch_size, seq_len, hidden_dim = teacher_hidden_states.shape
        losses = {}
        device = teacher_hidden_states.device
        
        # === 1. 基本的な出力マッチング（スケール適応型） ===
        # 出力スケールの不一致を動的に調整
        teacher_scale = teacher_hidden_states.detach().abs().mean()
        student_scale = student_hidden_states.detach().abs().mean()
        scale_ratio = teacher_scale / (student_scale + 1e-8)
        
        # スケール調整済みのVector Norm Loss
        scaled_student = student_hidden_states * scale_ratio
        content_loss = torch.linalg.vector_norm(teacher_hidden_states - scaled_student, dim=-1).mean() * (hidden_dim ** -0.5)
        losses['content'] = content_loss
        
        # === 2. コサイン類似度Loss（方向一致） ===
        cos_loss = 1 - torch.cosine_similarity(teacher_hidden_states, student_hidden_states, dim=-1).mean()
        losses['cosine'] = cos_loss
        
        # === 3. コンテキスト関係性Loss（正規化版） ===
        # 正規化してスケール不変にする
        t_norm = F.normalize(teacher_hidden_states, dim=-1)
        s_norm = F.normalize(student_hidden_states, dim=-1)
        
        teacher_context = torch.bmm(t_norm, t_norm.transpose(1, 2))
        student_context = torch.bmm(s_norm, s_norm.transpose(1, 2))
        
        # 全体的な関係性パターン
        context_loss = torch.norm(teacher_context - student_context, p='fro').mean() / (seq_len ** 2)
        losses['context'] = context_loss
        
        # === 4. 局所・大域特徴のマッチング ===
        # 局所特徴（チャンク単位の情報）
        chunk_size = max(1, hidden_dim // 16)
        local_t = teacher_hidden_states.view(batch_size, seq_len, -1, chunk_size).mean(dim=-2)
        local_s = student_hidden_states.view(batch_size, seq_len, -1, chunk_size).mean(dim=-2)
        local_loss = F.mse_loss(local_t, local_s) * (chunk_size ** 0.5)
        losses['local'] = local_loss
        
        # 大域特徴（シーケンス全体の情報）
        global_t = teacher_hidden_states.mean(dim=1)
        global_s = student_hidden_states.mean(dim=1)
        global_loss = F.mse_loss(global_t, global_s) * (hidden_dim ** 0.5)
        losses['global'] = global_loss
        
        # === 5. 時間的依存性の模倣 ===
        if seq_len > 1:
            # 時間的変化の一致
            teacher_diff = teacher_hidden_states[:, 1:] - teacher_hidden_states[:, :-1]
            student_diff = student_hidden_states[:, 1:] - student_hidden_states[:, :-1]
            temp_loss = 1 - F.cosine_similarity(
                teacher_diff.view(batch_size, -1),
                student_diff.view(batch_size, -1),
                dim=-1
            ).mean()
            losses['temporal'] = temp_loss
        
        # === 6. スペクトル特性のマッチング ===
        # 隠れ層の固有値分布を一致させる
        t_flat = teacher_hidden_states.reshape(-1, hidden_dim)
        s_flat = student_hidden_states.reshape(-1, hidden_dim)

        # バッチサイズが十分大きい場合のみSVD計算
        if t_flat.size(0) >= hidden_dim // 4:
            try:
                # 上位kの特異値を比較
                k = min(8, hidden_dim // 4)
                _, t_s, _ = torch.svd(t_flat.t() @ t_flat, some=True)
                _, s_s, _ = torch.svd(s_flat.t() @ s_flat, some=True)
                
                # 正規化された特異値の分布一致
                spectral_loss = F.mse_loss(t_s[:k]/t_s[0], s_s[:k]/s_s[0])
                losses['spectral'] = spectral_loss
            except:
                # SVDが収束しない場合は代替手段
                t_cov = (t_flat.t() @ t_flat) / t_flat.size(0)
                s_cov = (s_flat.t() @ s_flat) / s_flat.size(0)
                spectral_loss = torch.norm(t_cov - s_cov, p='fro') / hidden_dim
                losses['spectral'] = spectral_loss
        
        
        # === 9. 層依存パラメータの設定 ===
        # 下位層は基本情報、上位層は高次特徴を重視
        #total_layers = getattr(args, 'total_layers', 24)  # デフォルト値
        relative_depth = layer_idx / n_layer
        
        # 層の深さに応じた重み付け
        layer_weight = 1.0#args.base_weight * (args.layer_decay ** layer_idx)
        
        # 上部層はコンテキスト関係をより重視
        context_importance = 1.0 * (1.0 + 0.5 * relative_depth)
        
        # 下部層は基本情報をより重視
        content_importance = 1.0 * (1.0 - 0.3 * relative_depth)

        #print(f'Spectral Loss = {losses.get("spectral", 0)}')
        
        # === 10. 最終的な重み付けLoss ===
        # 基本Lossの組み合わせ
        combined_loss = (
            content_importance * losses['content'] +
            2.0 * losses['cosine'] +
            context_importance * losses['context'] +
            getattr(args, 'local_weight', 0.5) * losses.get('local', 0) +
            getattr(args, 'global_weight', 0.5) * losses.get('global', 0) +
            getattr(args, 'temporal_weight', 1.0) * losses.get('temporal', 0)+
            getattr(args, 'spectral_weight', 0.3) * losses.get('spectral', 0)
        )
        
        # オプション: 検証用にすべての損失コンポーネントを返す
        if getattr(args, 'return_components', False):
            return layer_weight * combined_loss, losses
        
        return layer_weight * combined_loss

    def comprehensive_attention_mimicking_loss(self, teacher_hidden_states, student_hidden_states, layer_idx=0, n_layer=32, args=None):
        """
        バッチサイズに依存しないよう、各サンプル毎に損失を計算し平均するバージョン
        比較対象の次元は (batch, context length, hidden dimension) です。
        """
        batch_size, seq_len, hidden_dim = teacher_hidden_states.shape
        losses = {}
        device = teacher_hidden_states.device

        # === 1. 出力マッチング (スケール適応型) ===
        # 各サンプルごとにスケールを計算（batch次元で平均していないので、バッチサイズに依存しません）
        teacher_scale = teacher_hidden_states.detach().abs().mean(dim=(1, 2), keepdim=True)  # (B,1,1)
        student_scale = student_hidden_states.detach().abs().mean(dim=(1, 2), keepdim=True)  # (B,1,1)
        scale_ratio = teacher_scale / (student_scale + 1e-8)
        
        scaled_student = student_hidden_states * scale_ratio
        # 各サンプル内で隠れ次元ごとのノルムを計算し、シーケンス次元で平均
        content_loss = torch.linalg.vector_norm(teacher_hidden_states - scaled_student, dim=-1)  # (B, seq_len)
        content_loss = content_loss.mean() * (hidden_dim ** -0.5)
        losses['content'] = content_loss

        # === 2. コサイン類似度 Loss (方向一致) ===
        cos_sim = torch.cosine_similarity(teacher_hidden_states, student_hidden_states, dim=-1)  # (B, seq_len)
        cos_loss = (1 - cos_sim).mean()
        losses['cosine'] = cos_loss

        # === 3. コンテキスト関係性 Loss (正規化版) ===
        t_norm = F.normalize(teacher_hidden_states, dim=-1)
        s_norm = F.normalize(student_hidden_states, dim=-1)
        # 各サンプルごとに (seq_len x seq_len) の関係行列を計算
        teacher_context = torch.bmm(t_norm, t_norm.transpose(1, 2))  # (B, seq_len, seq_len)
        student_context = torch.bmm(s_norm, s_norm.transpose(1, 2))  # (B, seq_len, seq_len)
        # 各サンプルごとに Frobenius ノルムを計算してから平均（dim=(1,2)を指定）
        context_loss = torch.norm(teacher_context - student_context, p='fro', dim=(1, 2))
        context_loss = context_loss.mean() / (seq_len ** 2)
        losses['context'] = context_loss

        # === 4. 局所・大域特徴のマッチング ===
        # 局所特徴: 隠れ次元をチャンクに分割し、各チャンクごとに平均
        chunk_size = max(1, hidden_dim // 16)
        num_chunks = hidden_dim // chunk_size
        local_t = teacher_hidden_states.view(batch_size, seq_len, num_chunks, chunk_size).mean(dim=-2)  # (B, seq_len, chunk_size)
        local_s = student_hidden_states.view(batch_size, seq_len, num_chunks, chunk_size).mean(dim=-2)
        local_loss = F.mse_loss(local_t, local_s, reduction='mean') * (chunk_size ** 0.5)
        losses['local'] = local_loss

        # 大域特徴: シーケンス全体で平均（各サンプル内での global 表現）
        global_t = teacher_hidden_states.mean(dim=1)  # (B, hidden_dim)
        global_s = student_hidden_states.mean(dim=1)
        global_loss = F.mse_loss(global_t, global_s, reduction='mean') * (hidden_dim ** 0.5)
        losses['global'] = global_loss

        # === 5. 時間的依存性の模倣 ===
        if seq_len > 1:
            teacher_diff = teacher_hidden_states[:, 1:] - teacher_hidden_states[:, :-1]  # (B, seq_len-1, hidden_dim)
            student_diff = student_hidden_states[:, 1:] - student_hidden_states[:, :-1]
            # 各サンプルごとにフラット化してコサイン類似度計算
            teacher_diff_flat = teacher_diff.view(batch_size, -1)
            student_diff_flat = student_diff.view(batch_size, -1)
            temporal_cos = torch.cosine_similarity(teacher_diff_flat, student_diff_flat, dim=-1)
            temporal_loss = (1 - temporal_cos).mean()
            losses['temporal'] = temporal_loss

        # === 6. スペクトル特性のマッチング ===
        # 各サンプルごとに共分散行列（隠れ次元×隠れ次元）を計算
        teacher_cov = teacher_hidden_states.transpose(1, 2) @ teacher_hidden_states  # (B, hidden_dim, hidden_dim)
        student_cov = student_hidden_states.transpose(1, 2) @ student_hidden_states

        teacher_cov = teacher_cov / seq_len
        student_cov = student_cov / seq_len
        spectral_loss = torch.norm(teacher_cov - student_cov, p='fro', dim=(1, 2)).mean() / hidden_dim
        losses['spectral'] = spectral_loss

        # === 7. 層依存パラメータの設定 ===
        relative_depth = layer_idx / n_layer
        layer_weight = 1.0  # 必要に応じて args.base_weight などを掛ける
        context_importance = 1.0 * (1.0 + 0.5 * relative_depth)
        content_importance = 1.0 * (1.0 - 0.3 * relative_depth)

        # === 8. 最終的な重み付け Loss の計算 ===
        combined_loss = (
            content_importance * losses['content'] +
            2.0 * losses['cosine'] +
            context_importance * losses['context'] +
            getattr(args, 'local_weight', 0.5) * losses.get('local', 0) +
            getattr(args, 'global_weight', 0.5) * losses.get('global', 0) +
            getattr(args, 'temporal_weight', 1.0) * losses.get('temporal', 0) +
            getattr(args, 'spectral_weight', 0.3) * losses.get('spectral', 0)
        )

        if getattr(args, 'return_components', False):
            return layer_weight * combined_loss, losses

        return layer_weight * combined_loss
    #@torch.compile
    def forward(self, 
        # hidden_states: torch.Tensor,
        # attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # past_key_value: Optional[Cache] = None,
        # output_attentions: bool = False,
        # use_cache: bool = False,
        # cache_position: Optional[torch.LongTensor] = None,
        # position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        kwargs['output_attentions'] = False

        # NOTE - instead of returning attentions here we return a special attention loss
        hidden_states = kwargs['hidden_states']

        if self.student_attn is not None:

            hidden_states = hidden_states.requires_grad_(True)
            v_first = self.v_first_state.shared_state.data[self.global_rank].clone()
            # print(f"AttentionWrapper: layer_idx={self.layer_idx}, attention_mask={self.attention_mask}")
            # print(f"kargs={kwargs}")
            if self.args.grad_cp == 1:
                if is_rwkv_7:
                    student_hidden_states,v_first = deepspeed.checkpointing.checkpoint(self.student_attn, hidden_states, v_first, self.attention_mask)
                else:
                    student_hidden_states = deepspeed.checkpointing.checkpoint(self.student_attn, hidden_states)
            else:
                if is_rwkv_7:
                    student_hidden_states,v_first = self.student_attn(hidden_states, v_first, self.attention_mask)
                else:
                    student_hidden_states = self.student_attn(hidden_states)
            self.v_first_state.shared_state.data[self.global_rank].copy_(v_first)
            #print(student_hidden_states)
            if self.args.stage != 1:
                return (student_hidden_states, None)
            # student_outputs = self.student_attn(hidden_states)



        with torch.no_grad():
            teacher_outputs = self.teacher_attn(*args, **kwargs)
        # special attention loss is the vector norm of the difference between the student and teacher attn outputs
        # student_hidden_states = student_outputs[0]
        teacher_hidden_states = teacher_outputs[0]


        
        # # 基本的な出力マッチングLoss
        # content_loss = torch.linalg.vector_norm(teacher_hidden_states - student_hidden_states, dim=-1).mean() * (teacher_hidden_states[0].size(-1) ** -0.5)

        # # コサイン類似度Loss
        # cos_loss = 1 - torch.cosine_similarity(teacher_hidden_states, student_hidden_states, dim=-1).mean()

        # # コンテキスト類似性Loss
        # teacher_context = torch.bmm(teacher_hidden_states, teacher_hidden_states.transpose(1, 2))
        # student_context = torch.bmm(student_hidden_states, student_hidden_states.transpose(1, 2))
        # context_loss = torch.norm(teacher_context - student_context, p='fro').mean() / (teacher_hidden_states.size(1) ** 2)

        # # 層ごとの重み付け
        # layer_weight = 1.0#args.base_weight * (args.layer_decay ** layer_idx)

        # content_weight = 1.0
        # cos_weight = 2.0
        # context_weight = 0.5


        # # 組み合わせLoss
        # special_attn_loss = layer_weight * (
        #     content_weight * content_loss + 
        #     cos_weight * cos_loss + 
        #     context_weight * context_loss
        # )

        #special_attn_loss = self.comprehensive_attention_mimicking_loss(teacher_hidden_states,student_hidden_states,self.layer_idx,self.args.n_layer,self.args)

        if self.student_attn is None:
            special_attn_loss = 0.0
        else:
            special_attn_loss = self.comprehensive_attention_mimicking_loss(teacher_hidden_states,student_hidden_states,self.layer_idx,self.args.n_layer,self.args)

            #special_attn_loss = torch.linalg.vector_norm(teacher_hidden_states - student_hidden_states, dim=-1).mean() * (teacher_hidden_states[0].size(-1) ** -0.5)
        
        #print(f'layer:{self.layer_idx} teacher_hidden_states = {teacher_hidden_states}')
        #print(f'layer:{self.layer_idx} student_hidden_states = {student_hidden_states}')
        #print(f'layer:{self.layer_idx} special_attn_loss = {special_attn_loss}')
        return (teacher_outputs[0], special_attn_loss, ) + teacher_outputs[2:]

class HybridModel(nn.Module):
    
    @staticmethod
    def get_rwkv_args(transformer_config):
        from argparse import Namespace
        args = Namespace()
        args.my_pos_emb = 0
        args.head_size_a = 64
        args.head_size_divisor = 8
        args.ctx_len = 4096
        args.n_layer = transformer_config.num_hidden_layers
        args.n_embd = transformer_config.hidden_size
        args.dim_att = transformer_config.hidden_size
        args.dim_ffn = transformer_config.intermediate_size
        args.pre_ffn = 0
        args.head_qk = 0
        args.tiny_att_dim = 0
        args.tiny_att_layer = -999
        args.vocab_size = transformer_config.vocab_size
        args.layers = [i for i in range(transformer_config.num_hidden_layers)]
        args.pad_id = transformer_config.eos_token_id
        args.stage = 4
        args.is_rwkv_att_only = True
        args.is_all_labels_kl = True
        args.init_with_llama = False
        return args
    
    def __init__(self, transformer_model, rwkv_args, tokenizer=None):
        super(HybridModel, self).__init__()
        stage = rwkv_args.stage
        if stage == 1:
            #Freeze the model
            transformer_model.requires_grad_(False)
        else:
            # for stage2 freeze transformer model
            #transformer_model.requires_grad_(False)
            # only train attention
            ##Unfreeze the model
            transformer_model.requires_grad_(True)
            if transformer_model.config.tie_word_embeddings:
                # copy untied embeddings
                transformer_model.get_output_embeddings().weight = nn.Parameter(transformer_model.get_input_embeddings().weight.clone())
                # untie the embeddings in the config, too
                transformer_model.tie_word_embeddings = False

        for layer_idx in range(transformer_model.config.num_hidden_layers):
            if layer_idx in rwkv_args.layers:
                #Only replace the attention layer with TimeMixer
                # EnableStudent = True
                # if layerprofile is not None:
                #     EnableStudent = False
                #     for i in range(len(layerprofile)):
                #         enablelayer = layerprofile[i]
                #         if enablelayer == layer_idx:
                #             EnableStudent = True

                # if EnableStudent:
                #     student_attn = TimeMixer(rwkv_args, layer_idx)
                # else:
                #     student_attn = None

                student_attn = TimeMixer(rwkv_args, layer_idx)



                llama_layer = transformer_model.model.layers[layer_idx]
                attn_wrapper = AttentionWrapper(student_attn,layer_idx,rwkv_args)
                llama_layer.self_attn = attn_wrapper
                gc.collect()
        self.model = transformer_model
        self.add_module("model", self.model)
        self.args = rwkv_args
        self.teacher_model = None  # 初始化为None，后续再设置
        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            if 'pad_token_id' not in self.tokenizer.__dict__:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        torch.cuda.empty_cache()
        self.client = None

    #
    def forward(
        self,
        input_ids,
        attention_mask,
        **kwargs,
    ):
        for layer_idx in range(self.model.config.num_hidden_layers):
            if layer_idx in self.args.layers:
                llama_layer = self.model.model.layers[layer_idx]
                attn_wrapper = llama_layer.self_attn
                attn_wrapper.attention_mask = attention_mask
        ret = self.model(input_ids, **kwargs)
        return ret
    
    def load_check_point(self, path):
        all_keys = set(self.state_dict().keys())
        incompatible_keys = set()
        #if the path is the file, load it directly
        #if the path is the directory, load the sharded files in the directory with suffix .pt or .bin
        if os.path.isdir(path):
            files = os.listdir(path)
            files = [os.path.join(path, f) for f in files if f.endswith('.pt') or f.endswith('.bin')]
        else:
            files = [path]
        for file in files:
            checkpoint = torch.load(file, map_location='cpu')
            self.load_state_dict(checkpoint, strict=False)
            print(f'load model from {file}')
            ckpt_keys = checkpoint.keys()
            #subtract the keys in the checkpoint from the all_keys
            #if the ckpt_key exists in the all_keys, remove it
            for ckpt_key in ckpt_keys:
                if ckpt_key in all_keys:
                    all_keys.remove(ckpt_key)
                else:
                    incompatible_keys.add(ckpt_key)
            del checkpoint
            gc.collect()
        print(f'Finish loading model from {path}')
        print(f'Incompatible keys: {incompatible_keys} missing keys: {all_keys}')
        
        
        return



