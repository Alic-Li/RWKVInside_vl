import deepspeed
import torch
import torch.nn.functional as F
import logging
#import cupy as cp
#from cupy.cuda import nccl
import json
import torch
from torch.optim import Adam
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.ops.lion import DeepSpeedCPULion, FusedLion
from profiler import time_function
import gc

from bitsandbytes.optim import Adam8bit,AdamW8bit,Lion8bit

def rank0_print(*args, **kwargs):
    if deepspeed.comm.get_rank() == 0:
        print(*args, **kwargs)
         

 

class L2Wrap(torch.autograd.Function):
        @staticmethod
        def forward(ctx, loss, y):
            ctx.save_for_backward(y)
            return loss

        @staticmethod
        def backward(ctx, grad_output):
            y = ctx.saved_tensors[0]
            # to encourage the logits to be close to 0
            factor = 1e-4 / (y.shape[0] * y.shape[1])
            maxx, ids = torch.max(y, -1, keepdim=True)
            gy = torch.zeros_like(y)
            gy.scatter_(-1, ids, maxx * factor)
            return (grad_output, gy)

@time_function
def train_step(model, batch, args, teacher_engine=None, tokenizer=None):
    # print(batch)
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask'].to(torch.int32)
    if 'labels' in batch:
        labels = batch['labels']
        # 验证labels的维度
        if labels.shape != input_ids.shape:
            raise ValueError(f"Labels shape {labels.shape} doesn't match input_ids shape {input_ids.shape}")
    else:
        # 直接创建左移的labels
        labels = torch.cat([input_ids[:, 1:], 
                          torch.full((input_ids.shape[0], 1), 
                                   tokenizer.pad_token_id, 
                                   device=input_ids.device)], dim=1)
        

    
    # 5. 非SFT模式的处理
    if args.stage == 2:
        teacher_logits, teacher_loss = get_teacher_outputs(teacher_engine, input_ids, attention_mask, labels, args)
        gc.collect()
        torch.cuda.empty_cache()
        student_outputs = get_student_outputs(
            model, args, input_ids, labels, attention_mask)
        
        #compute_kl_loss_ultra_efficient
        loss, kl_loss, student_ce_loss = compute_kl_loss_ultra_efficient(
            student_outputs, teacher_logits, labels, args,attention_mask=attention_mask,temperature=1.0)
        # loss, kl_loss, student_ce_loss = compute_kl_loss(
        #     student_outputs, teacher_logits, labels, args,attention_mask=attention_mask)

        #loss = L2Wrap.apply(loss, student_outputs.logits)

    elif args.stage == 1:
        ##print('get student outputs')
        student_outputs = get_student_outputs(
            model, args, input_ids, labels, attention_mask)
        #print('get_attn_loss')
        loss, kl_loss, student_ce_loss = get_attn_loss(
            args, student_outputs)
        teacher_loss = None
        
    return loss, teacher_loss, kl_loss, student_ce_loss
    
@time_function
def get_attn_loss(args, student_outputs):
    attn_from_wrapper = [student_outputs.attentions[i] for i in args.layers]
    # print(f'attn_from_wrapper {attn_from_wrapper}')
    loss = torch.stack(attn_from_wrapper, dim=0).mean()
    kl_loss = None
    student_cross_entropy_loss = None
    return loss,kl_loss,student_cross_entropy_loss
firsttime_student = True
firsttime_teacher = True
@time_function
def get_student_outputs(model, args, input_ids, labels, attention_mask):
    # print(f'student :attention_mask {attention_mask}')
    #print('getting Student Logits.')
    # global firsttime_student
    # if firsttime_student:
    #     firsttime_student = False
    #     with torch.no_grad():
    #         student_outputs = model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask, 
    #         labels=labels, use_cache=False, 
    #         output_attentions=args.stage==1)
    student_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            labels=labels, use_cache=False, 
            output_attentions=args.stage==1)
        
    return student_outputs
@time_function
def get_teacher_outputs(teacher_model, input_ids, attention_mask, labels, args):
    with torch.no_grad():
        teacher_outputs = teacher_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=False, use_cache=False) #, use_cache=False, output_hidden_states=False
        teacher_logits = teacher_outputs.logits
        teacher_loss = teacher_outputs.loss
 
    return teacher_logits,  teacher_loss


@time_function #Modified Temperature
def compute_kl_loss(student_outputs, teacher_logits, labels, args, attention_mask=None, chunk_size=4096,temperature=1.0):
    student_logits = student_outputs.logits  # shape: [batch_size, seq_len, vocab_size]
    vocab_student = student_logits.shape[-1]
    vocab_teacher = teacher_logits.shape[-1]

    # Truncate teacher logits if necessary
    if vocab_teacher > vocab_student:
        teacher_logits = teacher_logits[:, :, :vocab_student]
    
    # 温度パラメータの取得（argsにtemperature属性があると仮定）
    #temperature = getattr(args, "temperature", 2.0)

      
    # 温度スケーリングを適用したロジットの計算
    #student_logits_scaled = student_logits / temperature
    
    # Compute softmax for student and teacher with temperature scaling
    log_probs_student = F.log_softmax(student_logits, dim=-1)  # [batch_size, seq_len, vocab_size]
    with torch.no_grad():
        teacher_logits_scaled = teacher_logits / temperature
        targets = F.softmax(teacher_logits_scaled, dim=-1)    # [batch_size, seq_len, vocab_size]
    
    # Compute KL divergence without reduction
    kl_div_all = F.kl_div(
        log_probs_student,
        targets,
        reduction='none'  # Keep the full tensor to apply mask
    )  # [batch_size, seq_len, vocab_size]
    
    # Sum across vocabulary dimension first
    kl_div_per_token = kl_div_all.sum(dim=-1)  # [batch_size, seq_len]
    
    if attention_mask is not None:
        # Apply attention mask and compute mean only over attended positions
        masked_kl = kl_div_per_token * attention_mask
        kl_loss = masked_kl.sum() / (attention_mask.sum() + 1e-6)  # Add small epsilon for numerical stability
    else:
        # If no mask provided, take mean over all tokens
        kl_loss = kl_div_per_token.mean()
    
    # 温度スケーリングによる勾配補正
    #kl_loss = (temperature ** 2) * kl_loss

    del log_probs_student, targets, kl_div_all, kl_div_per_token
    
    # Get cross entropy loss from student outputs
    student_cross_entropy_loss = student_outputs.loss
    
    # Combine losses using weights from args
    loss = args.kl_weight * kl_loss + args.ce_weight * student_cross_entropy_loss
    
    del student_logits, teacher_logits, labels
    if attention_mask is not None:
        del attention_mask
    return loss, kl_loss, student_cross_entropy_loss
@time_function
def compute_kl_loss_ultra_efficient(student_outputs, teacher_logits, labels, args, attention_mask=None, chunk_size=1024, temperature=1.0):
    """
    勾配を正しく保持する極限最適化版
    """
    student_logits = student_outputs.logits
    vocab_student = student_logits.shape[-1]
    vocab_teacher = teacher_logits.shape[-1]
    
    if vocab_teacher > vocab_student:
        teacher_logits = teacher_logits[:, :, :vocab_student]
    
    batch_size, seq_len, vocab_size = student_logits.shape
    
    # 勾配を保持するテンソルとして累積
    kl_sum = torch.zeros(1, device=student_logits.device, requires_grad=True)
    total_tokens = 0
    
    for i in range(0, seq_len, chunk_size):
        end_idx = min(i + chunk_size, seq_len)
        
        # チャンク抽出
        student_chunk = student_logits[:, i:end_idx, :]
        teacher_chunk = teacher_logits[:, i:end_idx, :]
        
        # KL計算
        log_probs = F.log_softmax(student_chunk, dim=-1)
        
        with torch.no_grad():
            teacher_scaled = teacher_chunk / temperature
            targets = F.softmax(teacher_scaled, dim=-1)
        
        kl_div = F.kl_div(log_probs, targets, reduction='none')
        kl_per_token = kl_div.sum(dim=-1)  # [batch_size, chunk_size]
        
        # Attention mask適用
        if attention_mask is not None:
            mask_chunk = attention_mask[:, i:end_idx]
            masked_kl = kl_per_token * mask_chunk
            kl_sum = kl_sum + masked_kl.sum()
            total_tokens += mask_chunk.sum().item()
        else:
            kl_sum = kl_sum + kl_per_token.sum()
            total_tokens += kl_per_token.numel()
        
        # 中間テンソル削除
        del student_chunk, teacher_chunk, log_probs, targets, kl_div, kl_per_token
        if attention_mask is not None:
            del mask_chunk, masked_kl
        
        #torch.cuda.empty_cache()
    
    # 正規化（勾配保持）
    kl_loss = kl_sum / (total_tokens + 1e-6)
    
    # Cross entropy loss
    student_cross_entropy_loss = student_outputs.loss
    
    # Combine losses
    loss = args.kl_weight * kl_loss + args.ce_weight * student_cross_entropy_loss
    
    # クリーンアップ
    del student_logits, teacher_logits, labels, kl_sum
    if attention_mask is not None:
        del attention_mask
    
    return loss, kl_loss, student_cross_entropy_loss


 



import torch
import torch.nn.functional as F
import logging
import deepspeed
from typing import Dict, Any
class Stats:
    def __init__(self):
        self.total_calls = 0
        self.total_iterations = 0  # 总搜索迭代次数
        self.total_cutoff_sum = 0  # cutoff点位置总和
        self.total_samples = 0     # 总样本数
        self.cutoff_positions = []  # 存储每次找到的 cutoff 位置
        self.iteration_counts = []  # 存储每次迭代次数
        
stats = Stats()



def configure_optimizer_stage2(model, args):
    lr_decay = set()
    lr_0x = set()
    lr_1x = set()
    lr_2x = set()
    lr_3x = set()
    #print(model.keys())
    for n, p in model.named_parameters():
        if 'embed' in n or 'lm_head' in n or '.norm.' in n:
            print(f'name={n} grad={p.requires_grad}')
        if not p.requires_grad:
            print(f'{n} not train skipp optimizer')
            continue
        if ('attn.w0' in n ) and (args.layerwise_lr > 0): #or 'attn.w1' in n or 'attn.w2' in n
            lr_2x.add(n)
            print(f'{n} 3x Learning Rate! Lets RocknLOL! ')
        elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
            if 'embed' in n or 'lm_head' in n or '.norm.' in n:
                print(f'{n} zero lr')
                lr_0x.add(n)
            else:
                lr_decay.add(n)
                print(f'{n} lr decay')
        else:
            #print(f'{n}')
            if 'embed' in n or 'lm_head' in n or '.norm.' in n:
                print(f'{n} zero lr')
                lr_0x.add(n)
            else:
                print(f'{n} 1x LR')
                lr_1x.add(n)

    #exit()
    lr_decay = sorted(list(lr_decay))
    lr_0x = sorted(list(lr_0x))
    lr_1x = sorted(list(lr_1x))
    lr_2x = sorted(list(lr_2x))
    lr_3x = sorted(list(lr_3x))
    param_dict = {n: p for n, p in model.named_parameters()}
    
    if args.layerwise_lr > 0:
        optim_groups = [
                {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                {"params": [param_dict[n] for n in lr_0x], "weight_decay": 0.0, "my_lr_scale": 0.0},
            ]
    else:
        optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

    if args.weight_decay > 0:
        optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
    #print(optim_groups)
    if args.deepspeed:
        if args.deepspeed_offload:
            optimizer = DeepSpeedCPUAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
            #optimizer = DeepSpeedCPULion(optim_groups, lr=args.lr_init, betas=args.betas )
        
        else:
            if args.bnb_optimizer_mode:
                optimizer =  AdamW8bit(optim_groups,  betas=args.betas, eps=args.adam_eps)
            else:
                optimizer = FusedAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
    else:
        optimizer = Adam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps)

    return optimizer


def configure_optimizer(model, args):
    lr_decay = set()
    lr_1x = set()
    lr_2x = set()
    lr_3x = set()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # if (("_w1" in n) or ("_w2" in n)) and (args.layerwise_lr > 0):
        #     lr_1x.add(n)
        # elif (("time_mix" in n) or ("time_maa" in n)) and (args.layerwise_lr > 0):
        #     lr_2x.add(n)
        # elif (("time_decay" in n) or ("time_daaaa" in n)) and (args.layerwise_lr > 0):
        #     lr_2x.add(n)
        # elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
        #     lr_1x.add(n)
        # elif ("time_first" in n) and (args.layerwise_lr > 0):
        #     lr_3x.add(n)
        # 
        if ('attn.w0' in n) and (args.layerwise_lr > 0): # or 'attn.w1' in n or 'attn.w2' in 
            lr_2x.add(n)
            print(f'{n} 3x Learning Rate! Lets RocknLOL! ')
        elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
            lr_decay.add(n)
            print(f'{n} LR Decay')
        else:
            lr_1x.add(n)
            print(f'{n}')

        
    #exit()

    lr_decay = sorted(list(lr_decay))
    lr_1x = sorted(list(lr_1x))
    lr_2x = sorted(list(lr_2x))
    lr_3x = sorted(list(lr_3x))
    param_dict = {n: p for n, p in model.named_parameters()}
    
    if args.layerwise_lr > 0:
        optim_groups = [
                {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
            ]
    else:
        optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

    if args.weight_decay > 0:
        optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]

    if args.deepspeed:
        if args.deepspeed_offload:
            optimizer = DeepSpeedCPUAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
        else:
            if args.bnb_optimizer_mode == 1:
                optimizer =  AdamW8bit(optim_groups,  betas=args.betas, eps=args.adam_eps)
            elif args.bnb_optimizer_mode == 2:
                optimizer =  Lion8bit(optim_groups,  betas=args.betas)
            else:
                optimizer = FusedAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
    else:
        optimizer = Adam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps)

    return optimizer