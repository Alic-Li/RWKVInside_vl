#Deepspeed Stage2 Optimized
# OpenMOSE

import sys
import os
import torch.nn as nn
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:False"
import random
import torch
import gc
from typing import List, Optional, Union
def setup_env():
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    rwkv_insidea_path = os.path.join(parent_dir, 'rwkv_inside')
    sys.path.append(rwkv_insidea_path)
    sys.path.append(parent_dir)
    print(f'add path: {rwkv_insidea_path} to sys.path')
    os.environ['RWKV_JIT_ON'] = '0'
    os.environ['RWKV_T_MAX'] = os.environ.get('RWKV_T_MAX', '4096')
    os.environ['RWKV_FLOAT_MODE'] = 'bf16'
    os.environ['RWKV_HEAD_SIZE_A'] = '64'
    
    os.environ['RWKV_CTXLEN'] = os.environ.get('RWKV_CTXLEN', '4096')
    if 'WKV' not in os.environ:
        os.environ['WKV'] = ''
    if "RWKV_TRAIN_TYPE" not in os.environ:
        os.environ["RWKV_TRAIN_TYPE"] = ''
    RWKV_VERSION = os.environ.get('RWKV_VERSION', 'v7')
    if RWKV_VERSION == 'v7':
        os.environ["RWKV_MY_TESTING"]='x070'
    else:
        os.environ["RWKV_MY_TESTING"]='x060'
    print(f'RWKV_VERSION is {RWKV_VERSION}')
    
setup_env()

import argparse
import yaml
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
#from hybrid_model import HybridModel,VFirstHolder
from train_functions import configure_optimizer_stage2, train_step
import datasets
import json
import math
import time
import wandb
from tqdm import tqdm
from profiler import timer, time_function
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb

def create_arg_parser():
    node_rank = int(os.environ.get('NODE_RANK', 0))
    num_gpus = int(os.environ.get('NUM_GPUS', 1))
    world_size = int(os.environ.get('WORLD_SIZE', 7))
    print(f'node_rank: {node_rank}, num_gpus: {num_gpus}, world_size: {world_size}')
    parser = argparse.ArgumentParser(description='MLM trainer')
    parser.add_argument('--config_file', type=str,default='configs/test_hybrid.yaml', help='training config file')
    parser.add_argument('--preprocessed_data',type=str,nargs='+',help='preprocessed data directory')
    parser.add_argument('--raw_data',type=str,nargs='+',help='raw data directory')
    parser.add_argument('--need_to_pad',action='store_true',default=False,help='whether to pad the input with other sample to fill the sample to max length')
    parser.add_argument('--output_dir', type=str, default='/data/rwkv/tmp',help='directory to save the trained model')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs to train the model')
    parser.add_argument('--max_seq_length', type=int, default=512, help='maximum sequence length to train the model')
    parser.add_argument('--num_devices', type=int, default = 1,help='number of devices to train the model')
    parser.add_argument('--has_group_norm', action='store_true',default=False,help='whether the Time Mixer has group norm')
    parser.add_argument('--gate_free',action='store_true',default=False,help='whether the Time Mixer has gate free')
    parser.add_argument('--min_len', type=int, default=0, help='minimum length of the input')
    parser.add_argument('--max_len', type=int, default=4096, help='maximum length of the input')
    parser.add_argument('--freeze_mlp', action='store_true',default=False,help='freeze the mlp layer')
    parser.add_argument('--teacher_model_id', type=str, default=None, help='teacher model id used to distill in stage2')
    
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate in the model')
    parser.add_argument('--grad_cp', type=int, default=0, help='gradient checkpoint in the model')
    parser.add_argument('--save_per_batches', type=int, default=10000, help='number of batches to save the model')
    parser.add_argument('--my_exit', type=int, default=300, help='exit condition in the model')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay in the model')
    parser.add_argument('--lr_init', type=float, default=6e-4, help='initial learning rate in the model')
    parser.add_argument('--lr_final', type=float, default=1e-5, help='final learning rate in the model')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 parameter in the Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.98, help='beta2 parameter in the Adam optimizer')
    parser.add_argument('--layerwise_lr', type=float, nargs='+', default=1, help='layerwise learning rate in the model')
    parser.add_argument('--adam_eps', type=float, default=1e-8, help='epsilon parameter in the Adam optimizer')
    parser.add_argument('--warmup_steps', type=int, default=50, help='warmup steps in the model')
    parser.add_argument('--epoch_begin', type=int, default=0, help='beginning epoch for the training')
    parser.add_argument('--epoch_count', type=int, default=150, help='total number of epochs for the training')
    parser.add_argument('--epoch_save', type=int, default=1, help='number of epochs after which the model is saved')
    parser.add_argument('--max_epochs', type=int, default=150, help='maximum number of epochs for the training')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='number of epochs after which the validation is checked')
    parser.add_argument('--val_check_interval', type=int, default=5000, help='number of epochs after which the validation is checked')
    parser.add_argument('--num_sanity_val_steps', type=int, default=0, help='number of validation steps for sanity check at the beginning of training')
    parser.add_argument('--log_every_n_steps', type=int, default=5000, help='number of steps after which the training progress will be logged')
    parser.add_argument('--enable_checkpointing', type=bool, default=False, help='flag to enable checkpointing')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='number of batches to accumulate before performing a backward/update pass')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='maximum gradient norm')
    parser.add_argument('--num_nodes', type=int, default=1, help='number of nodes for distributed training')
    parser.add_argument('--micro_bsz', type=int,default=2, help='micro batch size for training')
    parser.add_argument('--real_bsz', type=int, help='real batch size for training')
    parser.add_argument('--my_pile_stage', type=int, default=0, help='pile stage in the model')
    parser.add_argument('--my_pile_edecay', type=float, default=0, help='pile exponential decay in the model')
    parser.add_argument('--weight_decay_final', type=float, default=-1, help='final weight decay in the model')
    parser.add_argument('--proj_dir', type=str, help='project directory to save the model and logs')
    parser.add_argument('--eval_every_steps', type=int, default=100, help='number of steps after which the model is evaluated')
    parser.add_argument('--wandb', type=str, default='hybrid_trainer', help='wandb project name')
    parser.add_argument('--run_name', type=str, default='hybrid_trainer_a800', help='run name for wandb logging')
    parser.add_argument('--strategy', type=str, default='deepspeed_stage_2_offload', help='strategy for distributed training')
    parser.add_argument("--ds_bucket_mb", default=200, type=int)  # deepspeed bucket size in MB. 200 seems enough
    parser.add_argument('--my_qa_mask', type=int, default=0)
    parser.add_argument('--optim',type=str,default='adam',help='optimizer')
    parser.add_argument('--train_type', type=str, default='', help='train type')
    parser.add_argument('--skip_steps',type=int,default=0,help='skip steps in the peft checkpoint')
    parser.add_argument('--full_params',action='store_true',help='full params update',default=False)
    parser.add_argument('--ckpt_file', type=str, default=None, help='checkpoint file')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='checkpoint directory')
    parser.add_argument('--ckpt_id', type=str, default=None, help='checkpoint id')
    # 添加DeepSpeed相关的参数
    parser.add_argument('--deepspeed', action='store_true', help='Enable DeepSpeed')
    parser.add_argument('--deepspeed_config', type=str, default=None, help='Path to DeepSpeed config file')
    parser.add_argument('--deepspeed_stage', type=int, default=2, choices=[0, 1, 2, 3], help='DeepSpeed ZeRO stage')
    parser.add_argument('--deepspeed_offload', action='store_true', help='Enable CPU offloading',default=False)
    parser.add_argument('--train_batch_size', type=int, default=None, help='train batch size')
    parser.add_argument('--world_size', type=int, help='world size')
    parser.add_argument('--local_rank', type=int, help='local rank')
    parser.add_argument('--stage', type=int, default=1,choices=[1,2,3], help='stage 1 only align attn output and stage 2 do kl-divergence,and stage 3 do SFT')
    parser.add_argument('--max_trained_tokens', type=int, default=100_000_000, help='max trained tokens')
    parser.add_argument('--terminate_at_loss', type=float, default=0, help='terminate the training at loss')

    parser.add_argument('--freeze_attention', type=int, default=0, help='Freeze Receptance,Key,Value')
    parser.add_argument('--use_bitsandbytes', type=int, default=0, help='apply 8bitlinear with bitsandbytes')
    parser.add_argument('--hybrid_attention_layers', type=int, default=8, help='Hybrid Attention Layers')
    parser.add_argument('--freeze_hybrid_attention', type=int, default=0, help='Freeze Hybrid Attention q,k,v')
    parser.add_argument('--allow_quant_frozen_layers', type=int, default=1, help='allow quant frozen layers')
    parser.add_argument('--quant_mode', type=str, default="int8", help='quant in peft mode except full,  can choose int8,nf4,none')
    parser.add_argument('--peftmode', type=str, default="full", help='peftmode full,lora,bone')
    parser.add_argument('--peft_r', type=int, default=32, help='peft block lora rank')
    parser.add_argument('--peft_scaling', type=float, default=0.5, help='peft block lora scaling')
    parser.add_argument('--peft_dropout', type=float, default=0.01, help='peft block lora dropout')

    parser.add_argument('--mlp_quant_mode', type=str, default="int8", help='MLP Quant mode int8,nf4')
    parser.add_argument('--bnb_optimizer_mode', type=int, default=1, help='Use Bitsandbytes 8bit optimizer AdamW:1 LION:2')
    #parser.add_argument('--deepspeed_lion_mode', type=int, default=0, help='Use Bitsandbytes 8bit optimizer AdamW')
    return parser

def lr_schedule(args, step):
    w_step = args.warmup_steps
    if args.lr_final == args.lr_init or args.epoch_count == 0:
        return args.lr_init
    
    decay_step = step - args.my_pile_edecay * args.epoch_steps
    decay_total = (args.epoch_count - args.my_pile_edecay) * args.epoch_steps
    progress = (decay_step - w_step + 1) / (decay_total - w_step)
    progress = min(1, max(0, progress))

    if args.lr_final == 0 or args.lr_init == 0:  # linear decay
        lr = args.lr_init + (args.lr_final - args.lr_init) * progress
    else:  # exp decay
        lr = args.lr_init * math.exp(math.log(args.lr_final / args.lr_init) * pow(progress, 1))

    if step < w_step:
        lr = lr * (0.01 + 0.99 * step / w_step)
    
    return lr

def weight_decay_schedule(args, progress):
    if args.weight_decay_final > 0:
        return args.weight_decay * math.exp(math.log(args.weight_decay_final / args.weight_decay) * progress)
    return args.weight_decay

def on_train_batch_start(args, model_engine, global_step, epoch):
    real_step = global_step + args.epoch_begin * args.epoch_steps

    # LR schedule
    lr = lr_schedule(args, real_step)
    
    # Weight decay schedule
    progress = (real_step - args.warmup_steps + 1) / ((args.epoch_count - args.my_pile_edecay) * args.epoch_steps - args.warmup_steps)
    progress = min(1, max(0, progress))
    wd_now = weight_decay_schedule(args, progress)

    # 更新优化器参数
    for param_group in model_engine.optimizer.param_groups:
        if param_group["weight_decay"] > 0:
            param_group["weight_decay"] = wd_now
        if args.layerwise_lr > 0:
            param_group["lr"] = lr * param_group["my_lr_scale"]
        else:
            param_group["lr"] = lr

    # 初始化日志（仅在第一步执行）
    if global_step == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "train_log.txt"), "a") as f:
            f.write(f"NEW RUN {time.strftime('%Y-%m-%d %H:%M:%S')}\n{vars(args)}\n")

    return lr, wd_now

    
#     return model
from bnbwrapper import quantize_and_replace_with_wrapper, quantize_mlp_layers_properly
# 在主训练循环开始前初始化tqdm
pbar = None
total_loss = 0
total_updates = 0
trained_tokens = 0
avg_loss = 0
def on_train_batch_end(args, batch_idx, model_engine,teacher_engine, loss, teacher_loss, kl_loss, student_cross_entropy_loss, global_step, epoch, last_log_time, token_per_step, is_accumulation_step, pbar):
    current_time = time.time()
    elapsed_time = current_time - last_log_time
    steps_per_second = 1 / elapsed_time
    kt_s = token_per_step * steps_per_second / 1000  # K tokens per second
    global total_loss
    global total_updates
    global trained_tokens
    global avg_loss
    total_loss += loss
    total_updates += 1
    avg_loss = total_loss / total_updates
    # 只在实际更新参数时更新进度条
    trained_tokens += token_per_step
    if is_accumulation_step and model_engine.global_rank == 0:
        if pbar is None:
            pbar = tqdm(total=args.epoch_steps, desc=f"Epoch {epoch}")
        
        pbar.update(1)
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'steps/s': f'{steps_per_second:.2f}',
            'kt/s': f'{kt_s:.2f}',
            'trained_tokens': f'{trained_tokens / 1e6:.2f} MT',
            'remained_tokens': f'{(args.max_trained_tokens - trained_tokens) / 1e6:.2f} MT'
        })
        timer.print_stats(global_step)
        if args.wandb:
            wandb.log({
                "loss": loss,
                "lr": model_engine.optimizer.param_groups[0]['lr'],
                "weight_decay": model_engine.optimizer.param_groups[0]['weight_decay'],
                "steps_per_second": steps_per_second,
                "kt/s": kt_s,
                "global_step": global_step,
                "Gtokens": global_step * token_per_step * args.accumulate_grad_batches / 1e9,
                "epoch": epoch,
                "teacher_loss": teacher_loss,
                "kl_loss": kl_loss,
                "student_cross_entropy_loss": student_cross_entropy_loss,
            })

    real_step = batch_idx
    if real_step % args.save_per_batches == 0 and real_step > 0 :
        #first check if the output_dir exists and deletes older checkpoints , we only keep latest 2 checkpoints
        if os.path.exists(args.output_dir):
            if model_engine.local_rank == 0:
                checkpoints = os.listdir(args.output_dir)
                #only list the directories   s
                checkpoints = [f for f in checkpoints if os.path.isdir(os.path.join(args.output_dir, f))]
                #sort by creation time  
                checkpoints.sort(key=lambda x: os.path.getctime(os.path.join(args.output_dir, x)))
                if len(checkpoints) > 2:
                    print(f'deleting older checkpoints {checkpoints[0]}')
                    import shutil
                    shutil.rmtree(os.path.join(args.output_dir, checkpoints[0]))    
        output_dir = f"{args.output_dir}/epoch_{epoch}_step_{real_step}"
        print(f'saving checkpoint to {output_dir}')
  
        # 在保存检查点的代码处使用上下文管理器
        with teacher_attn_manager.temporarily_remove_teacher_attn():
            try:
                print(f"Saving checkpoint to {output_dir} at epoch {epoch} step {real_step} rank {model_engine.global_rank}")
                model_engine.save_checkpoint(output_dir, f'epoch_{epoch}_step_{real_step}')
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
                import traceback
                traceback.print_exc()

    return current_time, pbar
import torch.distributed as dist
def setup_distributed():
    dist.init_process_group(backend='rccl')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
import contextlib
from typing import List

class TeacherAttnManager:
    def __init__(self, model_engine, layers: List[int]):
        self.model_engine = model_engine
        self.layers = layers
        self.stored_teacher_attns = {}
        self.stored_vfirst_state = {}
        self.stored_kfirst_state = {}
        
    @contextlib.contextmanager
    def temporarily_remove_teacher_attn(self):
        """
        上下文管理器，临时移除所有层的teacher_attn,v_first_state并在退出时恢复
        """
        try:
            # 保存并移除所有teacher_attn
            for layer_idx in self.layers:
                attention_wrapper = self.model_engine.module.model.model.layers[layer_idx].self_attn
                if hasattr(attention_wrapper, 'teacher_attn'):
                    self.stored_teacher_attns[layer_idx] = attention_wrapper.teacher_attn
                    # 移除teacher_attn模块
                    if hasattr(attention_wrapper, '_modules') and 'teacher_attn' in attention_wrapper._modules:
                        del attention_wrapper._modules['teacher_attn']
                    attention_wrapper.teacher_attn = None
                if hasattr(attention_wrapper, 'v_first_state'):
                    self.stored_vfirst_state[layer_idx] = attention_wrapper.v_first_state
                    attention_wrapper.v_first_state = None
                if hasattr(attention_wrapper, 'k_first_state'):
                    self.stored_kfirst_state[layer_idx] = attention_wrapper.k_first_state
                    attention_wrapper.k_first_state = None
            
            yield  # 允许在此上下文中执行代码
            
        finally:
            # 恢复所有teacher_attn
            for layer_idx, stored_attn in self.stored_teacher_attns.items():
                attention_wrapper = self.model_engine.module.model.model.layers[layer_idx].self_attn
                attention_wrapper.teacher_attn = stored_attn
                # 重新注册为子模块
                if hasattr(attention_wrapper, 'add_module') and not hasattr(attention_wrapper, 'teacher_attn'):
                    attention_wrapper.add_module("teacher_attn", stored_attn)
                v_first_state = self.stored_vfirst_state.get(layer_idx, None)
                k_first_state = self.stored_kfirst_state.get(layer_idx, None)
                if v_first_state is not None:
                    attention_wrapper.v_first_state = v_first_state
                if k_first_state is not None:
                    attention_wrapper.k_first_state = k_first_state
            # 清空存储的引用
            self.stored_teacher_attns.clear()

class TeacherAttnManager_:
    def __init__(self, model_engine, layers: List[int]):
        self.model_engine = model_engine
        self.layers = layers
        self.stored_teacher_attns = {}
        self.stored_vfirst_state = {}
        
    @contextlib.contextmanager
    def temporarily_remove_teacher_attn(self):
        """
        上下文管理器，临时移除所有层的teacher_attn,v_first_state并在退出时恢复
        """
        try:
            # 保存并移除所有teacher_attn
            for layer_idx in self.layers:
                attention_wrapper = self.model_engine.module.model.model.layers[layer_idx].self_attn
                if hasattr(attention_wrapper, 'teacher_attn'):
                    self.stored_teacher_attns[layer_idx] = attention_wrapper.teacher_attn
                    # 移除teacher_attn模块
                    if hasattr(attention_wrapper, '_modules') and 'teacher_attn' in attention_wrapper._modules:
                        del attention_wrapper._modules['teacher_attn']
                    attention_wrapper.teacher_attn = None
                if hasattr(attention_wrapper, 'v_first_state'):
                    self.stored_vfirst_state[layer_idx] = attention_wrapper.v_first_state
                    attention_wrapper.v_first_state = None
            
            yield  # 允许在此上下文中执行代码
            
        finally:
            # 恢复所有teacher_attn
            for layer_idx, stored_attn in self.stored_teacher_attns.items():
                attention_wrapper = self.model_engine.module.model.model.layers[layer_idx].self_attn
                attention_wrapper.teacher_attn = stored_attn
                # 重新注册为子模块
                if hasattr(attention_wrapper, 'add_module') and not hasattr(attention_wrapper, 'teacher_attn'):
                    attention_wrapper.add_module("teacher_attn", stored_attn)
                v_first_state = self.stored_vfirst_state.get(layer_idx, None)
                if v_first_state is not None:
                    attention_wrapper.v_first_state = v_first_state
            # 清空存储的引用
            self.stored_teacher_attns.clear()
import ctypes
def force_cpu_memory_cleanup():
    """
    CPUメモリを強制的にクリア
    """
    print("CPUメモリクリーンアップ開始...")
    
    # 1. PyTorchのガベージコレクション
    gc.collect()
    
    # 2. PyTorchの内部キャッシュをクリア
    torch.cuda.empty_cache()  # GPU側
    
    # 3. CPUメモリプールも解放を試行
    if hasattr(torch, '_C') and hasattr(torch._C, '_cuda_emptyCache'):
        torch._C._cuda_emptyCache()
    
    # 4. より積極的なガベージコレクション
    for _ in range(3):
        collected = gc.collect()
        print(f"  GC回収: {collected} オブジェクト")
    
    # 5. 低レベルメモリ操作（Linux/Mac）
    try:
        libc = ctypes.CDLL("libc.so.6")  # Linux
        libc.malloc_trim(0)
        print("  malloc_trim実行完了")
    except:
        try:
            libc = ctypes.CDLL("libc.dylib")  # Mac
            libc.malloc_trim(0)
            print("  malloc_trim実行完了")
        except:
            print("  malloc_trim未対応")
    
    print("CPUメモリクリーンアップ完了")
if __name__ == '__main__':
    parser = create_arg_parser()
    args = parser.parse_args()
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    print(args)
    # if args.num_nodes > 1:
    deepspeed.init_distributed()
        # setup_distributed()
    # 加载配置
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)

    # 设置设备和数据类型
    dtype = torch.bfloat16
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    DeviceID = f'cuda:{args.local_rank}'
    args.DeviceID = DeviceID

    print(DeviceID)
    #exit()
    
    # 加载模型和分词器
    transformer_model = AutoModelForCausalLM.from_pretrained(config['Llama']['model_id'],
                                                             torch_dtype=dtype, device_map='cpu',low_cpu_mem_usage=True,attn_implementation="sdpa")
    

    args.freeze_attention = config['freeze_attention']
    args.hybrid_attention_layers = config['hybrid_attention_layers']
    args.freeze_hybrid_attention = config['freeze_hybrid_attention']
    args.allow_quant_frozen_layers = config['allow_quant_frozen_layers']
    args.quant_mode = config['quant_mode']
    args.peftmode = config['peftmode']
    args.peft_r = config['peft_r']
    args.peft_scaling = config['peft_scaling']
    args.peft_dropout = config['peft_dropout']
    args.mlp_quant_mode = config['mlp_quant_mode']
    args.bnb_optimizer_mode = config['bnb_optimizer_mode']
    args.transformer_layers = config['RWKV']['transformer_layers']
    
    tokenizer = AutoTokenizer.from_pretrained(config['Llama']['model_id'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 设置参数
    args.my_pos_emb = 0
    args.head_size_divisor = 8
    args.ctx_len = 4096
    args.n_layer = transformer_model.config.num_hidden_layers
    args.n_embd = transformer_model.config.hidden_size
    args.config = transformer_model.config
    
    args.dim_ffn = transformer_model.config.intermediate_size
    args.num_attention_heads = transformer_model.config.num_attention_heads
    args.num_key_value_heads = transformer_model.config.num_key_value_heads
    args.num_key_value_heads = transformer_model.config.num_key_value_heads
    args.rms_norm_eps = transformer_model.config.rms_norm_eps
    args.head_size_a = getattr(transformer_model.config, 'head_dim', transformer_model.config.hidden_size // transformer_model.config.num_attention_heads)
    args.dim_att = transformer_model.config.num_attention_heads * args.head_size_a
    args.is_attention_bias = getattr(transformer_model.config, 'attention_bias', True)
    args.is_attention_output_bias = getattr(transformer_model.config, 'attention_output_bias', False)
    args.pre_ffn = 0
    args.head_qk = 0
    args.tiny_att_dim = 0
    args.tiny_att_layer = -999
    args.vocab_size = transformer_model.config.vocab_size
    args.layers = config['RWKV']['layers']
    args.pad_id = tokenizer.eos_token_id
    args.betas = (args.beta1, args.beta2)
    args.kl_weight = config['kl_weight']
    args.ce_weight = config['ce_weight']
    args.enable_AKL = config.get('enable_AKL', False)
    args.model_file = config['model_file']
    args.real_bsz = args.train_batch_size
    args.is_sft = config.get('is_sft', False)
    args.is_all_labels_kl = config.get('is_all_labels_kl', False)



    if args.bnb_optimizer_mode > 0:
        args.deepspeed_stage = 1
        args.deepspeed_offload = False

    os.environ["RWKV_HEAD"] = str(int(args.n_embd // args.head_size_a))
    os.environ["RWKV_HEAD_SIZE_A"] = str(int(args.head_size_a))
    os.environ["RWKV_MIRCO_BSZ"] = str(int(args.micro_bsz))

    os.environ['RWKV_ATTN_PEFTMODE'] = str(args.peftmode)
    os.environ['RWKV_ATTN_QUANT'] = str(args.quant_mode)
    os.environ['RWKV_ATTN_PEFT_R'] = str(args.peft_r)
    os.environ['RWKV_ATTN_PEFT_SCALING'] = str(args.peft_scaling)
    os.environ['RWKV_ATTN_PEFT_DROPOUT'] = str(args.peft_dropout)

    from hybrid_model import HybridModel,VFirstHolder,remove_original_weights_for_lora_bone ,KFirstHolder

    # 初始化混合模型
    #if args.stage == 1:
    teacher_attn_module_list = torch.nn.ModuleList()
    for layer_idx in range(transformer_model.config.num_hidden_layers):
        llama_layer = transformer_model.model.layers[layer_idx]
        teacher_attn_module_list.append(llama_layer.self_attn)
    for n,p in teacher_attn_module_list.named_parameters():
        p.requires_grad = False
    model = HybridModel(transformer_model, args, tokenizer)
    
    pname = 'model.model.layers.15.self_attn.student_attn.receptance.weight'
    
    #model = quantize_and_replace_with_wrapper(model, patterns=["mlp"], threshold=0)
    #teacher_attn_module_list = quantize_and_replace_with_wrapper(teacher_attn_module_list, patterns=["mlp"], threshold=0)
    model = quantize_mlp_layers_properly(model,args,device=DeviceID)

    #model = model.to(dtype=torch.bfloat16,device=DeviceID)

    force_cpu_memory_cleanup()

    

    # model = model.to(device=DeviceID)

    for name,param in model.named_parameters():
        #if 'mlp' in name:
            print(f'name = {name} {param.dtype} {param.device}')
    #exit()
    gc.collect()
    torch.cuda.empty_cache()

    #time.sleep(30)
    
    def SearchTensor(model,keyname):
        for name,param in model.named_parameters():
            if keyname in name:
                return param
        return None
    
    weight_mul_r = 1.0
    weight_mul_k = 1.0
    weight_mul_v = 1.0
    weight_mul_o = 1.0 
    with torch.no_grad():
        for i in range(args.n_layer):
            print(f'layer = {i} transfer to student')
            for name,param in model.named_parameters():
                #print(name)
                if f'model.layers.{i}.self_attn.student_attn' in name:
                    if 'receptance.weight' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.q_proj.weight')
                        if s != None:
                            if s.shape == param.shape:
                                #param = s.clone()
                                param.copy_(s*weight_mul_r)
                                #print(param)
                                print('param copied from teacher')
                                #exit()
                            else:
                                print('shape is not same')
                        else:
                            print('not found')
                    elif 'receptance.bias' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.q_proj.bias')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_r)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')

                    if 'key.weight' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.k_proj.weight')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_k)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')
                    elif 'key.bias' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.k_proj.bias')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_k)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')

                    
                    if 'value.weight' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.v_proj.weight')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_v)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')
                    elif 'value.bias' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.v_proj.bias')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_v)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')


                    if 'output.weight' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.o_proj.weight')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_o)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')
                    elif 'output.bias' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.o_proj.bias')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_o)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')




                    if 'q_proj.weight' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.q_proj.weight')
                        if s != None:
                            if s.shape == param.shape:
                                #param = s.clone()
                                param.copy_(s*weight_mul_r)
                                #print(param)
                                print('param copied from teacher')
                                #exit()
                            else:
                                print('shape is not same')
                        else:
                            print('not found')
                    elif 'q_proj.bias' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.q_proj.bias')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_r)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')

                    if 'k_proj.weight' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.k_proj.weight')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_k)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')
                    elif 'k_proj.bias' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.k_proj.bias')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_k)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')

                    
                    if 'v_proj.weight' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.v_proj.weight')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_v)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')
                    elif 'v_proj.bias' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.v_proj.bias')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_v)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')


                    if 'o_proj.weight' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.o_proj.weight')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_o)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')
                    elif 'o_proj.bias' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.o_proj.bias')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_o)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')
                    









                    
                    if 'r_norm.weight' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.q_norm.weight')
                        if s != None:
                            if s.shape == param.shape:
                                #param = s.clone()
                                param.copy_(s)
                                #print(param)
                                print('param copied from teacher')
                                #exit()
                            else:
                                print('shape is not same')
                        else:
                            print('not found')
                    if 'q_norm.weight' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.q_norm.weight')
                        if s != None:
                            if s.shape == param.shape:
                                #param = s.clone()
                                param.copy_(s)
                                #print(param)
                                print('param copied from teacher')
                                #exit()
                            else:
                                print('shape is not same')
                        else:
                            print('not found')
                    if 'k_norm.weight' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.k_norm.weight')
                        if s != None:
                            if s.shape == param.shape:
                                #param = s.clone()
                                param.copy_(s)
                                #print(param)
                                print('param copied from teacher')
                                #exit()
                            else:
                                print('shape is not same')
                        else:
                            print('not found')
    
    if args.local_rank == 0:
        print(model)
        # 打印几个关键参数的统计信息
        #print parameter:model.model.layers.27.self_attn.student_attn.ln_x.weight
        for name, param in model.named_parameters():
            if name == pname:
                mean_of_param = param.mean().item()
                std_of_param = param.std().item()
                print(f"Parameter {name}: mean={mean_of_param:.6f}, std={std_of_param:.6f}")
    # 设置模型参数的训练状态
    if args.stage == 2 or args.stage == 3:#3 means sft
        print('all params are trainable')
        print(f'freeze mlp is {args.freeze_mlp}')

        # チェックポイントをロード
        checkpoint = torch.load(args.ckpt_file, map_location='cpu', mmap=True)

        # モデル形式に応じてstate_dictを取得
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # 指定したモジュールを含まないパラメータのみをフィルタリング
        filtered_state_dict = {}
        # for key, value in state_dict.items():
        #     filtered_state_dict[key] = value.to(dtype=torch.bfloat16)
        #     print(f'{key} {filtered_state_dict[key].dtype}')
        #     # embed、lm_head、normを含むキーをスキップ
        #     if not any(excluded in key for excluded in ["embed", "lm_head", ".norm."]):
        #          filtered_state_dict[key] = value.to(dtype=torch.bfloat16)
        #          print(f'{key} {filtered_state_dict[key].dtype}')
        #     else:
        #          print(f'{key} {value.dtype}')

        for key in state_dict.keys():
            if not any(excluded in key for excluded in ["embed", "lm_head", ".norm.","mlp.weight","qweight",'scales']):
                # この時点でのみCPUメモリにロード
                value = state_dict[key]
                filtered_state_dict[key] = value.to(dtype=torch.bfloat16)
                print(f'{key} converted to bfloat16')
            else:
                print(f'{key} skipped')
            

        # フィルタリングしたパラメータをロード（strict=Falseで欠けているパラメータを許容）
        model.load_state_dict(filtered_state_dict, strict=False)

        


        # 除外したパラメータ数と残ったパラメータ数を出力
        print(f"除外したパラメータ: {len(state_dict) - len(filtered_state_dict)}")
        print(f"ロードしたパラメータ: {len(filtered_state_dict)}")

        del checkpoint
        del filtered_state_dict
        del state_dict
        gc.collect()

        if args.grad_cp == 1:
            print('enable gradient checkpointing')
            model.model.gradient_checkpointing_enable()


        
        for name, param in model.named_parameters():
            #print(name)
            Attention = 0
            for i in range(args.n_layer):
                t = f'layers.{i}.'
                if t in name and i in args.transformer_layers:
                    Attention = 1
                    break
                elif t in name:
                    Attention = 0
                    break
            if Attention == 0 and args.freeze_attention and ('receptance' in name or 'key' in name or 'value' in name):
                param.requires_grad = False
                print(f'{name} Frozen')
            elif Attention==1 and args.freeze_hybrid_attention and ('self_attn.student_attn' in name and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'o_proj' in name or 'q_norm' in name or 'k_norm' in name)):
                param.requires_grad = False
                print(f'{name} Frozen')
            elif args.freeze_mlp and 'mlp' in name and 'lora' not in name:
                param.requires_grad = False
                print(f'{name} Frozen')
            else:
                param.requires_grad = True
                print(f'{name} will Train')
        #exit()
        for name, param in model.named_parameters():
            if   'lm_head' in name or '.norm.' in name:
                param.requires_grad = False
                print(f'frozen {name}')
            # if   'emb' in name:
            #     param.requires_grad = False
            #     print(f'frozen {name}')

        lora_base_modules = set()
        if args.peftmode != 'full':
            print('freeze original weight if peft linears')
            
            # まず、LoRAモジュールを持つベースモジュール名を収集
            
            for name, param in model.named_parameters():
                if 'lora_A' in name or 'lora_B' in name:
                    # "blocks.0.att.receptance.lora_A.weight" -> "blocks.0.att.receptance"
                    base_module_name = name.rsplit('.lora_', 1)[0]
                    lora_base_modules.add(base_module_name)
                    # LoRAパラメータ自体は学習可能にする
                    param.requires_grad = True
                    print(f'{name} LoRA param - will train!')

                elif 'bone' in name:
                    # "blocks.0.att.receptance.lora_A.weight" -> "blocks.0.att.receptance"
                    base_module_name = name.rsplit('.bone', 1)[0]
                    lora_base_modules.add(base_module_name)
                    # LoRAパラメータ自体は学習可能にする
                    param.requires_grad = True
                    print(f'{name} Bone param - will train!')
            
            # LoRAモジュールに対応する元のweight/biasをフリーズ
            for name, param in model.named_parameters():
                for base_module in lora_base_modules:
                    # 元のweightをフリーズ
                    if name == f"{base_module}.weight":
                        param.requires_grad = False
                        print(f'{name} Frozen (has LoRA)')
                    # biasが存在する場合はフリーズ
                    elif name == f"{base_module}.bias":
                        param.requires_grad = False
                        print(f'{name} Frozen (has LoRA)')

        # 最終的な学習可能パラメータの確認
        print("\n=== Final trainable parameters ===")
        trainable_params = 0
        total_params = 0
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"  {name}: {param.shape}")


        gc.collect()
        torch.cuda.empty_cache()

        print(f"\nTrainable params: {trainable_params:,} / Total params: {total_params:,}")
        print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
        print(f'current gpu memory BEFORE quant: {torch.cuda.memory_summary(device=None, abbreviated=False)}')

        #exit()

        #model.dummy_param = torch.nn.Parameter(torch.zeros(1, requires_grad=True))

        

        #exit()
    else:
        if args.ckpt_file is not None:
            model.load_check_point(args.ckpt_file)  
        print(f'Only self_attn params are trainable')
        for name,param in model.named_parameters():

            if 'self_attn.student_attn' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False


    #exit()

    # 准备数据加载器
    if args.preprocessed_data is not None:
        print(f'load preprocessed data from {args.preprocessed_data}')
        from data.multi_source_datasets import data_collator_with_pad 
        from functools import partial
        from torch.utils.data.distributed import DistributedSampler
        pad_token_id = tokenizer.pad_token_id
        data_collator = partial(data_collator_with_pad, max_seq_length=args.max_seq_length,pad_token_id=pad_token_id)
        
        # 加载所有训练集数据
        train_datasets = []
        for data_path in args.preprocessed_data:  # 最后一个路径作为验证集
            ds = datasets.load_from_disk(data_path)
            train_datasets.append(ds)
        
        # 合并所有训练集
        train_ds = datasets.concatenate_datasets(train_datasets)
        
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=args.world_size,
            rank=args.local_rank,
            shuffle=True
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_ds, 
            batch_size=args.micro_bsz, 
            sampler=train_sampler,  # 使用分布式 sampler
            num_workers=1, 
            pin_memory=True, 
            drop_last=True, 
            collate_fn=data_collator
        )
        val_dataloader = None
        if args.local_rank == 0:
            print(f'load preprocessed data from {args.preprocessed_data} done')
    elif args.raw_data is not None:
        print(f'load raw data from {args.raw_data}')
        from data.raw_dataset import load_datasets_from_directories,TypedDataset,TypedStreamingCLMDataCollator
        all_ds,feature_types = load_datasets_from_directories(args.raw_data,tokenizer)
        typed_dataset = TypedDataset(all_ds, feature_types)
        # print(all_ds)
        # con_ds = datasets.concatenate_datasets(all_ds)
        # data_collator = StreamingCLMDataCollator(tokenizer=tokenizer, max_length=args.max_seq_length)
        data_collator = TypedStreamingCLMDataCollator(tokenizer=tokenizer, 
                                                  max_length=args.max_seq_length, 
                                                  min_length=args.max_seq_length, 
                                                  typed_dataset=typed_dataset,
                                                  need_to_pad=args.need_to_pad)
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            typed_dataset,
            num_replicas=args.world_size,
            rank=args.local_rank,
            shuffle=True
        )
        train_dataloader = torch.utils.data.DataLoader(
            typed_dataset, 
            batch_size=args.micro_bsz, 
            sampler=train_sampler,  # 使用分布式 sampler
            num_workers=4, 
            pin_memory=True, 
            drop_last=True, 
            collate_fn=data_collator
        ) 
        val_dataloader = None
        # if args.local_rank == 0:
        #     print(f'load preprocessed data from {args.raw_data} done') 

    # 设置DeepSpeed配置
    if args.deepspeed:
        if args.deepspeed_config:
            # 如果提供了 DeepSpeed 配置文件，直接加载它
            with open(args.deepspeed_config, 'r') as f:
                ds_config = json.load(f)
        else:
            # 否则，根据命令行参数创建配置
            if args.deepspeed_offload:
                ds_config = {
                    "distributed_backend": "rccl",
                    "train_batch_size": args.train_batch_size,
                    "bf16": {
                        "enabled": True
                    },
                    "fp32_reduce_scatter": True,
                    "zero_optimization": {
                    # "ignore_frozen_weights": True,
                        "stage": args.deepspeed_stage,
                        
                        "offload_optimizer": {
                            "device": "cpu",
                            "pin_memory": False,
                            "buffer_count": 1,
                            'ratio':1.0
                        },
                     
                        "allgather_partitions": True,
                        "sub_group_size": 1e7,
                        "overlap_comm": False,
                        "reduce_scatter": True,
                        "reduce_bucket_size": 1e6,
                        "contiguous_gradients": False
                    },
                    # "compile": {
                    #     "deepcompile": True,
                    # },
                    "gradient_clipping": args.gradient_clip_val,
                    "gradient_checkpointing": args.grad_cp == 1,
                    # "activation_checkpointing": {
                    #     "partition_activations": True,
                    #     "cpu_checkpointing": False,  # CPUチェックポイントをオフ
                    #     "contiguous_memory_optimization": True,  # メモリ最適化は有効
                    #     "number_checkpoints": None,
                    #     "synchronize_checkpoint_boundary": False,
                    #     "profile": False
                    # },
                    #"grad_accum_dtype": "bf16",
                    "zero_force_ds_cpu_initialization": True,
                    "zero_allow_untested_optimizer": True,
                    "gradient_accumulation_steps": args.accumulate_grad_batches if args.accumulate_grad_batches > 1 else None,
                    "wall_clock_breakdown": False,
                    "dump_state": True
                }
            else:
                ds_config = {
                    "distributed_backend": "rccl",
                    "train_batch_size": args.train_batch_size,
                    "bf16": {
                        "enabled": True
                    },
                  #  "fp32_reduce_scatter": True,
                    "zero_optimization": {
                    # "ignore_frozen_weights": True,
                        "stage": args.deepspeed_stage,

                    #    "allgather_partitions": True,
                        #"sub_group_size": 1e7,
                        "overlap_comm": True,
                     #   "reduce_scatter": True,
                        #"reduce_bucket_size": 1e6,
                        "contiguous_gradients": False
                    },
                    # "compile": {
                    #     "deepcompile": True,
                    # },
                    "gradient_clipping": args.gradient_clip_val,
                    "gradient_checkpointing": args.grad_cp == 1,
                    #"grad_accum_dtype": "bf16",
                    "zero_allow_untested_optimizer": True,
                    "gradient_accumulation_steps": args.accumulate_grad_batches if args.accumulate_grad_batches > 1 else None,
                   # "wall_clock_breakdown": False,
                   # "dump_state": True
                }
        if not args.deepspeed_offload:
            ds_config['zero_optimization']['offload_optimizer'] = None
            ds_config['zero_optimization']['offload_param'] = None
        # 手动配置优化器
        print(f'configuring optimizer with args {args}')

        if args.quant_mode != "none":
            for name, m in model.named_modules():
                Attention = 0
                for i in range(args.n_layer):
                    t = f'layers.{i}.'
                    if t in name and i in args.transformer_layers:
                        Attention = 1
                        break
                    elif t in name:
                        Attention = 0
                        break
                #print(f'{name} {param.dtype}')
                if Attention == 0 and args.freeze_attention and ('self_attn.student_attn' in name and ('receptance' in name or 'key' in name or 'value' in name)):
                    if hasattr(m, "quant") and callable(getattr(m, "quant")):
                        m.quant(args.quant_mode,DeviceID)
                        #print(f'{name} Quant on {DeviceID}. frozen RWKV')
                elif Attention == 1 and args.freeze_hybrid_attention and ('self_attn.student_attn' in name and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'o_proj' in name or 'q_norm' in name or 'k_norm' in name)):
                    if hasattr(m, "quant") and callable(getattr(m, "quant")):
                        m.quant(args.quant_mode,DeviceID)
                        #print(f'{name} Quant on {DeviceID} frozen GQA')
                else:
                    for base_module in lora_base_modules:
                        if name == f"{base_module}" and hasattr(m, "quant") and callable(getattr(m, "quant")):
                            m.quant(args.quant_mode,DeviceID)
                            #print(f'{name} Quant on {DeviceID} train peft')

        for name, m in model.named_parameters():
            print(f'{name} requires_grad = {m.requires_grad}')

        


        if args.quant_mode != 'none':
            model = remove_original_weights_for_lora_bone(model)




        model=model.to(device=DeviceID)
        optimizer = configure_optimizer_stage2(model, args)
        #exit()
        if args.local_rank == 0:
            print(f'optimizer is {optimizer}')
            num_total_params = sum(p.numel() for p in model.parameters())
            num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # for n, p in model.named_parameters():
            #     if p.requires_grad:
            #         print(f'param {n} is trainable')
            print(f'num_total_params: {num_total_params}, num_trainable_params: {num_trainable_params}, percent: {num_trainable_params / num_total_params * 100:.2f}%')
            #print current gpu memory
            print(f'current gpu memory BEFORE initializing deepspeed: {torch.cuda.memory_summary(device=None, abbreviated=False)}')
            # model.model = torch.compile(model.model,fullgraph=True)
            # 初始化 DeepSpeed
            print(f'initializing deepspeed with config {ds_config}')
        
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,  
            optimizer=optimizer,
            config=ds_config
        )
        del model
        del transformer_model
        del optimizer
        del teacher_attn_module_list
        gc.collect()
        torch.cuda.empty_cache()
        
        #print('sleep')
        #time.sleep(30)
        #exit()
        
        # 添加验证代码
        for name, param in model_engine.module.named_parameters():
            if name == pname:
                with deepspeed.zero.GatheredParameters(param):
                    if args.local_rank == 0:  # 只在 rank 0 打印
                        print(f"Parameter {name}:")
                        print(f"  - mean: {param.mean().item():.6f} versus {mean_of_param:.6f}")
                        print(f"  - std: {param.std().item():.6f} versus {std_of_param:.6f}")
                    break
            
            
        #we only init  teacher related stuff when is_sft is False
        #init the VFirstHolder with (B,T,C) shape
        #vfirst_holder = VFirstHolder(args.micro_bsz, args.max_seq_length, args.dim_att,model_engine.world_size,device=DeviceID)




        vfirst_holder = VFirstHolder(args.micro_bsz, args.max_seq_length,args.num_key_value_heads,args.head_size_a,device=DeviceID)
        vfirst_holder.requires_grad_(False)

        kfirst_holder = KFirstHolder(args.micro_bsz, args.max_seq_length,args.num_key_value_heads,args.head_size_a,device=DeviceID)
        kfirst_holder.requires_grad_(False)

        #if cxa079 is no need v_first
        # vfirst_holder = VFirstHolder(args.micro_bsz, 1, args.dim_att,device=DeviceID)
        # vfirst_holder.requires_grad_(False)


        #Zero 2 will hold the model in one GPU process
        print(f'Zero 2 will hold the model in one GPU process,set the vfirst_holder to model_engine')
        for layer_idx in args.layers:
            attn_wrapper = model_engine.module.model.model.layers[layer_idx].self_attn
            attn_wrapper.v_first_state = vfirst_holder
            attn_wrapper.k_first_state = kfirst_holder
        timer.initialize_with_engine(model_engine)
        #print current gpu memory
        if args.local_rank == 0:
            print(f'current gpu memory AFTER initializing deepspeed: {torch.cuda.memory_summary(device=None, abbreviated=False)}')
        if args.stage == 2 and args.is_sft == False:
            if args.local_rank == 0:
                print(f'initializing teacher model')
                print(f'current gpu memory BEFORE initializing teacher model: {torch.cuda.memory_summary(device=None, abbreviated=False)}')
     
            teacher_model_id = args.teacher_model_id
            if teacher_model_id is None:
                teacher_model_id = config['Llama']['model_id']
            print(f'initializing teacher model with id {teacher_model_id}')

            # Int8量子化設定
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,  # 4bitではなく8bitに変更
                int8_threshold=6.0,  # Int8量子化の閾値（デフォルト: 6.0）
                llm_int8_has_fp16_weight=False,  # FP16の重みを保持しない
                llm_int8_enable_fp32_cpu_offload=False  # CPU offloadを無効化
            )

            teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_model_id,
                quantization_config=quantization_config,  # 量子化設定を有効化
                torch_dtype=torch.bfloat16,  # Int8でも計算時の型指定は必要
                device_map=DeviceID,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa"
            )

            teacher_model.eval()

            teacher_model = torch.compile(teacher_model)
            if args.local_rank == 0:
                print('freeze teacher_model')
                print(f'teacher_model is {teacher_model}')
            for name, param in teacher_model.named_parameters():
                param.requires_grad = False

            teacher_engine = teacher_model
      
            if args.local_rank == 0:
                print(f'current gpu memory AFTER initializing teacher model: {torch.cuda.memory_summary(device=None, abbreviated=False)}')
                # 将处理好的teacher model设置到model_engine中
                # model_engine.module.set_teacher_model(teacher_engine.module)
                print(f'current gpu memory AFTER setting teacher model: {torch.cuda.memory_summary(device=None, abbreviated=False)}')
            # 清理不需要的引用
            #del teacher_model
            #teacher_engine=teacher_model
            torch.cuda.empty_cache()

        else:
            #Other stage we don't need teacher model
            #SFT or DPO
            teacher_engine = None
    else:
        # 如果不使用 DeepSpeed，使用普通的优化器
        print('not using deepspeed, EXIT')
        exit()
    # 初始化NCCL组
    # model.client = initialize_nccl_client(args)

    # 只在主进程上初始化wandb
    if args.wandb and model_engine.global_rank == 0:
        print(f'init wandb, project is {args.wandb}, name is {args.run_name}')
        wandb.init(project=args.wandb, name=args.run_name, config=args)
        print(f'begin training with {args.max_epochs} epochs')
    # 初始化一些变量
    args.epoch_steps = len(train_dataloader) // (args.accumulate_grad_batches)
    global_step = 0
    last_log_time = time.time()
    token_per_step = args.max_seq_length * args.micro_bsz * args.world_size

    # 训练循环
    # 创建管理器实例
    terminate = False
    teacher_attn_manager = TeacherAttnManager(model_engine, args.layers)
    for epoch in range(args.max_epochs):
        model_engine.train()
        if model_engine.global_rank == 0:
            pbar = tqdm(total=args.epoch_steps, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(train_dataloader):
            
            lr, wd_now = on_train_batch_start(args, model_engine, global_step, epoch)

            batch = {k: v.to(model_engine.device) for k, v in batch.items()}
            
            # 前向传播
            loss, teacher_loss, kl_loss, student_cross_entropy_loss = train_step(model_engine, batch, args, teacher_engine, tokenizer)
            
            #CAUTION: The v_first will NEVER be synchronized for first batch. Just treat it as an outlier.

            model_engine.backward(loss)

            is_accumulation_step = (batch_idx + 1) % args.accumulate_grad_batches == 0

            if is_accumulation_step:
               global_step += 1
            model_engine.step()

            # 每一步都调用 on_train_batch_end，但只在累积步骤结束时更新进度条
            last_log_time, pbar = on_train_batch_end(
                args, batch_idx, model_engine,teacher_engine, loss.item(), teacher_loss, kl_loss, student_cross_entropy_loss,
                global_step, epoch, last_log_time, token_per_step, is_accumulation_step, pbar
            )

            if trained_tokens >= args.max_trained_tokens:
                terminate = True
                break
        



        # 保存检查点
        if args.output_dir:
            if args.deepspeed:
                
                # 在保存检查点的代码处使用上下文管理器
                with teacher_attn_manager.temporarily_remove_teacher_attn():
                    try:
                        print(f"Saving checkpoint to {args.output_dir} at epoch {epoch} rank {model_engine.global_rank}")
                        model_engine.save_checkpoint(args.output_dir, f"checkpoint-epoch{epoch}")
                    except Exception as e:
                        print(f"Error saving checkpoint: {e}")
                        import traceback
                        traceback.print_exc()
                
                # if args.local_rank == 0:
                #     print(f'saving epoch checkpoint to {args.output_dir}')
                # #temporarily set attention_wrapper's teacher_attn to None
                # # 遍历所有层
                # for layer_idx in args.layers:
                #     # 获取当前层的 AttentionWrapper
                #     if args.local_rank == 0:
                #         print(f'set teacher attn to None {layer_idx}')
                #     attention_wrapper = model_engine.module.model.model.layers[layer_idx].self_attn
                #     # 设置 teacher_attn to None
                #     attention_wrapper.teacher_attn = None
                # model_engine.save_checkpoint(args.output_dir, f"checkpoint-epoch{epoch}")
                # # 遍历所有层
                # for layer_idx in args.layers:
                #     # 获取当前层的 AttentionWrapper
                #     if args.local_rank == 0:
                #         print(f'set teacher attn for layer {layer_idx}')
                #     attention_wrapper = model_engine.m