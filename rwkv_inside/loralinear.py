#For Faster Convergence from RWKV-LM-RLHF
#GANBAROU

#2025 OpenMOSE
import torch
from torch.nn import functional as F
import functools
import torch.nn as nn
from einops import rearrange
import os, math, gc, importlib
from torch._lowrank import svd_lowrank
import bitsandbytes as bnb
from bitsandbytes.functional import QuantState
from bitsandbytes.autograd._functions import matmul as i8matmul
import time



def rwkv_quantize(quant_type, weight):
    #global NowCurrentlyGPUNo
    if quant_type=='4bit':
        weight= weight.to(torch.bfloat16)
        qweight, qstate= bnb.functional.quantize_4bit((weight.data))
    elif quant_type=='nf4':
        weight= weight.to(torch.bfloat16)
        qweight, qstate= bnb.functional.quantize_nf4((weight.data))
    elif quant_type=='fp4':
        weight= weight.to(torch.bfloat16)
        qweight, qstate= bnb.functional.quantize_fp4((weight.data))
    elif quant_type=='int8':
        qweight, qstate= bnb.functional.quantize((weight.data.to(dtype=torch.float32)))
    return qweight, qstate

def rwkv_dequantize(quant_type, weight, qstate):
        if quant_type=='4bit':
            deweight= bnb.functional.dequantize_4bit(weight.data,quant_state=qstate).to(torch.bfloat16)
        elif quant_type=='nf4':
            deweight= bnb.functional.dequantize_nf4(weight.data,quant_state=qstate).to(torch.bfloat16)
        elif quant_type=='fp4':
            deweight= bnb.functional.dequantize_fp4(weight.data,quant_state=qstate).to(torch.bfloat16)
        elif quant_type=='int8':
            deweight= bnb.functional.dequantize(weight.data,state=qstate).to(torch.bfloat16)
        return deweight

def LinearForward(self,x,passthrough = False):
    if self.is_quant:
            if self.bonemode: # Covered All quantize method. currently slow implementation
                #with torch.no_grad():
                temporal_weight = rwkv_dequantize(self.quant_type, self.Qweight, self.qstate)

                if passthrough:
                    return F.linear(x, temporal_weight)
                w = rearrange(temporal_weight, '(a r1) (b r2) -> a b r1 r2', r1 = self.r, r2 = self.r)@self.bone+self.bone
                w = rearrange(w, 'a b r1 r2 ->(a r1) (b r2) ')
                return x @ (w + temporal_weight).t()
            
            else: #LoRA
                #with torch.no_grad():
                w = rwkv_dequantize(self.quant_type, self.Qweight, self.qstate)
                if passthrough:
                    return F.linear(x, w)
                
                if self.doramode:
                    lora_weight = self.lora_B @ self.lora_A
                    weight_combined = w + self.scaling * lora_weight
                    norm = weight_combined.norm(dim=0, keepdim=True) + 1e-6
                    norm = norm.detach() 
                    W_eff = (self.lora_M * weight_combined) / norm  # shape: (out_features, in_features)
                    out = F.linear(x, W_eff)
                    return out

                if self.loramode:
                    #print('quant linear mode lora')
                    #with torch.no_grad():
                    BaseOutput = F.linear(x, w)
                    
                    return ( 
                            BaseOutput + self.scaling *
                            F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)) 
                #print('quant linear mode')
                return F.linear(x, w)
                
    else: # Non Quant mode
        if passthrough:
                return F.linear(x, self.weight)
        if self.bonemode:
            if passthrough:
                return F.linear(x, self.weight)
            w = rearrange(self.weight, '(a r1) (b r2) -> a b r1 r2', r1 = self.r, r2 = self.r)@self.bone+self.bone
            w = rearrange(w, 'a b r1 r2 ->(a r1) (b r2) ')
            return F.linear(x,self.weight+w)
        if self.doramode:
            lora_weight = self.lora_B @ self.lora_A
            weight_combined = self.weight + self.scaling * lora_weight
            norm = weight_combined.norm(dim=0, keepdim=True) + 1e-6
            norm = norm.detach()  
            W_eff = (self.lora_M * weight_combined) / norm 
            out = F.linear(x, W_eff)
            return out
        if self.loramode:
            #print('non quant linear mode lora')
            return (
                F.linear(x, self.weight) + self.scaling *
                F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)) 
        #print('non quant lienar')
        return F.linear(x, self.weight)

class LoraLinear(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, bias: bool,peftmode:str):
        super().__init__()

        #peftmode = str(os.environ["RWKV_ATTN_PEFTMODE"])
        #peftmode = int(os.environ["RWKV_ATTN_PEFTMODE"])
        rank = int(os.environ["RWKV_ATTN_PEFT_R"])
        scaling = float(os.environ["RWKV_ATTN_PEFT_SCALING"])
        dropout = float(os.environ["RWKV_ATTN_PEFT_DROPOUT"])


        self.peftmode = peftmode
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias == True:
            self.bias = nn.Parameter(torch.empty((out_features)))
            self.biasmode = True
        else:
            self.biasmode = False

        if self.peftmode == 'bone':
            self.r = rank
            self.bone = nn.Parameter(torch.zeros(in_features//self.r, self.r, self.r))
            self.bonemode = True
        elif self.peftmode == 'lora' or self.peftmode == 'dora':
            self.doramode = False
            if self.peftmode == 'dora':
                #DoRA: Weight-Decomposed Low-Rank Adaptation
                with torch.no_grad():
                    m_init = self.weight.norm(dim=0, keepdim=True)
                self.lora_M = nn.Parameter(m_init) #momemtum
                self.doramode = True
            r = int(rank)
            dropout = float(dropout)
            self.lora_A = nn.Parameter(torch.empty(r, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, r))
            self.lora_dropout = nn.Dropout(dropout)
            self.scaling = scaling
            self.r = r
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_A)
            nn.init.zeros_(self.lora_B)
            self.bonemode = False
            self.loramode = True 
        else:
            self.loramode = False
            self.doramode = False
            self.bonemode = False

        self.is_quant = False
    def dora_init(self):
        self.doramode = True
        if self.is_quant:
            temporal_weight = rwkv_dequantize(self.quant_type, self.Qweight, self.qstate)
            with torch.no_grad():
                m_init = temporal_weight.norm(dim=0, keepdim=True) #temporal_weight#.norm(p=2, dim=0, keepdim=True)
            self.lora_M = nn.Parameter(m_init) #momemtum
            print(self.lora_M)
        else:
            with torch.no_grad():
                    m_init = self.weight.norm(dim=0, keepdim=True)
            self.lora_M = nn.Parameter(m_init) #momemtum

    def quant(self, quant_type,target_gpu):
        self.is_quant = True
        self.quant_type = quant_type
        with torch.no_grad():
            #print(f'input weight = {self.weight}')
            #print(f'From Weight Device = {self.weight.device}')
            #self.tintinprpr = nn.Parameter(torch.zeros(1))
            print(f'quant type = {self.quant_type}')
            self.Qweight, self.qstate= rwkv_quantize(self.quant_type, self.weight.to(device=target_gpu))
            # weightをパラメータから削除
            # del self._parameters['weight'] 
            
            # # 属性も削除
            # if hasattr(self, 'weight'):
            #     delattr(self, 'weight')
            del self.weight

            #print(f'{self.Qweight.shape}')
            
           # print(f'quant weight = {self.Qweight}')
           # w = rwkv_dequantize(quant_type,self.Qweight,self.qstate)
            #print('quant')

           # print(f'diff = {self.weight - w}')

            #del self.weight# Because Latest Pytorch-lightning forced to BF16 type. thats why delete
            #exit()

    def forward(self, x,passthrough = False):
        if self.biasmode == True:
            return LinearForward(self,x,passthrough) + self.bias
        else:
            return LinearForward(self,x,passthrough)