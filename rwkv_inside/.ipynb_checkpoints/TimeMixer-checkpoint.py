import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load
#HEAD_SIZE = 64
import sys
import os

parent_dir = os.path.dirname(os.path.abspath(__file__))
print(f'parent_dir: {parent_dir}')
is_wind_cuda = False

#from tritonbighead import RUN_CUDA_RWKV7g
from backstepping_longhead import RUN_CUDA_RWKV7g



    
class RWKV_Tmix_x070_Mose(torch.nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
   
        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd


        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

          

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            #D_DECAY_LORA = 64
            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            print(f'D_DECAY_LORA={D_DECAY_LORA}')
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = 64
            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            print(f'D_AAA_LORA={D_AAA_LORA}')
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            D_MV_LORA = 32
            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            print(f'D_MV_LORA={D_MV_LORA}')
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)


            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)


            # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
            # self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            # self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.output.weight.data.zero_()
            

    
    #@torch.compile
    def forward(self, x, v_first,attention_mask):
        B, T, C = x.size()
        #removed tokenshift
        H = self.n_head
        r = self.receptance(x)
        w = -F.softplus(-(self.w0 + torch.tanh(x @ self.w1) @ self.w2)) - 0.6 # soft-clamp to (-inf, -0.5) modified -0.5->-0.6
        k = self.key(x)
        v = self.value(x)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (x @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (x @ self.a1) @ self.a2) # a is "in-context learning rate"

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)
        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a,self.head_size,attention_mask)

        x = x.view(B, T, C)
        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x) 
        return x, v_first







class RWKV_Tmix_x070_Mose_v2(torch.nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
   
        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd


        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

          

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            #D_DECAY_LORA = 64
            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            print(f'D_DECAY_LORA={D_DECAY_LORA}')
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = 64
            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            print(f'D_AAA_LORA={D_AAA_LORA}')
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            D_MV_LORA = 32
            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            print(f'D_MV_LORA={D_MV_LORA}')
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)


            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!
            self.out_scale = nn.Parameter(torch.tensor(-11.5129))


            # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
            # self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            # self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.output.weight.data.zero_()
            

    
    #@torch.compile
    def forward(self, x, v_first,attention_mask):
        B, T, C = x.size()
        #removed tokenshift
        H = self.n_head
        r = self.receptance(x)
        w = -F.softplus(-(self.w0 + torch.tanh(x @ self.w1) @ self.w2)) - 0.6 # soft-clamp to (-inf, -0.5) modified -0.5->-0.6
        k = self.key(x)
        v = self.value(x)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (x @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (x @ self.a1) @ self.a2) # a is "in-context learning rate"

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)
        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a,self.head_size,attention_mask)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C) # Groupnorm is back!

        #x = x.view(B, T, C)
        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x) * (torch.sigmoid(self.out_scale)*1e5) #hopefully absorb MLP scale
        return x, v_first
    










def repeat_kv_(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

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


def is_nan(t, name):
    if torch.isnan(t).any():
        return True 
    else:
        return False
    # if torch.isnan(t).any():
    #     print(f"⚠️ NaN detected in {name}")
    # if torch.isinf(t).any():
    #     print(f"⚠️ Inf detected in {name}")
    # if t.abs().max() > 1e4:
    #     print(f"⚠️ Large value in {name}: max={t.abs().max().item()}")

def check_abs_max(t, name, threshold=1e4):
    max_abs = t.abs().max().item()
    print(f"[{name}] abs max: {max_abs:.4e}")
    if max_abs > threshold:
        print(f"⚠️ WARNING: {name} has large value (>{threshold}) → may cause NaN soon!")

class SafeGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-4):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps)

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        x = x.clamp(min=-1e4, max=1e4)
        out = self.norm(x)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return out
class RWKV_Tmix_x070_Mose_cxa073(torch.nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
   
        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size


        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads



        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd


        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.layer_architecture = nn.Parameter(torch.tensor(73.0))

        
            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            print(f'D_DECAY_LORA={D_DECAY_LORA}')
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            print(f'D_AAA_LORA={D_AAA_LORA}')
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            print(f'D_MV_LORA={D_MV_LORA}')
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)

            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))


            #Change to GQA Style
            self.receptance = nn.Linear(C, self.num_attention_heads * self.head_size, bias=self.args.is_attention_bias)
            self.key = nn.Linear(C, self.num_key_value_heads * self.head_size, bias=self.args.is_attention_bias)
            self.value = nn.Linear(C, self.num_key_value_heads * self.head_size, bias=self.args.is_attention_bias)
            self.output = nn.Linear(self.num_attention_heads * self.head_size, C, bias=self.args.is_attention_output_bias)
            #SafeGroupNorm
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!
            #self.ln_x = SafeGroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!
            #self.out_scale = nn.Parameter(torch.tensor(-11.5129))

            #later copy from teacher weights
            #maybe can copy from any GQA Models
            self.receptance.weight.data.zero_()
            self.key.weight.data.zero_()
            self.value.weight.data.zero_()
            self.output.weight.data.zero_()
           

    
    #@torch.compile
    def forward(self, x, v_first,attention_mask):
        B, T, C = x.size()
        #removed tokenshift
        H = self.n_head
        r = self.receptance(x)
        w = -F.softplus(-(self.w0 + torch.tanh(x @ self.w1) @ self.w2)) -0.6#- 0.6 # soft-clamp to (-inf, -0.5) modified -0.5->-0.6
        k = self.key(x)
        v = self.value(x)


        k = k.view(B, T, self.num_key_value_heads, self.head_size)
        v = v.view(B, T, self.num_key_value_heads, self.head_size)

        # repeat k/v heads if n_kv_heads < n_heads
        #modified repeat_kv B,T,H_kv,D) -> B,T,H,D -> B,T,C
        k = repeat_kv(k, self.num_key_value_groups)#reshape(B,T,-1) #(B,T,C)
        v = repeat_kv(v, self.num_key_value_groups)#reshape(B,T,-1) #(B,T,C)

        k = k.view(B, T, -1)
        v = v.view(B, T, -1)

        #so now all B,T,C tensors

        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (x @ self.v1) @ self.v2) # add value residual



        a = torch.sigmoid(self.a0 + (x @ self.a1) @ self.a2) # a is "in-context learning rate"

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        # check_abs_max(r, "r")
        # check_abs_max(w, "w")
        # check_abs_max(k, "k")
        # check_abs_max(v, "v")
        # check_abs_max(-kk, "-kk")
        # check_abs_max(kk * a, "kk*a")




        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a,self.head_size,attention_mask)# * 2



        #print(f'layerid = {self.layer_id} Afterwkv x = {is_nan(x,"x")}')
        x = self.ln_x(x.view(B * T, C)).view(B, T, C) # Groupnorm is back!
        #x = x.view(B,T,-1)

        #print(f'layerid = {self.layer_id} Afterln x = {is_nan(x,"x")}')

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)

        #print(f'layerid = {self.layer_id} Afterwkv sum = {is_nan(x,"x")}')
        x = self.output(x)# * (torch.sigmoid(self.out_scale)*1e5) #hopefully absorb MLP scale initial=1.0

        #print(f'layerid = {self.layer_id} r = {is_nan(r,"r")} w = {is_nan(w,"w")} k = {is_nan(k,"k")} v = {is_nan(v,"v")} kk = {is_nan(kk,"kk")} x = {is_nan(x,"x")}')
        return x, v_first
    






class RWKV_Tmix_x070_Mose_cxa074(torch.nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
   
        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size


        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads



        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd


        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.layer_architecture = nn.Parameter(torch.tensor(73.0))

        
            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            print(f'D_DECAY_LORA={D_DECAY_LORA}')
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            print(f'D_AAA_LORA={D_AAA_LORA}')
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            print(f'D_MV_LORA={D_MV_LORA}')
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)

            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))


            #Change to GQA Style
            self.receptance = nn.Linear(C, self.num_attention_heads * self.head_size, bias=self.args.is_attention_bias)
            self.key = nn.Linear(C, self.num_key_value_heads * self.head_size, bias=self.args.is_attention_bias)
            self.value = nn.Linear(C, self.num_key_value_heads * self.head_size, bias=self.args.is_attention_bias)
            self.output = nn.Linear(self.num_attention_heads * self.head_size, C, bias=self.args.is_attention_output_bias)
            #SafeGroupNorm
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!
            #self.ln_x = SafeGroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!
            #self.out_scale = nn.Parameter(torch.tensor(-11.5129))

            #later copy from teacher weights
            #maybe can copy from any GQA Models
            self.receptance.weight.data.zero_()
            self.key.weight.data.zero_()
            self.value.weight.data.zero_()
            self.output.weight.data.zero_()
           

    
    #@torch.compile
    def forward(self, x, v_first,attention_mask):
        B, T, C = x.size()

        
        #removed tokenshift
        H = self.n_head
        r = self.receptance(x)
        w = -F.softplus(-(self.w0 + torch.tanh((x) @ self.w1) @ self.w2)) - 0.6 # soft-clamp to (-inf, -0.5) modified -0.5->-0.6
        k = self.key(x)
        v = self.value(x)


        k = k.view(B, T, self.num_key_value_heads, self.head_size)
        v = v.view(B, T, self.num_key_value_heads, self.head_size)

        # repeat k/v heads if n_kv_heads < n_heads
        #modified repeat_kv B,T,H_kv,D) -> B,T,H,D -> B,T,C
        k = repeat_kv(k, self.num_key_value_groups)#reshape(B,T,-1) #(B,T,C)
        v = repeat_kv(v, self.num_key_value_groups)#reshape(B,T,-1) #(B,T,C)

        k = k.view(B, T, -1)
        v = v.view(B, T, -1)

        #so now all B,T,C tensors

        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (x @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (x @ self.a1) @ self.a2) # a is "in-context learning rate"

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)


        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a,self.head_size,attention_mask)
        #print(f'layerid = {self.layer_id} Afterwkv x = {is_nan(x,"x")}')

        x = self.ln_x(x.view(B * T, C)).view(B, T, C) # Groupnorm is back!
        #x = x.view(B,T,-1)



        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)

        #x = x * 2.0

        x = self.output(x)

        return x, v_first


class RWKV_Tmix_x070_Mose_cxa075(torch.nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
   
        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size


        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads



        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd


        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.layer_architecture = nn.Parameter(torch.tensor(73.0))

        
            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            print(f'D_DECAY_LORA={D_DECAY_LORA}')
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            print(f'D_AAA_LORA={D_AAA_LORA}')
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            print(f'D_MV_LORA={D_MV_LORA}')
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)

            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            #D_GATE_LORA = 128
            D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(torch.zeros(D_GATE_LORA, C))


            #Change to GQA Style
            self.receptance = nn.Linear(C, self.num_attention_heads * self.head_size, bias=self.args.is_attention_bias)
            self.key = nn.Linear(C, self.num_key_value_heads * self.head_size, bias=self.args.is_attention_bias)
            self.value = nn.Linear(C, self.num_key_value_heads * self.head_size, bias=self.args.is_attention_bias)
            self.output = nn.Linear(self.num_attention_heads * self.head_size, C, bias=self.args.is_attention_output_bias)
            #SafeGroupNorm
            #self.ln_x = nn.GroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!
            #self.ln_x = SafeGroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!
            #self.out_scale = nn.Parameter(torch.tensor(-11.5129))

            #later copy from teacher weights
            #maybe can copy from any GQA Models
            self.receptance.weight.data.zero_()
            self.key.weight.data.zero_()
            self.value.weight.data.zero_()
            self.output.weight.data.zero_()
           

    
    #@torch.compile
    def forward(self, x, v_first,attention_mask):
        B, T, C = x.size()
        #removed tokenshift
        H = self.n_head
        r = self.receptance(x)
        w = -F.softplus(-(self.w0 + torch.tanh(x @ self.w1) @ self.w2)) -0.5
        k = self.key(x)
        v = self.value(x)


        k = k.view(B, T, self.num_key_value_heads, self.head_size)
        v = v.view(B, T, self.num_key_value_heads, self.head_size)

        # repeat k/v heads if n_kv_heads < n_heads
        #modified repeat_kv B,T,H_kv,D) -> B,T,H,D -> B,T,C
        k = repeat_kv(k, self.num_key_value_groups)#reshape(B,T,-1) #(B,T,C)
        v = repeat_kv(v, self.num_key_value_groups)#reshape(B,T,-1) #(B,T,C)

        k = k.view(B, T, -1)
        v = v.view(B, T, -1)

        #so now all B,T,C tensors

        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (x @ self.v1) @ self.v2) # add value residual

        g_delta = torch.sigmoid(x @ self.g1) @ self.g2
        g = 1.0 + g_delta

        a = torch.sigmoid(self.a0 + (x @ self.a1) @ self.a2) # a is "in-context learning rate"

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        # check_abs_max(r, "r")
        # check_abs_max(w, "w")
        # check_abs_max(k, "k")
        # check_abs_max(v, "v")
        # check_abs_max(-kk, "-kk")
        # check_abs_max(kk * a, "kk*a")
        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a,self.head_size,attention_mask)# * 2

        #x = self.ln_x(x.view(B * T, C)).view(B, T, C) # Groupnorm is back!
        x = x.view(B,T,-1)

        #print(f'layerid = {self.layer_id} Afterln x = {is_nan(x,"x")}')

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)

        #print(f'layerid = {self.layer_id} Afterwkv sum = {is_nan(x,"x")}')
        x = self.output(x*g)# * (torch.sigmoid(self.out_scale)*1e5) #hopefully absorb MLP scale initial=1.0

        #print(f'layerid = {self.layer_id} r = {is_nan(r,"r")} w = {is_nan(w,"w")} k = {is_nan(k,"k")} v = {is_nan(v,"v")} kk = {is_nan(kk,"kk")} x = {is_nan(x,"x")}')
        return x, v_first
    
class Qwen3RMSNorm(nn.Module):
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


class RWKV_Tmix_x070_Mose_cxa076(torch.nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
   
        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size


        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.rms_norm_eps = args.rms_norm_eps

        self.RKNormMode = True



        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd


        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.layer_architecture = nn.Parameter(torch.tensor(73.0))

        
            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            print(f'D_DECAY_LORA={D_DECAY_LORA}')
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            print(f'D_AAA_LORA={D_AAA_LORA}')
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            print(f'D_MV_LORA={D_MV_LORA}')
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)

            #self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            #self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            #D_GATE_LORA = 128
            D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))


            #Change to GQA Style
            self.receptance = nn.Linear(C, self.num_attention_heads * self.head_size, bias=self.args.is_attention_bias)
            self.key = nn.Linear(C, self.num_key_value_heads * self.head_size, bias=self.args.is_attention_bias)
            self.value = nn.Linear(C, self.num_key_value_heads * self.head_size, bias=self.args.is_attention_bias)
            self.output = nn.Linear(self.num_attention_heads * self.head_size, C, bias=self.args.is_attention_output_bias)

            if self.RKNormMode == True:
                self.r_norm = Qwen3RMSNorm(self.head_size, eps=self.rms_norm_eps) 
                self.k_norm = Qwen3RMSNorm(self.head_size, eps=self.rms_norm_eps) 


            #later copy from teacher weights
            #maybe can copy from any GQA Models
            self.receptance.weight.data.zero_()
            self.key.weight.data.zero_()
            self.value.weight.data.zero_()
            self.output.weight.data.zero_()


    
           

    
    #@torch.compile
    def forward(self, x, v_first,attention_mask):
        B, T, C = x.size()
        #removed tokenshift
        H = self.n_head


        if self.RKNormMode == True:
            r = self.r_norm(self.receptance(x).view(B,T,H,-1))
            k = self.k_norm(self.key(x).view(B,T,self.num_key_value_heads,-1))
        else:
            r = self.receptance(x)
            k = self.key(x)

        
        w = -F.softplus(-(self.w0 + torch.tanh(x @ self.w1) @ self.w2)) -0.5
        
        v = self.value(x)


        k = k.view(B, T, self.num_key_value_heads, self.head_size)
        v = v.view(B, T, self.num_key_value_heads, self.head_size)

        # repeat k/v heads if n_kv_heads < n_heads
        #modified repeat_kv B,T,H_kv,D) -> B,T,H,D -> B,T,C
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        k = k.view(B, T, -1)
        v = v.view(B, T, -1)

        #so now all B,T,C tensors

        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (x @ self.v1) @ self.v2) # add value residual

        g = torch.sigmoid(x @ self.g1) @ self.g2

        a = torch.sigmoid(self.a0 + (x @ self.a1) @ self.a2) # a is "in-context learning rate"

        #kk = k * self.k_k removed k_k
        kk = F.normalize(k.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        #k = k * (1 + (a-1) * self.k_a)
        k = k * (1.0 - w + a)

        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a,self.head_size,attention_mask)
        x = x.view(B,T,-1)
        #Attention Scaling
        x = x * (self.head_size ** -0.5) 

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)

        x = self.output(x*g)

        #print(f'layerid = {self.layer_id} r = {is_nan(r,"r")} w = {is_nan(w,"w")} k = {is_nan(k,"k")} v = {is_nan(v,"v")} kk = {is_nan(kk,"kk")} x = {is_nan(x,"x")}')
        return x, v_first
    


    
# class RWKV_Tmix_x070_Mose(torch.nn.Module):
#     def __init__(self, args, layer_id):
#         super().__init__()
#         self.args = args
#         self.layer_id = layer_id
   
#         self.head_size = args.head_size_a
#         self.n_head = args.dim_att // self.head_size
#         assert args.dim_att % self.n_head == 0
#         H = self.n_head
#         N = self.head_size
#         C = args.n_embd


#         with torch.no_grad():
#             ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
#             ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
#             ddd = torch.ones(1, 1, C)
#             for i in range(C):
#                 ddd[0, 0, i] = i / C

          

#             def ortho_init(x, scale):
#                 with torch.no_grad():
#                     shape = x.shape
#                     if len(shape) == 2:
#                         gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
#                         nn.init.orthogonal_(x, gain=gain * scale)
#                     elif len(shape) == 3:
#                         gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
#                         for i in range(shape[0]):
#                             nn.init.orthogonal_(x[i], gain=gain * scale)
#                     else:
#                         assert False
#                     return x

#             #D_DECAY_LORA = 64
#             D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
#             print(f'D_DECAY_LORA={D_DECAY_LORA}')
#             self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
#             self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
#             decay_speed = torch.ones(C)
#             for n in range(C):
#                 decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
#             self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

#             D_AAA_LORA = 64
#             D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
#             print(f'D_AAA_LORA={D_AAA_LORA}')
#             self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
#             self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
#             self.a0 = nn.Parameter(torch.zeros(1,1,C))

#             D_MV_LORA = 32
#             D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
#             print(f'D_MV_LORA={D_MV_LORA}')
#             self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
#             self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
#             self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)


#             self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
#             self.k_a = nn.Parameter(torch.ones(1,1,C))
#             self.r_k = nn.Parameter(torch.zeros(H,N))

#             self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
#             self.receptance = nn.Linear(C, C, bias=False)
#             self.key = nn.Linear(C, C, bias=False)
#             self.value = nn.Linear(C, C, bias=False)
#             self.output = nn.Linear(C, C, bias=False)


#             # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
#             # self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
#             # self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
#             # self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
#             # self.output.weight.data.zero_()
            

    
#     #@torch.compile
#     def forward(self, x, v_first,attention_mask):
#         B, T, C = x.size()
#         #removed tokenshift
#         H = self.n_head
#         r = self.receptance(x)
#         w = -F.softplus(-(self.w0 + torch.tanh(x @ self.w1) @ self.w2)) - 0.6 # soft-clamp to (-inf, -0.5) modified -0.5->-0.6
#         k = self.key(x)
#         v = self.value(x)
#         if self.layer_id == 0:
#             v_first = v # store the v of the first layer
#         else:
#             v = v + (v_first - v) * torch.sigmoid(self.v0 + (x @ self.v1) @ self.v2) # add value residual
#         a = torch.sigmoid(self.a0 + (x @ self.a1) @ self.a2) # a is "in-context learning rate"

#         kk = k * self.k_k
#         kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
#         k = k * (1 + (a-1) * self.k_a)
#         x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a,self.head_size,attention_mask)

#         x = x.view(B, T, C)
#         x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
#         x = self.output(x) 
#         return x, v_first