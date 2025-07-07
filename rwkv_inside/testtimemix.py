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
        k = repeat_kv(k, self.num_key_value_groups)#later need reshape(B,T,-1) #(B,T,C)
        v = repeat_kv(v, self.num_key_value_groups)#later need reshape(B,T,-1) #(B,T,C)

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

        #attention_mask is dummy
        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a,self.head_size,attention_mask) 

        x = self.ln_x(x.view(B * T, C)).view(B, T, C) # Groupnorm is back!
 

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)

 
        x = self.output(x) 

        return x, v_first
    
