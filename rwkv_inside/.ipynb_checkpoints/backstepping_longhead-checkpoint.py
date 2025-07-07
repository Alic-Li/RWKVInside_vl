import os, torch as th
from torch.utils.cpp_extension import load

print('backstepping longhead mode with CUDA or HIP')

CHUNK_LEN = 8

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])
HEAD = int(os.environ["RWKV_HEAD"])
BATCH_SIZE = int(os.environ["RWKV_MIRCO_BSZ"])

class RWKV7_longhead(th.autograd.Function):
    @staticmethod
    def forward(ctx, q,w,k,v,a,b,s0):
        B,T,H,C = w.shape
        assert T%CHUNK_LEN == 0
        if not th.compiler.is_compiling():
            assert hasattr(th.ops.wind_backstepping_longhead, 'forward'), 'Requires a load kernel from load_backstepping_longhead(head_size)'
            assert all(i.dtype==th.bfloat16 for i in [w,q,k,v,a,b,s0])
            assert all(i.is_contiguous() for i in [w,q,k,v,a,b,s0])
            assert all(i.shape == w.shape for i in [w,q,k,v,a,b])
            assert list(s0.shape) == [B,H,C,C]
        B,T,H,C = w.shape
        y = th.empty_like(v)
        sT = th.empty_like(s0)
        if any(i.requires_grad for i in [w,q,k,v,a,b,s0]):
            s = th.empty(B,H,T//CHUNK_LEN,C,C, dtype=th.float32,device=w.device)
            sa = th.empty(B,T,H,C, dtype=th.float32,device=w.device)
        else:
            s = sa = None
        th.ops.wind_backstepping_longhead.forward(w,q,k,v,a,b, s0,y,s,sa,sT)
        ctx.save_for_backward(w,q,k,v,a,b,s,sa)
        return y, sT
    @staticmethod
    def backward(ctx, dy, dsT):
        w,q,k,v,a,b,s,sa = ctx.saved_tensors
        B,T,H,C = w.shape
        if not th.compiler.is_compiling():
            assert all(i.dtype==th.bfloat16 for i in [dy,dsT])
            assert all(i.is_contiguous() for i in [dy,dsT])

        dv,ds0 = [th.empty_like(x) for x in [v,dsT]]
        dw,dq,dk,da,db = [th.zeros(B,T,H,C, device=w.device) for i in range(5)]
        th.ops.wind_backstepping_longhead.backward(w,q,k,v,a,b, dy,s,sa,dsT, dw,dq,dk,dv,da,db,ds0)
        return dq,dw,dk,dv,da,db,ds0

def attn_backstepping_longhead(r,w,k,v,a,b, s0 = None):
    B,T,H,C = w.shape
    if s0 is None: s0 = th.zeros(B,H,C,C, dtype=th.bfloat16,device=w.device)
    return RWKV7_longhead.apply(r,w,k,v,a,b, s0)

def load_backstepping_longhead(head_size, batchsz_times_heads_estimate = 8*64):
    if hasattr(th.ops.wind_backstepping_longhead, 'forward'): return
    device_props = th.cuda.get_device_properties(th.cuda.current_device())
    if 'AMD' in device_props.name:
        value_chunk_size = 16
        # i hard coded gfx942=mi300x, because gfx1030 cannot compile
        CUDA_FLAGS = [f'-D_C_={head_size}', f'-D_K_={value_chunk_size}', f'-D_CHUNK_LEN_={CHUNK_LEN}', '-O3', '-ffast-math', '-DAMD', '--offload-arch=gfx942']
    else:
        value_chunk_size = 64
        if th.cuda.get_device_properties(th.cuda.current_device()).multi_processor_count >= batchsz_times_heads_estimate * head_size / 32:
            value_chunk_size = 32
        CUDA_FLAGS = ['-res-usage', f'-D_C_={head_size} -D_K_={value_chunk_size}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    path = os.path.dirname(__file__) + '/cuda/'
    load(name="wind_backstepping_longhead", sources=[os.path.join(path,'backstepping_longhead.cu'), os.path.join(path,'backstepping_longhead.cpp')], is_python_module=False, verbose=False, extra_cuda_cflags=CUDA_FLAGS)
    assert hasattr(th.ops.wind_backstepping_longhead, 'forward')

load_backstepping_longhead(HEAD_SIZE,BATCH_SIZE*HEAD)

def RUN_CUDA_RWKV7g(r,w,k,v,a,b, HEAD_SIZE, mask=None,  dot_prec = 'fp32'):
    #mask and dot_prec is dummy for RWKV-LM-RLHF
    B,T,HC = w.shape
    C = HEAD_SIZE
    H = HC//C
    r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
    s0 = th.zeros(B,H,C,C, dtype=th.bfloat16,device=w.device)
    return attn_backstepping_longhead(r,w,k,v,a,b,s0)[0].view(B,T,HC)