# test_mask.py
import torch
#from TimeMixer import RUN_CUDA_RWKV7g



from fla.ops.rwkv7 import chunk_rwkv7#,fused_recurrent_rwkv7
@torch.jit.ignore
def RUN_CUDA_RWKV7g_FLA(r, w, k, v, a, b, attention_mask=None, HEAD_SIZE = 64): 
    B, T, HC = w.shape
    C = HEAD_SIZE
    H = HC // C
    
    r, w, k, v, a, b = [i.view(B, T, H, C) for i in [r, w, k, v, a, b]]
    w = -w.float().exp().to(r)  

    if attention_mask is not None:
        v = v * attention_mask[:, :, None, None].to(dtype=r.dtype)
    
    v= v.to(dtype=r.dtype)
    
    o, _ = chunk_rwkv7(r, w, k, v, a, b, scale=1.0, initial_state=None, 
                       output_final_state=False, head_first=False)
    
    return o.view(B,T,HC)


import torch as th
import triton
import triton.language as tl

@triton.jit
def IND4(a,b,c,d,nb,nc,nd):
    return ((a*nb+b)*nc+c)*nd+d
@triton.jit
def IND5(a,b,c,d,e,nb,nc,nd,ne):
    return (((a*nb+b)*nc+c)*nd+d)*ne+e

@triton.jit
def _prod(a,b): return a*b

# inv(I-A) where A is a strictly lower triangular nxn matrix
@triton.jit
def tri_minv(A, n:tl.constexpr, prec:tl.constexpr):
    i = tl.arange(0,n)
    prod = (i[None,:]==i[:,None]).to(tl.float32)
    for j in range(n-1):
        prod += tl_dot(prec, prod, (A*((i[None,:]==j)*(i[:,None]>i[None,:]))).trans())
    return prod.trans()

@triton.jit
def fw_attn_triton(w_,q_,k_,v_,a_,b_, s0_,y_,s_,sT_, B:tl.constexpr,T:tl.constexpr,H:tl.constexpr,C:tl.constexpr,dT:tl.constexpr, prec:tl.constexpr):
    bi = tl.program_id(1)
    hi = tl.program_id(0)

    i = tl.arange(0,C)[None,:]
    state = tl.load(s0_+IND4(bi,hi,i.trans(),i, H,C,C)).to(tl.float32)
    for t0 in range(T//dT):
        t = t0*dT+tl.arange(0,dT)[:,None]
        sw = tl.load(w_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sq = tl.load(q_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sk = tl.load(k_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sv = tl.load(v_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sa = tl.load(a_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sb = tl.load(b_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)

        w = (-sw.exp()).exp()
        fw = tl.reduce(w, 0, _prod, keep_dims=True)
        incl_pref = tl.cumprod(w,axis=0)
        non_incl_pref = incl_pref / w
        inv_incl_pref = 1 / incl_pref

        wq = sq * incl_pref
        wa = sa * non_incl_pref
        kwi = sk * inv_incl_pref
        bwi = sb * inv_incl_pref

        mask1 = (t > t.trans())
        ab = tl_dot(prec, wa, bwi.trans()) * mask1
        ak = tl_dot(prec, wa, kwi.trans()) * mask1

        ab_inv = tri_minv(ab, dT, prec)

        ab_u = tl_dot(prec, ak, sv) + tl_dot(prec, wa, state.trans())
        u = tl_dot(prec, ab_inv, ab_u)
        mask2 = (t >= t.trans())
        qk = tl_dot(prec, wq, kwi.trans()) * mask2
        qb = tl_dot(prec, wq, bwi.trans()) * mask2
        yy = tl_dot(prec, qk, sv) + tl_dot(prec, qb, u) + tl_dot(prec, wq, state.trans())
        tl.store(y_+IND4(bi,t,hi,i, T,H,C), yy.to(tl.bfloat16))

        tl.store(s_+IND5(bi,hi,t0,i.trans(),i, H,T//dT,C,C), state.to(tl.float32))
        state = state * fw + tl_dot(prec, sv.trans(), kwi*fw) + tl_dot(prec, u.trans(), bwi*fw)
    tl.store(sT_+IND4(bi,hi,i.trans(),i, H,C,C), state.to(tl.bfloat16))

@triton.jit
def bw_attn_triton(w_,q_,k_,v_,a_,b_, dy_,s_,dsT_, dw_,dq_,dk_,dv_,da_,db_,ds0_, B:tl.constexpr,T:tl.constexpr,H:tl.constexpr,C:tl.constexpr,dT:tl.constexpr, prec:tl.constexpr):
    bi = tl.program_id(1)
    hi = tl.program_id(0)

    i = tl.arange(0,C)[None,:]
    dstate = tl.load(dsT_+IND4(bi,hi,i.trans(),i, H,C,C)).to(tl.float32)

    for t0 in range(T//dT-1,-1,-1):
        t = t0*dT+tl.arange(0,dT)[:,None]
        state = tl.load(s_+IND5(bi,hi,t0,i.trans(),i, H,T//dT,C,C)).to(tl.float32)

        sw = tl.load(w_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sq = tl.load(q_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sk = tl.load(k_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sv = tl.load(v_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sa = tl.load(a_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sb = tl.load(b_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sdy = tl.load(dy_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)

        dw_fac = -sw.exp()
        w = dw_fac.exp()
        fw = tl.reduce(w, 0, _prod, keep_dims=True)
        incl_pref = tl.cumprod(w,axis=0)
        non_incl_pref = incl_pref / w
        inv_incl_pref = 1 / incl_pref

        wq = sq * incl_pref
        wa = sa * non_incl_pref
        kwi = sk * inv_incl_pref
        bwi = sb * inv_incl_pref

        mask1 = (t > t.trans())
        ab = tl_dot(prec, wa, bwi.trans()) * mask1
        ak = tl_dot(prec, wa, kwi.trans()) * mask1

        ab_inv = tri_minv(ab, dT, prec)

        ab_u = tl_dot(prec, ak, sv) + tl_dot(prec, wa, state.trans())
        u = tl_dot(prec, ab_inv, ab_u)
        mask2 = (t >= t.trans())
        qk = tl_dot(prec, wq, kwi.trans()) * mask2
        qb = tl_dot(prec, wq, bwi.trans()) * mask2

        du = tl_dot(prec, qb.trans(), sdy) + tl_dot(prec, bwi*fw, dstate.trans())
        dab_u = tl_dot(prec, ab_inv.trans(), du)

        dv = tl_dot(prec, qk.trans(), sdy) + tl_dot(prec, kwi*fw, dstate.trans()) + tl_dot(prec, ak.trans(), dab_u)
        tl.store(dv_+IND4(bi,t,hi,i, T,H,C), dv.to(tl.bfloat16))

        dab = tl_dot(prec, tl_dot(prec, ab_inv.trans(), du), u.trans()) * mask1
        dak = tl_dot(prec, dab_u, sv.trans()) * mask1
        dab_u_state = tl_dot(prec, dab_u, state)
        da = non_incl_pref * (tl_dot(prec, dab, bwi) + tl_dot(prec, dak, kwi) + dab_u_state)
        tl.store(da_+IND4(bi,t,hi,i, T,H,C), da.to(tl.bfloat16))

        dqb = tl_dot(prec, sdy, u.trans()) * mask2
        dqk = tl_dot(prec, sdy, sv.trans()) * mask2
        dy_state = tl_dot(prec, sdy, state)
        dq = incl_pref * (tl_dot(prec, dqb, bwi) + tl_dot(prec, dqk, kwi) + dy_state)
        tl.store(dq_+IND4(bi,t,hi,i, T,H,C), dq.to(tl.bfloat16))

        fw_u_dstate = fw * tl_dot(prec, u, dstate)
        db = inv_incl_pref * (tl_dot(prec, dab.trans(), wa) + tl_dot(prec, dqb.trans(), wq) + fw_u_dstate)
        tl.store(db_+IND4(bi,t,hi,i, T,H,C), db.to(tl.bfloat16))

        fw_v_dstate = fw * tl_dot(prec, sv, dstate)
        dk = inv_incl_pref * (tl_dot(prec, dak.trans(), wa) + tl_dot(prec, dqk.trans(), wq) + fw_v_dstate)
        tl.store(dk_+IND4(bi,t,hi,i, T,H,C), dk.to(tl.bfloat16))

        dw0 = fw * tl.sum(state*dstate, axis=0,keep_dims=True)
        for k in range(t0*dT,t0*dT+dT):
            lmask = (t<k).trans()
            A = (tl_dot(prec, dab*lmask, bwi) + tl_dot(prec, dak*lmask, kwi)) * wa * (t>k)
            A += (tl_dot(prec, dqb*lmask, bwi) + tl_dot(prec, dqk*lmask, kwi)) * wq * (t>=k)
            A += (fw_v_dstate*kwi + fw_u_dstate*bwi) * (t<k)
            A += dab_u_state*wa * (t>k) + dy_state*wq * (t>=k)
            dw = tl.sum(A, axis=0,keep_dims=True) + dw0

            wk = tl.load(w_+IND4(bi,k,hi,i, T,H,C)).to(tl.float32)
            dw *= -wk.exp()
            tl.store(dw_+IND4(bi,k,hi,i, T,H,C), dw.to(tl.bfloat16))

        dstate = dstate * fw + tl_dot(prec, sdy.trans(), wq) + tl_dot(prec, dab_u.trans(), wa)
    tl.store(ds0_+IND4(bi,hi,i.trans(),i, H,C,C), dstate.to(tl.bfloat16))


class TritonRWKV7(th.autograd.Function):
    @staticmethod
    def forward(ctx, w,q,k,v,z,b,s0, dot_prec):
        K = 16
        B,T,H,C = w.shape
        s0 = th.zeros(B,H,C,C, dtype=w.dtype,device=w.device) if s0 is None else s0
        y = th.empty_like(v)
        sT = th.empty_like(s0)
        s = th.zeros(B,H,T//K,C,C, dtype=th.float32,device=w.device)
        fw_attn_triton[(H,B)](w,q,k,v,z,b, s0,y,s,sT, B,T,H,C,K, dot_prec)
        ctx.dot_prec = dot_prec
        ctx.save_for_backward(w,q,k,v,z,b,s)
        return y, sT
    @staticmethod
    def backward(ctx, dy, dsT):
        K = 16
        w,q,k,v,z,b,s = ctx.saved_tensors
        B,T,H,C = w.shape
        dw,dq,dk,dv,dz,db,ds0 = [th.empty_like(x) for x in [w,q,k,v,z,b,dsT]]
        bw_attn_triton[(H,B)](w,q,k,v,z,b, dy,s,dsT, dw,dq,dk,dv,dz,db,ds0, B,T,H,C,K, ctx.dot_prec)
        return dw,dq,dk,dv,dz,db,ds0,None

@triton.jit
def tl_dot(prec:tl.constexpr, a, b):
    if prec == 'fp32':
        return tl.dot(a.to(tl.float32),b.trans().to(tl.float32).trans(), allow_tf32=False)
    elif prec == 'tf32':
        return tl.dot(a.to(tl.float32),b.trans().to(tl.float32).trans(), allow_tf32=True)
    elif prec == 'bf16':
        return tl.dot(a.to(tl.bfloat16),b.trans().to(tl.bfloat16).trans(), allow_tf32=True)
    else:
        tl.static_assert(False)

def RUN_CUDA_RWKV7g(r,w,k,v,a,b, mask=None, HEAD_SIZE=64, dot_prec = 'fp32'):
    B,T,HC = w.shape
    C = HEAD_SIZE
    H = HC//C
    r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
    
    if mask is not None:
        mask_view = mask.view(B, T, 1, 1)
        r = r * mask_view
        k = k * mask_view
        v = v * mask_view
        a = a * mask_view
        b = b * mask_view
    
    s0 = th.zeros(B,H,C,C, dtype=th.bfloat16,device=w.device)
    return TritonRWKV7.apply(w,r,k,v,a,b,s0,dot_prec)[0].view(B,T,HC)

def test_wkv_with_mask(mode = 'triton'):
    print("Running test_wkv_with_mask...")
    
    B, T, HC = 2, 32, 128  # batch_size=2, seq_len=32, hidden_size=128
    device = "cuda:0"

    tensors = {
        name: 0.1 * torch.randn(B, T, HC, dtype=torch.bfloat16, device=device).contiguous()
        for name in ['q', 'w', 'k', 'v', 'a', 'b']
    }
    

    mask_rightpad = torch.ones(B, T, dtype=torch.int32, device=device)
    mask_leftpad = torch.ones(B, T, dtype=torch.int32, device=device)

    mask_rightpad[1, T//2:] = 0
    mask_leftpad[1, 0:T//2] = 0

    print(f'RightPad = {mask_rightpad}')
    print(f'LeftPad = {mask_leftpad}')
    
    print("Running without mask...")
    if mode == 'fla':
        print('fla')
        output1 = RUN_CUDA_RWKV7g_FLA(
            tensors['q'], tensors['w'], tensors['k'],
            tensors['v'], tensors['a'], tensors['b'],
            None
        )
    else:
        output1 = RUN_CUDA_RWKV7g(
            tensors['q'], tensors['w'], tensors['k'],
            tensors['v'], tensors['a'], tensors['b'],
            None
        )
    
    print("Running with rightpad_mask...")
    if mode == 'fla':
        print('fla')
        output_right = RUN_CUDA_RWKV7g_FLA(
            tensors['q'], tensors['w'], tensors['k'],
            tensors['v'], tensors['a'], tensors['b'],
            mask_rightpad
        )
    else:
        output_right = RUN_CUDA_RWKV7g(
            tensors['q'], tensors['w'], tensors['k'],
            tensors['v'], tensors['a'], tensors['b'],
            mask_rightpad
        )

    print("Running with leftpad_mask...")
    if mode == 'fla':
        print('fla')
        output_left = RUN_CUDA_RWKV7g_FLA(
            tensors['q'], tensors['w'], tensors['k'],
            tensors['v'], tensors['a'], tensors['b'],
            mask_leftpad
        )
    else:
        output_left = RUN_CUDA_RWKV7g(
            tensors['q'], tensors['w'], tensors['k'],
            tensors['v'], tensors['a'], tensors['b'],
            mask_leftpad
        )
    
    print("\nValidating results:")

    first_seq_same_right = torch.allclose(output1[0], output_right[0], rtol=1e-3)
    print(f"First sequence unchanged right: {first_seq_same_right}")

    first_seq_same_left = torch.allclose(output1[0], output_left[0], rtol=1e-3)
    print(f"First sequence unchanged left: {first_seq_same_left}")
    
    # 检查第二个序列的masked部分是否不同
    masked_changed_right = not torch.allclose(
        output1[1, T//2:], 
        output_right[1, T//2:], 
        rtol=1e-3
    )

    masked_changed_left = not torch.allclose(
        output1[1, 0:T//2], 
        output_left[1, 0:T//2], 
        rtol=1e-3
    )
    print(f"Masked part changed right: {masked_changed_right}")

    print(f"Masked part changed left: {masked_changed_left}")
    
    return first_seq_same_right and first_seq_same_left and masked_changed_right and masked_changed_left,output1, output_left, output_right

if __name__ == "__main__":

    print('RWKV v7 Kernel Padding Check')
    print('FLA Mode')
    success,fla_full, fla_left,fla_right = test_wkv_with_mask('fla')
    if success:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
    print('Triton Mode')
    success,triton_full,triton_left,triton_right= test_wkv_with_mask('triton')
    if success:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")

    print(fla_full)
    print(triton_full)

    print(f' fla - triton full sum ={torch.sum(fla_full - triton_full)}')
    print(f' fla - triton left sum ={torch.sum(fla_left - triton_left)}')
    print(f' fla - triton right sum ={torch.sum(fla_right - triton_right)}')