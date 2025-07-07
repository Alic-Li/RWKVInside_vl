import torch

# 現在の設定を確認
print(f"Flash SDP enabled: {torch.backends.cuda.flash_sdp_enabled()}")
print(f"Mem Efficient SDP enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
print(f"Math SDP enabled: {torch.backends.cuda.math_sdp_enabled()}")

# どのバックエンドが使われるかテスト
def check_sdpa_backend():
    # テスト用のテンソルを作成
    batch_size, seq_len, num_heads, head_dim = 2, 1024, 32, 64
    dtype = torch.bfloat16
    device = torch.device("cuda")
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                    dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                    dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                    dtype=dtype, device=device)
    
    # SDPAを実行してバックエンドを確認
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=True,
        enable_mem_efficient=True
    ):
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
    
    # CUDAイベントで確認（デバッグ用）
    print(f"Output shape: {out.shape}")
    print(f"Output dtype: {out.dtype}")

check_sdpa_backend()