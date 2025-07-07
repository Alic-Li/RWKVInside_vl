import torch
from transformers import AutoModelForCausalLM

# === 1. モデル読み込み ===
model_id = "/home/client/Projects/llm/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
rotary_emb = model.model.rotary_emb  # これは動作してる前提

# === 2. Transformers内のcos/sin取得 ===
seq_len = 1024
position_ids = torch.arange(seq_len, device=rotary_emb.inv_freq.device).unsqueeze(0)
cos_ref, sin_ref = rotary_emb(position_ids, position_ids)  # shape: [1, seq_len, head_dim]
cos_ref = cos_ref.squeeze(0)  # [seq_len, head_dim]
sin_ref = sin_ref.squeeze(0)

# === 3. 自前RoPE計算関数 ===
def compute_rope_cache(seq_len, head_dim, device, dtype, rope_theta=1000000.0):
    half_dim = head_dim // 2
    freq_seq = torch.arange(half_dim, dtype=dtype, device=device)
    inv_freq = 1.0 / (rope_theta ** (freq_seq / half_dim))  # [half_dim]

    positions = torch.arange(seq_len, dtype=dtype, device=device)  # [seq_len]
    freqs = torch.outer(positions, inv_freq)  # [seq_len, half_dim]
    emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, head_dim]

    cos = emb.cos()  # [seq_len, head_dim]
    sin = emb.sin()
    return cos, sin

# === 4. rotary_dim（RoPE適用次元）だけで比較するよう修正 ===
rotary_dim = rotary_emb.inv_freq.shape[0] * 2
dtype = rotary_emb.inv_freq.dtype
device = rotary_emb.inv_freq.device

# === 5. 自前計算（rotary_dimを明示） ===
cos_own, sin_own = compute_rope_cache(seq_len, rotary_dim, device, dtype, rope_theta=1e6)

# === 6. Transformers側の [seq_len, head_dim] から rotary_dim 部分だけ切り出す
cos_ref = cos_ref[:, :rotary_dim]
sin_ref = sin_ref[:, :rotary_dim]

# === 7. 差分再計算
cos_diff = (cos_ref - cos_own).abs().mean()
sin_diff = (sin_ref - sin_own).abs().mean()

print(f"Cos diff (mean abs): {cos_diff.item():.8f}")
print(f"Sin diff (mean abs): {sin_diff.item():.8f}")