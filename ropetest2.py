import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM

# === モデル設定 ===
model_path = "/home/client/Projects/llm/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
rotary_emb = model.model.rotary_emb

# === 位置情報 ===
seq_len = 1024
position_ids = torch.arange(seq_len, device=rotary_emb.inv_freq.device).unsqueeze(0)
cos_ref, sin_ref = rotary_emb(position_ids.to(dtype=torch.bfloat16), position_ids)
cos_ref = cos_ref.squeeze(0)  # [seq_len, head_dim]
sin_ref = sin_ref.squeeze(0)

# === パラメータ取得 ===
rotary_dim = rotary_emb.inv_freq.shape[0] * 2
print(f'rotary_dim = {rotary_dim}')
device = rotary_emb.inv_freq.device
dtype = rotary_emb.inv_freq.dtype
print(f'dtype = {dtype}')
rope_theta = getattr(model.config, "rope_theta", 1e6)
print(f'rope_theta = {rope_theta}')

# === Qwen3 RoPE 再現 ===
def compute_qwen3_rope_cache(seq_len, rotary_dim, device, dtype, rope_theta):
    half_dim = rotary_dim // 2
    freq_seq = torch.arange(half_dim, dtype=dtype, device=device)
    inv_freq = 1.0 / (rope_theta ** (freq_seq / half_dim))
    positions = torch.arange(seq_len, dtype=dtype, device=device)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin, inv_freq

cos_own, sin_own, inv_freq_own = compute_qwen3_rope_cache(seq_len, rotary_dim, device, dtype, rope_theta)

# === トリミングして比較（必要に応じて） ===
cos_ref_cut = cos_ref[:, :rotary_dim]
sin_ref_cut = sin_ref[:, :rotary_dim]

# === 差分評価 ===
cos_diff = (cos_ref_cut - cos_own).abs()
sin_diff = (sin_ref_cut - sin_own).abs()

print("\n=== 🔍 RotaryEmbedding 差分レポート ===")
print(f"cos_diff.mean() = {cos_diff.mean().item():.8f}")
print(f"cos_diff.max()  = {cos_diff.max().item():.8f}")
print(f"sin_diff.mean() = {sin_diff.mean().item():.8f}")
print(f"sin_diff.max()  = {sin_diff.max().item():.8f}")

tol = 1e-4
print("\n=== ✅ torch.allclose 比較 ===")
print(f"cos_ref ≈ cos_own: {torch.allclose(cos_ref_cut, cos_own, atol=tol)}")
print(f"sin_ref ≈ sin_own: {torch.allclose(sin_ref_cut, sin_own, atol=tol)}")

# === inv_freq 検証 ===
print("\n=== 📉 inv_freq[:5] 比較 ===")
print("ref: ", rotary_emb.inv_freq[:5].cpu().numpy())
print("own: ", inv_freq_own[:5].cpu().numpy())

# === 可視化: 序盤数次元の差分を時系列で表示 ===
def plot_diff(ref, own, label, dim=0):
    diff = (ref - own).abs()
    plt.figure(figsize=(10, 3))
    plt.plot(diff[:, dim].cpu(), label=f"{label} abs diff (dim={dim})")
    plt.title(f"{label} Difference Across Positions (Dim {dim})")
    plt.xlabel("Position")
    plt.ylabel("Absolute Error")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_diff(cos_ref_cut, cos_own, "Cos", dim=0)
plot_diff(sin_ref_cut, sin_own, "Sin", dim=0)

# === 差分ヒートマップ（全体） ===
def plot_heatmap(diff_tensor, title):
    import seaborn as sns
    import numpy as np
    plt.figure(figsize=(10, 5))
    sns.heatmap(diff_tensor[:128, :16].cpu().numpy(), cmap="viridis", cbar=True)
    plt.title(title + " (first 128 positions × 16 dims)")
    plt.xlabel("Dim")
    plt.ylabel("Position")
    plt.tight_layout()
    plt.show()

plot_heatmap(cos_diff, "Cosine Error Heatmap")
plot_heatmap(sin_diff, "Sine Error Heatmap")

