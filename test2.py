import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

def calculate_token_level_loss(model, tokenizer, text, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    トークンレベルでロスを計算する関数
    
    Args:
        model: 言語モデル
        tokenizer: トークナイザー
        text: ロスを計算したいテキスト
        device: 計算に使用するデバイス
        
    Returns:
        tokens: トークンのリスト
        token_losses: 各トークンのロス値
    """
    # 文章をトークン化
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # トークンごとのロスを計算するため、各トークンのIDを取得
    tokens = []
    for i in range(input_ids.shape[1]):
        tokens.append(tokenizer.decode(input_ids[0, i]))
    
    # ラベルを設定（次のトークンを予測）
    labels = input_ids.clone()
    
    # フォワードパスを実行
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        
    # ロス計算のため、モデル出力からロジットを取得
    logits = outputs.logits
    
    # 各トークンのロスを計算
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    shift_logits = logits[0, :-1, :].contiguous()
    shift_labels = labels[0, 1:].contiguous()
    token_losses = loss_fct(shift_logits, shift_labels)
    
    # トークンとそれぞれのロスを返す
    return tokens[:-1], token_losses.float().cpu().numpy()

def visualize_token_losses(tokens, token_losses, title="トークンごとのロス"):
    """
    トークンごとのロスを可視化する関数
    
    Args:
        tokens: トークンのリスト
        token_losses: 各トークンのロス値
        title: グラフのタイトル
    """
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(token_losses)), token_losses)
    plt.xticks(range(len(token_losses)), tokens, rotation=90)
    plt.title(title)
    plt.xlabel("トークン")
    plt.ylabel("ロス")
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()
    
    # ロスが高い上位5つのトークンを表示
    top_indices = np.argsort(token_losses)[-5:][::-1]
    print(f"\n{title} - ロスが高い上位5つのトークン:")
    for idx in top_indices:
        print(f"トークン: '{tokens[idx]}', ロス: {token_losses[idx]:.4f}")

# メインコード
torch.random.manual_seed(0)
model_path = "../Phi-4-mini-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 分析したい文章を指定
text_to_analyze = "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."

# トークンレベルでロスを計算
tokens, token_losses = calculate_token_level_loss(model, tokenizer, text_to_analyze)

# 結果を可視化
visualize_token_losses(tokens, token_losses, "バナナとドラゴンフルーツの回答のトークンごとのロス")

# 平均ロスも計算
avg_loss = np.mean(token_losses)
print(f"\n文章全体の平均ロス: {avg_loss:.4f}")
print(f"トークン数: {len(tokens)}")
print(f"ロスの最大値: {np.max(token_losses):.4f}")
print(f"ロスの最小値: {np.min(token_losses):.4f}")