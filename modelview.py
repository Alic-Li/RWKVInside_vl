import torch

# モデルファイルのロード
state_dict = torch.load('TestOutput/pytorch_model.bin', map_location=torch.device('cpu'))

# 各レイヤーの情報を表示
for key, tensor in state_dict.items():
    # キー名
    print(f"\nKey: {key}")
    
    # テンソルの次元サイズ
    print(f"Shape: {tensor.shape}")
    
    # データ型
    print(f"Data Type: {tensor.dtype}")
    
    # メモリサイズ（バイト）
    print(f"Memory Size: {tensor.element_size() * tensor.nelement()} bytes")