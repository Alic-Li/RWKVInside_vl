import os
import torch
from safetensors.torch import load_file
from collections import OrderedDict

def convert_weight_names(state_dict):
    """
    重みの名前を変換する関数
    （主に safetensors 側の変換用）
    """
    name_mapping = {
        'model.': '',
        'layers.': 'blocks.',
        'self_attn.': 'att.',
        'mlp.': 'ffn.',
        'gate_up_proj' : 'gate_up',
        'down_proj': 'down',
        'gate_proj': 'gate',
        'up_proj': 'up',
        'input_layernorm': 'ln1',
        'post_attention_layernorm': 'ln2',
        'lm_head': 'head',
        'r_norm.': 'ln_r.',
        'q_norm.': 'ln_r.',
        'k_norm.': 'ln_k.',
        'norm.': 'ln_out.',
        'embed_tokens.': 'emb.'
    }
    
    new_state_dict = OrderedDict()
    
    with torch.no_grad():
        for old_key, tensor in state_dict.items():
            new_key = old_key
            for old_pattern, new_pattern in name_mapping.items():
                new_key = new_key.replace(old_pattern, new_pattern)
            # テンソルをBF16に変換し、grad情報なしでコピー
            print(f'old = {old_key} new = {new_key}')
            new_state_dict[new_key] = tensor.detach().clone().to(torch.bfloat16)
    
    return new_state_dict

def convert_adapter_weight_names(state_dict):
    """
    重みの名前を変換する関数（Adapter.bin 用）
    例: model.model.layers.23.self_attn.student_attn.v0 -> blocks.23.att.v0
    """
    # こちらの mapping は Adapter のみを想定して必要最低限に
    # model.model. → （空文字に）
    # layers.     → blocks.
    # self_attn.student_attn. → att.
    adapter_name_mapping = {
        'model.model.': '',
        'model.':'',
        'layers.': 'blocks.',
        'self_attn.student_attn.': 'att.',
        'lm_head': 'head',
        'layers.': 'blocks.',
        'self_attn.time_mixer.': 'att.',
        'mlp.': 'ffn.',
        'gate_up_proj' : 'gate_up',
        'down_proj': 'down',
        'gate_proj': 'gate',
        'up_proj': 'up',
        'input_layernorm': 'ln1',
        'post_attention_layernorm': 'ln2',
        'lm_head': 'head',
        'r_norm.': 'ln_r.',
        'q_norm.': 'ln_r.',
        'k_norm.': 'ln_k.',
        'norm.': 'ln_out.',
        'embed_tokens.': 'emb.'
    }
    
    new_state_dict = OrderedDict()
    
    with torch.no_grad():
        for old_key, tensor in state_dict.items():
            print(f'Adapter File {old_key}')
            new_key = old_key
            for old_pattern, new_pattern in adapter_name_mapping.items():
                new_key = new_key.replace(old_pattern, new_pattern)
            # テンソルをBF16に変換
            # if 'emb.' in new_key or 'head' in new_key or 'ln_out' in new_key:
            #     print(f'skipped {new_key}')
            # else:
            new_state_dict[new_key] = tensor.detach().clone().to(torch.bfloat16)

    #exit()
    
    return new_state_dict

def merge_safetensors(input_dir):
    """
    指定されたディレクトリ内のすべてのsafetensorファイルを読み込み、キー名変換後にマージする
    """
    safetensor_files = [f for f in os.listdir(input_dir) if f.endswith('.safetensors')]
    
    if not safetensor_files:
        raise FileNotFoundError("Inputフォルダにsafetensorsファイルが見つかりません。")
    
    print(f"処理するファイル数: {len(safetensor_files)}")
    
    merged_weights = OrderedDict()
    
    for file_name in safetensor_files:
        safetensor_path = os.path.join(input_dir, file_name)
        print(f"\n処理中のファイル: {file_name}")
        
        try:
            with torch.no_grad():
                # safetensorsファイルを読み込み
                current_weights = load_file(safetensor_path)
                
                # 重み名を変換
                converted_weights = convert_weight_names(current_weights)
                #converted_weights = current_weights
                
                # 重複をチェック
                overlap = set(merged_weights.keys()) & set(converted_weights.keys())
                if overlap:
                    print(f"\n警告: {file_name} で重複する重みが見つかりました:")
                    for key in overlap:
                        print(f"  {key}")
                    choice = input("重複する重みを上書きしますか？ (y/n): ").strip().lower()
                    if choice != 'y':
                        print("処理を中断します。")
                        return None
                
                # 重みをマージ（BF16形式、grad情報なし）
                for key, tensor in converted_weights.items():
                    print(f'safetensor {key}')
                    merged_weights[key] = tensor.clone()
                    
                print(f"{file_name} の処理が完了しました。")
                print(f"現在の総重み数: {len(merged_weights)}")
            
        except Exception as e:
            print(f"ファイル {file_name} の処理中にエラーが発生しました: {str(e)}")
            return None
        
    #exit()
    
    return merged_weights

def main():
    input_dir = "/home/client/Projects/llm/Qwen3-8B"
    adapter_file = "/home/client/Projects/output/Qwen3-8B/hxa079-stage2/pytorch_model.bin/pytorch_model.bin"  # Adapter の PyTorch モデルファイル
    output_file = "/home/client/Projects/output/Qwen3-8B/hxa079-stage2-hybrid.pth"
    
    try:
        print("safetensorファイルのマージを開始します...")
        with torch.no_grad():
            merged_weights = merge_safetensors(input_dir)





            # ---- Adapter.bin を読み込み、キー変換してマージ ----
            print(f"\nPyTorchモデルファイル {adapter_file} を読み込みます...")
            
            if not os.path.isfile(adapter_file):
                raise FileNotFoundError(f"Adapterファイルが見つかりません: {adapter_file}")
            
            adapter_state = torch.load(adapter_file, map_location="cpu")
            print(f"Adapterファイル読み込み完了。キー数: {len(adapter_state)}")
            
            print("Adapter用のキー変換を実行します...")
            converted_adapter_weights = convert_adapter_weight_names(adapter_state)


            #Check Existing RWKV Layer
            RWKVLayers = 0
            for i in range(96):
                SearchWord = f'blocks.{i}.att.receptance'
                for key in converted_adapter_weights.keys():
                    #print(f'checking {key}')
                    if SearchWord in key:
                        RWKVLayers = i + 1
            print(f'RWKVBlock = {RWKVLayers}')
            #exit()

            TotalLayers = 0
            for i in range(96):
                SearchWord = f'blocks.{i}.'
                for key in merged_weights.keys():
                    #print(f'checking {key}')
                    if SearchWord in key:
                        TotalLayers = i + 1
            print(f'TotalLayers = {TotalLayers}')

            GQALayer = TotalLayers - RWKVLayers

            print(f'GQALayers = {GQALayer}')

            

            remove_keys = []
            for i in range(TotalLayers):
                att=f'blocks.{i}.att'
                for k in merged_weights.keys():
                    if i < RWKVLayers:
                        if ("q_proj" in k or "k_proj" in k or "v_proj" in k  or "o_proj" in k   or "qkv_proj" in k) and att in k:
                            remove_keys.append(k)














            
            
            if merged_weights is None:
                print("マージ処理が中断されました。")
                return
            
            # # ---- ここで q_proj, k_proj, v_proj を含むキーを削除 ----
            # print("\nq_proj, k_proj, v_proj を含むキーを削除します...")
            # remove_keys = [k for k in merged_weights.keys() 
            #                if ("q_proj" in k or "k_proj" in k or "v_proj" in k  or "o_proj" in k   or "qkv_proj" in k)]
            print(remove_keys)
            #exit()
            for k in remove_keys:
                del merged_weights[k]
                print(f'{k} deleted')
            print(f"削除したキー数: {len(remove_keys)}")

            #exit()
            
            
            
            # マージ時の重複チェック
            overlap = set(merged_weights.keys()) & set(converted_adapter_weights.keys())
            if overlap:
                print(f"\n警告: Adapter と既存のモデルで重複するキーが見つかりました:")
                for key in overlap:
                    print(f"  {key}")
                choice = input("重複するキーを上書きしますか？ (y/n): ").strip().lower()
                if choice != 'y':
                    print("処理を中断します。")
                    return
            
            # Adapter の重みをマージ
            for key, tensor in converted_adapter_weights.items():
                merged_weights[key] = tensor.clone()
            
            print(f"Adapter のマージが完了しました。現在の総重み数: {len(merged_weights)}")
            have_head = False
            for key, tensor in merged_weights.items():
                print(f'merged key = {key}')
                if 'head' in key:
                    print(f'have head {key}')
                    have_head = True

            if have_head == False:
                merged_weights['head.weight'] = merged_weights['emb.weight'].clone()
            
            # ---- 最終的にマージした重みを保存 ----
            print(f"\nマージした重みを {output_file} に保存します...")
            torch.save(merged_weights, output_file, _use_new_zipfile_serialization=True)
            print(f"保存完了: {output_file}")
            
            # 最終的な重み名一覧を表示
            print("\n最終的な重み名一覧:")
            for key in merged_weights.keys():
                print(f"  {key}")
            
    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()
