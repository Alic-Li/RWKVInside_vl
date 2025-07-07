import torch
import torch.nn.functional as F
import math

def round_up_to_multiple(x, base):
    return base * math.ceil(x / base)

def pad_tensor_to_shape(tensor, target_shape, name=""):
    assert len(tensor.shape) == len(target_shape)
    pad = []
    for i in reversed(range(len(tensor.shape))):
        pad_size = target_shape[i] - tensor.shape[i]
        if pad_size < 0:
            raise ValueError(f"Dimension {i} of {name} is larger than target.")
        pad.extend([0, pad_size])
    
    if any(pad):
        print(f"[{name}] Before: {tuple(tensor.shape)} → After: {tuple(target_shape)}")
        return F.pad(tensor, pad, "constant", 0)
    else:
        print(f"[{name}] No change needed: {tuple(tensor.shape)}")
        return tensor

def process_all_layers(input_path, output_path, multiple=512):
    with torch.no_grad():
        print(f"Loading weights from: {input_path}")
        print(f"Padding all intermediate dimensions to be a multiple of {multiple}")
        weights = torch.load(input_path, map_location="cpu")
        new_weights = {}

        for name, tensor in weights.items():
            shape = tensor.shape

            if "ffn.gate.weight" in name or "ffn.up.weight" in name:
                inter_dim, in_dim = shape  # 転置
                if inter_dim % multiple != 0:
                    new_inter_dim = round_up_to_multiple(inter_dim, multiple)
                    tensor = pad_tensor_to_shape(tensor, (new_inter_dim, in_dim), name)
                else:
                    print(f"[{name}] No change needed: {tuple(shape)}")
                new_weights[name] = tensor

            elif "ffn.down.weight" in name:
                out_dim, inter_dim = shape
                if inter_dim % multiple != 0:
                    new_inter_dim = round_up_to_multiple(inter_dim, multiple)
                    tensor = pad_tensor_to_shape(tensor, (out_dim, new_inter_dim), name)
                else:
                    print(f"[{name}] No change needed: {tuple(shape)}")
                new_weights[name] = tensor

            else:
                new_weights[name] = tensor

        torch.save(new_weights, output_path)
        print(f"Saved updated weights to: {output_path}")

process_all_layers('/workspace/output/rekaflash3/PRWKV7-reka-flash3-21B-Stage2-preview.pth',"/workspace/output/rekaflash3/PRWKV7-reka-flash3-21B-Stage2-preview-ffnaligned.pth")
