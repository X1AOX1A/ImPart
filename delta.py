import os
from tqdm import tqdm
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_delta_to_ptm(model_pth, delta_pth):
    print(f"New Load {my_dtype}")
    tokenizer = AutoTokenizer.from_pretrained(model_pth)
    model = AutoModelForCausalLM.from_pretrained(model_pth, torch_dtype=my_dtype)
    delta = torch.load(delta_pth)
    for name, param in tqdm(model.named_parameters()):
        if name not in delta:
            delta[name] = param
    model.load_state_dict(delta)
    return tokenizer, model

def decomposition(masked_input_tensor, dim=None):
    U , S , V = torch.svd(masked_input_tensor.to(torch.float32))
    if dim is not None:
        U , S , V = U[:, :dim],S[:dim] ,V[:, :dim]
    return U, S, V 


def svd_delta(base_model, finetuned_model, dim_attn, save_pth):
    device = "cpu"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=my_dtype
    ).to(device)
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        finetuned_model,
        torch_dtype=my_dtype
    ).to(device)
    
    param_dict = dict()
    for k,v in tqdm(base_model.state_dict().items()):
        if "self_attn" in k or "mlp" in k:
            if ".weight" in k:
                delta = finetuned_model.state_dict()[k] - v
                delta.to("cuda")
                dim = dim_attn
                
                if "mlp" in k:
                    dim = int(dim * 1.45)
                U,S,V = decomposition(delta, dim=dim)
                U = U.to(device)
                S = S.to(device)
                V = V.to(device)
                k = k.replace(".weight", "")
                
                param_dict[k + ".base"] = v
                param_dict[k + ".U"] = U.data.to(my_dtype)
                param_dict[k + ".S"] = S.data.to(my_dtype)
                param_dict[k + ".V"] = V.data.to(my_dtype)

    torch.save(param_dict, save_pth)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--svd', action='store_true', help='compute svd of delta weight between base model and finetuned model')
    parser.add_argument('--merge', action='store_true', help='reconstruct finetuned model by adding delta weight on base model')
    parser.add_argument('--dim', type=int, default=2000, help='num singular rank to save')
    parser.add_argument('--delta_pth', type=str, default="", help='processed delta weight')
    parser.add_argument('--save_pth', type=str, default="", help='path to save the svd outcome of delta weight or reconstructed model')
    parser.add_argument('--finetuned_model', type=str, default="vanillaOVO/WizardMath-13B-V1.0")
    parser.add_argument('--base_model', type=str, default="meta-llama/Llama-2-13b-hf")
    parser.add_argument('--dtype', type=str, choices=["fp16", "bf16"], help='fp16 or bf16')
    args = parser.parse_args()

    my_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    if args.svd:
        svd_delta(
            base_model=args.base_model,
            finetuned_model=args.finetuned_model,
            dim_attn=args.dim,
            save_pth=args.save_pth
        )

    elif args.merge:
        print(f"Dtype:{my_dtype}")
        print(f"Finetuned model:{args.finetuned_model}")
        print(f"Delta Path: {args.delta_pth}")
        tokenizer, model = load_delta_to_ptm(
            model_pth=args.finetuned_model,
            delta_pth=args.delta_pth
        )
        tokenizer.save_pretrained(args.save_pth)
        model.save_pretrained(
            args.save_pth,
            safe_serialization=False
        )