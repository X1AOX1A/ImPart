import warnings
warnings.filterwarnings("ignore")
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import pandas as pd

import logging
def setup_logger(log_file=None):
    # Create a logger for this file
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # Set the logging level
    # Prevent propagation to the root logger
    logger.propagate = False
    # Create a StreamHandler for console output
    console_handler = logging.StreamHandler()
    # Define a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d_%H-%M-%S')
    console_handler.setFormatter(formatter)
    # Add the handler to the logger
    logger.addHandler(console_handler)
    # Create a FileHandler for writing log files
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w')  # overwrite
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


@torch.no_grad()
def merge_base_usv(svd_delta):
    logger.info("Merging svd_delta...")
    merged_weight = dict()
    names = [name.split(".base")[0] for name in list(svd_delta.keys()) if ".base" in name]
    for name in names:
        # must merge on cuda, otherwise the result will be different
        base = svd_delta[name + ".base"].to("cuda")
        u = svd_delta[name + ".U"].to("cuda")
        s = svd_delta[name + ".S"].to("cuda")
        v = svd_delta[name + ".V"].to("cuda")
        merged = base + u @ torch.diag(s) @ v.t()
        merged_weight[name + ".weight"] = merged.to(dtype_16).to("cpu")
    return merged_weight


def replace_weight_and_save_model(ftm, tokenizer, merged_weight, save_dir):
    logger.info("Saving model...")
    for k, v in ftm.named_parameters():
        if k not in merged_weight:
            merged_weight[k] = v
    ftm.load_state_dict(merged_weight)
    ftm.save_pretrained(save_dir, safe_serialization=False)
    tokenizer.save_pretrained(save_dir)
    logger.info(f"Model saved to {save_dir}")


def cal_sparsity_ratios(singular_values, target_uv_sparsity_ratio=0.95,
                   preprune_ratio=0.5, C=1):
    """Algorithm1: Sparsity Ratios Computations
    Args:
        singular_values: list of singular values, length n
        target_uv_sparsity_ratio: target sparsity ratio for U or V (aka alpha)
        preprune_ratio: preprune ratio (aka beta)
        C: rescale parameter
    Returns:
        sparsity_ratio_list: list of sparsity ratios for each singular value **after shifting** (length <= n)
    """
    assert preprune_ratio>=0

    # rescale
    singular_values = [s**C for s in singular_values]
    n = len(singular_values)

    # preprune
    singular_values = singular_values[:int(n * (1-preprune_ratio))]
    s_max = max(singular_values)

    # gamma = (alpha - beta) / (1 - beta)
    gamma = (target_uv_sparsity_ratio - preprune_ratio) / (1 - preprune_ratio)

    # importance-aware sparsification
    sparsity_ratio_list = [(1 - s / s_max) * gamma for s in singular_values]
    assert sum(sparsity_ratio_list) / len(sparsity_ratio_list) <= gamma

    # shift boundary to meet target sparsity ratio
    i = len(sparsity_ratio_list) - 1
    while sum(sparsity_ratio_list) / len(sparsity_ratio_list) < gamma:
        sparsity_ratio_list[i] = 1
        i -= 1

    assert all(0 <= m <= 1 for m in sparsity_ratio_list)
    return sparsity_ratio_list[:i+1]


def apply_sparsify(matrix, sparsity_ratio_list):
    """Apply column-wise sparsification to a matrix"""
    m, n = matrix.shape
    assert n == len(sparsity_ratio_list)
    mask = torch.ones(m, n).to(matrix.device).to(dtype_16)
    for col in range(n):
        p = int(m * sparsity_ratio_list[col])
        indices = torch.randperm(m)[:p]
        mask[indices, col] = 0
    return matrix * mask, mask


def cal_overall_alpha_qt(attn_uv_alpha_qt, mlp_uv_alpha_qt,
                         include_s=False, hidden_size=5120, intermediate_size=13824):
    """Given the sparsity ratio of Attn and MLP, calculate the final sparsity ratio"""
    # 4 for qkvo, 4 for gate,up,down
    denomitor = hidden_size*hidden_size*4 + intermediate_size*hidden_size*3
    nominator = 4*hidden_size*hidden_size*2*(1-attn_uv_alpha_qt) + \
                3*intermediate_size*hidden_size*(1-mlp_uv_alpha_qt) + \
                3*hidden_size*hidden_size*(1-mlp_uv_alpha_qt)
    if include_s:
        nominator += 4*hidden_size*(1-attn_uv_alpha_qt) + 3*hidden_size*(1-mlp_uv_alpha_qt)
    return round(1 - nominator / denomitor, 4)


import numpy as np
def cal_uv_alpha_qt(sparsity_ratio_list, n=5120):
    """Calculate the overall sparsity+quant ratio for U and V"""
    keep_ratio_list = np.array([1 - m for m in sparsity_ratio_list])
    keep_ratio = (
        keep_ratio_list[:2].sum()/2     # 8bit, first 2 rank
        + keep_ratio_list[2:34].sum()/4 # 4bit, 3-34 rank
        + keep_ratio_list[34:].sum()/8  # 2bit, else
    ) / n
    return 1 - keep_ratio


def find_uv_sparsify_ratio(singular_values, target_uv_alpha_qt=0.95,
                           preprune_ratio=0.5, C=1, tol=1e-4):
    """Algorithm3: Find the sparsity ratio that meets the target alpha_qt"""
    low, high = 2*preprune_ratio-1, 1.0 # init mid as preprune_ratio, as mid>=preprune_ratio
    while high - low > tol:
        mid = 0.5 * (low + high)
        sparsity_ratio_list = cal_sparsity_ratios(
            singular_values=singular_values,
            target_uv_sparsity_ratio=mid,
            preprune_ratio=preprune_ratio,
            C=C
        )
        uv_alpha_qt = cal_uv_alpha_qt(sparsity_ratio_list, n=len(singular_values))
        if uv_alpha_qt < target_uv_alpha_qt:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def find_preprune_ratio(target_uv_alpha_qt, n):
    # rank_to_drop = x
    # (2/2 + 32/4 + (n-34-x)/8) / n = 1-target_uv_alpha_qt
    # => x = n + 38 - 8 * (1-target_uv_alpha_qt) * n
    rank_to_drop = n + 38 - 8 * (1-target_uv_alpha_qt) * n
    rank_to_keep = n - rank_to_drop
    preprune_ratio = rank_to_drop / n
    return preprune_ratio, int(rank_to_keep)


def main(args):
    logger.info("Loading svd_delta...")
    svd_delta = torch.load(args.svd_dir)

    global dtype_16
    dtype_16 = svd_delta["model.layers.0.self_attn.q_proj.base"].dtype
    logger.info(f"dtype_16: {dtype_16}")

    dataframe = []
    for weight_type in args.weight_types:
        for layer_num in range(args.layer_num):
            # get weight
            weight_name = args.weight_name.format(weight_type=weight_type, layer_num=layer_num)
            u = svd_delta[weight_name+".U"]
            s = svd_delta[weight_name+".S"]
            v = svd_delta[weight_name+".V"]
            n = s.shape[0]

            # cal sparsity ratios
            target_uv_alpha_qt = args.attn_uv_alpha_qt if "self_attn" in weight_type else args.mlp_uv_alpha_qt
            preprune_ratio = args.attn_preprune_ratio if "self_attn" in weight_type else args.mlp_preprune_ratio
            if args.C == "only_lowrank":
                preprune_ratio, dim = find_preprune_ratio(
                    target_uv_alpha_qt=target_uv_alpha_qt, n=n)
                target_uv_sparsity_ratio = preprune_ratio
            else:
                preprune_ratio = args.attn_preprune_ratio if "self_attn" in weight_type else args.mlp_preprune_ratio
                # binary search to find sparsity ratio
                target_uv_sparsity_ratio = find_uv_sparsify_ratio(
                    singular_values=s.tolist(),
                    target_uv_alpha_qt=target_uv_alpha_qt,
                    preprune_ratio=preprune_ratio,
                    C=args.C
                )
                # cal sparsity ratios
                sparsity_ratio_list = cal_sparsity_ratios(
                    singular_values=s.tolist(),
                    target_uv_sparsity_ratio=target_uv_sparsity_ratio,  # alpha
                    preprune_ratio=preprune_ratio,                      # beta
                    C=args.C                                            # C
                )
                dim = len(sparsity_ratio_list)

            # preprune
            u, s, v = u[:, :dim], s[:dim], v[:, :dim]

            # sparsify
            if args.C != "only_lowrank":
                # u = (apply_sparsify(u, sparsity_ratio_list) / torch.tensor([1-m for m in sparsity_ratio_list])).to(dtype_16)
                # v = (apply_sparsify(v, sparsity_ratio_list) / torch.tensor([1-m for m in sparsity_ratio_list])).to(dtype_16)
                u, u_mask = apply_sparsify(u, sparsity_ratio_list)
                v, v_mask = apply_sparsify(v, sparsity_ratio_list)
                u = (u / torch.tensor([1-m for m in sparsity_ratio_list])).to(dtype_16)
                v = (v / torch.tensor([1-m for m in sparsity_ratio_list])).to(dtype_16)

            # replace weight
            svd_delta[weight_name+".U"] = u
            svd_delta[weight_name+".S"] = s
            svd_delta[weight_name+".V"] = v
            if args.C != "only_lowrank":
                svd_delta[weight_name + ".U_mask"] = u_mask
                svd_delta[weight_name + ".V_mask"] = v_mask

            # log
            if args.C == "only_lowrank":
                actual_uv_sparsity_ratio = 1 - preprune_ratio
                actual_uv_alpha_qt = cal_uv_alpha_qt([0]*dim, n)   # not mask
            else:
                actual_uv_sparsity_ratio = sum(sparsity_ratio_list+[1]*(n-dim)) / n
                actual_uv_alpha_qt = cal_uv_alpha_qt(sparsity_ratio_list, n)

            assert math.isclose(actual_uv_alpha_qt, target_uv_alpha_qt, abs_tol=1e-3)
            dataframe.append([weight_name+".U", list(u.shape), preprune_ratio,
                              actual_uv_alpha_qt, target_uv_alpha_qt,
                              actual_uv_sparsity_ratio, target_uv_sparsity_ratio])
            logger.info(f"{weight_name}.U | shape: {list(u.shape)} "
                        f"| preprune_ratio: {round(preprune_ratio, 4)} "
                        f"| uv_alpha_qt(actual|target): {round(actual_uv_alpha_qt, 4)}|{round(target_uv_alpha_qt, 4)} "
                        f"| uv_sparsity_ratio(actual|target): {round(actual_uv_sparsity_ratio, 4)}|{round(target_uv_sparsity_ratio, 4)}")

    df = pd.DataFrame(dataframe, columns=["weight_name", "shape", "preprune_ratio",
                                          "actual_uv_alpha_qt", "target_uv_alpha_qt",
                                          "actual_uv_sparsity_ratio", "target_uv_sp"])
    df.to_csv(os.path.join(args.log_dir, "sparsity_quant_ratio.csv"), index=False)

    # merge and save model
    logger.info("Saving svd_delta...")
    torch.save(svd_delta, os.path.join(args.save_dir, "svd_delta.pt"))
    ftm = AutoModelForCausalLM.from_pretrained(args.ftm_dir, torch_dtype=dtype_16)
    tokenizer = AutoTokenizer.from_pretrained(args.ftm_dir)
    merged_weight = merge_base_usv(svd_delta)
    replace_weight_and_save_model(
        ftm=ftm, tokenizer=tokenizer,
        merged_weight=merged_weight, save_dir=args.save_dir
    )


import argparse
class DebugConfig(argparse.Namespace):
    ftm_dir = "vanillaOVO/WizardMath-13B-V1.0"
    svd_dir = "saves/precompuated_svd/wizardmath-13b/delta_full.pt"
    save_dir = "saves/cache/debug"
    log_dir = "saves/eval/debug"

    # model args
    weight_types = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    layer_num = 40
    weight_name = "model.layers.{layer_num}.{weight_type}"

    # mask args
    compression_ratio_qt = 32   # alpha_qt = 1 - 1 / CR_qt
    preprune_ratio = 0.80       # beta, none if C=="only_lowrank"
    C = 1.0                     # or "only_lowrank"


if __name__ == "__main__":
    # args = DebugConfig()
    import json
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    args = OmegaConf.load(args.config)
    logger = setup_logger(os.path.join(args.log_dir, "sparsify_quant.log"))

    # update sparsity ratio for U and V in attn and mlp
    args.overall_alpha_qt =  1 - 1 / args.compression_ratio_qt
    args.attn_uv_alpha_qt = (1+args.overall_alpha_qt) / 2
    args.mlp_uv_alpha_qt = (0.55 + 1.45*args.overall_alpha_qt) / 2

    if args.C == "only_lowrank":
        assert getattr(args, "preprune_ratio") is None
        args.attn_preprune_ratio = (1+args.overall_alpha_qt) / 2
        args.mlp_preprune_ratio = (0.55 + 1.45*args.overall_alpha_qt) / 2
    else:
        assert getattr(args, "preprune_ratio") is not None
        args.attn_preprune_ratio = args.preprune_ratio
        args.mlp_preprune_ratio = 1 - (1-args.preprune_ratio) * 1.45 \
            if args.preprune_ratio > 0 else 0
        assert args.attn_preprune_ratio <= 1 and args.mlp_preprune_ratio >= 0

    # check overall sparsity+quant ratio
    model_config = AutoConfig.from_pretrained(args.ftm_dir)
    hidden_size, intermediate_size = model_config.hidden_size, model_config.intermediate_size
    _overall_alpha_qt = cal_overall_alpha_qt(
        args.attn_uv_alpha_qt, args.mlp_uv_alpha_qt,
        hidden_size=hidden_size, intermediate_size=intermediate_size)
    import math
    assert math.isclose(_overall_alpha_qt, args.overall_alpha_qt, abs_tol=1e-3), \
        f"Overall sparsity+quant ratio not match: Target ({args.overall_alpha_qt}) vs  Actual ({_overall_alpha_qt})"

    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(json.dumps(OmegaConf.to_container(args), indent=4))
    main(args)