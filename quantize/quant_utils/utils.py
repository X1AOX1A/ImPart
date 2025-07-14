import torch
import argparse
from datasets import load_dataset
from transformers import default_data_collator,AutoTokenizer
from yaml import safe_load, safe_dump
try:
    from llava.model import *
    from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
except: pass
import os

def load_llava(path,device):
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    
    
    image_processor = None

    if 'llava' in path.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device)
        if device != 'auto':
            vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    
    return model

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, help='llama model to load', nargs="+")
    parser.add_argument('--dataset', type=str, choices=['wikitext2', 'ptb', 'c4', 'metaMath'], help='Where to extract calibration data from.')
    parser.add_argument('--my_dtype', type=str, help='dtype', choices=["fp16", "bf16"])
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration data samples.')
    parser.add_argument('--percdamp', type=float, default=.01, help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--nearest', action='store_true', help='Whether to run the RTN baseline.')
    parser.add_argument('--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16], help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument('--trits', action='store_true', help='Whether to use trits for quantization.')
    parser.add_argument('--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--eval', action='store_true', help='evaluate quantized model.')
    parser.add_argument('--test-generation', action='store_true', help='test generation.')
    parser.add_argument('--save', type=str, default='', help='Save quantized checkpoint under this name.')
    parser.add_argument('--save_safetensors', type=str, default='', help='Save quantized `.safetensors` checkpoint under this name.')
    parser.add_argument('--load', type=str, default='', help='Load quantized model.')
    parser.add_argument('--benchmark', type=int, default=0, help='Number of tokens to use for benchmarking.')
    parser.add_argument('--check', action='store_true', help='Whether to compute perplexity during benchmarking for verification.')
    parser.add_argument('--sym', action='store_true', help='Whether to perform symmetric quantization.')
    parser.add_argument('--act-order', action='store_true', help='Whether to apply the activation order GPTQ heuristic')
    parser.add_argument('--true-sequential', action='store_true', help='Whether to run in true sequential model.')
    parser.add_argument('--new-eval', action='store_true', help='Whether to use the new PTB and C4 eval')
    parser.add_argument('--layers-dist', type=str, default='', help='Distribution of layers across GPUs. e.g. 2:1:1 for 2 layers on GPU 0, 1 layer on GPU 1, and 1 layer on GPU 2. Any remaining layers will be assigned to your last GPU.')
    parser.add_argument('--observe',
                        action='store_true',
                        help='Auto upgrade layer precision to higher precision, for example int2 to int4, groupsize 128 to 64. \
            When this feature enabled, `--save` or `--save_safetensors` would be disable.')
    parser.add_argument('--quant-directory', type=str, default=None, help='Specify the directory for export quantization parameters to toml format. `None` means no export by default.')
    parser.add_argument('--save_compressed_delta_dir', type=str, default=None)
    parser.add_argument('--saved_delta_path', type=str, default=None)
    parser.add_argument('--attn_fp16_col', type=int, default=0)
    parser.add_argument('--mlp_fp16_col', type=int, default=0)
    parser.add_argument('--attn_int8_col', type=int, default=0)
    parser.add_argument('--mlp_int8_col', type=int, default=0)
    parser.add_argument('--attn_int4_col', type=int, default=0)
    parser.add_argument('--mlp_int4_col', type=int, default=0)
    parser.add_argument('--attn_int3_col', type=int, default=0)
    parser.add_argument('--mlp_int3_col', type=int, default=0)
    parser.add_argument('--attn_int2_col', type=int, default=0)
    parser.add_argument('--mlp_int2_col', type=int, default=0)
    parser.add_argument('--attn_int1_col', type=int, default=0)
    parser.add_argument('--mlp_int1_col', type=int, default=0)
    parser.add_argument('--save_trained_path', type=str, default=None)     
    parser.add_argument('--bits', nargs='+', type=int)
    
    # training args
    parser.add_argument("--dataset_name", type=str, default="c4")
    parser.add_argument("--subset", type=str, default="en")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--stu_model", type=str, default=None)
    parser.add_argument("--teacher_model", type=str, default=None)
    parser.add_argument("--param_dict", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)           
    
    # adding for config
    parser.add_argument("--config", type=str, help="the path of the config path(yaml)")
    parser.add_argument("--baseModel", type=str, help="the path of base model")
    parser.add_argument("--find_fusion", type=bool, default=False, help="find the fusion rank")
    parser.add_argument("--use_async_find", type=bool, default=False, help="finding use async")
    parser.add_argument("--only_u", type=bool, default=False, help="if true, will only quant u")
    parser.add_argument("--only_v", type=bool, default=False, help="if true, will only quant v")
    parser.add_argument("--rank_config_path", type=str, help="if given, will use the config")
    parser.add_argument("--no_quant", type=bool, default=False, help="if true, will not quant")
    parser.add_argument("--compute_first_layer_loss", type=bool, default=False, help="if true, will compute the origin model and quant model loss")
    parser.add_argument("--v_target_bit", type=float, default=1, help="How many bit to quant v")
    parser.add_argument("--u_target_bit", type=float, default=1, help="How many bit to quant u")
    parser.add_argument("--u_bit", type=int, default=None, help="the bit to quant u")
    parser.add_argument("--v_bit", type=int, default=None, help="the bit to quant v")
    args = parser.parse_args()

    if args.config is not None:
        args = load_args(args)

    assert not (args.only_u and args.only_v), "you can only choose one to be true"
    assert (not any((args.only_u, args.only_v)) and args.no_quant) or not args.no_quant, "if you not want to quant, set only_u or v to be false"
    assert not args.compute_first_layer_loss or (args.compute_first_layer_loss and args.load)
    if args.only_u:
        print("You are only quant u !!!")
    if args.only_v:
        print("You are only quant v !!!")
    if args.no_quant:
        print("using no quant !!!")
    if args.compute_first_layer_loss:
        print(f"we will eval the quant model from path {args.load} with origin model {args.model[0]}")
    if args.u_bit is not None or args.v_bit is not None:
        assert args.find_fusion, "if you set u_bit or v_bit, you should set find_fusion to be true"
        if args.u_bit is not None:
            print("setting u to use ", args.u_bit, "bit")
        if args.v_bit is not None:
            print("setting v to use ", args.v_bit, "bit")

    if not any([args.compute_first_layer_loss,args.benchmark,args.eval,args.test_generation]):
        with open(os.path.join(os.path.dirname(args.save_compressed_delta_dir), 'config.yaml'), 'w') as f:
            safe_dump(vars(args), f)
    return args

def load_args(args):
    with open(args.config, 'r') as f:
        config = safe_load(f)
    for key, value in config.items():
        assert hasattr(args, key), f"Invalid config key: {key}"
        setattr(args, key, value)
    return args

def _preprocess(tokenizer, examples, max_length=128):
    
    # import pdb; pdb.set_trace()
    texts = []
    for question , answer in zip(examples["question"], examples["answer"]):
        text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n ### Instruction:\n{question}\n\n### Response: Let's think step by step. {answer}"
        texts.append(text)
    # examples["text"]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=max_length
    )

def get_dataset(dataset_name, subset, split, size=None):
    if size is None:
        dataset = load_dataset(dataset_name, subset)[split]
        
    else:
        dataset = load_dataset(dataset_name, subset, streaming=True)[split]
        # dataset = load_dataset("gsm8k", "main", streaming=True)[split]
        dataset = dataset.take(size)
        
    
    dataset = load_dataset("json", data_files={"train": "/home/pingbowen/workspace/lora-fusion/UltraEval/datasets/gsm8k/data/dev.jsonl"},split="train",streaming=True)
    dataset = dataset.take(size) # [split]
    # import pdb; pdb.set_trace()
    return dataset

def get_dataloader(dataset, tokenizer, batch_size, num_workers=4, max_length=128):
    dataset = dataset.map(
        lambda examples: _preprocess(tokenizer, examples, max_length),
        batched=True,
        batch_size=batch_size,
        remove_columns=['question', 'answer'],
        # remove_columns=["text", "timestamp", "url"],
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=default_data_collator,
    )
    return dataloader