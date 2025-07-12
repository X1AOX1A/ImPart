"""
The code is modified based on the implementation of Delta-CoMe (https://github.com/thunlp/Delta-CoMe).
"""
import time
import numpy as np
import torch
import torch.nn as nn
import quant
from tqdm import tqdm

from impart_gptq import GPTQ, Observer
from utils import find_layers, DEV, set_seed, get_wikitext2, get_ptb, get_c4, get_ptb_new, get_c4_new, get_loaders, export_quant_table, gen_conditions
from texttable import Texttable
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import gc
from typing import List, Optional, Tuple, Union, Dict
import math
import types
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    rotate_half,
    apply_rotary_pos_emb,
)
try:
    from transformers.models.llama.modeling_llama import repeat_kv
except:
    pass
from utils.utils import parse_args, load_llava
try:
    from model.multiLlama import multiLlamaFroCausalLM
    from model.evalLlama import evalLlamaFroCausalLM
except:
    from model.multiOldLlama import multiLlamaFroCausalLM
    from model.evalOldLlama import evalLlamaFroCausalLM

from datetime import datetime
from multiprocessing import Pool
import os

def get_llama(model):

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    if isinstance(model, list) and len(model) > 1:
        config = AutoConfig.from_pretrained(model[0])
        config._attn_implementation = "eager"
        multi_model = multiLlamaFroCausalLM(config)
        for name in model:
            multi_model.add_model(AutoModelForCausalLM.from_pretrained(name, torch_dtype=my_dtype))
        model = multi_model
    else:
        config = AutoConfig.from_pretrained(model)
        config._attn_implementation = "eager"
        model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=my_dtype, config=config)
    model.seqlen = 2048
    return model


class Delta(nn.Module):

    def __init__(self, base, U, U_mask, S, V, V_mask, name):
        super().__init__()
        self.register_buffer("base", base)
        self.register_buffer("U", None)
        self.register_buffer("U_mask", None)
        self.register_buffer("V", None)
        self.register_buffer("V_mask", None)
        self.register_buffer("S", None)

        self.register_buffer("all_U", None)
        self.register_buffer("all_V", None)
        self.register_buffer("all_S", None)

        self.register_buffer("U_total", U)
        self.register_buffer("U_mask_total", U_mask)
        self.register_buffer("V_total", V)
        self.register_buffer("V_mask_total", V_mask)
        self.register_buffer("S_total", S)
        self.cur_col = 0
        self.pre_col = 0
        self.have_post = True
        self.name = name

    def pre_quant(self, cur_col, pre_col=0, typing=None, check=True):
        # self.base = self.base + self.U[:,:cur_col] @ torch.diag(self.S[:cur_col]) @ self.V[:,:cur_col].T
        # import pdb; pdb.set_trace()
        if check:
            assert self.have_post, "last pre_quant havn't done, please check code"
            self.have_post = False

        # if cur_col <= self.U_total.shape[1]:
        #     self.cur_col = cur_col
        # else:
        #     print("now bit rank out of range!!!, use the max range")
        #     self.cur_col = self.U_total.shape[1]
        self.cur_col = cur_col
        self.pre_col = pre_col
        if typing is None:
            assert isinstance(cur_col, pre_col.__class__), "cur_col and pre should be same class"
            if isinstance(cur_col, int):
                self.U = self.U_total[:,pre_col:cur_col]
                self.U_mask = self.U_mask_total[:,pre_col:cur_col]
                if self.S_total.dim() == 2:
                    self.S = self.S_total[: , pre_col:cur_col]
                else:
                    self.S = self.S_total[pre_col:cur_col]
                self.V = self.V_total[:,pre_col:cur_col]
                self.V_mask = self.V_mask_total[:,pre_col:cur_col]
            elif isinstance(cur_col, (list, tuple)):
                assert len(cur_col) == len(pre_col), "cur and pre length should be same"
                self.U = None
                self.S = None
                self.V = None
                for i, j in zip(pre_col, cur_col):
                    U = self.U_total[:,i:j]
                    U_mask = self.U_mask_total[:,i:j]
                    V = self.V_total[:,i:j]
                    V_mask = self.V_mask_total[:,i:j]
                    if self.S_total.dim() == 2:
                        S = self.S_total[: , i:j]
                    else:
                        S = self.S_total[i:j]
                    
                    if self.U is None:
                        self.U = U
                        self.U_mask = U_mask
                        self.S = S
                        self.V = V
                        self.V_mask = V_mask
                    else:
                        self.U = torch.cat([self.U, U], dim = -1)
                        self.U_mask = torch.cat([self.U_mask, U_mask], dim = -1)
                        self.S = torch.cat([self.S, S], dim = -1)
                        self.V = torch.cat([self.V, V], dim = -1)
                        self.V_mask = torch.cat([self.V_mask, V_mask], dim = -1)
                
        elif typing.lower() == "all":
            self.U = self.U_total
            if self.S_total.dim() == 2:
                self.S = self.S_total
            else:
                self.S = self.S_total
            self.V = self.V_total
            self.cur_col = None #self.V.shape[-1]
            self.pre_col = None #0
        else:
            assert False , "get unknow typing"
        
    def post_quant(self, bit, name, quant_type):
        # import pdb; pdb.set_trace()
        self.have_post = True
        if quant_type.lower() == "u":
            if args.save_trained_path is not None:
                tmp[name + f".U_{bit}"] = self.U
                tmp[name + f".S_{bit}"] = self.S
                tmp[name + f".V_{bit}"] = self.V
                
                if tmp.get(name + ".base") is None:
                    tmp[name + ".base"] = self.base
            if self.all_U is None:
                self.all_U = self.U
                self.all_V = self.V
                self.all_S = self.S
            else:
                self.all_U = torch.cat([self.all_U, self.U], dim=-1)
                self.all_V = torch.cat([self.all_V, self.V], dim=-1)
                self.all_S = torch.cat([self.all_S, self.S], dim=-1)
        else:
            if isinstance(self.cur_col, int):
                self.V_total[:,self.pre_col:self.cur_col] = self.V
            elif isinstance(self.cur_col, (list, tuple)):
                begin = 0
                for i, j in zip(self.pre_col, self.cur_col):
                    num = j - i
                    self.V_total[:,i:j] = self.V[:, begin:begin + num]
                    begin += num
        self.U = None
        self.S = None
        self.V = None

    def clear(self):
        self.U_total = None
        self.V_total = None
        self.S_total = None
        self.U = None
        self.S = None
        self.V = None

    def forward(self, x, gptq: GPTQ=None, quant_type=None, index=None):
        # TODO: This can be faster
        # import pdb; pdb.set_trace()
        if gptq is not None:
            if quant_type == "V":
                # 为查询混合精度量化做准备
                if index is None:
                    gptq.weights = self.S_total
                else:
                    gptq.weights = self.S_total.abs().sum(dim =0)

                for i in range(x.size(0)):
                    gptq.add_batch(x[i].data, x[i].data) #[seq, dim]  #和y无关，只与x有关
            else:
                y = x.clone()
                if index is None:
                    # w_ = (torch.diag(self.S) @ self.V.T).to(x.dtype)
                    w_ = (self.S.unsqueeze(1) * self.V.T).to(x.dtype) # [r, in_dim]
                else:
                    # w_ = (torch.diag(self.S[index]) @ self.V.T).to(x.dtype)
                    w_ = (self.S[index].unsqueeze(1) * self.V.T).to(x.dtype)
                y = y @ w_.T # [batch, seq, r]
            # import pdb; pdb.set_trace()
                for i in range(y.size(0)):
                    gptq.add_batch(y[i].data, y[i].data)

        if self.all_U is not None:
            if index is None:
                # w = (self.base + self.U @ torch.diag(self.S) @ self.V.T).to(x.dtype)
                w = (self.base + ((self.all_U * self.all_S.unsqueeze(0)) @ self.all_V.T).to(self.base.dtype)).to(x.dtype)
            else:
                # w = (self.base[index] + self.U @ torch.diag(self.S[index]) @ self.V.T).to(x.dtype)
                w = (self.base + ((self.all_U * self.all_S[index].unsqueeze(0)) @ self.all_V.T).to(self.base.dtype)).to(x.dtype)
            return x @ w.T


        w = (self.base).to(x.dtype)

        return x @ w.T

    def get_in_out_shape(self):
        if self.S_total.dim() == 2:
            return (self.base[0].shape[0], self.base[0].shape[1])
        return (self.base.shape[0], self.base.shape[1])
    
def llama_quant_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # import pdb; pdb.set_trace()
    index = kwargs.get("index", None)
    bsz, q_len, _ = hidden_states.size()

    quant_type = gptqs.get("quant_type",None)
    gptq_q , gptq_k , gptq_v , gptq_o = None, None, None, None
    
    for k,v in gptqs.items():
        if "q_proj" in k:
            gptq_q = v
        elif "k_proj" in k:
            gptq_k = v
        elif "v_proj" in k:
            gptq_v = v
        elif "o_proj" in k:
            gptq_o = v   
             

    # import pdb; pdb.set_trace()
    query_states = self.q_proj(hidden_states,gptq_q,quant_type, index)
    key_states = self.k_proj(hidden_states,gptq_k,quant_type, index)
    value_states = self.v_proj(hidden_states,gptq_v,quant_type, index)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    if "llama-3" in args.model[0].lower():
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
    
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output,gptq_o,quant_type,index)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def llama_quant_mlp_forward(self, x, index=None):
    
    quant_type = gptqs.get("quant_type",None)
    gptq_up , gptq_gate , gptq_down  = None, None, None
    for k,v in gptqs.items():
        if "up_proj" in k:
            gptq_up = v
        elif "gate_proj" in k:
            gptq_gate = v
        elif "down_proj" in k:
            gptq_down = v     
    
    return self.down_proj(self.act_fn(self.gate_proj(x,gptq_gate,quant_type, index=index)) * self.up_proj(x,gptq_up,quant_type, index=index),gptq_down,quant_type, index=index)


def enable_llama_quant_forward(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_llama_quant_forward(
                module,
            )
        
        if isinstance(module, LlamaAttention): 
            model._modules[name].forward = types.MethodType(
                llama_quant_attn_forward, model._modules[name]
            )
            
        if isinstance(module, LlamaMLP): 
            model._modules[name].forward = types.MethodType(
                llama_quant_mlp_forward, model._modules[name]
            )

def set_delta(model,state_dict):
    col = {}
    for name, module in model.named_modules():
        if "vision" in name:
            continue        
        if "self_attn" in name or "mlp" in name:
            for subname, submodule in module.named_children():
                if "proj" in subname:
                    setattr(module, subname, None)
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    base = state_dict[name + "." + subname + ".base"]
                    U = state_dict[name + "." + subname + ".U"]
                    U_mask = state_dict[name + "." + subname + ".U_mask"]
                    V = state_dict[name + "." + subname + ".V"]
                    V_mask = state_dict[name + "." + subname + ".V_mask"]
                    S = state_dict[name + "." + subname + ".S"]
                    col[name] = (U.shape, S.shape, V.shape)
                    setattr(module, subname, Delta(base=base, U=U, U_mask=U_mask, S=S, V=V, V_mask=V_mask, name=name + "." + subname))
                    # import pdb; pdb.set_trace()

    print(col)
    return model

def get_index_dict(args):
    index_dict = {}
    
    for name in ["self_attn","mlp"]:
        for bit in args.bits:
            if bit == 8:
                if "self_attn" in name:
                    pre_col, cur_col = args.attn_fp16_col, args.attn_fp16_col + args.attn_int8_col
                else:
                    pre_col, cur_col = args.mlp_fp16_col , args.mlp_fp16_col + args.mlp_int8_col
            elif bit == 4:
                if "self_attn" in name:
                    pre_col, cur_col = args.attn_fp16_col + args.attn_int8_col, args.attn_fp16_col + args.attn_int8_col + args.attn_int4_col  
                else: 
                    pre_col, cur_col = args.mlp_fp16_col + args.mlp_int8_col, args.mlp_fp16_col + args.mlp_int8_col + args.mlp_int4_col  
            elif bit == 3:
                if "self_attn" in name:
                    pre_col, cur_col = args.attn_fp16_col + args.attn_int8_col + args.attn_int4_col ,args.attn_fp16_col + args.attn_int8_col + args.attn_int4_col + args.attn_int3_col  
                else: 
                    pre_col, cur_col = args.mlp_fp16_col + args.mlp_int8_col + args.mlp_int4_col , args.mlp_fp16_col + args.mlp_int8_col + args.mlp_int4_col + args.mlp_int3_col  
            elif bit == 2:
                if "self_attn" in name:
                    pre_col, cur_col = args.attn_fp16_col + args.attn_int8_col + args.attn_int4_col + args.attn_int3_col ,args.attn_fp16_col + args.attn_int8_col + args.attn_int4_col + args.attn_int3_col + args.attn_int2_col  
                else: 
                    pre_col, cur_col = args.mlp_fp16_col + args.mlp_int8_col + args.mlp_int4_col + args.mlp_int3_col , args.mlp_fp16_col + args.mlp_int8_col + args.mlp_int4_col + args.mlp_int3_col + args.mlp_int2_col   
            elif bit == 1:
                if "self_attn" in name:
                    pre_col, cur_col = args.attn_fp16_col + args.attn_int8_col + args.attn_int4_col + args.attn_int3_col + args.attn_int2_col   ,args.attn_fp16_col + args.attn_int8_col + args.attn_int4_col + args.attn_int3_col + args.attn_int2_col + args.attn_int1_col  
                else: 
                    pre_col, cur_col = args.mlp_fp16_col + args.mlp_int8_col + args.mlp_int4_col + args.mlp_int3_col + args.mlp_int2_col   , args.mlp_fp16_col + args.mlp_int8_col + args.mlp_int4_col + args.mlp_int3_col + args.mlp_int2_col + args.mlp_int1_col                  
            if  "self_attn" in name and index_dict.get(f"self_attn_{bit}") is None: # index for delta-compression
                index_dict[f"self_attn_{bit}"] = (pre_col, cur_col)
            elif "mlp" in name and index_dict.get(f"mlp_{bit}") is None:
                index_dict[f"mlp_{bit}"] = (pre_col, cur_col)
    print(index_dict)
    return index_dict                    


gptqs = {} 
@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    print('Starting ...')
    state_dict = torch.load(args.saved_delta_path)
    num = 1
    if "names" in state_dict:
        num = len(state_dict["names"])
        del model
        model = get_llama([os.path.join(os.path.dirname(args.saved_delta_path), i) for i in state_dict["names"]])
        setattr(args, "svd_names", state_dict["names"])
    print(f"find {num} different svd names",)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    # import pdb; pdb.set_trace()
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)
    #args.nsamples *= num
    state_dict = load_warpper(state_dict, model, args.baseModel)

    dtype = next(iter(model.parameters())).dtype

    # import pdb; pdb.set_trace()
    # inps = torch.zeros((args.nsamples * num, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)  # todo: "cpu"
    inps = torch.zeros((args.nsamples * num, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev).to("cpu")
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp  # todo: to cpu
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for index in range(num):
        for batch in dataloader:
            try:
                if num != 1:
                    model(batch[0].to(dev), index = index)
                else:
                    model(batch[0].to(dev))
            except ValueError:
                pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    # import pdb; pdb.set_trace()
    # outs = torch.zeros_like(inps, dtype=dtype, device=dev)  #  cpu
    outs = torch.zeros_like(inps, dtype=dtype, device=dev).to("cpu")
    attention_mask = cache['attention_mask'].repeat(args.batch_size, 1, 1, 1)
    position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    observer = Observer()
    
    
    # import pdb; pdb.set_trace()
    all_find_bits = {}
    
    model = set_delta(model,state_dict)
    enable_llama_quant_forward(model)
    if not args.find_fusion:
        index_dict = get_index_dict(args)
    else:
        os.makedirs(os.path.join(os.path.dirname(args.save_compressed_delta_dir) , "find_mixquant_config"), exist_ok=True)

    bits = args.bits
    for i in tqdm(range(len(layers))):
    # for i in tqdm(range(1)):
        # import pdb; pdb.set_trace()
        layer_begin = time.time()
        layer = layers[i].to(dev)
        all_index_dict = {}
        # 处理16bit
        if (args.attn_fp16_col != 0 or args.mlp_fp16_col != 0) and not args.find_fusion:
            params = ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj','self_attn.o_proj','mlp.up_proj', 'mlp.gate_proj','mlp.down_proj']
            for param in params:
                cur_col = args.attn_fp16_col if "self_attn" in param else args.mlp_fp16_col
                layer.get_submodule(param).pre_quant(pre_col=0, cur_col=cur_col)
                layer.get_submodule(param).post_quant(bit=16, name=f"model.layers.{i}." + param, quant_type="U")

        if args.true_sequential:
                sequential = [['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],['self_attn.o_proj'],['mlp.up_proj', 'mlp.gate_proj'],['mlp.down_proj']]

        for names in sequential: #进行quant     
            if args.find_fusion:
                if args.rank_config_path is not None:
                    if names == sequential[0]:
                        from yaml import safe_load
                        with open(os.path.join(args.rank_config_path, f"layer {i}.yaml"), 'r') as f:
                            all_index_dict = safe_load(f)
                        for key in all_index_dict:
                            key = key.split('_')
                            name = "_".join(key[:-1])
                            bit = int(key[-1])
                            if name not in all_find_bits:
                                all_find_bits[name] = []
                            all_find_bits[name].append(bit)
                        for name in all_find_bits:
                            all_find_bits[name].sort(reverse=True)
                else:
                    from find_bits import UV_find_optimal
                    print("finding rank")
                    gptqs["quant_type"] =  "V"
                    quant_type = "V"
                    for name in names:
                        layer.get_submodule(name).pre_quant(pre_col=0, cur_col=0, typing="all", check=False)
                        name = f"model.layers.{i}." + name        
                        gptq = GPTQ(model.get_submodule(name), quant_type=quant_type,observe=args.observe)
                        gptqs[name] = gptq
                    
                    for j in range(0,args.nsamples * num, args.batch_size):
                        if num == 1:
                            # layer(inps[j:j+args.batch_size], attention_mask=attention_mask, position_ids=position_ids)  # inps to cuda
                            layer(inps[j:j+args.batch_size], attention_mask=attention_mask, position_ids=position_ids)  # inps to cuda
                        else:
                            # layer(inps[j:j+args.batch_size].to("cuda"), attention_mask=attention_mask, position_ids=position_ids, index=j // args.nsamples)
                            layer(inps[j:j+args.batch_size].to("cuda"), attention_mask=attention_mask, position_ids=position_ids, index=j // args.nsamples)
                    funs = []
                    pool = Pool(3)
                    find_time = time.time()
                    for k, v in gptqs.items():
                        if "proj" in k:
                            delta: Delta = model.get_submodule(k) 
                            gptq: GPTQ  = v
                            loss: torch.Tensor = gptq.get_quant_loss(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order)
                            in_s, out_s = delta.get_in_out_shape()
                            assert args.v_target_bit == args.u_target_bit, "not implement for u, v bit not equal"
                            if args.u_bit is not None or args.v_bit is not None:
                                assert args.u_bit is not None, "not implement for v"
                                t = torch.arange(0,17) * in_s / in_s / out_s # in_s代表V的输入维度
                                t[1:] = t[1:] + args.u_bit * out_s / in_s / out_s # 加入U的输入维度， 0bit默认u也是0bit
                                t = t * args.v_target_bit # 选择目标比特数
                            else:
                                t = torch.arange(0,17) * (in_s + out_s) * args.v_target_bit / in_s / out_s # 这里存储空间包含了相同精度的 U
                            # UV_find_optimal(loss.cpu().numpy(), t.cpu().numpy())
                            if args.use_async_find:
                                funs.append((k , pool.apply_async(UV_find_optimal, args=(loss.cpu().numpy(), t.cpu().numpy()))))
                            else:
                                layer_bits, pre_cols, cur_cols, count = UV_find_optimal(loss.cpu().numpy(), t.cpu().numpy())
                                print(f"finding {k} done, using {count} ranks")
                                layer_bits = layer_bits.tolist()
                                if 0 in layer_bits:
                                    layer_bits.remove(0)
                                all_find_bits[k] = layer_bits
                                for idx, bit in enumerate(layer_bits):
                                    all_index_dict[k + f'_{bit}'] = [pre_cols[idx], cur_cols[idx]]
                    pool.close()
                    pool.join()
                    for k, f in funs:
                        layer_bits, pre_cols, cur_cols, count = f.get()
                        print(f"finding {k} done, using {count} ranks")
                        layer_bits = layer_bits.tolist()
                        if 0 in layer_bits:
                            layer_bits.remove(0)
                        all_find_bits[k] = layer_bits
                        for idx, bit in enumerate(layer_bits):
                            all_index_dict[k + f'_{bit}'] = [pre_cols[idx], cur_cols[idx]]
                    gptqs.clear()
                    print(f"finding over, cost times: {(time.time() - find_time):.2f}s")

            if args.no_quant:
                for name in names:
                    layer_name = f"model.layers.{i}." + name 
                    if args.find_fusion:
                        bits = all_find_bits[layer_name]
                    print(f'{name}  {i+1}/{len(layers)}..')
                    for bit in bits:
                        if args.find_fusion:
                            pre_col = all_index_dict[layer_name + f'_{bit}'][0]
                            cur_col = all_index_dict[layer_name + f'_{bit}'][1]
                        else:
                            pre_col = index_dict[f"self_attn_{bit}"][0] if "self_attn" in name else index_dict[f"mlp_{bit}"][0]
                            cur_col = index_dict[f"self_attn_{bit}"][-1] if "self_attn" in name else index_dict[f"mlp_{bit}"][-1]
                        layer.get_submodule(name).pre_quant(pre_col=pre_col,cur_col=cur_col)
                        layer.get_submodule(name).post_quant(bit=bit,name=f"model.layers.{i}." + name,quant_type="U")
            else:
                for ii in range(2):
                    gptqs.clear()
                    quant_type = "V" if ii == 0 else "U"
                    gptqs["quant_type"] = quant_type

                    if args.only_u and quant_type != "U":
                        continue
                    if args.only_v and quant_type != "V":
                        continue
            
                    
                    if quant_type == "V":
                        # V可以提前算，对于U的混合精度则不行
                        for name in names:
                            name = f"model.layers.{i}." + name
                            gptq = GPTQ(model.get_submodule(name), quant_type=quant_type, observe=args.observe)
                            # import pdb; pdb.set_trace()
                            gptqs[name] = gptq
                        # 计算海森矩阵
                        for j in range(0, args.nsamples * num, args.batch_size):
                            # import pdb; pdb.set_trace()
                            if num == 1:
                                # outs[j:j+args.batch_size] = layer(inps[j:j+args.batch_size], attention_mask=attention_mask, position_ids=position_ids)[0]
                                outs[j:j+args.batch_size] = layer(inps[j:j+args.batch_size].to("cuda"), attention_mask=attention_mask, position_ids=position_ids)[0].to("cpu")
                            else:
                                # outs[j:j+args.batch_size] = layer(inps[j:j+args.batch_size], attention_mask=attention_mask, position_ids=position_ids, index=j // args.nsamples)[0]
                                outs[j:j+args.batch_size] = layer(inps[j:j+args.batch_size].to("cuda"), attention_mask=attention_mask, position_ids=position_ids, index=j // args.nsamples)[0].to("cpu")

                    for name in names:
                        layer_name = f"model.layers.{i}." + name 
                        if args.find_fusion:
                            bits = all_find_bits[layer_name]
                        print(f'Quantizing  {name}  {i+1}/{len(layers)}..')
                        print('+----------------+--------------+------------+-----------+-------+-------+-----+')
                        print('|       name     | weight_error | fp_inp_SNR | q_inp_SNR | time  | rank  | bit |')
                        print('+================+==============+============+===========+=======+=======+=====+')
                        for bit in bits:
                            if args.find_fusion:
                                pre_col = all_index_dict[layer_name + f'_{bit}'][0]
                                cur_col = all_index_dict[layer_name + f'_{bit}'][1]
                                if quant_type == "U" and args.u_bit is not None:
                                    for bit in bits[1:]:
                                        pre_col_tmp = all_index_dict[layer_name + f'_{bit}'][0]
                                        pre_col = pre_col + pre_col_tmp
                                        cur_col_tmp = all_index_dict[layer_name + f'_{bit}'][1]
                                        cur_col = cur_col + cur_col_tmp
                                    
                            else:
                                pre_col = index_dict[f"self_attn_{bit}"][0] if "self_attn" in name else index_dict[f"mlp_{bit}"][0]
                                cur_col = index_dict[f"self_attn_{bit}"][-1] if "self_attn" in name else index_dict[f"mlp_{bit}"][-1]
                            # import pdb; pdb.set_trace()
                            layer.get_submodule(name).pre_quant(pre_col=pre_col,cur_col=cur_col)
                            # import pdb; pdb.set_trace()

                            if quant_type == "U":
                                gptqs.clear()
                                gptqs["quant_type"] = quant_type
                                gptq = GPTQ(model.get_submodule(layer_name), quant_type=quant_type, observe=args.observe)
                                gptqs[layer_name] = gptq
                                for j in range(0, args.nsamples * num, args.batch_size):
                                    if num == 1:
                                        # outs[j:j+args.batch_size] = layer(inps[j:j+args.batch_size], attention_mask=attention_mask, position_ids=position_ids)[0]
                                        outs[j:j+args.batch_size] = layer(inps[j:j+args.batch_size].to("cuda"), attention_mask=attention_mask, position_ids=position_ids)[0].to("cpu")
                                    else:
                                        # outs[j:j+args.batch_size] = layer(inps[j:j+args.batch_size], attention_mask=attention_mask, position_ids=position_ids, index=j // args.nsamples)[0]
                                        outs[j:j+args.batch_size] = layer(inps[j:j+args.batch_size].to("cuda"), attention_mask=attention_mask, position_ids=position_ids, index=j // args.nsamples)[0].to("cpu")

                            elif quant_type == "V":
                                gptqs[layer_name].layer = model.get_submodule(layer_name)
                                gptqs[layer_name].weights = model.get_submodule(layer_name).S
                            else:
                                assert False , f"unknow quant type {quant_type}"

                            for k,v in gptqs.items():
                                if "proj" in k and k==layer_name:
                                    gptq = v
                                    if args.u_bit is not None and quant_type == "U":
                                        bit = args.u_bit
                                    gptq.quantizer.configure(bit, perchannel=True, sym=args.sym, mse=False)
                                    scale, zero, g_idx, error = gptq.fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=k.rsplit(".")[-1] + f".{quant_type}", bit=bit)
                                    quantizers['%s.%s.%d' % (k,gptqs['quant_type'],bit)] = (gptqs[layer_name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), bit, args.groupsize)
                                    # import pdb; pdb.set_trace()
                                    if not any((args.only_u, args.only_v)):
                                        layer.get_submodule(name).post_quant(bit=bit,name=f"model.layers.{i}." + name,quant_type=quant_type)
                                    else:
                                        layer.get_submodule(name).post_quant(bit=bit,name=f"model.layers.{i}." + name,quant_type="U")
                                    # import pdb; pdb.set_trace()

                            if args.u_bit is not None and args.find_fusion and quant_type == "U":
                                break
                            
                        print('+----------------+--------------+------------+-----------+-------+-------+-----+')
                        print('\n')
                    gptqs.clear()

        for j in range(0, args.nsamples * num, args.batch_size):
                if num == 1:
                    # outs[j:j+args.batch_size] = layer(inps[j:j+args.batch_size], attention_mask=attention_mask, position_ids=position_ids)[0]
                    outs[j:j+args.batch_size] = layer(inps[j:j+args.batch_size].to("cuda"), attention_mask=attention_mask, position_ids=position_ids)[0].to("cpu")
                else:
                    # outs[j:j+args.batch_size] = layer(inps[j:j+args.batch_size], attention_mask=attention_mask, position_ids=position_ids, index=j // args.nsamples)[0]
                    outs[j:j+args.batch_size] = layer(inps[j:j+args.batch_size].to("cuda"), attention_mask=attention_mask, position_ids=position_ids, index=j // args.nsamples)[0].to("cpu")
        inps, outs = outs, inps
            
        for names in sequential:
            for name in names:
                    layer.get_submodule(name).clear()
        # layer
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        if args.find_fusion:
            from yaml import safe_dump
            with open(os.path.join(os.path.dirname(args.save_compressed_delta_dir) , "find_mixquant_config", f"layer {i}.yaml"), "w") as f:
                safe_dump(all_index_dict, f)
        print(f"layer {i} over, costing time {(time.time() - layer_begin):.2f}s")
    
    model.config.use_cache = use_cache

    return quantizers, model

@torch.no_grad()
def find_optimal_bits(layers: Delta, dataloder, dev):
    pass

@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...') 
    if hasattr(testenc, "input_ids"):
        testenc = testenc.input_ids
        nsamples = testenc.numel() // model.seqlen
    else:
        testenc = torch.cat([i[0] for i in testenc], dim = 0).reshape(1,-1)
        nsamples = args.nsamples

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask'].repeat(args.batch_size,1,1,1)
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = quant.Quantizer()
                quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantizer.quantize(W).to(next(iter(layer.parameters())).dtype)

        for j in range(0,args.nsamples, args.batch_size):
            outs[j:j+args.batch_size] = layer(inps[j:j+args.batch_size], attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache
    return float(ppl.cpu().item())

# TODO: perform packing on GPU
def llama_pack(model, quantizers, wbits, groupsize):
    layers = find_layers(model,[Delta])
    index_dict = get_index_dict(args)
    quant.make_quant_linear(model, quantizers, args.bits, groupsize,index_dict=index_dict)
    qlayers = find_layers(model, [quant.MixquantLinear])
    # import pdb; pdb.set_trace()
    print('Packing ...')
    for name in qlayers:
        qlayers[name].pack(quantizers,name,layers,index_dict)
    print('Done.')
    return model


def load_quant(model, checkpoint, wbits, groupsize=-1, fused_mlp=True, eval=True, warmup_autotune=False):
    from transformers import LlamaConfig, LlamaForCausalLM, modeling_utils
    config = LlamaConfig.from_pretrained(model)
    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()
    state_dict = torch.load(args.saved_delta_path)
    state_dict = load_warpper(state_dict, model, args.baseModel)
    model = set_delta(model,state_dict)
    layers = find_layers(model,[Delta])
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    index_dict = get_index_dict(args)
    quant.make_quant_linear(model, layers, args.bits, groupsize,index_dict=index_dict)
    del layers
    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint))
    if eval:
        quant.make_mix_quant_attn(model)
        quant.make_quant_norm(model)
        if fused_mlp:
            quant.make_fused_mlp(model)

    if warmup_autotune:
        quant.autotune_warmup_linear(model, transpose=not (eval))
        if eval and fused_mlp:
            quant.autotune_warmup_fused(model)
    model.seqlen = 2048
    print('Done.')

    return model


def llama_multigpu(model, gpus, gpu_dist):
    model.model.embed_tokens = model.model.embed_tokens.to(gpus[0])
    if hasattr(model.model, 'norm') and model.model.norm:
        model.model.norm = model.model.norm.to(gpus[0])
    import copy
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[0])

    cache = {'mask': None, 'position_ids': None}

    class MoveModule(nn.Module):

        def __init__(self, module, invalidate_cache):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
            self.invalidate_cache=invalidate_cache

        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)

            if cache['mask'] is None or cache['mask'].device != self.dev or self.invalidate_cache:
                cache['mask'] = kwargs['attention_mask'].to(self.dev)
            kwargs['attention_mask'] = cache['mask']

            if cache['position_ids'] is None or cache['position_ids'].device != self.dev or self.invalidate_cache:
                cache['position_ids'] = kwargs['position_ids'].to(self.dev)
            kwargs['position_ids'] = cache['position_ids']
            
            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.model.layers
    from math import ceil
    if not gpu_dist:
        pergpu = ceil(len(layers) / len(gpus))
        for i in range(len(layers)):
            layers[i] = MoveModule(layers[i].to(0 if i == 0 or i == len(layers) -1 else gpus[(i-1) // pergpu]), i==0)
    else:
        assert gpu_dist[0] >= 2, "At least two layers must be on GPU 0."
        assigned_gpus = [0] * (gpu_dist[0]-1)
        for i in range(1, len(gpu_dist)):
            assigned_gpus = assigned_gpus + [i] * gpu_dist[i]

        remaining_assignments = len(layers)-len(assigned_gpus) - 1
        if remaining_assignments > 0:
            assigned_gpus = assigned_gpus + [-1] * remaining_assignments

        assigned_gpus = assigned_gpus + [0]

        for i in range(len(layers)):
            layers[i] = MoveModule(layers[i].to(gpus[assigned_gpus[i]]), i==0)

    model.gpus = gpus


def benchmark(model, input_ids, check=False):
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)
    torch.cuda.synchronize()

    cache = {'past': None}

    def clear_past(i):

        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None

        return tmp

    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()

    max_memory = 0
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(input_ids[:, i:i + 1], past_key_values=cache['past'], attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1)))
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            if hasattr(model, 'gpus'):
                mem_allocated = sum(torch.cuda.memory_allocated(gpu) for gpu in model.gpus) / 1024 / 1024
            else:
                mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            max_memory = max(max_memory, mem_allocated)
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
            cache['past'] = list(out.past_key_values)
            del out
        sync()
        print('Median:', np.median(times))
        if check:
            print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())
            print('max memory(MiB):', max_memory)

@torch.no_grad()
def load_warpper(svd, model, base_model) -> Dict[str, torch.Tensor]:
    if "U" not in svd:
        return svd
    assert base_model is not None or isinstance(base_model, str), f"need the path of base model to switch format"
    print(f"loading base model from {base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=my_dtype)
    new_state = {}
    for name, module in base_model.named_modules():  
            for subname, submodule in module.named_children():
                final_name = name + "." + subname + ".weight"
                if final_name in svd['U']:
                    new_state[name + "." + subname + ".base"] = submodule.weight.data
                    if isinstance(svd['sigmas'][final_name], list):
                        new_state[name + "." + subname + ".S"] = torch.stack(svd['sigmas'][final_name])
                        tmp_val = new_state[name + "." + subname + ".S"].abs()
                        idx = tmp_val.sum(dim = 0).argsort(descending=True)
                        new_state[name + "." + subname + ".S"] = new_state[name + "." + subname + ".S"][:, idx]
                        new_state[name + "." + subname + ".U"] = svd['U'][final_name][:, idx]
                        new_state[name + "." + subname + ".V"] = svd['V'][final_name][:, idx]
                    else:
                        new_state[name + "." + subname + ".S"] = svd['sigmas'][final_name]
                        new_state[name + "." + subname + ".U"] = svd['U'][final_name]
                        new_state[name + "." + subname + ".V"] = svd['V'][final_name]
    return new_state

@torch.no_grad()
def save_compressed_delta(save_compressed_delta_dir,model,index = None):
    compressed_delta = dict()

    for name, module in tqdm(model.named_modules()):
        # import pdb; pdb.set_trace()
        if "vision_tower" in name:
            continue
        
        if "self_attn" in name or "mlp" in name:
            for subname, submodule in module.named_children():
                if "proj" in subname:
                    # import pdb; pdb.set_trace()
                    base = model.get_submodule(name + "." + subname).base
                    U = model.get_submodule(name + "." + subname).all_U
                    S = model.get_submodule(name + "." + subname).all_S
                    V = model.get_submodule(name + "." + subname).all_V

                    if U is None:
                        print(f"{name} not find quant")
                        continue

                    base = base.to("cuda").to(my_dtype)
                    U = U.to("cuda").to(my_dtype)
                    S = S.to("cuda").to(my_dtype)
                    V = V.to("cuda").to(my_dtype)

                    if index is None:
                        # import pdb; pdb.set_trace()
                        delta = (U @ torch.diag(S) @ V.t())
                    else:
                        delta = (U @ torch.diag(S[index]) @ V.t())
                    
                    if args.save_trained_path is not None:
                        tmp[name + "." + subname + f".U_{args.bits[-1]}"] = U
                        if index is None:
                            tmp[name + "." + subname + f".S_{args.bits[-1]}"] = S
                        else:
                            tmp[name + "." + subname + f".S_{args.bits[-1]}"] = S[index]
                        tmp[name + "." + subname + f".V_{args.bits[-1]}"] = V

                    # signs = torch.sign(delta)
                    # mask = signs == 0
                    # signs[mask] = 1
                    # delta = signs * coeff_dict[name + "." + subname + ".coeff"]
                    '''
                    sign_u,sign_v = torch.sign(U) , torch.sign(V)
                    mask_u , mask_v = sign_u == 0 , sign_v == 0 
                    sign_u[mask_u] = 1 
                    sign_v[mask_v] = 1
                    U , V = sign_u * coeff_u, sign_v * coeff_v
                    '''

                    compressed_delta[name + "." + subname + ".weight"] = (base + delta).to(my_dtype).to("cpu")
    
    torch.save(compressed_delta, save_compressed_delta_dir)
    if args.save_trained_path is not None:
        torch.save(tmp, args.save_trained_path)


@torch.no_grad()
def compare_loss(fisrt_model, second_model, dataloader, dev):
    print('Starting ...')
    num = 1
    print(f"find {num} different svd names",)
    fisrt_model.config.use_cache = False
    second_model.config.use_cache = False
    layers = fisrt_model.model.layers
    fisrt_model.model.embed_tokens = model.model.embed_tokens.to(dev)
    fisrt_model.model.norm = model.model.norm.to(dev)

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples * num, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}
    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for index in range(num):
        for batch in dataloader:
            try:
                if num != 1:
                    model(batch[0].to(dev), index = index)
                else:
                    model(batch[0].to(dev))
            except ValueError:
                pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    fisrt_model.model.embed_tokens = model.model.embed_tokens.cpu()
    fisrt_model.model.norm = model.model.norm.cpu()
    attention_mask = cache['attention_mask'].repeat(args.batch_size, 1,1,1)
    position_ids = cache['position_ids']

    def get_output(layer):
        layer.to(dev)
        outputs = {}
        for j in range(0,args.nsamples * num, args.batch_size):
            output = layer(inps[j:j+args.batch_size], attention_mask=attention_mask, position_ids=position_ids)[-1]
            for i in output:
                if i not in outputs:
                    outputs[i] = output[i]
                else:
                    outputs[i] = torch.cat([outputs[i], output[i]], dim = 0)
        layer.to("cpu")
        return outputs
    fisrt_model_output = get_output(fisrt_model.model.layers[0])
    second_model_output = get_output(second_model.model.layers[0])
    loss = {}
    all_loss = 0
    for i in fisrt_model_output:
        loss[i] = fisrt_model_output[i].to(DEV) - second_model_output[i].to(DEV)
        loss[i] = float(loss[i].norm().cpu())
        all_loss += loss[i]
    loss["sum"] = all_loss
    return loss


if __name__ == '__main__':
    now = datetime.now()
    log_message = f"{now.year}-{now.month:02d}-{now.day:02d} {now.hour:02d}:{now.minute:02d}:{now.second:02d}"
    print("<< Begin Time>> :", log_message)
    args = parse_args()

    my_dtype = torch.float16 if args.my_dtype == "fp16" else torch.bfloat16
    print("<< My Dtype>> :", my_dtype)
    print("<< My Dtype>> :", my_dtype)

    assert args.nsamples % args.batch_size == 0 , ""
    index_dict = dict()
    if args.layers_dist:
        gpu_dist = [int(x) for x in args.layers_dist.split(':')]
    else:
        gpu_dist = []
    print(args.model)
    if type(args.load) is not str:
        args.load = args.load.as_posix()

    if args.load:
        # model=""
        # setattr(model, "seqlen", 2048)
        pass
        # model = load_quant(args.model[0], args.load, args.wbits, args.groupsize)
    else:
        if "llava" not in args.model[0].lower():
            model = get_llama(args.model[0])
            model.eval()

            if "llama-3" in args.model[0].lower():
                model.seqlen = 4096
        else:
            model = load_llava(args.model[0],"cuda" if torch.cuda.is_available() else "cpu")
            if not hasattr(model, 'seqlen'):
                model.seqlen = 2048
            # import pdb ; pdb.set_trace()

    dataloader, testloader = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model[0], seqlen=2048) # model.seqlen
    # import pdb ; pdb.set_trace()
    if not args.load and args.wbits < 16 and not args.nearest:
        tick = time.time()

        # if args.saved_delta_path is not None:
        tmp = dict()

        quantizers, model = llama_sequential(model, dataloader, DEV)

        if args.save_compressed_delta_dir is not None:
                if args is not None and hasattr(args, "svd_names") and args.svd_names is not None:
                    for index, name in enumerate(args.svd_names):
                        save_compressed_delta(os.path.join(os.path.dirname(args.save_compressed_delta_dir), name + '.svd'), model, index)
                else:
                    save_compressed_delta(args.save_compressed_delta_dir, model, index=None)

        print(time.time() - tick)

    if args.compute_first_layer_loss:
        from yaml import safe_dump

        config = AutoConfig.from_pretrained(args.model[0])
        config._attn_implementation = "eager"
        model = evalLlamaFroCausalLM.from_pretrained(args.model[0], torch_dtype=my_dtype, config=config)
        model.eval()
        quant_model = evalLlamaFroCausalLM.from_pretrained(args.load, torch_dtype=my_dtype, config=config)
        quant_model.eval()
        model.seqlen = 2048
        quant_model.seqlen = 2048
        loss = compare_loss(model, quant_model, dataloader, DEV)
        print(loss)
        ppl = llama_eval(quant_model, dataloader, DEV)
        loss["ppl"] = ppl
        with open(os.path.join(os.path.dirname(args.load), "first_layer_loss.yaml"), 'w') as f:
            safe_dump({"layer 1" : loss}, f)

    # import pdb; pdb.set_trace()
    if args.benchmark:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            llama_multigpu(model, gpus, gpu_dist)
        else:
            model = model.to(DEV)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, :args.benchmark]
            benchmark(model, input_ids, check=args.check)

    if args.eval:
        datasets = ['wikitext2', 'ptb', 'c4']
        if args.new_eval:
            datasets = ['wikitext2', 'ptb-new', 'c4-new']
        for dataset in datasets:
            dataloader, testloader = get_loaders(dataset, seed=args.seed, model=args.model[0], seqlen=model.seqlen)
            print(dataset)
            llama_eval(model, testloader, DEV)

    if args.test_generation:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            llama_multigpu(model, gpus, gpu_dist)
        else:
            model = model.to(DEV)

        from transformers import LlamaTokenizer, TextStreamer
        tokenizer = LlamaTokenizer.from_pretrained(args.model[0], use_fast=False)
        input_ids = tokenizer(["The capital of New Mexico is"], return_tensors="pt").input_ids.to(gpus[0])
        streamer = TextStreamer(tokenizer)
        with torch.no_grad():
            generated_ids = model.generate(input_ids, streamer=streamer)

    if args.quant_directory is not None:
        export_quant_table(quantizers, args.quant_directory)

    if not args.observe and args.save:
        # import pdb ; pdb.set_trace()
        llama_pack(model, quantizers, args.wbits, args.groupsize)
        torch.save(model.state_dict(), args.save)

    if not args.observe and args.save_safetensors:
        llama_pack(model, quantizers, args.wbits, args.groupsize)
        from safetensors.torch import save_file as safe_save
        state_dict = model.state_dict()
        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        safe_save(state_dict, args.save_safetensors)
    now = datetime.now()

    log_message = f"{now.year}-{now.month:02d}-{now.day:02d} {now.hour:02d}:{now.minute:02d}:{now.second:02d}"
    print("ending time:", log_message)