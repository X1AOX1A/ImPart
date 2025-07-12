import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import gc
from task_vector import TaskVector


class Merger:
    def __init__(self, merge_method: str = None):
        """
        Methods for model merge
        :param merge_method: str
            e.g. avg
            e.g. ta
            e.g. ties_0.8_1.0
        """
        if merge_method:
            tmp = merge_method.split("_")
            self.merge_config = {"merge_method": tmp[0]}
            if tmp[0] == "ties":
                self.merge_config["param_value_mask_rate"] = float(tmp[1])
        else:
            self.merge_config = None
            print("No merge method")

    @staticmethod
    def copy_params_to_model(params: dict, model: nn.Module):
        merged_model = copy.deepcopy(model)
        for param_name, param_value in merged_model.named_parameters():
            if param_name in params:
                param_value.data.copy_(params[param_name])
        return merged_model

    @torch.no_grad()
    def task_arithmetic(self, models_to_merge_task_vectors: list):
        print("TA start merging!!!!!!!!")
        merged_task_vector = copy.deepcopy(models_to_merge_task_vectors[0])
        for index in range(1, len(models_to_merge_task_vectors)):
            merged_task_vector = merged_task_vector + models_to_merge_task_vectors[index]
        return merged_task_vector

    def ties_merging(self, models_to_merge_task_vectors: list, param_value_mask_rate: float = 0.8):
        # import pdb; pdb.set_trace()
        def mask_smallest_magnitude_param_values(
                flattened_models_to_merge_param: torch.Tensor,
                param_value_mask_rate: float = 0.8
        ):
            num_mask_params = int(flattened_models_to_merge_param.shape[1] * param_value_mask_rate)
            # import pdb; pdb.set_trace()
            kth_values, _ = flattened_models_to_merge_param.abs().kthvalue(k=num_mask_params, dim=1, keepdim=True)
            mask = flattened_models_to_merge_param.abs() >= kth_values
            return flattened_models_to_merge_param * mask

        def get_param_signs(flattened_models_to_merge_param: torch.Tensor):
            param_signs = torch.sign(flattened_models_to_merge_param.sum(dim=0))
            majority_sign = torch.sign(param_signs.sum(dim=0))
            param_signs[param_signs == 0] = majority_sign
            return param_signs

        def disjoint_merge(
                flattened_models_to_merge_param: torch.Tensor,
                param_signs: torch.Tensor
        ):
            param_to_preserve_mask = ((param_signs.unsqueeze(dim=0) > 0) & (flattened_models_to_merge_param > 0)) | (
                        (param_signs.unsqueeze(dim=0) < 0) & (flattened_models_to_merge_param < 0))
            param_to_preserve = flattened_models_to_merge_param * param_to_preserve_mask
            num_models_param_preserved = (param_to_preserve != 0).sum(dim=0).float()
            merged_flattened_param = torch.sum(param_to_preserve, dim=0) / torch.clamp(num_models_param_preserved, min=1.0)
            return merged_flattened_param

        merged_task_vector_param_dict = {}
        layer_name_list = list(models_to_merge_task_vectors[0].task_vector_param_dict.keys())
        print("TIES start merging!!!!!!!!")
        with torch.no_grad():
            for layer_idx in range(len(layer_name_list)):
                param_name = layer_name_list[layer_idx]
                print(param_name)
                # Tensor, original shape
                param_original_shape = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape
                # Tensor, shape (num_models_to_merge, num_params), flattened parameters of individual models that need to be merged
                flattened_models_to_merge_param = torch.vstack([task_vector.task_vector_param_dict[param_name].flatten() for task_vector in models_to_merge_task_vectors]).to("cuda")
                # for idx in range(len(models_to_merge_task_vectors)):
                #     print("Delete Single layer in Task Vector.")
                #     del models_to_merge_task_vectors[idx].task_vector_param_dict[param_name]
                #     gc.collect()
                flattened_models_to_merge_param = mask_smallest_magnitude_param_values(
                    flattened_models_to_merge_param=flattened_models_to_merge_param,
                    param_value_mask_rate=param_value_mask_rate)
                param_signs = get_param_signs(
                    flattened_models_to_merge_param=flattened_models_to_merge_param
                )
                merged_flattened_param = disjoint_merge(
                    flattened_models_to_merge_param=flattened_models_to_merge_param,
                    param_signs=param_signs
                )

                merged_task_vector_param_dict[param_name] = merged_flattened_param.reshape(param_original_shape).to("cpu")
                print("Delete Single layer Task Vector. The vector has been Flatten and Vstack")
                del merged_flattened_param

        return TaskVector(task_vector_param_dict=merged_task_vector_param_dict)


    @torch.no_grad()
    def merge(self, models_to_merge_task_vectors: list):
        if self.merge_config["merge_method"] == "ta":
            return self.task_arithmetic(
                models_to_merge_task_vectors=models_to_merge_task_vectors
            )
        elif self.merge_config["merge_method"] == "ties":
            return self.ties_merging(
                models_to_merge_task_vectors=models_to_merge_task_vectors,
                param_value_mask_rate=self.merge_config["param_value_mask_rate"],
            )
        else:
            raise NotImplementedError(f"unsupported for merging_method_name {self.merge_config['merge_method']}!")



if __name__ == '__main__':
    from fire import Fire
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from utils.utils import smart_tokenizer_and_embedding_resize

    to_merge = {
        "ptm": "/nfsdata/yany/download/merge/from_llama2_13b/meta-llama/Llama-2-13b-hf",
        "math": "/nfsdata/yany/download/merge/from_llama2_13b/vanillaOVO/WizardMath-13B-V1.0",
        "code": "/nfsdata/yany/download/merge/from_llama2_13b/code/magicoder-llama-2-13b-bf16"
    }

    def run(merge_method=None):
        ptm_model = AutoModelForCausalLM.from_pretrained(to_merge["ptm"], torch_dtype=torch.float16, device_map="cpu")
        math_model = AutoModelForCausalLM.from_pretrained(to_merge["math"], torch_dtype=torch.float16, device_map="cpu")
        code_model = AutoModelForCausalLM.from_pretrained(to_merge["code"], torch_dtype=torch.float16, device_map="cpu")
        ptm_tokenizer = AutoTokenizer.from_pretrained(to_merge["ptm"])
        math_tokenizer = AutoTokenizer.from_pretrained(to_merge["math"])
        code_tokenizer = AutoTokenizer.from_pretrained(to_merge["code"])

        model_dict = {"ptm": ptm_model, "math": math_model, "code": code_model}
        tokenizer_dict = {"ptm": ptm_tokenizer, "math": math_tokenizer, "code": code_tokenizer}

        for task in model_dict:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                model=model_dict[task],
                tokenizer=tokenizer_dict[task],
            )

        exclude_param_names_regex = []
        models_to_merge = [math_model, code_model]
        tv_to_merge = []
        for _ in range(len(models_to_merge)):
            print("Get Task Vector")
            tv_to_merge.append(
                TaskVector(
                    pretrained_model=ptm_model,
                    finetuned_model=models_to_merge[0],
                    exclude_param_names_regex=exclude_param_names_regex
                )
            )
            del models_to_merge[0]  # 用完一个 ftm model 删一个 ftm model
            gc.collect()

        merger = Merger(merge_method=merge_method)
        merged_model = merger.merge(pretrained_model=ptm_model, models_to_merge_task_vectors=tv_to_merge)

    Fire(run)

    # e.g. python src/merge.py --merge_method ta_0.3
