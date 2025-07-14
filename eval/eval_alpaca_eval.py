import os
import sys
import logging
import datasets
import torch
import json
from vllm import SamplingParams
from src.utils.evaluate_llms_utils import batch_data

def get_instruction_prompt(instruction, model_name="WizardLM-13B-V1.0"):
    if "WizardLM" in model_name:
        return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {instruction} ASSISTANT:"



def eval_alpaca_eval(llm, llm_name, batch_size, logger: logging.Logger, start_index=0, end_index=sys.maxsize, save_gen_results_folder=None):
    # end_index = 64
    eval_set = datasets.load_from_disk("math_code_data/alpaca_eval")

    instructions = []
    reference_outputs = []
    for example in eval_set:
        # dictionary with 'instruction', 'output': 'generator' and 'dataset' as keys
        instructions.append(example["instruction"])
        reference_outputs.append(example)

    instructions = instructions[start_index:end_index]
    reference_outputs = reference_outputs[start_index:end_index]
    prompts = [get_instruction_prompt(instruction=prompt, model_name=llm_name) for prompt in instructions]

    batch_prompts = batch_data(prompts, batch_size)
    batch_reference_outputs = batch_data(reference_outputs, batch_size)

    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048)
    logger.info(f"sampling params is {sampling_params}")

    os.makedirs(save_gen_results_folder, exist_ok=True)
    output_file = f"{save_gen_results_folder}/{llm_name}_alpacaeval.json"
    outputs = []
    for batch_in, batch_ref in zip(batch_prompts, batch_reference_outputs):
        # import pdb
        # pdb.set_trace()
        completions = llm.generate(batch_in, sampling_params)
        for tmp_in, tmp_ref, output in zip(batch_in, batch_ref, completions):
            generated_text = output.outputs[0].text
            outputs.append({
                "instruction": tmp_ref["instruction"],
                "output": generated_text,
                "generator": llm_name,
                "dataset": tmp_ref["dataset"]
            })
    #     write_jsonl(output_file, generated_outputs, append=True)
    #
    # outputs = [o for o in stream_jsonl(output_file)]

    logger.info(f"save to {output_file}")
    with open(output_file, "w", encoding="utf-8") as fout:
        json.dump(outputs, fout)

    del llm
    torch.cuda.empty_cache()


if __name__ == '__main__':
    from fire import Fire
    from vllm import LLM

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    def test_single_model(model_pth, llm_name, tokenizer_pth=None, tensor_parallel_size=1, batch_size=64, save_gen_results_folder="."):
        tokenizer_pth = tokenizer_pth if tokenizer_pth else model_pth
        llm = LLM(model=model_pth, tokenizer=tokenizer_pth, tensor_parallel_size=tensor_parallel_size)
        eval_alpaca_eval(
            llm=llm,
            llm_name=llm_name,
            batch_size=batch_size,
            logger=logger,
            save_gen_results_folder=save_gen_results_folder
        )

    Fire(test_single_model)

    # e.g. PYTHONPATH=/home/yany/0.project/MultiTaskMerge/ python src/eval/eval_alpaca_eval.py --model_pth /data1/yany2/download/merge/from_llama2_13b/WizardLMTeam/WizardLM-13B-V1.2 --llm_name WizardLM-13B-V1.2 --tensor_parallel_size 2 --batch_size 64

    ## PATH
    # WizardMath-7B: /data1/yany2/download/merge/from_llama2_7b/WizardLMTeam/WizardMath-7B-V1.0