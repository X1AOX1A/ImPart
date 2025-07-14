import sys
import logging
import torch
import os
from tqdm import tqdm
from vllm import SamplingParams
from human_eval.data import write_jsonl, read_problems, stream_jsonl

def batch_data(data_list, batch_size=1):
    return [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]

def get_code_task_prompt(input_text, model_name):
    # LLaMA-2-13b based: resource: https://huggingface.co/layoric/llama-2-13b-code-alpaca#llama-2-13b-code-alpaca
    if "llama-2-13b-code-alpaca" in model_name:
        return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n\n### Instruction:\nCreate a Python script for this problem:\n{input_text}\n\n### Response:"

    # LLaMA-2-7b based: resource: https://huggingface.co/mrm8488/llama-2-coder-7b
    elif "llama-2-coder-7b" in model_name:
        return f"You are a coding assistant that will help the user to resolve the following instruction:\n### Instruction: {input_text}\n\n### Solution:\n"

    # Mistral-7b-v0.1 based: resource: https://huggingface.co/Nondzu/Mistral-7B-codealpaca-lora
    elif "Mistral-7B-codealpaca-lora" in model_name:
        return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{input_text}\n\n### Response:"

    # WizardCoder based: resource: https://huggingface.co/WizardLMTeam/WizardCoder-Python-13B-V1.0
    elif "WizardCoder" in model_name:
        # return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{input_text}\n\n### Response:"
        return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n\n### Instruction:\nCreate a Python script for this problem:\n{input_text}\n\n### Response:"

    elif "magicoder" in model_name:
        return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n\n### Instruction:\nCreate a Python script for this problem:\n{input_text}\n\n### Response:"
        # return f"[INST]Create a Python script for this problem:\n{input_text}\n\n[/INST]"

    else:
        print("model name not founded, use << llama-2-13b-code-alpaca >> Template as default")
        return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n\n### Instruction:\nCreate a Python script for this problem:\n{input_text}\n\n### Response:"


def eval_humaneval(llm, llm_name, batch_size, logger: logging.Logger, start_index=0, end_index=sys.maxsize, save_gen_results_folder=None):
    print(f"Using prompt: {get_code_task_prompt('<<question>>', model_name=llm_name)}")
    problems = read_problems()
    task_ids = sorted(problems.keys())[start_index: end_index]
    prompts = [problems[task_id]['prompt'] for task_id in task_ids]
    prompts = [prompt.replace('    ', '\t') for prompt in prompts]
    prompts = [get_code_task_prompt(prompt, model_name=llm_name) for prompt in prompts]

    batch_task_ids = batch_data(task_ids, batch_size=batch_size)
    batch_prompts = batch_data(prompts, batch_size=batch_size)
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048)

    # shutil.rmtree(save_gen_results_folder, ignore_errors=True)
    os.makedirs(save_gen_results_folder, exist_ok=True)
    output_file = f"{save_gen_results_folder}/{llm_name}.jsonl"
    if os.path.exists(output_file):
        os.remove(output_file)
    for batch_in, batch_id in tqdm(zip(batch_prompts, batch_task_ids)):
        with torch.no_grad():
            completions = llm.generate(batch_in, sampling_params)
        completion_seqs = []
        for completion, id in zip(completions, batch_id):
            gen_seq = completion.outputs[0].text
            # import pdb
            # pdb.set_trace()
            completion_seq = gen_seq.split("### Response:")[-1]
            completion_seq = completion_seq.replace('\t', '    ')
            all_code = gen_seq.replace('\t', '    ')
            completion_seqs.append({'task_id': id, 'completion': completion_seq, 'all_code': all_code})

        # import pdb
        # pdb.set_trace()

        write_jsonl(output_file, completion_seqs, append=True)

    codes = [c for c in stream_jsonl(output_file)]
    logger.info(f"find {len(codes)} codes in {output_file}")

    outputs = []
    for code in codes:
        completion = code['completion']
        completion = completion.replace("\r", "")
        completion = completion.strip()
        if '```python' in completion:
            logger.info("completion matches ```python")
            def_line = completion.index('```python')
            completion = completion[def_line:].strip()
            completion = completion.replace('```python', '')
            try:
                next_line = completion.index('```')
                completion = completion[:next_line].strip()
            except:
                logger.info("wrong completion")
        if "__name__ == \"__main__\"" in completion:
            logger.info("completion matches __name__ == \"__main__\"")
            try:
                next_line = completion.index('if __name__ == "__main__":')
                completion = completion[:next_line].strip()
            except:
                logger.info("wrong completion")
        if "# Example usage" in completion:
            logger.info("completion matches # Example usage")
            next_line = completion.index('# Example usage')
            completion = completion[:next_line].strip()
        # the following codes are used to deal with the outputs of code-alpaca
        if "The solution is:" in completion:
            logger.info("completion matches The solution is:")
            def_line = completion.index("The solution is:")
            completion = completion[def_line:].strip()
            completion = completion.replace('The solution is:', '')
            try:
                next_line = completion.index('\n\nThe answer is:')
                completion = completion[:next_line].strip()
            except:
                completion = completion.strip()
                logger.info("maybe wrong completion")
        if "The answer is:" in completion:
            logger.info("completion matches The answer is:")
            def_line = completion.index("The answer is:")
            completion = completion[def_line:].strip()
            completion = completion.replace('The answer is:', '')
            try:
                next_line = completion.index('\n\nThe answer is:')
                completion = completion[:next_line].strip()
            except:
                completion = completion.strip()
                logger.info("maybe wrong completion")
        code['completion'] = completion
    outputs += codes

    logger.info(f"save to {save_gen_results_folder}/humaneval_processed.jsonl")
    write_jsonl(f"{save_gen_results_folder}/humaneval_processed.jsonl", outputs)

    del llm
    torch.cuda.empty_cache()

if __name__ == '__main__':
    from fire import Fire
    from vllm import LLM

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    def test_single_model(model_pth, tokenizer_pth=None, tensor_parallel_size=1, batch_size=64, save_gen_results_folder=None, my_dtype="bf16"):
        tokenizer_pth = tokenizer_pth if tokenizer_pth else model_pth
        my_dtype = "float16" if my_dtype == "fp16" else "bfloat16"
        llm = LLM(
            model=model_pth,
            tokenizer=tokenizer_pth,
            tensor_parallel_size=tensor_parallel_size,
            dtype=my_dtype
        )
        eval_humaneval(
            llm=llm,
            llm_name="WizardCoder",
            batch_size=batch_size,
            logger=logger,
            save_gen_results_folder=save_gen_results_folder
        )

    Fire(test_single_model)