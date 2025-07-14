import json
import sys
import logging
import jsonlines
import torch
from vllm import SamplingParams
from evaluate_llms_utils import batch_data, remove_boxed, last_boxed_only_string, process_results

def get_math_prompt(instruction, model_name="WizardMath-7B-V1.0"):
    if model_name == "WizardMath-7B-V1.0":
        return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response: Let's think step by step."

def eval_math(llm, test_data_path, batch_size, save_gen_results_folder, logger: logging.Logger, start_index=0, end_index=sys.maxsize):
    hendrycks_math_ins = []
    hendrycks_math_answers = []
    prompt_template = get_math_prompt("<empty instructions>")
    logger.info(f"math test prompt is {prompt_template}")
    with open(test_data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            hendrycks_math_ins.append(get_math_prompt(item["instruction"]))
            solution = item['output']
            temp_ans = remove_boxed(last_boxed_only_string(solution))
            hendrycks_math_answers.append(temp_ans)

    hendrycks_math_ins = hendrycks_math_ins[start_index:end_index]
    hendrycks_math_answers = hendrycks_math_answers[start_index:end_index]
    batch_hendrycks_math_ins = batch_data(hendrycks_math_ins, batch_size=batch_size)

    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048, stop=stop_tokens)
    logger.info(f"sampling params is {sampling_params}")

    total_idx = start_index
    for idx, prompt in enumerate(batch_hendrycks_math_ins):
        batch_res = []
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]
        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            generated_text = output.outputs[0].text
            # res_completions.append(generated_text)
            batch_res.append({"idx": total_idx, "gen": generated_text})
            total_idx += 1
        with jsonlines.open(f"{save_gen_results_folder}/math_gen.jsonl", mode='a') as f_out:
            f_out.write_all(batch_res)

    results = []
    invalid_outputs = []
    outputs = []
    with open(f"{save_gen_results_folder}/math_gen.jsonl", mode='r') as f_in:
        res_completions = [json.loads(line)["gen"] for line in f_in if line.strip()]
    for idx, (prompt, completion, prompt_answer) in enumerate(
            zip(hendrycks_math_ins, res_completions, hendrycks_math_answers)):
        res, valid = process_results(prompt, completion, prompt_answer, invalid_outputs)
        results.append(res)
        outputs.append({"idx": idx, "correct": res, "valid": valid, "prompt": prompt, "completion": completion, "answer": prompt_answer})
    with jsonlines.open(f"{save_gen_results_folder}/math_gen_result.jsonl", mode='w') as f_out:
        f_out.write_all(outputs)
    accuracy = sum(results) / len(results)
    logger.info(f"invalid outputs length is {len(invalid_outputs)}, invalid_outputs are {invalid_outputs}")
    logger.info(f"data index starts from {start_index}, ends at {end_index}")
    logger.info(f"MATH test data length is {len(results)}, accuracy is {accuracy}")

    del llm
    torch.cuda.empty_cache()

    return accuracy


if __name__ == '__main__':
    from fire import Fire
    from vllm import LLM

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    gpu_num = torch.cuda.device_count()
    def test_single_model(model_pth, tokenizer_pth=None, batch_size=512, save_gen_results_folder=None, my_dtype="fp16"):
        my_dtype = "float16" if my_dtype == "fp16" else "bfloat16"
        print(f"< My Dtype: {my_dtype}>")
        llm = LLM(
            model_pth,
            tokenizer=tokenizer_pth,
            tensor_parallel_size=gpu_num,
            dtype=my_dtype,
        )
        test_data_pth = "eval/data/MATH_test.jsonl"
        eval_math(
            llm=llm,
            test_data_path=test_data_pth,
            batch_size=batch_size,
            logger=logger,
            save_gen_results_folder=save_gen_results_folder
        )

    Fire(test_single_model)
