import re
import sys
import logging
import json
import jsonlines
import torch
from fraction import Fraction
from vllm import SamplingParams
from evaluate_llms_utils import batch_data, is_number


def extract_answer_number(completion):
    if "the answer is" in completion.lower():
        extract_ans = completion.lower().split('the answer is:')[-1].strip()
    elif "Therefore," in completion or "So," in completion:  # 这里逗号必须有
        extract_ans = completion.split('Therefore,')[-1].strip().split('So,')[-1].strip()
    else:
        print("No \"the answer is\", \"Therefore\" or \"So\" in completion, cannot extract answer!!!")
        return None

    match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
    if match:
        if '/' in match.group():
            denominator = match.group().split('/')[1]
            numerator = match.group().split('/')[0]
            if is_number(denominator) == True and is_number(numerator) == True:
                if denominator == '0':
                    return round(float(numerator.replace(',', '')))
                else:
                    frac = Fraction(match.group().replace(',', ''))
                    num_numerator = frac.numerator
                    num_denominator = frac.denominator
                    return round(float(num_numerator / num_denominator))
            else:
                return None
        else:
            if float(match.group().replace(',', '')) == float('inf'):
                return None
            return round(float(match.group().replace(',', '')))
    else:
        print(f"No number in {extract_ans}, cannot extract answer!!!")
        return None

def get_math_prompt(instruction, model_name="WizardMath-7B-V1.0"):
    if "WizardMath-7B-V1.0" in model_name:
        return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response: Let's think step by step."

def extract(save_gen_results_folder, gsm8k_ins, gsm8k_answers):
    y_preds = []
    results = []
    invalid_outputs = []
    outputs = []
    with open(f"{save_gen_results_folder}/gsm8k_gen.jsonl", mode='r') as f_in:
        res_completions = [json.loads(line)["gen"] for line in f_in if line.strip()]
    for idx, (prompt, completion, prompt_answer) in enumerate(
            zip(gsm8k_ins, res_completions, gsm8k_answers)):
        y_pred = extract_answer_number(completion)
        # res, valid = process_results(prompt, completion, prompt_answer, invalid_outputs)
        y_preds.append(y_pred)
        if y_pred != None:
            res = (float(y_pred) == float(prompt_answer))
            valid = True
        else:
            res = False
            valid = False
            temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
            invalid_outputs.append(temp)
        results.append(res)
        outputs.append({
            "idx": idx,
            "y_pred": y_pred,
            "answer": prompt_answer,
            "correct": res,
            "valid": valid,
            "prompt": prompt,
            "completion": completion,
        })
    with jsonlines.open(f"{save_gen_results_folder}/gsm8k_gen_result.jsonl", mode='w') as f_out:
        f_out.write_all(outputs)
    accuracy = sum(results) / len(results)
    logger.info(f"invalid outputs length is {len(invalid_outputs)}, invalid_outputs are {invalid_outputs}")
    # logger.info(f"data index starts from {start_index}, ends at {end_index}")
    logger.info(f"GSM8K test data length is {len(results)}, new test accuracy is {accuracy}")
    return accuracy


def eval_gsm8k(llm, test_data_path, batch_size, save_gen_results_folder, logger: logging.Logger=None, start_index=0, end_index=sys.maxsize):
    # end_index = 256
    gsm8k_ins = []
    gsm8k_answers = []
    prompt_template = get_math_prompt("<empty instructions>")
    if not logger:
        logger = logging.getLogger()
    logger.info(f"gsm8k test prompt is {prompt_template}")
    with open(test_data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            gsm8k_ins.append(get_math_prompt(item["question"]))
            temp_ans = item['answer'].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            gsm8k_answers.append(temp_ans)

    gsm8k_ins = gsm8k_ins[start_index:end_index]
    gsm8k_answers = gsm8k_answers[start_index:end_index]
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)

    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=1024, stop=stop_tokens)
    # sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=1024)
    logger.info(f"sampling params is {sampling_params}")

    total_idx = start_index
    for idx, prompt in enumerate(batch_gsm8k_ins):
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
        with jsonlines.open(f"{save_gen_results_folder}/gsm8k_gen.jsonl", mode='a') as f_out:
            f_out.write_all(batch_res)

    # import pdb; pdb.set_trace()
    accuracy = extract(save_gen_results_folder, gsm8k_ins, gsm8k_answers)
    del llm
    torch.cuda.empty_cache()
    return accuracy


if __name__ == '__main__':
    from fire import Fire
    from vllm import LLM

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    gpu_num = torch.cuda.device_count()
    def test_single_model(model_pth, tokenizer_pth=None, batch_size=512, save_gen_results_folder=None, my_dtype="bf16"):
        my_dtype = "float16" if my_dtype == "fp16" else "bfloat16"
        print(f"< My Dtype: {my_dtype}>")
        tokenizer_pth = tokenizer_pth if tokenizer_pth else model_pth
        llm = LLM(
            model=model_pth,
            tokenizer=tokenizer_pth,
            tensor_parallel_size=gpu_num,
            dtype=my_dtype
        )
        test_data_pth = "eval/data/gsm8k_test.jsonl"
        eval_gsm8k(llm=llm, test_data_path=test_data_pth, batch_size=batch_size, save_gen_results_folder=save_gen_results_folder, logger=logger)
        logger.info(f"model pth: {model_pth}")

    Fire(test_single_model)
