import json
import re
import logging
from human_eval.data import write_jsonl

def read_jsonl(jsonl_pth):
    data = []
    with open(jsonl_pth, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def extract_code_until_last_return(text):
    lines = text.strip().split('\n')
    last_return_index = -1
    return_pattern = re.compile(r'^\s*return\b')  # 匹配以 return 开头的行，确保不是 returns
    for i, line in enumerate(lines):
        if return_pattern.search(line):
            last_return_index = i
    if last_return_index != -1:
        # 截取到最后一个 return 行，包括该行
        extracted_lines = lines[:last_return_index + 1]
    else:
        # 如果没有找到 return，返回原始文本
        extracted_lines = lines
    # 重新组合成字符串
    # import pdb; pdb.set_trace()
    return '\n'.join(extracted_lines).strip()


def post_process(codes, logger):
    # raise NotImplementedError
    outputs = []
    ext_tag = False
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
            ext_tag = True

        # e.g. My added extract method
        if '```' in completion:
            logger.info("completion matches ```, no python tags")
            def_line = completion.index('```')
            completion = completion[def_line:].strip()
            completion = completion.replace('```', '', 1)
            try:
                next_line = completion.index('```')
                completion = completion[:next_line].strip()
            except:
                logger.info("wrong completion")
            ext_tag = True
        if 'Python script you requested:' in completion:
            logger.info("completion matches \'Python script you requested:\'")
            def_line = completion.index('Python script you requested:')
            completion = completion[def_line:].strip()
            completion = completion.replace('Python script you requested:', '', 1)
            ext_tag = True
        # e.g. My added extract method

        if "__name__ == \"__main__\"" in completion:
            logger.info("completion matches __name__ == \"__main__\"")
            try:
                next_line = completion.index('if __name__ == "__main__":')
                completion = completion[:next_line].strip()
            except:
                logger.info("wrong completion")
            ext_tag = True
        if "# Example usage" in completion:
            logger.info("completion matches # Example usage")
            next_line = completion.index('# Example usage')
            completion = completion[:next_line].strip()
            ext_tag = True
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
            ext_tag = True
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
            ext_tag = True
        code['completion'] = extract_code_until_last_return(completion)
        if not ext_tag:
            print(f"<<{completion}>> hasn't followed any extract pattern!!!, maybe wrong!!!")
    outputs += codes
    return outputs

def new_file(input_jsonl="WizardCoder.jsonl"):
    logger = logging.getLogger()
    data = read_jsonl(input_jsonl)
    outputs = post_process(data, logger)
    write_jsonl(input_jsonl.replace(input_jsonl.split("/")[-1], "processed_new.jsonl"), outputs, append=True)

if __name__ == '__main__':
    from fire import Fire
    Fire(new_file)
