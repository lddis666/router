import os
import json

import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import re

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

client2 = OpenAI(
    # base_url="http://localhost:8802/v1",
    base_url="https://ark.cn-beijing.volces.com/api/v3/",
    # api_key="token-abc123",
    api_key="606f1ad7-6633-4c7e-87e8-2e8ab460d003"
    
)
def get_72b_response(prompt):
    print("------------")
    print(prompt)
    completion = client2.chat.completions.create(
        # model="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8",
        # model = "deepseek-v3-250324",
        model = "deepseek-r1-250120",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=1.2,
        top_p=0.8,
        max_tokens=1024,
        extra_body={
            "repetition_penalty": 1.05,
        },
    )
    # print(completion.choices[0].message.content)
    print("------------")
    print(completion.choices[0].message.content)
    print("------------")
    return completion.choices[0].message.content

check_prompt = '''
As a network device expert, please determine whether the following answer correctly responds to the user's question based on the router configuration manual. Only output "Correct" or "Incorrect" without any explanation or additional information. You must make your own judgment based on the Router configuration manual provided to you, not on your own knowledge.
---
## User question:
{User_Question}
---
## Proposed answer:
{answer}
---
## Router configuration manual:
{Manual_Excerpt}
---
Please answer Correct or Incorrect.
Your answer:
'''.strip()


import os
import json

def extract_funcdef_and_clis(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        func_def = data.get("FuncDef", "").strip()
        clis = data.get("CLIs", [])
        clis_text = "\n".join([cli for cli in clis])
        para = data.get("ParaDef", [])
        para_text = "\n".join([str(dic) for dic in para])
        return  "CLIs:\n" + clis_text + "\nFuncDef:\n" + func_def + "\nParaDef:\n" + para_text
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return None

# def extract_funcdef_and_clis(json_path):
#     with open(json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     return json.dumps(data,indent=4,ensure_ascii=False)


def extract_commands_from_json(json_str):
    """
    从JSON字符串中提取commands列表。
    :param json_str: 纯JSON字符串
    :return: 命令字符串列表，若不存在则返回空列表
    """
    try:
        obj = json.loads(json_str)
        return obj.get('commands', [])
    except json.JSONDecodeError:
        return []

def get_response(input_text, lora = True, system = False):
    # system
    system_prompt = '''
You are a router command assistant. Please generate the required router configuration commands based on my input, and output ONLY in JSON format as shown below:

{
  "commands": [
    "command 1",
    "command 2",
    "command 3"
    // The number of commands may vary depending on the requirements
  ]
}

Do not provide any explanation, description, or code block markers. Only return the pure JSON object.
'''
    if system:
        messages = [{"role": "system","content":system_prompt},{"role": "user", "content": input_text}]
    else:
        messages = [{"role": "user", "content": input_text}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(inputs, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            # adapter_names = ['__base__']
            adapter_names = ['expert'] if lora else ['__base__']
        )
    return tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

with open('./question_test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


model_name = 'Qwen/Qwen2.5-7B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype='auto', device_map = 'auto').eval()
model = PeftModel.from_pretrained(model,"./qwen",adapter_name='expert',torch_dtype='auto')
# model.to("cuda:0")



# correct = 0
# for i in tqdm(data):
#     path = i['path']
#     question = i['question']
#     Def, Cli, Text = extract_funcdef_and_clis(path)

#     response = get_response(question,lora=True)

#     judge = get_72b_response(check_prompt.format(
#         User_Question=question,
#         answer=response,
#         Manual_Excerpt=Text
#     ))
#     if "correct" in judge.lower():
#         correct += 1

# print(f"Accuracy: {correct / len(data) * 100:.2f}%")


from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_sample(i):
    try:
        path = i['path']
        question = i['question']
        Text = extract_funcdef_and_clis(path)

        response = get_response(question, lora=True, system=True)
        cli_list = '\n'.join(extract_commands_from_json(response))
        judge = get_72b_response(check_prompt.format(
            User_Question=question,
            answer=cli_list,
            Manual_Excerpt=Text
        ))
        return not 'incorrect' in judge.lower()
    except Exception as e:
        print(f"Error processing sample: {e}")
        return False

# 线程数可根据CPU数和每次API响应速度调整，通常4~8左右合适
MAX_WORKERS = 10

correct = 0
results = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # 提交所有任务
    futures = [executor.submit(process_sample, i) for i in data]
    # 显示进度条
    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        print(result)
        results.append(result)

correct = sum(results)
print(f"Accuracy: {correct / len(data) * 100:.2f}%")
        
