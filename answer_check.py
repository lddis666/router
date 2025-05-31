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
from similarity_search import get_retrieved_prompt

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
        model = "deepseek-v3-250324",
        # model = "deepseek-r1-250120",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
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
        para = data.get("ParaDef", [])
        if clis:
            clis_text = "\n".join([cli for cli in clis])
        else: 
            clis_text = "No CLIs found"
        if not func_def:
            func_def = "No FuncDef found"
        
        if para:
            para_text = "\n".join([str(dic) for dic in para])
        else:
            para_text = "No ParaDef found"
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

def get_response(input_text, lora = True, system = False, model_name = 'r1'):
    print("input_text")
    print(input_text)
    # system
    if system:
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

    Only the most 1-3 essential CLI commands are needed, a complete list of all CLI is not required.
    Do not provide any explanation, description, or code block markers. Only return the pure JSON object. 
    '''

        # messages = [{"role": "system","content":system_prompt},{"role": "user", "content": input_text}]
        messages = [{"role": "user", "content": system_prompt+"\n"+input_text}]
    else:
        messages = [{"role": "user", "content": input_text}]
    # inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # inputs = tokenizer(inputs, return_tensors="pt").to(model.device)
    # with torch.no_grad():
    #     outputs = model.generate(
    #         **inputs,
    #         max_new_tokens=1024,
    #         do_sample=False,
    #         # temperature=0.7,
    #         # top_p=0.9,
    #         # top_k=50,
    #         # adapter_names = ['__base__']
    #         adapter_names = ['expert'] if lora else ['__base__']
    #     )
    # return tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

    completion = client2.chat.completions.create(
        # model="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8",
        # model = "deepseek-v3-250324",
        model = "deepseek-r1-250528" if model_name == 'r1' else "deepseek-v3-250324",
        messages=messages,
        temperature=0.7,
        top_p=0.8,
        max_tokens=1024,
        extra_body={
            "repetition_penalty": 1.05,
        },
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

with open('./question_test_huawei.json', 'r', encoding='utf-8') as f:

# with open('/root/router/rewrite_question.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


# model_name = 'Qwen/Qwen2.5-7B-Instruct'
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype='auto', device_map = 'auto').eval()
# model = PeftModel.from_pretrained(model,"./qwen",adapter_name='expert',torch_dtype='auto').eval()
# model.to("cuda:0")



# correct = 0
# for i in tqdm(data):
#     path = i['path']
#     question = i['question']
#     Text = extract_funcdef_and_clis(path)

#     response = get_response(question,lora=False,  system=True, model_name='r1')
#     response = response.strip('```').strip("json").strip()
#     cli_list = '\n'.join(extract_commands_from_json(response))


#     # response = get_response(question.replace('Nokia','huawei'),lora=True,  system=True)
#     # response = response.strip('```').strip("json").strip()
#     # retrieved_prompt = get_retrieved_prompt(question, extract_commands_from_json(response))
#     # cli_list = get_response(retrieved_prompt, lora=False, system=False)
#     # cli_list = cli_list.strip('```').strip("json").strip()

#     judge = get_72b_response(check_prompt.format(
#         User_Question=question,
#         answer=cli_list,
#         Manual_Excerpt=Text
#     ))
#     if not 'incorrect' in judge.lower():
#         correct += 1

# print(f"Accuracy: {correct / len(data) * 100:.2f}%")


def get_intent_prompt(question):
    ROUTER_MODEL = "Nokia 7750 SR"  # or "Huawei NE40E" based on your requirement
#     prompt = '''
# You are a network configuration manual expert responsible for documenting CLI commands for Nokia 7750 SR routers.

# Given a user's real-world configuration requirement, your task is to extract the **most essential CLI command** that directly solves the user's request, and generate its **manual-style configuration documentation** as if it appeared in the official CLI reference guide.

# You must output ONLY a JSON object with the following structure:

# {
#     "FuncDef": "<A concise technical explanation of what this CLI command achieves and how it directly addresses the user's problem.>",
#     "CLIs": [
#         "<The most critical CLI command syntax required to implement the solution>"
#     ],
#     "ParaDef": [
#         {
#             "Parameters": "<parameter name>",
#             "Info": "<detailed technical description of this parameter and valid options or effects>"
#         }
#     ]
# }

# Instructions and constraints:
# - This JSON describes the **documentation for a single CLI command that is central to solving the user's configuration problem.**
# - Do NOT include mode entry commands (e.g., entering BGP context or interface view), only the command itself.
# - Do NOT include supporting or optional commands unless they are absolutely necessary.
# - Do NOT explain or summarize anything outside the JSON — your response must be clean, parseable JSON.

# User input:

# '''
    prompt = '''

You are a CLI configuration manual expert for Huawei NE40E routers, responsible for writing authoritative and precise command documentation.

The user will describe their real-world configuration needs in natural or conversational language. Your task is to accurately identify the user's true, standardized configuration intent, and extract the **single most essential CLI command** that directly fulfills the requirement. Then, generate configuration documentation for that command in the style of the official CLI reference manual.

You must output ONLY a JSON object in the following structure:

{
    "FuncDef": "<A concise technical explanation of what this CLI command achieves and how it directly addresses the user's problem.>",
    "CLIs": [
        "<The most critical CLI command syntax required to implement the solution>"
    ],
    "ParaDef": [
        {
            "Parameters": "<parameter name>",
            "Info": "<detailed technical description of this parameter and valid options or effects>"
        }
    ]
}

Instructions and constraints:
- This JSON must describe only one CLI command that is central to solving the user's configuration requirement.
- Do NOT include mode entry commands (e.g., entering interface, BGP, or service context).
- Do NOT include supporting or optional commands unless absolutely necessary to implement the core requirement.
- Do NOT output anything other than the JSON object — no explanations, comments, or summaries.
- If the user's input is ambiguous or conversational, make a professional, technically sound assumption about their intended configuration goal and select the most appropriate command accordingly.

Your goal is to act as a CLI manual expert and output precise, professional documentation for the most relevant configuration command.

User input:

'''
    return prompt+ question
    # return prompt.format(ROUTER_MODEL=ROUTER_MODEL, question=question).strip()


from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_sample(i):
    try:
        path = i['path']
        question = i['question']
        Text = extract_funcdef_and_clis(path)

        # response = get_response(question,lora=False,  system=False, model_name='r1')
        # cli_list = response
        # response = response.strip().strip('```').strip("json").strip()
        # cli_list = '\n'.join(extract_commands_from_json(response))

        # response = get_response(get_intent_prompt(question),lora=True,  system=False, model_name='v3')
        # response = response.strip().strip('```').strip("json").strip()

        response = None

        retrieved_prompt = get_retrieved_prompt(question, response)
        cli_list = get_response(retrieved_prompt, lora=False, system=False)
        cli_list = cli_list.strip('```').strip("json").strip()

        judge = get_72b_response(check_prompt.format(
            User_Question=question,
            answer=cli_list,
            Manual_Excerpt=Text
        ))


        # response = get_response(question.replace('Nokia 7750 SR','Huawei NE40E'),lora=True,  system=True, model_name='v3')
        # response = response.strip().strip('```').strip("json").strip()
        # retrieved_prompt = get_retrieved_prompt(question, extract_commands_from_json(response))
        # cli_list = get_response(retrieved_prompt, lora=False, system=False)
        # cli_list = cli_list.strip('```').strip("json").strip()

        # judge = get_72b_response(check_prompt.format(
        #     User_Question=question,
        #     answer=cli_list,
        #     Manual_Excerpt=Text
        # ))


        return not 'incorrect' in judge.lower()
    except Exception as e:
        print(f"Error processing sample: {e}")
        return False

# 线程数可根据CPU数和每次API响应速度调整，通常4~8左右合适
MAX_WORKERS = 100

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
        
