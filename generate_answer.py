import os
import json

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

client2 = OpenAI(
    base_url="http://localhost:8802/v1",
    api_key="token-abc123",
)
def get_72b_response(prompt):
    completion = client2.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8",
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
    return completion.choices[0].message.content


prompt = '''
You are a network administrator and need to use Huawei router to perform CLI command line configuration on the network. You need to be enthusiastic and careful to answer questions about the CLI command line configuration of Huawei routers, and I will provide you with a configuration document for reference.

Reference configuration document:
{cli_config}

Requirements:
1. When you output a command, use a Shell-style block of code to display it. No words such as <huawei> are required at the beginning of each command.
2. Do not have "according to the provided documents" and other phrases, please directly generate your response to the question.
3. Don't just give a command, please provide relevant explanations and instructions to help users who are not familiar with the configuration to quickly learn and understand.

User question:
{question}

Your answer:
'''.strip()


def extract_funcdef_and_clis(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return json.dumps(data,indent=4,ensure_ascii=False)


with open('./question.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

result_list = []
for i in tqdm(data):
    path = i['path']
    question = i['question']
    cli_config = extract_funcdef_and_clis(path)

    answer = get_72b_response(prompt.format(cli_config = cli_config, question = question))
    result_list.append({"instruction": question, "path": path, "input":"","output":answer})

with open('dataset.json', 'w', encoding='utf-8') as f:
    json.dump(result_list, f, indent=4, ensure_ascii=False)

