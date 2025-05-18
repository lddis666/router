import os
import json

import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import re

def extract_funcdef_and_clis(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return (json.dumps(data,indent=4,ensure_ascii=False),json_path)


def collect_all_texts(root_dir):
    """
    遍历目录下所有JSON文件
    """
    result_texts = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(subdir, file)
                text = extract_funcdef_and_clis(json_path)
                if text:
                    result_texts.append(text)


    return result_texts

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
        temperature=3,
        top_p=0.8,
        max_tokens=1024,
        extra_body={
            "repetition_penalty": 1.05,
        },
    )
    # print(completion.choices[0].message.content)
    return completion.choices[0].message.content
huawei_clis = collect_all_texts('./BGP配置命令行数据/huawei')


prompt = '''
你是一个华为路由器CLI命令考试的考官，你的任务是考察网络管理员是否能在实际场景中正确运用指定的CLI命令。

以下是待考察CLI命令的配置手册说明：
{cli_config}

你请遵循以下要求：

1. 从上面的 "CLIs" 命令中选取一条；
2. 提出一句简洁的问题，用以考察网络管理员是否掌握该命令的使用方法；
3. 若待考察命令包含“ParaDef”中的参数，需在问题中给出明确的具体数值或内容，不能使用“某前缀”、“一个AS”等模糊描述；
4. 问题不得直接提及命令名称，且不得使用“你、我、他、网络管理员”等人称，请直接表述问题本身；
5. 问题应具有通用性，考察角度可适用于不同品牌的设备，但问题中应明确指出华为路由器，以提示考生使用华为CLI作答；
6. 问题中不要包含有关"ParentView"的描述,不要含有“假设”、“如果”等类似话术。
7. 最终输出仅为一句问题，不需要给出答案。问题使用英文输出。

请输出英文问题：

'''.strip()



 
result_list = []
for cli, path in tqdm(huawei_clis):
    for i in range(5):
        response = get_72b_response(prompt.format(cli_config=cli))
        result_list.append({"question": response, "path": path})

with open('question.json', 'w', encoding='utf-8') as f:
    json.dump(result_list, f, indent=4, ensure_ascii=False)