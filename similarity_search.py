from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class CommandRetriever:
    def __init__(self, huawei_commands,nokia_commands, model_name1='CyCraftAI/CmdCaliper-base', model_name2='all-MiniLM-L6-v2'):

        self.huawei_commands = huawei_commands
        self.nokia_commands = nokia_commands
        self.model_cli = SentenceTransformer(model_name1)
        self.model_def = SentenceTransformer(model_name2)
        self.embeddings_cli = self.model_cli.encode([cmd for cmd in self.huawei_commands['cli']], convert_to_tensor=True)
        self.embeddings_def = self.model_def.encode([cmd for cmd in self.nokia_commands['def']], convert_to_tensor=True)

    def retrieve_cli(self, query, top_k=3):
        query_emb = self.model_cli.encode([query], convert_to_tensor=True)


        # 余弦相似度
        sim_scores = cosine_similarity(query_emb.cpu(), self.embeddings_cli.cpu())[0]
        top_indices = np.argsort(sim_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((idx, float(sim_scores[idx]), self.huawei_commands['cli'][idx]))
        return results

    def retrieve_def(self, query, top_k=3):
        query_emb = self.model_def.encode([query], convert_to_tensor=True)
        # 余弦相似度
        sim_scores = cosine_similarity(query_emb.cpu(), self.embeddings_def.cpu())[0]
        top_indices = np.argsort(sim_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((idx, float(sim_scores[idx]), self.nokia_commands['def'][idx]))
        return results


    def get_huawei_item(self, idx):
        return self.huawei_commands['cli'][idx], self.huawei_commands['def'][idx], self.huawei_commands['text'][idx]

    def get_nokia_item(self, idx):
        return self.nokia_commands['cli'][idx], self.nokia_commands['def'][idx], self.nokia_commands['text'][idx]

    def search(self, query, top_k1=2, top_k2=5):
        results = []
        def_results = self.retrieve_def(query, top_k2)
        for idx_def, score_def, def_cmd in def_results:
            print("检索到的def")
            print(def_cmd)
            # print(f"DEF: {def_cmd} (Score: {score_def:.4f})")
            _, _, text = self.get_nokia_item(idx_def)
            results.append(text)



        # cli_results = self.retrieve_cli(query, top_k1)
        # # return cli_results
        # for idx, score, cmd in cli_results:
        #     # print(f"CLI: {cmd} (Score: {score:.4f})")
        #     # print(idx)
        #     print("匹配的cli")
        #     print(cmd)

        #     _, Def, _ = self.get_huawei_item(idx)
        #     print("查询的def")
        #     print(Def)
        #     # print(Def)
        #     def_results = self.retrieve_def(Def, top_k2)
            # for idx_def, score_def, def_cmd in def_results:
            #     print("检索到的def")
            #     print(def_cmd)
            #     # print(f"DEF: {def_cmd} (Score: {score_def:.4f})")
            #     _, _, text = self.get_nokia_item(idx_def)
            #     results.append(text)

        return results
                
            

import os
import json
def extract_funcdef_and_clis(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        func_def = data.get("FuncDef", "").strip()
        clis = data.get("CLIs", [])
        clis_text = "\n".join([cli for cli in clis])
        return func_def , clis_text, json.dumps(data,indent=4,ensure_ascii=False)
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return None

def collect_all_texts(root_dir):
    """
    遍历目录下所有JSON文件，返回每个文件中FuncDef和CLIs拼接后的字符串列表
    """
    result_dict = {'cli':[],'def':[],'text':[]}

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(subdir, file)
                output = extract_funcdef_and_clis(json_path)
                if output:
                    result_dict['cli'].append(output[1])
                    result_dict['def'].append(output[0])
                    result_dict['text'].append(output[2])
    return result_dict

huawei_texts = collect_all_texts('./BGP配置命令行数据/huawei')
nokia_texts = collect_all_texts('./BGP配置命令行数据/nokia')

# retriever = CommandRetriever(huawei_texts,nokia_texts,'CyCraftAI/CmdCaliper-base','all-MiniLM-L6-v2')

retriever = CommandRetriever(huawei_texts,nokia_texts,'BAAI/bge-large-en','BAAI/bge-large-en')




# retriever = HuaweiCommandRetriever(huawei_texts,'CyCraftAI/CmdCaliper-base')
# llm_command = 'peer 10.10.1.2 route-update-interval 25'

rag_prompt = '''

You are an expert assistant in configuring network devices. Based on the user's request and the retrieved content from the router configuration manual below, generate appropriate CLI configuration commands.

---

User Request:  
{user_question}

Retrieved Reference Content (from the configuration manual):  
{retrieved_docs}

---

Generate CLI configuration commands that fulfill the user's request using the information from the reference content. Please respond directly to the user's question. Do not mention phrases like "based on the information you provided", "according to the documentation", or anything similar.

'''.strip()


def get_retrieved_prompt(user_question, llm_commands):

    results = []
    for llm_command in llm_commands:
        print("原cli:")
        print(llm_command)
        results+=(retriever.search(llm_command, top_k1=5, top_k2=5))
    retrieved_docs = '\n'.join(list(set(results)))
    return rag_prompt.format(user_question=user_question, retrieved_docs=retrieved_docs)

# for similarity, cmd in top_matches:
#     print(f"华为命令: {cmd}（相似度: {similarity:.4f}）")