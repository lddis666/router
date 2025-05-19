from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class HuaweiCommandRetriever:
    def __init__(self, huawei_commands,nokia_commands, model_name1='CyCraftAI/CmdCaliper-base', model_name2='all-MiniLM-L6-v2'):

        self.huawei_commands = huawei_commands
        self.nokia_commands = nokia_commands
        self.model_cli = SentenceTransformer(model_name1)
        self.model_def = SentenceTransformer(model_name2)
        self.embeddings_cli = self.model_cli.encode([cmd for cmd in self.huawei_commands['cli']], convert_to_tensor=True)
        self.embeddings_def = self.model_def.encode([cmd for cmd in self.nokia_commands['def']], convert_to_tensor=True)

    def retrieve_cli(self, query, top_k=3):
        query_emb = self.embeddings_cli.encode([query], convert_to_tensor=True)


        # 余弦相似度
        sim_scores = cosine_similarity(query_emb.cpu(), self.embeddings_cli.cpu())[0]
        top_indices = np.argsort(sim_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((idx, float(sim_scores[idx]), self.huawei_commands['cli'][idx]))
        return results

    def retrieve_def(self, query, top_k=3):
        query_emb = self.embeddings_def.encode([query], convert_to_tensor=True)
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



retriever = HuaweiCommandRetriever(huawei_texts,'CyCraftAI/CmdCaliper-base')
llm_command = 'peer 10.10.1.2 route-update-interval 25'
top_matches = retriever.retrieve(llm_command, top_k=2)

for similarity, cmd in top_matches:
    print(f"华为命令: {cmd}（相似度: {similarity:.4f}）")