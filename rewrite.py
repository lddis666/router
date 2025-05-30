import json
from openai import OpenAI
question = json.load(open("/root/router/question_test_nokia_2.json", "r"))


client2 = OpenAI(
    # base_url="http://localhost:8802/v1",
    base_url="https://ark.cn-beijing.volces.com/api/v3/",
    # api_key="token-abc123",
    api_key="606f1ad7-6633-4c7e-87e8-2e8ab460d003"
    
)


# rewrite_prompt = '''

# You are an experienced network engineer assisting in optimizing a Retrieval-Augmented Generation (RAG) system. Some user questions are too similar to entries in the knowledge base, which leads to overly direct matches. Your task is to rewrite such questions by increasing their linguistic complexity while keeping their technical meaning unchanged.

# Specifically, you should take the following input question—originally asked by a network administrator—and rewrite it into a more complex, realistic version. You must simulate a practical operational scenario in the rewritten question (e.g., a deployment, troubleshooting situation, or configuration validation during a change window). Your goal is to make the question sound natural, nuanced, and less likely to overlap with static documentation.

# Guidelines:
# - Preserve the original meaning and technical intent.
# - Add a brief but realistic network operations context or background to the question.
# - Use more sophisticated or varied phrasing (e.g., passive voice, subordinate clauses, terminology-rich expression).
# - Avoid reusing key phrases or the surface structure of the original sentence.
# - The tone may be slightly conversational, reflecting how an actual network engineer might pose the question.
# - Output exactly one rewritten question sentence. Do not include explanations, lists, or formatting.

# Input question:
# {question}

# Output:

# '''.strip()

rewrite_prompt = '''

You are an experienced network engineer assisting in optimizing a Retrieval-Augmented Generation (RAG) system. In practice, some user questions are too similar to entries in the underlying knowledge base, resulting in overly literal or direct matches. Your task is to rewrite these questions to increase their linguistic complexity and realism, while preserving their original technical intent.

Please follow these guidelines when rewriting:

- The technical meaning of the original question must be fully preserved.
- Introduce a concise but realistic operational context, such as a deployment scenario, a configuration change window, troubleshooting effort, or system validation.
- Use more natural, varied, and sophisticated phrasing — for example, include subordinate clauses, passive constructions, or idiomatic technical expressions.
- Avoid reusing key phrases or surface structure from the original question.
- The output should be a full paragraph that sounds like something a network engineer would naturally write or say during operations.
- Do not include any explanations, lists, or additional commentary — **output only the rewritten paragraph**.

Input question:  
{question}

Output:


'''.strip()

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


result = []
for item in question:
    question = item["question"] 
    path = item["path"]
    rewritten_question = get_72b_response(rewrite_prompt.format(question=question))
    result.append({
        "path": path,
        "question": rewritten_question
    })
with open("rewrite_question.json", "w") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

