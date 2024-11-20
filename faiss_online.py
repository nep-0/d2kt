import numpy as np
import faiss
from openai import OpenAI
import numpy as np

client = OpenAI(
    api_key="sk-",
    base_url="https://api.siliconflow.cn/v1",
)

faiss_read_index = faiss.read_index('data/faiss.index')

def answer(user_input):
    response = client.embeddings.create(
        input=user_input,
        model="BAAI/bge-m3"
    )

    query_vector = np.array(response.data[0].embedding)

    distances, indices = faiss_read_index.search(np.array([query_vector]).astype('float32'), k=2)
    # print(f"Indices of nearest neighbors: {indices}")
    # print(f"Distances: {distances}")
    lines = []
    context = []
    with open('./data./knowledge/运动鞋店铺知识库.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for index in indices[0]:
        # print(f"Index: {index}, Content: {lines[index]}")
        context.append(lines[index])

    completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer questions based on context only."},
                {"role": "assistant", "content": "<Context>"+context[0]+"</Context>"},
                {"role": "assistant", "content": "<Context>"+context[1]+"</Context>"},
                {"role": "user", "content": "<UserInput>"+user_input+"</UserInput>"},
            ],
            stream=True,
        )

    for chunk in completion:
        print(chunk.choices[0].delta.content, end="", flush=True)


user_input = "有什么颜色？"

while True:
    answer(input("\nUser: "))