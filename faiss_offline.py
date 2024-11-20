import faiss
from openai import OpenAI
import numpy as np
import sqlite3

client = OpenAI(
    api_key="sk-",
    base_url="https://api.siliconflow.cn/v1",
)

faiss_index = faiss.IndexFlatL2(1024)
conn = sqlite3.connect('data/text.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS `ai_context`
(
    `id`           INT(11) NOT NULL PRIMARY KEY,
    `text`         text NOT NULL,
    `gmt_create`   DATETIME NOT NULL DEFAULT now,
    `gmt_modified` DATETIME NOT NULL DEFAULT now
);
''')

lines = []
embeddings = []
with open('./data./knowledge/运动鞋店铺知识库.txt', 'r', encoding='utf-8') as f:
    for line in f:
        lines.append(line)

for i in range(0, len(lines)):
    response = client.embeddings.create(
        input=lines[i],
        model="BAAI/bge-m3"
    )
    embeddings.append(response.data[0].embedding)
    c.execute("INSERT INTO ai_context (id, text) VALUES ("+str(i)+",'"+lines[i]+"')")
    # print(response.data[0].embedding)

print(np.array(embeddings).astype(np.float32))
print(np.array(embeddings).astype(np.float32).shape)

faiss_index.add(np.array(embeddings).astype(np.float32))
faiss.write_index(faiss_index, "data/faiss.index")
conn.commit()
conn.close()
