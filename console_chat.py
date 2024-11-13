import os
from openai import OpenAI

client = OpenAI(
    api_key="sk-",
    base_url="https://api.siliconflow.cn/v1",
)

# Whether the user wants to end the conversation
def isExiting(user_input):
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {"role": "system", "content": "You are an assistant trying to figure out whether a user wants to end a conversation. The user input is surrounded by <UserInput> and </UserInput> tags. If the user wants to end the conversation, say 'Y'. Otherwise, say 'N'."},
            {"role": "user", "content": "<UserInput>"+user_input+"</UserInput>"},
        ],
    )

    return completion.choices[0].message.content == "Y"

msgs = [
    {"role": "system", "content": "You are a helpful assistant."},
]

while True:
    user_input = input("\nUser: ")
    msgs.append({"role": "user", "content": user_input})
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=msgs,
        stream=True,
    )
    msg = ""
    for chunk in completion:
        print(chunk.choices[0].delta.content, end="", flush=True)
        msg += chunk.choices[0].delta.content
    msgs.append({"role": "assistant", "content": msg})

    if isExiting(user_input):
        break
