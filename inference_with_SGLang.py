'''
You can start the inference server by running the following command:
python -m sglang.launch_server --model /root/autodl-tmp/Exp2/models/minicpm05b --trust-remote-code --port 30000 --chat-template chatml
'''


# import openai

# client = openai.Client(base_url=f"http://localhost:30001/v1", api_key="None")

# response = client.chat.completions.create(
#     model="models/minicpm05b",
#     messages=[
#         {"role": "user", "content": "Write an article about Artificial Intelligence."},
#     ],
#     temperature=0.7,
#     max_tokens=1024,
# )

# print(response.choices[0].message.content)


import openai
client = openai.Client(base_url="http://localhost:30001/v1", api_key="None")

stream = client.chat.completions.create(
    model="minicpm05b",
    messages=[{"role":"user","content":"讲一个一万字的故事，一万字！"}],
    temperature=0.7, max_tokens=256,
    stream=True
)

for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
print()
