import tiktoken

encoding = tiktoken.get_encoding('o200k_base')

prompt = """
You are Krishna, offering thoughtful guidance.

Use the provided context.
Be compassionate and practical.
Answer in approximately 200 tokens.
"""

tokens = encoding.encode(prompt)

print('prompt Text: ', prompt)
print('total tokens: ', len(prompt))