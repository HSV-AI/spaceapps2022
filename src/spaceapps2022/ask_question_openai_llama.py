"""
This file is an example of how to import llama-cpp and make calls.

It should not be used directly in any other code.
"""

from openai import OpenAI


# This is a basic prompt to show building the context of the RAG and asking the question
SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer
as helpfully as possible, while being safe. Your answers should not include any harmful, 
unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your
responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of 
answering something not correct. If you don't know the answer to a question, please don't share 
false information.

Generate the next agent response by answering the question. Answer it as succinctly as possible. 
You are provided several documents with titles. If the answer comes from different documents 
please mention all possibilities in your answer and use the titles to separate between topics 
or domains. If you cannot answer the question from the given documents, please state that you 
do not have an answer.

CONTEXT:

Blog Post page 1: In this part, we will wrap the Transformer model with HuggingFace pipeline so that we can pass 
the rules to the Transformer model. To craft and pass the rules to the Transformer model, 
we can use the LangChain Prompt Template. In this prompt template, we can tell how the 
LLM should behave. This is shown in the pre_prompt variable. Next, we give some information 
or context to the LLM to refine our prediction. For example, we can tell the LLM that we are 
dicussing Apple product such as iphone, ipad and mac book, instead of discussing Apple as a 
fruit. With the context, we can pass in the relevant question as shown in the prompt variable.
"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What type of model do we pass the rules to?"}
]

client = OpenAI(base_url="http://localhost:8000/v1", api_key="llama.cpp")

completion = client.chat.completions.create(
    model="llama-2-7b-chat.Q4_K_M.gguf",
    messages=messages
    )
print(completion.choices[0].message.content)

print("\nNow with the streaming option:\n")

stream = client.chat.completions.create(
    model="llama-2-7b-chat.Q4_K_M.gguf",
    messages=messages,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
