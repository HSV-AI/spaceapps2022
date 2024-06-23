"""
This file is an example of how to import llama-cpp and make calls.

It should not be used directly in any other code.
"""

from llama_cpp import Llama

# Here are a few models that I've downloaded to try
CODE_LLAMA = "codellama-7b-instruct.Q4_K_M.gguf"
LLAMA_2 = "llama-2-7b-chat.Q4_K_M.gguf"
LLAMA_2_13B = "llama-2-13b-chat.Q4_K_M.gguf"


# Using the defaults here except for moving some of the layers to the GPU availalbe on this laptop.
llm = Llama(model_path=LLAMA_2_13B, n_gpu_layers=20,)


# This is a basic prompt to show building the context of the RAG and asking the question
PROMPT = """[INST] <<SYS>>You are a helpful, respectful and honest assistant. Always answer
as helpfully as possible, while being safe. Your answers should not include any harmful, 
unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your
responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of 
answering something not correct. If you don't know the answer to a question, please don't share 
false information.<</SYS>>

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

Question:

What type of model do we pass the rules to?
[/INST]"""

# Since the default of llama-cpp-python uses a 512 token context length, we need
# to check and see how much our prompt uses.
input_length = len(llm.tokenize(bytes(PROMPT, "utf-8")))

print(f'Context length of prompt: {input_length}')

# To generate the answer, simply call the llm
output = llm(
    PROMPT, # Prompt
    max_tokens=None # Generate tokens until we run out
)

# The output structure has a lot of additional stuff in it:
print('\nTotal output structure:')
print(output)

print('\nOutput answer:')
print(output['choices'][0]['text'])
