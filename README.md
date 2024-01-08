# Langchain
---
In LangChain, a chain is an end-to-end wrapper around multiple individual components, providing a way to accomplish a common use case by combining these components in a specific sequence. The most commonly used type of **chain** is the LLMChain, which consists of a **PromptTemplate**, a **model** (either an LLM or a ChatModel), and an optional **output parser**.

The LLMChain works as follows:

Takes (multiple) input variables.
Uses the PromptTemplate to format the input variables into a prompt.
Passes the formatted prompt to the model (LLM or ChatModel).
If an output parser is provided, it uses the OutputParser to parse the output of the LLM into a final format.

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

llm = OpenAI(model="text-davinci-003", temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

#text-davinci-003 is the openai model used at temperature 0.9 keeping it creative
#the input_variable formatted into the prompt is 'product'

chain = LLMChain(llm=llm, prompt=prompt)
# next chain the large language model with the prompt

# Run the chain only specifying the input variable which is product as stated earlier in the code
print(chain.run("eco-friendly water bottles"))
```
