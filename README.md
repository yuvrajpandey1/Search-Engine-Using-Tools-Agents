
# GenAI Search Engine

With the help of OpenAI LLM model GPT-4.o building App in such a way that this model will get integrate with multiple tools easily to interact with the worlds data to get information by stablishing tools.And making customized tools with the help of OpenAI.



# Documentation 

[Tools](https://python.langchain.com/v0.1/docs/modules/tools/)

Tools are interfaces that an agent, chain, or LLM can use to interact with the world. They combine a few things:

* The name of the tool 
* A description of what the tool is
* JSON schema of what the inputs to the tool are 
* The function to call
* Whether the result of a tool should be returned directly to the user

It is useful to have all this information because this information can be used to build action-taking systems! The name, description, and JSON schema can be used to prompt the LLM so it knows how to specify what action to take, and then the function to call is equivalent to taking that action.

Importantly, the name, description, and JSON schema (if used) are all used in the prompt. Therefore, it is really important that they are clear and describe exactly how the tool should be used. You may need to change the default name, description, or JSON schema if the LLM is not understanding how to use the tool.

## Default Tools

```bash
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

```

[Agents](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/)

The core idea of agents is to use a language model to choose a sequence of actions to take. In chains, a sequence of actions is hardcoded (in code). In agents, a language model is used as a reasoning engine to determine which actions to take and in which order.

This categorizes all the available agents along a few dimensions are :

* Intended Model Type: Whether this agent is intended for Chat Models (takes in messages, outputs message) or LLMs (takes in string, outputs string). The main thing this affects is the prompting strategy used. You can use an agent with a different type of model than it is intended for, but it likely won't produce results of the same quality.

* Supports Chat History: Whether or not these agent types support chat history. If it does, that means it can be used as a chatbot. If it does not, then that means it's more suited for single tasks. Supporting chat history generally requires better models, so earlier agent types aimed at worse models may not support it.

* Supports Multi-Input Tools: Whether or not these agent types support tools with multiple inputs. If a tool only requires a single input, it is generally easier for an LLM to know how to invoke it. Therefore, several earlier agent types aimed at worse models may not support them.

* Supports Parallel Function Calling: Having an LLM call multiple tools at the same time can greatly speed up agents whether there are tasks that are assisted by doing so. However, it is much more challenging for LLMs to do this, so some agent types do not support this.

* Required Model Params: Whether this agent requires the model to support any additional parameters. Some agent types take advantage of things like OpenAI function calling, which require other model parameters. If none are required, then that means that everything is done via prompting.







 








## Important Libraries Used

 - [ArxivQueryRun](https://python.langchain.com/api_reference/community/tools/langchain_community.tools.arxiv.tool.ArxivQueryRun.html)
 - [WikipediaQueryRun](https://python.langchain.com/api_reference/community/tools/langchain_community.tools.wikipedia.tool.WikipediaQueryRun.html)
- [WikipediaAPIWrapper](https://python.langchain.com/api_reference/community/utilities/langchain_community.utilities.wikipedia.WikipediaAPIWrapper.html)
 - [ArxivAPIWrapper](https://python.langchain.com/api_reference/community/utilities/langchain_community.utilities.arxiv.ArxivAPIWrapper.html)
 - [FAISS](https://python.langchain.com/docs/integrations/vectorstores/faiss/)
 - [OpenAIEmbeddings](https://python.langchain.com/docs/integrations/text_embedding/openai/)
 - [RecursiveCharacterTextSplitter](https://dev.to/eteimz/understanding-langchains-recursivecharactertextsplitter-2846)






## Plateform or Providers

 - [LangChain-OpenAI](https://python.langchain.com/docs/integrations/providers/openai/)
 - [LangChain Hub](https://smith.langchain.com/hub)

## Model

 - LLM - Llama3-8b-8192


## Installation

Install below libraries

```bash
  pip install langchain
  pip install langchain_community
  pip install arxiv
  pip install OpenAI
  pip install wikipedia
  pip install langchain_openai

```
    
## Tech Stack

**Client:** Python, LangChain PromptTemplate, OpenAI

**Server:** Anaconda Navigator, Jupyter Notebook


## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`API_KEY`

`LANGCHAIN_API_KEY`
`OpenAI_API_KEY`
`GROQ_API_KEY`



## Examples
Initialize Vector Embedding
```javascript
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # dimensions=1024
)
```

## Recursively split by character

```javascript
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,

)
```
## Instantiate Facebook AI Similarity Search (FAISS)

```javascript
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
) 
```

