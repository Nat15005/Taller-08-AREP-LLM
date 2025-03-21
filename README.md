# Taller-08-AREP-LLM Introduction to Creating RAGs (Retrieval-Augmented Generators) with OpenAI

## Overview

This lab introduces the fundamental concepts and practical implementation of Retrieval-Augmented Generators (RAGs) using OpenAI’s tools and Pinecone as the vector database. By the end of this lab, students will have hands-on experience building RAGs and will deliver two GitHub repositories showcasing their work.

## Objectives

- Understand the basics of Retrieval-Augmented Generators (RAGs).
- Use OpenAI for generating embeddings and interacting with language models.
- Use Pinecone as a vector database for storing and retrieving embeddings.
- Build a simple RAG application.

## Pre-Lab Preparation

Before starting the lab, familiarize yourself with the basic concepts of RAGs and how they combine retrieval and generation for enhanced language model performance.

## Lab Instructions

### 1. Setup

### Installation

To install the necessary packages, run the following commands:

```python
pip install openai pinecone-client
```

### Set OpenAI API Key

Set your OpenAI API key in the environment:

```python
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
```

### 2. Using OpenAI for Embeddings

### Generate Embeddings

Use OpenAI to generate embeddings for your text data:

```python
import openai

def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=[text], model=model)
    return response['data'][0]['embedding']

text = "This is a sample text for embedding."
embedding = get_embedding(text)
print(embedding)
```

### 3. Using Pinecone as a Vector Database

### Install Pinecone

Install the Pinecone client:

```python
pip install pinecone-client
```

### Initialize Pinecone

Initialize Pinecone with your API key:

```python
import pinecone

pinecone.init(api_key="your_pinecone_api_key", environment="us-west1-gcp")
```

### Create and Index Embeddings

Create an index and store embeddings:

```python
index_name = "example-index"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)

index = pinecone.Index(index_name)

# Example embedding (replace with actual embeddings from OpenAI)
embedding = get_embedding("This is a sample text for embedding.")
index.upsert([("example-id", embedding)])
```

### 4. Building the RAG Application

### Retrieve Relevant Documents

Query the Pinecone index to retrieve relevant documents:

```python
query_embedding = get_embedding("Query text for retrieval.")
query_response = index.query([query_embedding], top_k=5)

for match in query_response['matches']:
    print(f"ID: {match['id']}, Score: {match['score']}")
```

### Generate Responses Using OpenAI

Use OpenAI to generate responses based on the retrieved documents:

```python
def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

retrieved_documents = ["Document 1 text", "Document 2 text"]  # Replace with actual retrieved documents
context = " ".join(retrieved_documents)
prompt = f"Context: {context}\n\nQuestion: What is the answer to the query?"
response = generate_response(prompt)
print(response)
```

### 5. Images

![image](https://github.com/user-attachments/assets/7c342090-3b26-44f6-a3f9-10cab92362af)

![image](https://github.com/user-attachments/assets/50102164-7b94-4774-a3b8-7aaadecc7dcd)

![image](https://github.com/user-attachments/assets/8e3794bd-3f88-4444-a8e0-718c9d6048ef)

![image](https://github.com/user-attachments/assets/e871f285-8013-48bc-a371-d76c1116a9db)

### 6. Conclusion

In this lab, it was learned how to:

- Use OpenAI to generate embeddings and interact with language models.
- Use Pinecone as a vector database for storing and retrieving embeddings.
- Build a simple RAG application that retrieves relevant documents and generates responses.

### Further Reading

- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Pinecone Documentation](https://www.pinecone.io/docs/)

## Repository Structure

```python
Taller-08-AREP-LLM/
│
├── README.md                
└── main.py                       
```

## How to Run
Running in Google Colab

- Open Google Colab and create a new notebook.

- Copy and paste the code snippets from this README into the notebook cells.

- Run each cell sequentially to install dependencies, set up the environment, and execute the code.

## Contributing

[https://python.langchain.com/docs/tutorials/llm_chain/](https://python.langchain.com/docs/tutorials/llm_chain/)


## Author

Developed by [Natalia Rojas](https://github.com/Nat15005)

