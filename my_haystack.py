import os 
from getpass import getpass
from haystack.components.generators import OpenAIGenerator
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from datasets import load_dataset
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack import Pipeline


if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass("Enter OpenAI API key: ")
generator = OpenAIGenerator(model="gpt-4o-mini")

## Document Store
document_store = InMemoryDocumentStore()

## Dataset
dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()

docs_with_embeddings = doc_embedder.run(docs)
document_store.write_documents(docs_with_embeddings["documents"])

## Retriever
retriever = InMemoryEmbeddingRetriever(document_store)


template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""

prompt_builder = PromptBuilder(template=template)

text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
basic_rag_pipeline = Pipeline()
basic_rag_pipeline.add_component("embedder", text_embedder)  # First reference to "embedder"
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", generator)

# Connect components using the same names
basic_rag_pipeline.connect("embedder", "retriever")  # Using same "embedder" name
basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
basic_rag_pipeline.connect("prompt_builder", "llm")

# Run pipeline with matching component name
question = "What does Rhodes Statue look like?"
response = basic_rag_pipeline.run({
    "embedder": {"text": question},  # Using same "embedder" name
    "prompt_builder": {"question": question}
})

print(response["llm"]["replies"][0])