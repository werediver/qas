import os
import sys
import time

from llama_index import PromptTemplate, VectorStoreIndex, SimpleDirectoryReader
from llama_index.service_context import ServiceContext, set_global_service_context
from llama_index.schema import BaseNode
from llama_index.storage.storage_context import StorageContext
# from llama_index.vector_stores.simple import SimpleVectorStore
from llama_index.vector_stores import ChromaVectorStore
from llama_index.vector_stores.types import (
    DEFAULT_PERSIST_DIR,
    DEFAULT_PERSIST_FNAME,
)
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import LLM, Ollama
from llama_index.embeddings import FastEmbedEmbedding

from llama_index.query_engine import CustomQueryEngine
from llama_index.retrievers import BaseRetriever

import chromadb

class RAGQueryEngine(CustomQueryEngine):
  retriever: BaseRetriever
  llm: LLM
  query_embed_template: str = "{query}"
  context_entry_template: str = "{content}"
  prompt_template: PromptTemplate

  def custom_query(self, query: str):
      nodes = self.retriever.retrieve(self.query_embed_template.format(query=query))
      context = "\n\n".join([self.format_context_node(n.node) for n in nodes])
      query = self.prompt_template.format(context=context, query=query)
      print(f"Augmented query: {query}") # Debug
      return self.llm.complete(query, formatted=False)

  def format_context_node(self, node: BaseNode):
    source_name = self.file_name_without_ext(node.source_node.metadata["file_name"])
    return self.context_entry_template.format(
      source=source_name,
      content=node.get_content()
    )

  def file_name_without_ext(self, path: str) -> str:
    file_name = os.path.basename(path)
    return os.path.splitext(file_name)[0]

def mark(prefix: str, suffix: str | None = None):
  class Mark:
    def __init__(self, prefix: str, suffix: str):
        self.prefix = prefix
        self.suffix = suffix
        self.t = 0.0

    def __enter__(self):
      print(self.prefix, end="")
      self.t = time.perf_counter()

    def __exit__(self, exc_type, exc_value, exc_tb):
      elapsed = time.perf_counter() - self.t
      print(self.suffix.format(elapsed))

  return Mark(prefix, suffix or "{:.2f} s")

def main():
  # TODO: If only one model is available via the API, detect and select it automatically.
  model_id = "mistral"
  service_ctx = ServiceContext.from_defaults(
    llm=Ollama(model=model_id),
    embed_model=FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5", max_length=512)
  )
  # set_global_service_context(service_ctx)

  chroma_client = chromadb.PersistentClient()
  chroma_collection = chroma_client.get_or_create_collection(
    "context", 
    metadata={"hnsw:space": "ip"} # Use the dot-product as the distance function
  )
  vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

  storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

  if chroma_collection.count() == 0:
    with mark("Scanning documents... "):
      docs_path = os.getenv("DOCS")
      assert docs_path is not None, "Environment variable DOCS must point to a directory with documents to index"
      documents = SimpleDirectoryReader(docs_path, required_exts=[".md"]).load_data()

      node_parser = SimpleNodeParser.from_defaults(
        chunk_size=512, 
        chunk_overlap=96,
        paragraph_separator="\n\n",
        include_metadata=False
      )
      nodes = node_parser.get_nodes_from_documents(documents)
  else:
     nodes = []

  # `VectorStoreIndex` requires an embedding model in `service_context`.
  vector_index = VectorStoreIndex(
    nodes, 
    service_context=service_ctx, 
    storage_context=storage_ctx,
    show_progress=True,
  )

  query_engine = RAGQueryEngine(
    llm=service_ctx.llm, 
    retriever=vector_index.as_retriever(), 
    query_embed_template="Represent this sentence for searching relevant passages: {query}",
    context_entry_template="From document \"{source}\":\n\n{content}",
    prompt_template=PromptTemplate(
      "Below are pieces of the context information followed by the text \"End of context.\"\n\n"
      "{context}\n\n"
      "End of context.\n\n"
      "Given the context information and not prior knowledge, answer the following query:\n\n"
      "{query}\n"
    )
  )

  # A non-augmented request (comment out the docs indexing code to speed up execution)
  # response = service_ctx.llm.complete("What is the moon?", formatted=False)

  print("Query: ", end="", flush=True)
  for q in sys.stdin:
    response = query_engine.query(q)
    print(f"Response: {response}")
    print("Query: ", end="", flush=True)

if __name__ == "__main__":
  main()
