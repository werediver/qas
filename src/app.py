import os
import sys

from llama_index.embeddings import FastEmbedEmbedding
from llama_index.indices import VectorStoreIndex
from llama_index.llms import Ollama
from llama_index.node_parser import SimpleNodeParser
from llama_index.prompts import PromptTemplate
from llama_index.readers import SimpleDirectoryReader
from llama_index.service_context import ServiceContext, set_global_service_context
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore

import chromadb
from qas.messages_to_prompt import make_mistral_messages_to_prompt_converter

from qas.query_engine import QueryEngine
from qas.utils import Mark

def main():
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
    with Mark("Scanning documents... "):
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

  query_engine = QueryEngine(
    query_embed_template="Represent this sentence for searching relevant passages: {query}",
    context_entry_template="From document \"{source}\":\n\n{content}",
    augmented_query_template=PromptTemplate(
      "Below are pieces of the context information followed by the text \"End of context.\"\n\n"
      "{context}\n\n"
      "End of context.\n\n"
      "Given the context information and not prior knowledge, answer the following query:\n\n"
      "{query}\n"
    ),
    messages_to_prompt=make_mistral_messages_to_prompt_converter(),
    retriever=vector_index.as_retriever(similarity_top_k=5),
    llm=service_ctx.llm, 
  )

  print("Query: ", end="", flush=True)
  for q in sys.stdin:
    response = query_engine.query(q.strip())
    print(f"Response: {response}")
    print("Query: ", end="", flush=True)

if __name__ == "__main__":
  main()
