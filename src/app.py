import sys
from typing import Any, Callable

from llama_index.embeddings import FastEmbedEmbedding
from llama_index.indices import VectorStoreIndex
from llama_index.llms import Ollama
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.postprocessor import SentenceTransformerRerank
from llama_index.prompts import PromptTemplate
from llama_index.schema import BaseNode
from llama_index.service_context import ServiceContext
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore

import chromadb

from qas.expand_query_transform import ExpandQueryTransform
from qas.ingestion.node_dedup import NodeDedup
from qas.ingestion.text_clean_up import TextCleanUp
from qas.messages_to_prompt import make_mistral_messages_to_prompt_converter
from qas.query_engine import QueryEngine
import config

def main():
  model_id = "mistral"
  node_parser=SentenceWindowNodeParser.from_defaults()
  embed_model=FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5", max_length=512)
  service_ctx = ServiceContext.from_defaults(
    llm=Ollama(model=model_id, request_timeout=60.0),
    node_parser=node_parser,
    embed_model=embed_model,
    transformations=[
      log_node_count("Initial node count: {count}"),
      node_parser,
      log_node_count("Node count after applying the node parser: {count}"),
      TextCleanUp(),
      log_node_count("Node count after removing tiny nodes: {count}"),
      NodeDedup(),
      log_node_count("Node count after deduplication: {count}"),
    ],
  )

  chroma_client = chromadb.PersistentClient()
  chroma_collection = chroma_client.get_or_create_collection(
    "context", 
  )
  vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

  storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

  if chroma_collection.count() == 0:
    documents = config.load_data()
    print(f"Total document count: {len(documents)}")
  else:
     documents = []

  vector_index = VectorStoreIndex.from_documents(
    documents,
    service_context=service_ctx,
    storage_context=storage_ctx,
    show_progress=True,
  )

  query_engine = QueryEngine(
    query_transform=ExpandQueryTransform(llm=service_ctx.llm),
    context_entry_template="From document \"{source}\":\n\n{content}",
    augmented_query_template1=PromptTemplate(
      "Below are pieces of the context information followed by the text \"End of context.\"\n\n"
      "{context}\n\n"
      "End of context.\n\n"

      # A Tree of Thought-like prompt.
      "Three experts with different mindsets who rarely agree with each other are reading the context and answering the following request by writing down one step of their independent thinking and sharing it with the group in turns, until they reach a conclusion.\n\n"

      "{query}\n"
    ),
    augmented_query_template2=PromptTemplate(
      "Below are pieces of the context information followed by the text \"End of context.\"\n\n"
      "{context}\n\n"
      "End of context.\n\n"
      "The opinions of other experts to consider critically:\n\n"
      "{response}\n\n"
      "Given the context information and not prior knowledge, answer the following query concise and to the point:\n\n"
      "{query}\n"
    ),
    messages_to_prompt=make_mistral_messages_to_prompt_converter(),
    retriever=vector_index.as_retriever(similarity_top_k=128),
    # Find suitable model list at https://www.sbert.net/docs/pretrained-models/ce-msmarco.html
    reranker=SentenceTransformerRerank(top_n=10, model="cross-encoder/ms-marco-MiniLM-L-12-v2"),
    llm=service_ctx.llm, 
  )

  print("🔴 Query: ", end="", flush=True)
  for q in sys.stdin:
    response = query_engine.query(q.strip())
    print(f"🟢 Response: {response}")
    print("🔴 Query: ", end="", flush=True)

def log_node_count(msg: str = "Node count: {count}") -> Callable[[list[BaseNode]], list[BaseNode]]:
  def _log_node_count(nodes: list[BaseNode], **kwargs: Any) -> list[BaseNode]:
    del kwargs
    print(msg.format(count=len(nodes)))
    return nodes

  return _log_node_count

if __name__ == "__main__":
  main()
