from os import getenv

from llama_index import PromptTemplate, VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import LLM, Ollama
from llama_index.embeddings import OllamaEmbedding
from llama_index.service_context import set_global_service_context

from llama_index.query_engine import CustomQueryEngine
from llama_index.retrievers import BaseRetriever

class RAGQueryEngine(CustomQueryEngine):
    retriever: BaseRetriever
    llm: LLM
    prompt_template: PromptTemplate

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        query = self.prompt_template.format(context_str=context_str, query_str=query_str)
        print(f"Query: {query}") # Debug
        return self.llm.complete(query, formatted=False)
    
def main():
  # TODO: If only one model is available via the API, detect and select it automatically.
  model_id = "mistral"
  service_ctx = ServiceContext.from_defaults(
     llm=Ollama(model=model_id),
     # TODO: Consider reducing the dimensionality of the embedding vectors.
     embed_model=OllamaEmbedding(model_name=model_id) # 4096 elements
  )
  # set_global_service_context(service_ctx)

  docs_path = getenv("DOCS")
  assert not docs_path is None, "Environment variable DOCS must point to a directory with documents to index"
  documents = SimpleDirectoryReader(docs_path, required_exts=[".md"]).load_data()

  node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
  nodes = node_parser.get_nodes_from_documents(documents)
  vector_index = VectorStoreIndex(nodes, service_context=service_ctx) # Requires an embedding model

  qa_prompt = PromptTemplate(
    "Below are pieces of the context information followed by the text \"End of context.\"\n\n"
    "{context_str}\n\n"
    "End of context.\n\n"
    "Given the context information and not prior knowledge, answer the following query:\n\n"
    "{query_str}\n"
  )

  query_engine = RAGQueryEngine(
     llm=service_ctx.llm, 
     retriever=vector_index.as_retriever(), 
     prompt_template=qa_prompt
  )

  # A non-augmented request (comment out the docs indexing code to speed up execution)
  response = service_ctx.llm.complete("What is the moon?", formatted=False)

  # response = query_engine.query("Which NAS cases are mentioned in the context?")

  print(f"Response: {response}")

if __name__ == "__main__":
  main()
