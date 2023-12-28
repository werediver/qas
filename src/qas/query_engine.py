from typing_extensions import override
import os

from llama_index.llms import LLM
from llama_index.prompts import PromptTemplate
from llama_index.query_engine import CustomQueryEngine
from llama_index.retrievers import BaseRetriever
from llama_index.schema import BaseNode

class QueryEngine(CustomQueryEngine):
  """
  A retrieval-augmented query engine.
  """

  query_embed_template: str = "{query}"

  context_entry_template: str = "{content}"
  """
  Template parameters:

  - `source`: the source name (the file name without extension)
  - `content`: the relevant source fragment
  """

  augmented_query_template: PromptTemplate
  """
  Template parameters:

  - `context`: the combined context entries
  - `query`: the original query
  """

  retriever: BaseRetriever
  llm: LLM

  @override
  def custom_query(self, query: str):
      nodes = self.retriever.retrieve(self.query_embed_template.format(query=query))
      context = "\n\n".join([self.format_context_node(n.node) for n in nodes])
      query = self.augmented_query_template.format(context=context, query=query)
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
