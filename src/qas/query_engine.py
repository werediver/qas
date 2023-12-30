from typing import List
from typing_extensions import override
import os

from llama_index.llms import LLM
from llama_index.llms.llm import MessagesToPromptType
from llama_index.prompts import PromptTemplate
from llama_index.query_engine import CustomQueryEngine
from llama_index.retrievers import BaseRetriever
from llama_index.schema import BaseNode

from llama_index.llms import ChatMessage, MessageRole
from pydantic import Field

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

  messages_to_prompt: MessagesToPromptType = Field(exclude=True)

  similarity_cutoff: float | None
  """
  In case the retriever doesn't support `similarity_cutoff`, it can be supplied to the query engine.

  The retrieved nodes with the similarity score below `similarity_cutoff` will be ignored.
  """

  retriever: BaseRetriever
  llm: LLM

  messages: List[ChatMessage] = []

  @override
  def custom_query(self, query: str):
      nodes = self.retriever.retrieve(self.query_embed_template.format(query=query))

      # scores = [n.score for n in nodes]
      # print(f"Context node scores: {scores}");
      if self.similarity_cutoff is not None:
         nodes = [node for node in nodes if node.score >= self.similarity_cutoff]

      context = "\n\n".join([self.format_context_node(n.node) for n in nodes])
      augmented_query = self.augmented_query_template.format(context=context, query=query)
      # print(f"Augmented query: {augmented_query}") # Debug

      prompt = self.messages_to_prompt(
          self.messages + [ChatMessage(role=MessageRole.USER, content=augmented_query)]
      )
      print(f"\nFull prompt: {prompt}") # Debug

      response = str(self.llm.complete(prompt)).strip()

      self.messages.append(ChatMessage(role=MessageRole.USER, content=query))
      self.messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=response))

      return response

  def format_context_node(self, node: BaseNode):
    source_node = node.source_node
    assert source_node is not None
    file_name = source_node.metadata["file_name"] or "Unknown"

    source_name = self.file_name_without_ext(file_name)

    return self.context_entry_template.format(
      source=source_name,
      content=node.get_content()
    )

  def file_name_without_ext(self, path: str) -> str:
    file_name = os.path.basename(path)
    return os.path.splitext(file_name)[0]
