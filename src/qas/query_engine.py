from typing import List
from typing_extensions import override
import os

from llama_index.llms import ChatMessage, MessageRole
from llama_index.llms import LLM
from llama_index.llms.llm import MessagesToPromptType
from llama_index.prompts import PromptTemplate
from llama_index.query_engine.custom import CustomQueryEngine, STR_OR_RESPONSE_TYPE
from llama_index.retrievers import BaseRetriever
from llama_index.schema import BaseNode
from pydantic import Field
import llama_index.node_parser.text.sentence_window as sentence_window

class QueryEngine(CustomQueryEngine):
  """
  A retrieval-augmented query engine.
  """

  # TODO: Consider moving into a custom retriever, because formatting early may not play well with hybrid search.
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

  # TODO: Consider moving into a custom retriever.
  similarity_cutoff: float | None
  """
  In case the retriever doesn't support `similarity_cutoff`, it can be supplied to the query engine.

  The retrieved nodes with the similarity score below `similarity_cutoff` will be ignored.
  """

  retriever: BaseRetriever
  llm: LLM

  messages: List[ChatMessage] = []

  @override
  def custom_query(self, query: str) -> STR_OR_RESPONSE_TYPE:
      context = self.retrieve_context(query)
      augmented_query = self.augmented_query_template.format(context=context, query=query)

      prompt = self.messages_to_prompt(
          self.messages + [ChatMessage(role=MessageRole.USER, content=augmented_query)]
      )

      response = str(self.llm.complete(prompt)).strip()

      # An attempt at refining the response

      context = self.retrieve_context(response)
      context += f"\n\nAn opinion of another expert:\n\n{response}"
      augmented_query = self.augmented_query_template.format(context=context, query=query)

      prompt = self.messages_to_prompt(
          self.messages + [ChatMessage(role=MessageRole.USER, content=augmented_query)]
      )

      response = str(self.llm.complete(prompt)).strip()

      self.messages.append(ChatMessage(role=MessageRole.USER, content=query))
      self.messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=response))

      return response

  def retrieve_context(self, query: str) -> str:
    nodes = self.retriever.retrieve(self.query_embed_template.format(query=query))

    if self.similarity_cutoff is not None:
        nodes = [node for node in nodes if node.score is not None and node.score >= self.similarity_cutoff]

    return "\n\n".join([self.format_context_node(n.node) for n in nodes])

  def format_context_node(self, node: BaseNode):
    source_node = node.source_node or node
    title = source_node.metadata.get("title") or self.file_name_without_ext(source_node.metadata.get("file_name")) or "Unknown"

    return self.context_entry_template.format(
      source=title,
      content= node.metadata.get(sentence_window.DEFAULT_WINDOW_METADATA_KEY) or node.get_content()
    )

  def file_name_without_ext(self, path: str | None) -> str | None:
    if path is not None:
      file_name = os.path.basename(path)
      return os.path.splitext(file_name)[0]
    else:
      return None
