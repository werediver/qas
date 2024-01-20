from typing import List
from typing_extensions import override
import os

from llama_index.llms import ChatMessage, MessageRole
from llama_index.llms import LLM
from llama_index.llms.llm import MessagesToPromptType
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.prompts import PromptTemplate
from llama_index.query_engine.custom import CustomQueryEngine, STR_OR_RESPONSE_TYPE
from llama_index.retrievers import BaseRetriever
from llama_index.schema import BaseNode
from pydantic import Field
import llama_index.node_parser.text.sentence_window as sentence_window

# TODO: Consider splitting into multiple component and chaining them together in a `QueryPipeline`.
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

  augmented_query_template1: PromptTemplate
  """
  Template parameters:

  - `context`: the combined context entries
  - `query`: the original query
  """

  # TODO: Consider supporting multiple refinement iterations.
  augmented_query_template2: PromptTemplate
  """
  A refinement template. Parameters:

  - `context`: the combined context entries
  - `response`: the initial response
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
  reranker: BaseNodePostprocessor | None
  llm: LLM

  messages: List[ChatMessage] = []

  @override
  def custom_query(self, query: str) -> STR_OR_RESPONSE_TYPE:
      context = self.retrieve_context(query)
      augmented_query = self.augmented_query_template1.format(context=context, query=query)

      prompt = self.messages_to_prompt(
          self.messages + [ChatMessage(role=MessageRole.USER, content=augmented_query)]
      )

      response = str(self.llm.complete(prompt)).strip()
      print(f"🟠 Initial response: {response}") # Debug

      # Refining the response

      # context = self.retrieve_context(response)
      augmented_query = self.augmented_query_template2.format(context=context, response=response, query=query)

      prompt = self.messages_to_prompt(
          self.messages + [ChatMessage(role=MessageRole.USER, content=augmented_query)]
      )

      response = str(self.llm.complete(prompt)).strip()

      # There is no big point in maintaining the chat history, because
      # supporting queries-in-context (e.g. "How bit is _that_ team?")
      # requires more sophisticated machinery.
      # self.messages.append(ChatMessage(role=MessageRole.USER, content=query))
      # self.messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=response))

      return response

  def retrieve_context(self, query: str) -> str:
    nodes = self.retriever.retrieve(self.query_embed_template.format(query=query))

    if self.similarity_cutoff is not None:
        # TODO: Handle the situation when all or too many nodes are cut off.
        nodes = [node for node in nodes if node.score is not None and node.score >= self.similarity_cutoff]

    if self.reranker:
      nodes = self.reranker.postprocess_nodes(nodes=nodes, query_str=query)

    # Put the most relevant entries in the end (of the prompt), where they may have more impact on the generation.
    nodes.reverse()

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
