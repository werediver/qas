from typing_extensions import override
import os

from llama_index.indices.query.query_transform.base import BaseQueryTransform
from llama_index.llms import ChatMessage, MessageRole
from llama_index.llms.base import BaseLLM
from llama_index.llms.llm import MessagesToPromptType
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.prompts import PromptTemplate
from llama_index.query_engine.custom import CustomQueryEngine, STR_OR_RESPONSE_TYPE
from llama_index.retrievers import BaseRetriever
from llama_index.schema import BaseNode, NodeWithScore, QueryBundle
from pydantic import Field
import llama_index.node_parser.text.sentence_window as sentence_window

class QueryEngine(CustomQueryEngine):
  """
  A retrieval-augmented query engine.
  """

  query_transform: BaseQueryTransform

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

  retriever: BaseRetriever
  reranker: BaseNodePostprocessor | None
  llm: BaseLLM

  messages: list[ChatMessage] = []

  @override
  def custom_query(self, query: str) -> STR_OR_RESPONSE_TYPE:
    query_bundle1 = self.query_transform.run(query)

    context_nodes = self.retriever.retrieve(query_bundle1)
    if self.reranker:
      context_nodes = self.reranker.postprocess_nodes(nodes=context_nodes, query_bundle=query_bundle1)
    context = self._format_context_nodes(context_nodes)

    augmented_query = self.augmented_query_template1.format(context=context, query=query)

    prompt = self.messages_to_prompt(
        self.messages + [ChatMessage(role=MessageRole.USER, content=augmented_query)]
    )

    response = str(self.llm.complete(prompt)).strip()

    # Refining the response

    query_bundle2 = QueryBundle(
      query_str=query,
      custom_embedding_strs=(query_bundle1.custom_embedding_strs or []) + split_expert_group_response(response),
    )
    context_nodes = self.retriever.retrieve(query_bundle2)
    if self.reranker:
      context_nodes = self.reranker.postprocess_nodes(nodes=context_nodes, query_bundle=query_bundle2)
    context = self._format_context_nodes(context_nodes)

    augmented_query = self.augmented_query_template2.format(context=context, response=response, query=query)

    prompt = self.messages_to_prompt(
        self.messages + [ChatMessage(role=MessageRole.USER, content=augmented_query)]
    )

    response = str(self.llm.complete(prompt)).strip()


    return response

  def _format_context_nodes(self, nodes: list[NodeWithScore]) -> str:
    # Put the most relevant entries in the end (of the prompt), where they may have more impact on the generation.
    return "\n\n".join([self._format_context_node(node_with_score.node) for node_with_score in reversed(nodes)])

  def _format_context_node(self, node: BaseNode):

    source_node = node.source_node or node
    title = source_node.metadata.get("title") or self._file_name_without_ext(source_node.metadata.get("file_name")) or "Unknown"

    return self.context_entry_template.format(
      source=title,
      content= node.metadata.get(sentence_window.DEFAULT_WINDOW_METADATA_KEY) or node.get_content()
    )

  def _file_name_without_ext(self, path: str | None) -> str | None:
    if path is not None:
      file_name = os.path.basename(path)
      return os.path.splitext(file_name)[0]
    else:
      return None

import re

EXPERT_MARKER = re.compile(r"^.*?Expert[^:]*:[\s_*]*|\n\n", flags=re.MULTILINE | re.IGNORECASE)

def split_expert_group_response(response: str) -> list[str]:
  return [s for s in [s.strip() for s in EXPERT_MARKER.split(response)] if s]
