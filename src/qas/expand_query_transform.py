from typing import Callable
from typing_extensions import override

import re

from llama_index.indices.query.query_transform.base import BaseQueryTransform
from llama_index.llms.base import BaseLLM
from llama_index.prompts import BasePromptTemplate, PromptTemplate
from llama_index.schema import QueryBundle

EXPAND_QUERY_TEMPLATE = PromptTemplate(
  "A search engine optimization expert is rephrasing the following request "
  "by writing down a few similar questions or requests in a numbered list:\n\n"
  "{query}\n"
)

NUMBERED_LIST_ITEM_MARKER = re.compile(r"^\s*\d+\.", flags=re.MULTILINE)

def split_num_list_response(response: str) -> list[str]:
  return [s for s in [s.strip() for s in NUMBERED_LIST_ITEM_MARKER.split(response)] if s]

class ExpandQueryTransform(BaseQueryTransform):
  _prompt_template: BasePromptTemplate
  """
  Template parameters:

  - `query`, the query to expand
  """

  _split_response: Callable[[str], list[str]]

  _llm: BaseLLM

  def __init__(
    self,
    llm: BaseLLM,
    prompt_template: PromptTemplate | None = None,
    split_response: Callable[[str], list[str]] | None = None,
  ):
    self._prompt_template = prompt_template or EXPAND_QUERY_TEMPLATE
    self._split_response = split_response or split_num_list_response
    self._llm = llm
    super().__init__()

  @override
  def _run(self, query_bundle: QueryBundle, metadata: dict) -> QueryBundle:
    del metadata
    prompt = self._prompt_template.format(query=query_bundle.query_str)
    response = str(self._llm.complete(prompt)).strip()
    alt_queries = self._split_response(response)
    return QueryBundle(
      query_str=query_bundle.query_str,
      custom_embedding_strs=[query_bundle.query_str] + alt_queries,
    )

  @override
  def _get_prompts(self) -> dict[str, BasePromptTemplate]:
    return {
      "expand_prompt_query": self._prompt_template,
    }

  @override
  def _update_prompts(self, prompts_dict: dict[str, BasePromptTemplate]) -> None:
      new_prompt_template = prompts_dict.get("expand_prompt_query")
      if new_prompt_template:
        self._prompt_template = new_prompt_template

