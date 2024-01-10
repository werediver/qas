from typing import Any, List
import re

from llama_index.schema import TransformComponent, BaseNode, TextNode

class TextCleanUp(TransformComponent):
  """
  Cleans the content of `TextNode` nodes up in place (!!).
  """

  _leading_empty_lines = re.compile(r"^([^\S\n]*\n)+")
  _excessive_empty_lines = re.compile(r"([^\S\n]*\n){2,}")
  _rouge_line_break = re.compile(r"(?<=[\w ])\n(?=[\w])")
  _trailing_whitespace = re.compile(r"\s+$")

  def __call__(self, nodes: List["BaseNode"], **kwargs: Any) -> List["BaseNode"]:
      del kwargs
      return [self.check_node(node) for node in nodes]

  def check_node(self, node: BaseNode):
    if isinstance(node, TextNode):
      # Note that the following clean-up can invalidate
      # `node.start_char_idx` and `.end_char_idx`.

      node.text = self.clean_up_text(node.text)
    return node

  def clean_up_text(self, s: str):
    s = self._leading_empty_lines.sub("", s)
    s = self._excessive_empty_lines.sub("\n\n", s)
    s = self._rouge_line_break.sub(" ", s)
    s = self._trailing_whitespace.sub("", s)
    return s
