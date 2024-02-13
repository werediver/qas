from typing import Any, List, Set

from llama_index.schema import TransformComponent, BaseNode
import llama_index.node_parser.text.sentence_window as sentence_window

class NodeDedup(TransformComponent):
  _keys: Set[Any] = set()

  def __call__(self, nodes: List["BaseNode"], **kwargs: Any) -> List["BaseNode"]:
      del kwargs
      return [node for node in nodes if self._register_node(node)]

  def _register_node(self, node: BaseNode) -> bool:
    key = self._get_key(node)
    if key in self._keys:
      return False
    else:
      self._keys.add(key)
      return True

  def _get_key(self, node: BaseNode):
    source_node = node.source_node or node
    source_id = source_node.metadata.get("file_name") or source_node.metadata.get("page_id")
    return (
      source_id,
      node.metadata.get(sentence_window.DEFAULT_WINDOW_METADATA_KEY) or node.get_content()
    )