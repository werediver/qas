from typing import Any, List, Set

from llama_index.schema import TransformComponent, BaseNode
import llama_index.node_parser.text.sentence_window as sentence_window

class NodeDedup(TransformComponent):
  _keys: Set[str] = set()

  def __call__(self, nodes: List["BaseNode"], **kwargs: Any) -> List["BaseNode"]:
      del kwargs
      return [node for node in nodes if self._register_node(node)]

  def _register_node(self, node: BaseNode) -> bool:
     key = node.metadata.get(sentence_window.DEFAULT_WINDOW_METADATA_KEY) or node.get_content()
     if key in self._keys:
        return False
     else:
        self._keys.add(key)
        return True