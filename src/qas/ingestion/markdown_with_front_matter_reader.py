from pathlib import Path
from typing import Any, Iterable
from itertools import chain

from llama_index import Document
from llama_index.readers.base import BaseReader

import frontmatter

PAGE_ID_KEY = "page_id"
ANCESTORS_KEY = "ancestors"
URL_KEY = "url"
SPACE_KEY = "space"

class MarkdownWithFrontMatterReader(BaseReader):
  def lazy_load_data(self, file_path: Path, extra_info: dict | None) -> Iterable[Document]:
    post = frontmatter.load(file_path)
    doc = Document(
      metadata={k: v for k, v in chain(extra_info.items(), post.metadata.items()) if k != ANCESTORS_KEY},
      excluded_embed_metadata_keys=[PAGE_ID_KEY, ANCESTORS_KEY, URL_KEY, SPACE_KEY],
      excluded_llm_metadata_keys=[PAGE_ID_KEY, ANCESTORS_KEY, URL_KEY],
      text=post.content,
    )
    return [doc]