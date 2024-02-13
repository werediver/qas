import os
from typing import List

from llama_index.readers import SimpleDirectoryReader
from llama_index.schema import Document

from qas.ingestion.markdown_with_front_matter_reader import MarkdownWithFrontMatterReader

INGESTION_LOCAL_PATH_ENV = "DOCS"

def load_data(
  docs_path: str | None = None,
  doc_exts: List[str] | None = [".md", ".txt"]
) -> List[Document]:
  docs_path = docs_path or os.getenv(INGESTION_LOCAL_PATH_ENV)
  assert docs_path, "The path for local document ingestion is not specified"
  return SimpleDirectoryReader(
    input_dir=docs_path,
    required_exts=doc_exts,
    file_extractor={
      ".md": MarkdownWithFrontMatterReader(),
    },
  ).load_data()
