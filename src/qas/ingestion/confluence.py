from typing import Callable, List

from llama_hub.confluence import ConfluenceReader
from llama_index.schema import Document

def load_data(
  url: str, 
  cloud: bool, 
  cql: str,
  patch_reader: Callable[[ConfluenceReader], None],
) -> List[Document]:
  """
  See `CONFLUENCE_API_TOKEN`, `CONFLUENCE_USERNAME`, `CONFLUENCE_PASSWORD`
  in `llama_hub.confluence` for authorization options.

  If that's not enough, use `patch_reader` to inject your own `Confluence` 
  (from `atlassian`) client instance.
  """

  reader = ConfluenceReader(base_url=url, cloud=cloud)

  patch_reader(reader)

  return reader.load_data(
    cql=cql,
    include_attachments=False, 
  )
