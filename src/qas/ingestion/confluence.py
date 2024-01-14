from typing import Callable, List

from llama_hub.confluence import ConfluenceReader
from llama_index.schema import Document
from requests import HTTPError

from qas.utils import Mark

def load_data(
  url: str, 
  cloud: bool, 
  space_keys: List[str],
  cql_queries: List[str],
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

  try:
    docs = []
    for space_key in space_keys:
      with Mark(f"Fetching Confluence space {space_key}... "):
        result = reader.load_data(
          space_key=space_key,
          include_attachments=False,
        )
      print(f"Fetched {len(result)} document(s)")
      docs.extend(result)
    for q in cql_queries:
      with Mark(f"Querying Confluence with \"{q}\"..."):
        # CQL results seem to be limited to 4k entries
        # (or it's and issue with the particular Confluence instance used for testing)
        result = reader.load_data(
          cql=q,
          include_attachments=False,
        )
      print(f"Fetched {len(result)} document(s)")
      docs.extend(result)
  except HTTPError as e:
    print(f"HTTP error with status code {e.response.status_code}")
    raise

  return docs
