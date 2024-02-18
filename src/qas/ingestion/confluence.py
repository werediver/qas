from random import random
from time import sleep
from typing import Callable

from llama_hub.confluence import ConfluenceReader
from llama_index.schema import Document
from requests import HTTPError

from qas.utils import Mark

def load_data(
  url: str,
  cloud: bool,
  space_keys: list[str],
  cql_queries: list[str],
  patch_reader: Callable[[ConfluenceReader], None],
) -> list[Document]:
  """
  See `CONFLUENCE_API_TOKEN`, `CONFLUENCE_USERNAME`, `CONFLUENCE_PASSWORD`
  in `llama_hub.confluence` for authorization options.

  If that's not enough, use `patch_reader` to inject your own `Confluence`
  (from `atlassian`) client instance.
  """

  reader = ConfluenceReader(base_url=url, cloud=cloud)

  patch_reader(reader)

  retry_limit = 5
  retry_delay = 5

  try:
    docs = []
    for space_key in space_keys:
      for retry in range(retry_limit):
        try:
          with Mark(f"Fetching Confluence space {space_key}... "):
            result = reader.load_data(
              space_key=space_key,
              include_attachments=False,
            )
          print(f"Fetched {len(result)} document(s)")
          docs.extend(result)
        except:
          print(f"Retrying ({retry + 1} attempt(s) failed)...")
          sleep(random() * retry_delay)
          continue
        break
    for q in cql_queries:
      for retry in range(retry_limit):
        try:
          with Mark(f"Querying Confluence with \"{q}\"..."):
            # CQL results seem to be limited to 4k entries
            # (or it's and issue with the particular Confluence instance used for testing)
            result = reader.load_data(
              cql=q,
              include_attachments=False,
            )
          print(f"Fetched {len(result)} document(s)")
          docs.extend(result)
        except:
          print(f"Retrying ({retry + 1} attempt(s) failed)...")
          sleep(random() * retry_delay)
          continue
        break
  except HTTPError as e:
    print(f"HTTP error with status code {e.response.status_code}")
    raise

  return docs
