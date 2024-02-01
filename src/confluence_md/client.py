from time import sleep
from typing import Callable

from atlassian.confluence import Confluence
from requests import HTTPError, ReadTimeout

from .types import *
from .retry import RetryInfo, exp_backoff, retry

# Confluence Server REST API docs: https://docs.atlassian.com/ConfluenceServer/rest/latest/

SKIP_LIMIT = 300

class _Skip(BaseModel):
  skip_count: int

def _default_retry(f: Callable):
  def on_failure(retry_info: RetryInfo):
    if isinstance(retry_info.failure_reason, HTTPError) and retry_info.failure_reason.response.status_code in range(500, 600):
      print(f"Operation failed due to an HTTP error with status code {retry_info.failure_reason.response.status_code}: {retry_info.failure_reason.response.text}")
      # The batch size is likely the culprit; only introduce a small delay.
      delay = exp_backoff(failure_count=retry_info.failure_count, base=0.5, slot=0.1)
    elif isinstance(retry_info.failure_reason, ReadTimeout):
      print(f"Operation failed due to a read timeout.")
      # Either the server struggles or some kind of rate-limiting is in action; introduce a large delay.
      delay = exp_backoff(failure_count=retry_info.failure_count, base=30.0, slot=15.0)
    else:
      # Likely unrecoverable.
      raise

    print(f"Retrying {retry_info.failure_count}/{retry_info.failure_limit} in {delay:.2f} s... ", end="", flush=True)
    sleep(delay)
    print("(now)")

  return retry(
    f,
    failure_limit=5,
    on_failure=on_failure,
  )

_T = TypeVar("_T")

class Client:
  _confluence: Confluence

  def __init__(self, confluence: Confluence):
    self._confluence = confluence

  def get_all_spaces(
    self,
    batch_size: int = 500,
    limit: int | None = None,
    expand: str | None = None
  ) -> List[Space]:

    @_default_retry
    def fetch(start: int, retry_info: RetryInfo) -> Response[Space]:
      return Response[Space].model_validate(
        self._confluence.get_all_spaces(
          start=start,
          limit=batch_size,
          expand=expand,
          space_type="global",
          space_status="current",
        )
      )

    return self._collect(fetch, limit=limit)

  def get_space(
    self,
    space_key: str,
    expand: str | None = None
  ) -> Space:
    return Space.model_validate(
      self._confluence.get_space(
        space_key=space_key,
        expand=expand,
      )
    )

  def get_space_page_count(self, space_key: str) -> int | None:
    response = Response[Any].model_validate(
      self._confluence.cql(
        cql=f"space={space_key} and type=page",
        limit=0
      )
    )
    return response.totalSize

  def get_space_content(
    self,
    space_key: str,
    batch_size: int = 50,
    limit: int | None = None,
    expand: str | None = None,
  ) -> List[Content]:

    @_default_retry
    def fetch(start: int, retry_info: RetryInfo) -> Response[Content] | _Skip:
      adjusted_batch_size = max(batch_size // (3 ** retry_info.failure_count), 1)

      if retry_info.failure_count > 0:
        print(f"At offset {start}.")

        print(f"Reducing batch size to {adjusted_batch_size} (default is {batch_size}).")
        if adjusted_batch_size == 1:
          if retry_info.extra.get("ready_to_skip"):
            # Skipping 1 page doesn't really help, at least not on
            # the particular problematic Confluence instance used for testing.
            print(f"🟠 Skipping {batch_size} page(s).")
            return _Skip(skip_count=batch_size)

          retry_info.extra["ready_to_skip"] = True

      return Response[Content].model_validate(
        self._confluence.get_all_pages_from_space_raw(
          space=space_key,
          start=start,
          limit=adjusted_batch_size,
          expand=expand,
        )
      )

    return self._collect(fetch, limit=limit)

  def _collect(
    self,
    fetch: Callable[[int], Response[_T] | _Skip],
    limit: int | None
  ) -> List[_T]:
    items: List[_T] = []
    skip_count = 0
    while limit is None or len(items) < limit:
      try:
        response = fetch(len(items) + skip_count)
      except Exception as e:
        item_count = len(items)
        if item_count > 0:
          print(f"Loading interrupted due to an error: {e}")
          print(f"Returning {item_count} loaded item(s).")
          break
        else:
          raise

      if isinstance(response, _Skip):
        skip_count += response.skip_count

        if skip_count >= SKIP_LIMIT:
          print(f"🟠 Skip limit ({SKIP_LIMIT}) reached. Returning {len(items)} loaded item(s).")
          break
      else:
        items.extend(response.results)

        has_more = bool(response.links.next)
        if not response.size or not has_more:
          break

    return items
