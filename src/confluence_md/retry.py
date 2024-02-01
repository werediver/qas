from random import random
from typing import Any, Callable
from pydantic import BaseModel

class RetryInfo(BaseModel):
  failure_count: int = 0
  failure_limit: int
  failure_reason: Any = None
  extra: dict = {}

def retry(f: Callable, failure_limit: int, on_failure: Callable[[RetryInfo], Any] | None) -> Callable:

  def inner(*args, **kwargs):
    retry_info = RetryInfo(failure_limit=failure_limit)
    while True:
      try:
        return f(
          *args,
          **kwargs,
          retry_info=retry_info,
        )
      except Exception as e:
        retry_info.failure_reason = e

        if retry_info.failure_count < retry_info.failure_limit:
          retry_info.failure_count += 1

          if on_failure:
            on_failure(retry_info)

          continue
        else:
          raise

  return inner

def exp_backoff(failure_count: int, base: float, slot: float) -> float:
  return base + ((1 << failure_count) - 1) * slot * (0.5 + random() / 2) # Always sample the upper half of the range.
