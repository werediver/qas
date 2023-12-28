from typing_extensions import override
from contextlib import AbstractContextManager
import time

class Mark(AbstractContextManager):
  """
  A context manager class that announces a task (by printing the "prefix" string)
  and its execution time (by formatting and printing the "suffix" string on the same line).
  """

  def __init__(self, prefix: str, suffix: str = "{:.2f} s"):
    self.prefix = prefix
    self.suffix = suffix
    self.t = 0.0

  @override
  def __enter__(self):
    print(self.prefix, end="", flush=True)
    self.t = time.perf_counter()
    return self

  @override
  def __exit__(self, exc_type, exc_value, exc_tb):
    elapsed = time.perf_counter() - self.t
    print(self.suffix.format(elapsed))
