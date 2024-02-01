import re

TRAILING_WHITESPACE = re.compile(r"[^\S\n]+$", flags=re.MULTILINE)
REDUNDANT_LINE_BREAKS = re.compile(r"\n{3,}")

def clean_text(s: str) -> str:
  """
  Remove trailing whitespace and redundant line breaks.
  """

  s = TRAILING_WHITESPACE.sub("", s)
  s = REDUNDANT_LINE_BREAKS.sub("\n\n", s)

  return s.strip()
