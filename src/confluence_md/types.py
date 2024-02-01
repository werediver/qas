from typing import Any, Generic, List, Literal, TypeVar
from pydantic import BaseModel, Field, HttpUrl

class Links(BaseModel):
  this: HttpUrl = Field(default=None, alias="self") # "https://www.collaboration.dtf.signify.com/rest/api/space?type=global"
  base: HttpUrl | None = None
  context: str | None = None

  next: str | None = None
  """
  Only the URL path, if there are more results.
  """

  webui: str | None = None
  """
  Only the URL path.
  """

  tinyui: str | None = None
  """
  Short permanent link, only the URL path.
  """

class Space(BaseModel):
  id: int
  key: str
  name: str
  type: Literal["global", "personal"]
  links: Links = Field(alias="_links")
  expandable: dict = Field(alias="_expandable")

  metadata: Any | None = None
  icon: List[Any] | None = None
  description: dict | None = None
  """
  The possible keys are "plain", "view".
  """

  retentionPolicy: List[Any] | None = None
  homepage: List[Any] | None = None
  """
  Only the URL path.
  """

class Content(BaseModel):
  id: str
  type: str
  """
  Possible values are "page", "blogpost".
  """
  status: str
  """
  Possible values: "current", "trashed"; "any" (when querying).
  """
  links: Links = Field(alias="_links")
  expandable: dict = Field(alias="_expandable")
  title: str | None = None
  body: dict | None = None
  """
  Possible keys are "anonymous_export_view", "export_view" (HTML), "styled_view" (HTML with CSS), "view" (HTML), "editor".
  """
  ancestors: List["Content"] | None = None
  """
  Ancestor pages ordered root-to-leaf, starting with the space front-page.
  """
  children: dict[str, "Content"] | None = None
  descendants: dict[str, "Content"] | None = None
  metadata: dict | None = None
  version: List[Any] | None = None

_T = TypeVar("_T")

class Response(BaseModel, Generic[_T]):
  results: List[_T]
  start: int
  limit: int
  size: int
  totalSize: int | None = None
  links: Links = Field(alias="_links")
