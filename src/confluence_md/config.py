from pydantic import HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
  model_config = SettingsConfigDict(extra="ignore", env_file=".env")

  url: HttpUrl
  """
  Confluence URL.
  """

  client_id: str
  """
  User name (e-mail).
  """

  access_token: str
  """
  Personal access token.
  """

  dump_dir: str
  """
  An existing path to save pages to.
  """