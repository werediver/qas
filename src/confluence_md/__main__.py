import os.path as p
from typing import List, Tuple
from urllib.parse import urljoin

from atlassian.confluence import Confluence

import frontmatter

from .config import Config
from .client import Client
from .types import Space
from .html2md import make_html2md
from .clean_text import clean_text

config = Config()

client = Client(Confluence(
  url=str(config.url),
  oauth2={
    "client_id": config.client_id,
    "token": {
      "access_token": config.access_token,
      "token_type": "Bearer",
    },
  }
))

spaces = client.get_all_spaces()

spaces_ext: List[Tuple[int, Space]] = [(client.get_space_page_count(scope.key) or 0, scope) for scope in spaces]
spaces_ext.sort(key=lambda item: item[0], reverse=True) # Start with the largest spaces.

expected_page_count = sum([page_count for page_count, _ in spaces_ext])
loaded_page_count = 0

failed_spaces: List[Tuple[int, Space]] = []

# for i, space in enumerate(spaces):
for i, (space_page_count, space) in enumerate(spaces_ext):
  print(f"[{i + 1:>3}/{len(spaces):>3}] Loading {space.key}, {space_page_count} page(s)...")

  try:
    pages = client.get_space_content(space.key, batch_size=100, expand="ancestors,body.export_view")

    page_count = len(pages)
    loaded_page_count += page_count

    print(f"Loaded {page_count} page(s).")
    print("Saving as files...")

    base_url = str(space.links.base) if space.links.base else config.url
    html2md = make_html2md(str(base_url))

    for page in pages:
      meta: dict[str, object] = {
        # Using "page_id" over just "id" for compatibility with `llama_hub.confluence.ConfluenceReader`.
        "page_id": page.id,
        "title": page.title,
        "url": urljoin(str(base_url), page.links.tinyui),
        "space": space.key,
      }
      if page.ancestors and len(page.ancestors) > 1:
        meta["ancestors"] = [ancestor.id for ancestor in page.ancestors[1:]] # Skip the space front-page.

      assert page.body is not None

      html = page.body["export_view"]["value"]
      md = clean_text(html2md.handle(html))

      post = frontmatter.Post(md, **meta)

      # NB: The file will be overwritten.
      fname = p.join(config.dump_dir, f"{page.id}.md")
      with open(fname, mode="wb") as f:
        frontmatter.dump(post, f)

  except Exception as e:
    failed_spaces.append((space_page_count, space))
    print(f"❌ Skipping space {space.key} due to errors.")
    continue

  print(f"Done.")

if failed_spaces:
  print("The following spaces couldn't be processed due to errors:")
  for page_count, space in failed_spaces:
    print(f"- {space.key}, {page_count} page(s)")

print(f"Successfully loaded {loaded_page_count / expected_page_count * 100:.2f}% of all pages")