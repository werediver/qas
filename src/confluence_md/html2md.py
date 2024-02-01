from html2text import HTML2Text

def make_html2md(base_url: str) -> HTML2Text:
  html2md = HTML2Text(
    baseurl=base_url,
    bodywidth=0,
  )

  # Note that the generated Markdown links will contain the link title, when present.
  # E.g. [DuckDuckGo](https://duckduckgo.com/ "Web search").
  html2md.ignore_links = True
  html2md.wrap_links = False
  html2md.ignore_images = True
  html2md.ignore_tables = True
  html2md.ul_item_mark = "-"

  return html2md
