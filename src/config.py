from llama_index.schema import Document

def load_data() -> list[Document]:
  import qas.ingestion.local

  return qas.ingestion.local.load_data()
