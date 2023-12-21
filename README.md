# A retrieval-augmented question-answering system

The goal of this project is to implement a retrieval-augmented (RAG) question-answering system that uses local documents or a remote wiki-like system (e.g. Confluence) to augment the user requests with relevant context before passing them to an LLM.

The current implementation relies on [Ollama](https://github.com/jmorganca/ollama) for both embedding and text generation.
