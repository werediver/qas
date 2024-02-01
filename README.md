# A retrieval-augmented question-answering system

The goal of this project is to implement a retrieval-augmented (RAG) question-answering system that uses local documents or a remote wiki-like system (e.g. Confluence) to augment the user requests with relevant context before passing them to an LLM.

The current implementation relies on [Ollama](https://github.com/jmorganca/ollama) for text generation and [fastembed](https://github.com/qdrant/fastembed) / [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) for text embedding.

Using an OpenAI API-like provider instead of Ollama (e.g. [LM Studio](https://lmstudio.ai/)) is easily possible.

## How to run

You'll need [PDM](https://github.com/pdm-project/pdm) and Ollama installed in your system.

First, install the project dependencies by executing the following command in the project directory:

```
pdm install
```

Then make sure Ollama has Mistral model downloaded by executing the following command:

```
ollama pull mistral
```

Make sure Ollama server is running by executing the following command or in any other way:

```
ollama serve
```

Finally, to run the app execute the following command:

```
env DOCS="path/to/txt/or/md/docs" pdm run src/app.py
```

## How to load Confluence pages

`confluence_md` package can be used as a CLI tool to download Confluence pages as Markdown files with metadata in stored YAML front matter.

Make sure to set the following environment variables put them in `.env` file in the project root:

- `URL`, Confluence server URL
- `CLIENT_ID`, Confluence user name
- `ACCESS_TOKEN`, personal access token
- `DUMP_DIR`, directory to write Markdown files to

To start downloading, execute the following command:

```
date; time pdm run python -m src.confluence_md
```
