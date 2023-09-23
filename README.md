# LangChain playground

## Pre-requisites

- Docker.
- An Elasticsearch cluster running 8.9 or greater.
- A working Python 3.11 or greater installation.
- An OpenAI API key to use LLM integrations.

## Setup

- Go the directory where you cloned the repository.

- Create a virtual env

  ```shell
  python3 -mvenv .venv
  ```

- Activate the virtual env (`source .venv/bin/activate` for bash/zsh, `.venv\bin\activate` on Windows)

- Install poetry

  ```shell
  pip install -U pip wheel poetry
  ```

- Install dependencies:
  
  ```shell
  poetry install
  ```

## Basic configuration

- Copy the file `config/sample.yaml` to `config/default.yaml` .
- Fill the `elasticsearch` stanza with the URL and credentials to access your Elasticsearch cluster.
- Set `elasticsearch.index` to the index where you will store data. :warning: **Make sure to not put an existing index here as it will be destroyed**.

## Indexing

Put at least one PDF in the `docs` folder and run `python index.py --nuke` to remove existing documents and start processing.

On the first execution the script will install any required dependency / model from HuggingFace.

## Search

Start the API with `python serve.py`.

You can visit the API docs at http://localhost:8000 and send a question through
the `ask` endpoint; if indexing went well, you should get results.

## LLM integration

To enable prompting the question to an LLM, you will need to set the following options:

- `openai.apikey`: your OpenAI API key.
- `openai.completionModel`: the completion model to use (`gpt-3.5-turbo-16k` by default).
- `llm`: set to `openai`.

After restarting the server, requests to `ask` should return the final response from the LLM using the results as a context.

## UI

A simple streamlit app to use the API is available by running `streamlit run ui.py` or `docker-compose up -d` .

Once started, it can be viewed at http://localhost:8501 .