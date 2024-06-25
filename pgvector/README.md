# Experimenting with pgvector as a database/vector database
## To run:

```bash
docker compose up --build -d
```

if you get errors, make sure you're logged in to docker in the terminal by running
```bash
docker login
```

## To play with and view the database:
1. exec into the pgvector container
```bash
docker exec -it <PG_Container_Id> /bin/bash
```

2. Use psql cli to access the running db
#### Connect to db
```bash
psql -U postgres
```

#### List
```bash
>>> \l
```

#### switch to specific db
```bash
>>> \c <DATABASE_NAME>
```

#### list tables
```bash
>>> \dt
```

#### describe tables
```bash
>>> \dt <TABLE_NAME>
```

#### enable pgvector extension for a database
```bash
>>> CREATE EXTENSION vector;
```

#### restore database from dump file (need to create the database first and run
```bash
psql -U postgres -d mydb -f backup.sql
```

## To access the notebook visit localhost:8888
The notebook will not save changes you make to any script since they are just in a docker container.

## To play with the embedding/llm service (ollama)
## Embedding models list and db name (<model-name>, <database name>)
    ('all-minilm:latest', 'minilm'),
    ('gte-base:latest', 'gte'),
    ('nomic-embed-text:latest', 'nomic'),
    ('qwen2:0.5b', 'qwen'),
    ('tinyllama:latest', 'tinyllama'),
    ('mxbai-embed-large:latest', 'mxbai'),
    ('gemma:2b-instruct-v1.1-q2_K', 'gemma'),
    ('phi3:mini-128k', 'phimini'),
    ('llama3:latest', 'llama'),
The models will mount to the volume after installation and be there upon future restarts. To install models:

1. exec into the container
```bash
docker exec -it ollama /bin/bash
```
OR
```bash
docker exec -it <OLLAMA Container  ID> /bin/bash
```

2. Install models (below command installs several small ones)
```bash
ollama pull tinyllama && 
    ollama pull gemma:2b-instruct-v1.1-q2_K && 
    ollama pull nomic-embed-text && 
    ollama pull all-minilm &&
    ollama pull mxbai-embed-large &&
    ollama pull phi3:mini-128k &&
    ollama pull qwen2:0.5b &&
    ollama pull llama3
```

3. Specify the embedding model upon request
```bash
curl http://localhost:11434/api/embeddings -d '{
  "model": "mxbai-embed-large",
  "prompt": "Represent this sentence for searching relevant passages: The sky is blue because of Rayleigh scattering"
}'
```
OR in python:
```python
ollama.embeddings(model='<MODEL_TAG ex: mxbai-embed-large>', prompt='Represent this sentence for searching relevant passages: The sky is blue because of Rayleigh scattering')
```
3. Some models are not available on Ollamas website (gte-base) and need to be loaded from another source. You can download them from hugging face and then use `ollama create -f path/to/Modelfile <What you want to name the model>
