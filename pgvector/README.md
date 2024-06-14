# Experimenting with pgvector as a database/vector database
## To run:

```bash
docker compose up --build -d
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


## To access the notebook visit localhost:8888

## To play with the embedding/llm service (ollama)
The models will mount to the volume after installation and be there upon future restarts. To install models:

1. exec into the container
```bash
docker exec -it ollama /bin/bash
```

2. Install models (below command installs several small ones)
```bash
ollama pull tinyllama && 
    ollama pull gemma:2b-instruct-v1.1-q2_K && ollama pull nomic-embed-text && 
    ollama pull all-minilm &&
    ollama pull mxbai-embed-large &&
    ollama pull phi3:mini-128k &&
    ollama pull qwen2:0.5b
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
