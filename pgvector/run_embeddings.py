import psycopg
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
import time
import os

# normal connection string for connecting to local/docker postgres
# "postgresql+psycopg://postgres:password@localhost:5432/test"
conn = psycopg.connect(
        "dbname=test user=postgres password=password host=localhost"
        )
with conn.cursor() as cur:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
with conn.cursor() as cur:
    cur.execute("SELECT * FROM langchain_pg_collection;")
    if not cur.fetchone():
        raise ValueError("Something wrong with that db.")
    conn.commit()
conn.close()


def create_database(db_name, host='localhost'):
    conn = psycopg.connect(
            f"dbname=postgres user=postgres password=password host={host}"
            )
    conn.autocommit = True

    with conn.cursor() as cur:
        # Create the new database if it does not exist
        cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
        exists = cur.fetchone()
        if not exists:
            cur.execute(f"CREATE DATABASE {db_name};")
    conn.close()

    conn = psycopg.connect(
            f"dbname={db_name} user=postgres password=password host={host}"
            )
    conn.autocommit = True

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.close()


models = [
    ('all-minilm:latest', '45 MB', 'minilm'),
    ('gte-base:latest', '117 MB', 'gte'),
    ('nomic-embed-text:latest', '274 MB', 'nomic'),
    ('qwen2:0.5b', '352 MB', 'qwen'),
    ('tinyllama:latest', '637 MB', 'tinyllama'),
    ('mxbai-embed-large:latest', '669 MB', 'mxbai'),
    ('gemma:2b-instruct-v1.1-q2_K', '1.2 GB', 'gemma'),
    ('phi3:mini-128k', '2.2 GB', 'phimini'),
    ('llama3:latest', '4.7 GB', 'llama'),
    ('command-r:35b-v0.1-q3_K_S', '4.7 GB', 'commandr'),
]

directory = 'pdfs/'
pdfs = []
for filename in os.listdir(directory):
    if filename.endswith('.pdf'):
        filepath = os.path.join(directory, filename)
        pdfs.append(filepath)

start_time = time.time()
for model in models:
    print(f'Begin embedding: {model}')
    db_name = model[2]
    create_database(db_name)
    embedding = OllamaEmbeddings(model=model[0])
    connection_str = f"postgresql+psycopg://postgres:password@localhost:5432/{db_name}"

    for pdf in pdfs:
        loader = PyMuPDFLoader(pdf)
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=80
                )
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)
        vectorstore = PGVector(
            embeddings=embedding,
            collection_name='mind',
            connection=connection_str,
            use_jsonb=True,
        )
        vectorstore.add_documents(chunks)

end_time = time.time()

print(f"SUCCESS!!!\n\t total time: {end_time-start_time} seconds")
