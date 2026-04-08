from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os

load_dotenv()

es = Elasticsearch(
    os.getenv("ELASTIC_HOST"),
    api_key=os.getenv("ELASTIC_API_KEY")
)

print(es.info())
