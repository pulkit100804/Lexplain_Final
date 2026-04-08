from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os

load_dotenv()

es = Elasticsearch(
    os.getenv("ELASTIC_HOST"),
    api_key=os.getenv("ELASTIC_API_KEY")
)

res = es.search(
    index="judgments_chunks",
    query={"match_all": {}},
    size=3
)

for hit in res["hits"]["hits"]:
    print(hit["_source"])
