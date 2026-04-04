import json
from elasticsearch import Elasticsearch, helpers

# 🔐 PUT YOUR NEW PASSWORD HERE
es = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "qwmA7r5qvDiYiOghRIaT"),
    verify_certs=False,
    request_timeout=60
)

INDEX_NAME = "judgments_chunks"
FILE_PATH = "paragraph_chunks.jsonl"


def create_index():
    try:
        if es.indices.exists(index=INDEX_NAME):
            print("Deleting old index...")
            es.indices.delete(index=INDEX_NAME)

        print("Creating new index...")

        es.indices.create(
            index=INDEX_NAME,
            body={
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        "source_file": {"type": "keyword"},
                        "paragraph_id": {"type": "integer"}
                    }
                }
            }
        )

        print("Index created successfully")

    except Exception as e:
        print("INDEX CREATION ERROR:", e)


def generate_docs():
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                doc = json.loads(line)

                yield {
                    "_index": INDEX_NAME,
                    "_id": doc["chunk_id"],
                    "_source": doc
                }

            except Exception as e:
                print("Skipping bad line:", e)


def main():
    try:
        print("Connecting to Elasticsearch...")
        print(es.info())

        create_index()

        print("Indexing started...")

        helpers.bulk(es, generate_docs())

        print("✅ INDEXING COMPLETE")

    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()