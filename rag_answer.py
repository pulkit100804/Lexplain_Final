import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from elasticsearch import Elasticsearch
import google.generativeai as genai

# ---------------- CONFIG ----------------
ES_URL = "https://localhost:9200"
ES_USER = "elastic"
ES_PASS = "qwmA7r5qvDiYiOghRIaT"

GEMINI_API_KEY = "AIzaSyDLJDTuv6hUV72m3v2UOc4wB08qkKkAj34"

INDEX_NAME = "judgments_chunks"

# ---------------- INIT ----------------
es = Elasticsearch(
    ES_URL,
    basic_auth=(ES_USER, ES_PASS),
    verify_certs=False
)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# ---------------- SEARCH ----------------
def search_es(query):
    response = es.search(
        index=INDEX_NAME,
        size=3,
        query={
            "match": {
                "text": query
            }
        }
    )
    return response["hits"]["hits"]

# ---------------- FULL CASE ----------------
def get_full_case(case_name):
    response = es.search(
        index=INDEX_NAME,
        size=1000,
        query={
            "term": {
                "source_file": case_name
            }
        }
    )

    hits = response["hits"]["hits"]

    if not hits:
        return ""

    hits.sort(key=lambda x: x["_source"].get("paragraph_id", 0))

    return " ".join([h["_source"]["text"] for h in hits])

# ---------------- RAG ----------------
def generate_answer(query):
    hits = search_es(query)

    contexts = []
    citations = []

    print("\n🔍 RETRIEVED CASES:\n")

    for i, hit in enumerate(hits):
        case_name = hit["_source"]["source_file"]
        print(f"{i+1}. {case_name}")

        full_case = get_full_case(case_name)

        if full_case:
            contexts.append(full_case[:1500])
            citations.append(case_name)

    if not contexts:
        print("❌ No context found")
        return

    prompt = f"""
You are a senior legal assistant.

Answer the legal question using ONLY the provided case contexts.

DO NOT summarize case-by-case.
Instead:

1. Extract the COMMON LEGAL PRINCIPLE
2. Combine reasoning across cases
3. Write like a legal explanation

FORMAT:

LEGAL PRINCIPLE:
<what law says>

REASONING:
<combined reasoning from cases>

CONCLUSION:
<final answer>

If answer is not in context, say:
"Not found in provided cases"

---------------------
QUESTION:
{query}

---------------------
CASE CONTEXTS:
{contexts}
"""

    response = model.generate_content(prompt)
    answer = response.text

    print("\n⚖️ FINAL ANSWER:\n")
    print(answer)

    print("\n📚 CITATIONS:")
    for c in citations:
        print("-", c)

# ---------------- RUN ----------------
if __name__ == "__main__":
    q = input("Enter legal query: ")
    generate_answer(q)