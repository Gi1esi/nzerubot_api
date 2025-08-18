import requests

resp = requests.post(
    "http://localhost:8000/rag/ask",
    json={"question": "what is a starch test?"}
)
print(resp.json())
