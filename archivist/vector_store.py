import json
import os

class VectorStore:
    def __init__(self, index_path="archivist_index.json"):
        self.index_path = index_path
        self.documents = []
        self.load()

    def add_document(self, source, content):
        doc = {
            "source": source,
            "content": content
        }
        self.documents.append(doc)

    def save(self):
        with open(self.index_path, 'w') as f:
            json.dump(self.documents, f, indent=2)
        print(f"Index saved to {self.index_path}")

    def load(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, 'r') as f:
                self.documents = json.load(f)
        else:
            self.documents = []

    def search(self, query):
        # Simple keyword search for now
        results = []
        query_terms = query.lower().split()
        for doc in self.documents:
            content_lower = doc['content'].lower()
            score = 0
            for term in query_terms:
                if term in content_lower:
                    score += 1
            if score > 0:
                results.append((doc, score))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results]
