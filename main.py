import os
from sentence_transformers import SentenceTransformer
import numpy as np

docs_folder = "docs"

model = SentenceTransformer("all-MiniLM-L6-v2")

all_chunks = []

files = os.listdir(docs_folder)

for file in files:
    if file.endswith(".txt"):
        file_path = os.path.join(docs_folder, file)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = content.split("\n")

        for chunk in chunks:
            if chunk.strip():
                all_chunks.append({
                    "text": chunk,
                    "source": file
                })

for chunk in all_chunks:
    chunk["embedding"] = model.encode(chunk["text"])

for chunk in all_chunks:
    print("Text:", chunk["text"])
    print("Source:", chunk["source"])
    print("Embedding:", chunk["embedding"])
    print("-" * 50)




from sklearn.metrics.pairwise import cosine_similarity

query = "Can I get a refund?"

query_embedding = model.encode(query)


# for chunk in all_chunks:
#     score = cosine_similarity(
#         query_embedding.reshape(1, -1), # type: ignore
#         chunk["embedding"].reshape(1, -1)
#     )[0][0]

#     chunk["score"] = score

# all_chunks.sort(key=lambda x: x["score"], reverse=True)

# for chunk in all_chunks:
#     print(f"Score: {chunk['score']:.4f}")
#     print(f"Source: {chunk['source']}")
#     print(f"Text: {chunk['text']}")
#     print("-" * 50)


import faiss
import numpy as np

embeddings = np.array(
    [chunk["embedding"] for chunk in all_chunks]
).astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

faiss.normalize_L2(embeddings)

index.add(embeddings)

print("Number of vectors:", index.ntotal)


query = "Can I get a refund?"

query_embedding = model.encode([query]).astype("float32") # type: ignore

faiss.normalize_L2(query_embedding)

scores, indices = index.search(query_embedding, 3)

for score, idx in zip(scores[0], indices[0]):
    chunk = all_chunks[idx]

    print(f"Score: {score:.4f}")
    print(f"Source: {chunk['source']}")
    print(f"Text: {chunk['text']}")
    print("-" * 50)