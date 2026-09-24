from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR

CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "zepto_policies"


# ---------------------------------------------------------
# Embedding model
# ---------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"

print("Loading embedding model...")
embedding_model = SentenceTransformer(MODEL_NAME)
print("Embedding model loaded.")


# ---------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------
client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


# ---------------------------------------------------------
# Load documents
# ---------------------------------------------------------
def load_documents():
    documents = []

    for i in range(1, 9):
        file_path = DOCS_DIR / f"doc_{i:02d}.txt"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing document: {file_path}"
            )

        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            raise ValueError(
                f"Document is empty: {file_path}"
            )

        documents.append(
            {
                "doc_id": f"doc_{i:02d}",
                "text": text
            }
        )

    return documents


# ---------------------------------------------------------
# Chunk documents
# ---------------------------------------------------------
def chunk_documents(documents):
    chunks = []

    for document in documents:
        text = document["text"]

        # Each policy is short, so one document = one chunk.
        chunks.append(
            {
                "chunk_id": f"{document['doc_id']}_chunk_01",
                "doc_id": document["doc_id"],
                "text": text
            }
        )

    return chunks


# ---------------------------------------------------------
# Build ChromaDB knowledge base
# ---------------------------------------------------------
def build_knowledge_base():
    documents = load_documents()

    print(f"Loaded {len(documents)} documents.")

    chunks = chunk_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

    ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "doc_id": chunk["doc_id"],
            "chunk_id": chunk["chunk_id"]
        }
        for chunk in chunks
    ]

    # Remove old data so the build remains deterministic.
    existing = collection.get()

    if existing["ids"]:
        collection.delete(
            ids=existing["ids"]
        )

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print("Knowledge base created successfully.")
    print(f"ChromaDB collection: {COLLECTION_NAME}")
    print(f"Total chunks stored: {collection.count()}")


# ---------------------------------------------------------
# Retrieve top-k chunks
# ---------------------------------------------------------
def retrieve(query, top_k=3):
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []

    for i in range(len(results["documents"][0])):
        retrieved.append(
            {
                "chunk_id": results["metadatas"][0][i]["chunk_id"],
                "doc_id": results["metadatas"][0][i]["doc_id"],
                "text": results["documents"][0][i],
                "distance": results["distances"][0][i]
            }
        )

    return retrieved


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    build_knowledge_base()

    print("\n===== TEST RETRIEVAL =====")

    test_query = "How long does Zepto delivery take?"

    results = retrieve(
        test_query,
        top_k=3
    )

    for result in results:
        print("\nChunk ID:", result["chunk_id"])
        print("Document:", result["doc_id"])
        print("Distance:", result["distance"])
        print("Text:", result["text"][:200])