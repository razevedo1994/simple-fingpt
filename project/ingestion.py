import uuid
from fastembed import (
    TextEmbedding,
    SparseTextEmbedding,
    LateInteractionTextEmbedding,
)
from qdrant_client import models
from storage.vector_storage import create_collection
from config.settings import (
    EMAIL,
    MAX_TOKENS,
    COLLECTION_NAME,
    QDRANT_ENDPOINT,
    DENSE_MODEL,
    SPARSE_MODEL,
    COLBERT_MODEL,
)
from utils.semantic_chunker import SemanticChunker
from utils.edgar_client import EdgarClient


client_qdrant = create_collection(
    collection=COLLECTION_NAME,
    endpoint=QDRANT_ENDPOINT,
)

edgar = EdgarClient(email=EMAIL)

data_10k = edgar.fetch_filing_data("AAPL", "10-Q")
text_10k = edgar.get_combined_data(data_10k)

data_10q = edgar.fetch_filing_data("AAPL", "10-Q")
text_10q = edgar.get_combined_data(data_10k)

chunker = SemanticChunker(max_tokens=MAX_TOKENS)

all_chunks = []
for data, text in [(data_10k, text_10q), (data_10q, text_10q)]:
    chunks = chunker.create_chunks(text)
    for chunk in chunks:
        all_chunks.append(
            {
                "text": chunk,
                "metadata": data["metadata"],
            }
        )

dense_model = TextEmbedding(DENSE_MODEL)
sparse_model = SparseTextEmbedding(SPARSE_MODEL)
colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL)

points = []
for chunk_data in all_chunks:
    chunk = chunk_data["text"]
    metadata = chunk_data["metadata"]

    dense_embedding = list(dense_model.passage_embed([chunk]))[0].tolist()
    sparse_embedding = list(sparse_model.passage_embed([chunk]))[0].as_object()
    colbert_embedding = list(colbert_model.passage_embed([chunk]))[0].tolist()

    point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector={
            "dense": dense_embedding,
            "sparse": sparse_embedding,
            "colbert": colbert_embedding,
        },
        payload={"text": chunk, "metadata": metadata},
    )
    points.append(point)

client_qdrant.upload_points(
    collection_name=COLLECTION_NAME,
    points=points,
    batch_size=5,
)


query_text = "what are the main financial risks?"
query_dense = list(dense_model.query_embed([query_text]))[0].tolist()
query_sparse = list(sparse_model.query_embed([query_text]))[0].as_object()
query_colbert = list(colbert_model.query_embed([query_text]))[0].tolist()

results = client_qdrant.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=[
        {
            "prefetch": [
                {
                    "query": query_dense,
                    "using": "dense",
                    "limit": 10,
                },
                {
                    "query": query_sparse,
                    "using": "sparse",
                    "limit": 10,
                },
            ],
            "query": models.FusionQuery(fusion=models.Fusion.RRF),
            "limit": 20,
        }
    ],
    query=query_colbert,
    using="colbert",
    limit=3,
)
max_score = max(result.score for result in results.points)

for r in results.points:
    normalized_score = r.score / max_score
    print(f"Score: {normalized_score}")
    print(f"Texto: {r.payload['text'][:100]}")
    print("-" * 80)
