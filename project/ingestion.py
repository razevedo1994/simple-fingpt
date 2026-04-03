import uuid
from fastembed import (
    TextEmbedding,
    SparseTextEmbedding,
    LateInteractionTextEmbedding,
)
from qdrant_client import models
from storage.vector_storage import create_collection
from config.settings import (
    FILE_PATH,
    COLLECTION_NAME,
    QDRANT_ENDPOINT,
    DENSE_MODEL,
    SPARSE_MODEL,
    COLBERT_MODEL,
)


client_qdrant = create_collection(
    collection=COLLECTION_NAME,
    endpoint=QDRANT_ENDPOINT,
)

with open(FILE_PATH, "r", encoding="utf-8") as file:
    content = file.read()

paragraphs = content.split("\n\n")
chunks = [p.strip() for p in paragraphs if len(p.strip()) > 50]

dense_model = TextEmbedding(DENSE_MODEL)
sparse_model = SparseTextEmbedding(SPARSE_MODEL)
colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL)

points = []
for chunk in chunks:
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
        payload={"text": chunk, "source": FILE_PATH},
    )
    points.append(point)

    client_qdrant.upload_points(collection_name=COLLECTION_NAME, points=points)


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
