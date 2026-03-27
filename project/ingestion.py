import uuid
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import models
from storage.vector_storage import create_collection
from config.settings import FILE_PATH, COLLECTION_NAME, QDRANT_ENDPOINT, DENSE_MODEL


client_qdrant = create_collection(
    collection=COLLECTION_NAME,
    endpoint=QDRANT_ENDPOINT,
)

with open(FILE_PATH, "r", encoding="utf-8") as file:
    content = file.read()

paragraphs = content.split("\n\n")
chunks = [p.strip() for p in paragraphs if len(p.strip()) > 50]

model = TextEmbedding(DENSE_MODEL)

points = []
for chunk in chunks:
    embedding = list(model.passage_embed([chunk]))[0].tolist()
    point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding,
        payload={"text": chunk, "source": FILE_PATH},
    )
    points.append(point)

    client_qdrant.upload_points(collection_name=COLLECTION_NAME, points=points)


query_text = "what are the main financial risks?"
query_embedding = list(model.query_embed([query_text]))[0].tolist()

results = client_qdrant.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding,
    limit=3,
)

for r in results.points:
    print(f"Score: {r.score}")
    print(f"Texto: {r.payload['text'][:100]}")
    print("-" * 80)
