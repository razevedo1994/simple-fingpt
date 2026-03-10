import uuid
from fastembed import TextEmbedding
from qdrant_client import models
from storage.vector_storage import create_collection
from config.settings import FILE_PATH, COLLECTION_NAME, QDRANT_ENDPOINT, MODEL_NAME


client_qdrant = create_collection(
    collection=COLLECTION_NAME,
    endpoint=QDRANT_ENDPOINT,
)

with open(FILE_PATH, "r", encoding="utf-8") as file:
    content = file.read()

paragraphs = content.split("\n\n")
chunks = [p.strip() for p in paragraphs if len(p.strip()) > 50]

model = TextEmbedding(MODEL_NAME)

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
