from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from config.settings import COLLECTION_NAME, FILE_PATH

qdrant = QdrantClient(
    url="http://localhost:6333",
)

try:
    if not qdrant.collection_exists(collection_name=COLLECTION_NAME):

        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE,
            ),
        )
except Exception as e:
    print(e)


with open(FILE_PATH, "r", encoding="utf-8") as file:
    content = file.read()
    print(content)
