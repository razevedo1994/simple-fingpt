from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from config.settings import COLLECTION_NAME

qdrant = QdrantClient(
    url="http://localhost:6333",
)

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=384,
        distance=models.Distance.COSINE,
    ),
)
