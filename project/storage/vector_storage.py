from qdrant_client import QdrantClient, models


def create_collection(collection: str, endpoint: str) -> None:
    qdrant = QdrantClient(
        url=endpoint,
    )

    try:
        if not qdrant.collection_exists(collection_name=collection):

            qdrant.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                ),
            )
    except Exception as e:
        print(e)
