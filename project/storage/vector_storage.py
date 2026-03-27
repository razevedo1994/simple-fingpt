from qdrant_client import QdrantClient, models


def create_collection(collection: str, endpoint: str) -> None:
    qdrant = QdrantClient(
        url=endpoint,
    )

    delete_collection(collection, qdrant)

    try:
        if not qdrant.collection_exists(collection_name=collection):

            qdrant.create_collection(
                collection_name=collection,
                vectors_config={
                    "dense": models.VectorParams(
                        size=384,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(),
                },
            )
    except Exception as e:
        print(e)

    return qdrant


def delete_collection(collection: str, qdrant_client: QdrantClient):
    qdrant_client.delete_collection(collection_name=collection)
