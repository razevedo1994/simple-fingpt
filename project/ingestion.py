from fastembed import TextEmbedding
from storage.vector_storage import create_collection
from config.settings import FILE_PATH, COLLECTION_NAME, QDRANT_ENDPOINT


create_collection(
    collection=COLLECTION_NAME,
    endpoint=QDRANT_ENDPOINT,
)

with open(FILE_PATH, "r", encoding="utf-8") as file:
    content = file.read()
    print(content)
