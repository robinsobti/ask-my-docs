import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.collections import collection
from .config import WEAVIATE_URL, DOCS_SCHEMA, COLLECTION_NAME
from weaviate.exceptions import WeaviateBaseError
from typing import List, Dict, Any

def get_client() -> weaviate.WeaviateClient:
    return weaviate.connect_to_local(host=WEAVIATE_URL.replace("http://", "").replace("https://", "").split(":")[0],
                                     port=int(WEAVIATE_URL.split(":")[-1]))

def create_collection_if_missing() -> collection:
    client = get_client()
    try:
        list_of_collections = [c for c in client.collections.list_all()]
        if COLLECTION_NAME not in list_of_collections:
            props = [
                    Property(name=p["name"], data_type=DataType.TEXT, description=p.get("description"))
                    for p in DOCS_SCHEMA["properties"]
                ]
            client.collections.create(
                    name=DOCS_SCHEMA["name"],
                    description=DOCS_SCHEMA["description"],
                    properties=props,
                    vectorizer_config=Configure.Vectorizer.none(),
                    inverted_index_config=Configure.inverted_index()
                )
        coll = client.collections.get(COLLECTION_NAME)
        return coll
    except WeaviateBaseError as wbe:
        raise RuntimeError(f'Error creating collection: {wbe}') from wbe
    
def upsert_batch(objs: List[Dict[str, Any]]) -> int:
    """
    Insert documents (BM25-only). If an ID exists, we overwrite it.
    Each obj must include: id (uuid string), text, title, source, doc_id, chunk_id
    """
    coll = create_collection_if_missing()
    count = 0
    # v0 simplicity: delete if exists, then insert
    for o in objs:
        # ensure required fields
        for k in ("id","text","title","source","doc_id","chunk_id"):
            if k not in o:
                raise ValueError(f"Missing field '{k}' in object: {o}")
        try:
            coll.data.delete_by_id(o["id"])
        except Exception:
            pass
        coll.data.insert(
            properties={
                "text": o["text"],
                "title": o["title"],
                "source": o["source"],
                "doc_id": o["doc_id"],
                "chunk_id": o["chunk_id"],
            },
            uuid=o["id"]
        )
        count += 1
    return count

def search_bm25(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    BM25 keyword search (no vectors).
    Returns list of dicts with score, title, text, source, id.
    """
    coll = create_collection_if_missing()
    res = coll.query.bm25(query=query, limit=k)
    hits = []
    for o in res.objects:
        hits.append({
            "id": o.uuid,
            "score": o.metadata.score,  # BM25 score
            "title": o.properties.get("title",""),
            "text": o.properties.get("text",""),
            "source": o.properties.get("source",""),
        })
    return hits
