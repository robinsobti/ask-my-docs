import time
import unittest
import uuid

_DEPENDENCIES_AVAILABLE = True
_DEPENDENCY_ERROR = None

try:
    from src.config import COLLECTION_NAME
    from src.retriever import retrieve
    from src.weaviate_store import create_collection_if_missing, upsert_batch
except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
    _DEPENDENCIES_AVAILABLE = False
    _DEPENDENCY_ERROR = exc


@unittest.skipUnless(_DEPENDENCIES_AVAILABLE, f"Missing dependency: {_DEPENDENCY_ERROR}")
class RetrieverIntegrationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.collection = create_collection_if_missing()
        except Exception as exc:  # pragma: no cover - depends on external service
            raise unittest.SkipTest(f"Weaviate not available: {exc}") from exc

    def test_bm25_returns_inserted_document(self) -> None:
        unique_keyword = f"integration-{uuid.uuid4().hex}"
        doc_uuid = str(uuid.uuid4())
        chunk_uuid = str(uuid.uuid4())
        document = {
            "id": doc_uuid,
            "text": f"This integration test document mentions {unique_keyword}.",
            "title": "Integration Test Document",
            "source": "integration-test",
            "doc_id": doc_uuid,
            "chunk_id": chunk_uuid,
        }

        upsert_batch([document])

        hits = []
        try:
            for _ in range(5):  # allow the index to update
                hits = retrieve(
                    collection=COLLECTION_NAME,
                    query=unique_keyword,
                    k=3,
                    mode="bm25",
                )
                if any(hit["id"] == doc_uuid for hit in hits):
                    break
                time.sleep(0.5)
        finally:
            try:
                self.collection.data.delete_by_id(doc_uuid)
            except Exception:
                pass

        self.assertTrue(hits, "Expected at least one retrieval hit.")
        inserted_hit = next((hit for hit in hits if hit["id"] == doc_uuid), None)
        self.assertIsNotNone(inserted_hit, "Inserted document was not returned.")
        self.assertIn(unique_keyword, inserted_hit["text"])


if __name__ == "__main__":
    unittest.main()
