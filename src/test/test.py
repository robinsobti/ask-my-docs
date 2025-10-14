import sys
import unittest
from types import ModuleType, SimpleNamespace

def _ensure_stub_module(name: str, module: ModuleType) -> None:
    if name not in sys.modules:
        sys.modules[name] = module

dotenv_stub = ModuleType("dotenv")
setattr(dotenv_stub, "load_dotenv", lambda *_args, **_kwargs: None)
_ensure_stub_module("dotenv", dotenv_stub)

weaviate_stub = ModuleType("weaviate")
setattr(weaviate_stub, "WeaviateClient", SimpleNamespace)
setattr(weaviate_stub, "connect_to_local", lambda *args, **kwargs: SimpleNamespace(collections=SimpleNamespace()))
_ensure_stub_module("weaviate", weaviate_stub)

classes_module = ModuleType("weaviate.classes")
_ensure_stub_module("weaviate.classes", classes_module)

config_module = ModuleType("weaviate.classes.config")


class _Configure:
    class Vectorizer:
        @staticmethod
        def none():
            return None

    @staticmethod
    def inverted_index():
        return None


class _Property:
    def __init__(self, name, data_type, description=None):
        self.name = name
        self.data_type = data_type
        self.description = description


class _DataType:
    TEXT = "text"


setattr(config_module, "Configure", _Configure)
setattr(config_module, "Property", _Property)
setattr(config_module, "DataType", _DataType)
_ensure_stub_module("weaviate.classes.config", config_module)

collections_module = ModuleType("weaviate.collections")
setattr(collections_module, "collection", object)
_ensure_stub_module("weaviate.collections", collections_module)

exceptions_module = ModuleType("weaviate.exceptions")


class _WeaviateBaseError(Exception):
    pass


setattr(exceptions_module, "WeaviateBaseError", _WeaviateBaseError)
_ensure_stub_module("weaviate.exceptions", exceptions_module)

from src.retriever import _extract_score


class ExtractScoreTestCase(unittest.TestCase):
    def test_returns_score_when_present(self) -> None:
        metadata = SimpleNamespace(score=0.42)
        self.assertAlmostEqual(_extract_score(metadata), 0.42)

    def test_falls_back_to_certainty(self) -> None:
        metadata = SimpleNamespace(certainty=0.73)
        self.assertAlmostEqual(_extract_score(metadata), 0.73)

    def test_converts_distance_to_similarity(self) -> None:
        metadata = SimpleNamespace(distance=0.2)
        self.assertAlmostEqual(_extract_score(metadata), 0.8)

    def test_returns_zero_when_no_metric(self) -> None:
        metadata = SimpleNamespace()
        self.assertEqual(_extract_score(metadata), 0.0)
        self.assertEqual(_extract_score(None), 0.0)


if __name__ == "__main__":
    unittest.main()