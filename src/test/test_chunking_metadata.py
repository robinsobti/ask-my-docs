from src.chunking import split_into_chunks


def test_split_into_chunks_preserves_book_name_metadata() -> None:
    doc = {
        "doc_id": "oxford-handbook",
        "title": "Oxford Handbook of Clinical Dentistry",
        "source": "/tmp/Oxford Handbook of Clinical Dentistry 6e.pdf",
        "text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 10,
        "metadata": {"book_name": "Oxford Handbook of Clinical Dentistry", "source": "/tmp/Oxford.pdf"},
    }

    chunks = list(split_into_chunks(doc, chunk_size=80, chunk_overlap=0))
    assert chunks, "expected chunks to be produced"
    for chunk in chunks:
        assert chunk["metadata"]["book_name"] == "Oxford Handbook of Clinical Dentistry"
        assert chunk["title"] == "Oxford Handbook of Clinical Dentistry"
