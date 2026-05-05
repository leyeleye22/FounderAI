import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.settings import get_settings
from app.services.retrieval.chroma_store import ChromaStore
from app.services.retrieval.ingestion import KnowledgeIngestionService


def main() -> None:
    settings = get_settings()
    knowledge_dir = ROOT_DIR / "knowledge"
    store = ChromaStore(
        persist_directory=settings.rag_dir,
        collection_name=settings.rag_collection,
        embedding_model=settings.embedding_model,
    )
    report = KnowledgeIngestionService(knowledge_dir=knowledge_dir, store=store).ingest()
    print(
        f"Ingested {report.documents} documents into {report.collection} "
        f"with {report.chunks} chunks."
    )


if __name__ == "__main__":
    main()
