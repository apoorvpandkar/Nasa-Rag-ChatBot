import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass  # Use built-in sqlite3 (works on macOS/Windows)
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")

    # Look for ChromaDB directories (directories starting with 'chroma')
    chroma_dirs = [d for d in current_dir.iterdir() 
                   if d.is_dir() and d.name.startswith("chroma")]

    for chroma_dir in chroma_dirs:
        try:
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False)
            )

            collections = client.list_collections()

            for collection in collections:
                key = f"{chroma_dir.name}::{collection.name}"

                try:
                    doc_count = collection.count()
                except Exception:
                    doc_count = "N/A"

                backends[key] = {
                    "directory": str(chroma_dir),
                    "collection_name": collection.name,
                    "display_name": f"{chroma_dir.name} / {collection.name} ({doc_count} docs)",
                    "doc_count": str(doc_count)
                }

        except Exception as e:
            error_msg = str(e)[:50]
            backends[str(chroma_dir)] = {
                "directory": str(chroma_dir),
                "collection_name": "",
                "display_name": f"{chroma_dir.name} (Error: {error_msg})",
                "doc_count": "0"
            }

    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str) -> Tuple:
    """Initialize the RAG system with specified backend (cached for performance)"""
    try:
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        # The collection persists the api_base it was built with (Vocareum), so
        # override the embedding function to follow the current environment.
        embedding_function = OpenAIEmbeddingFunction(
            api_key=os.environ.get("CHROMA_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            model_name="text-embedding-3-small",
            api_base=os.environ.get("OPENAI_API_BASE"),
        )
        collection = client.get_collection(collection_name, embedding_function=embedding_function)
        return collection, True, None
    except Exception as e:
        return None, False, str(e)


def retrieve_documents(collection, query: str, n_results: int = 3,
                       mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    where_filter = None

    if mission_filter and mission_filter.lower() not in ("all", ""):
        where_filter = {"mission": {"$eq": mission_filter}}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )

    return results


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    context_parts = ["=== RELEVANT CONTEXT FROM NASA DOCUMENTS ===\n"]

    for i, (doc, metadata) in enumerate(zip(documents, metadatas), 1):
        mission = metadata.get("mission", "unknown")
        mission = mission.replace("_", " ").title()

        source = metadata.get("source", "unknown")
        category = metadata.get("document_category", "general")
        category = category.replace("_", " ").title()

        header = f"[Source {i}] Mission: {mission} | File: {source} | Category: {category}"
        context_parts.append(header)

        max_length = 800
        if len(doc) > max_length:
            doc_text = doc[:max_length] + "... [truncated]"
        else:
            doc_text = doc

        context_parts.append(doc_text)
        context_parts.append("")  # blank line between sources

    return "\n".join(context_parts)