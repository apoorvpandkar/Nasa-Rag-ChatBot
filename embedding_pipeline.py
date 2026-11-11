#!/usr/bin/env python3
"""
ChromaDB Embedding Pipeline for NASA Space Mission Data - Text Files Only

This script reads parsed text data from various NASA space mission folders and creates
a permanent ChromaDB collection with OpenAI embeddings for RAG applications.
Optimized to process only text files to avoid duplication with JSON versions.

Supported data sources:
- Apollo 11 extracted data (text files only)
- Apollo 13 extracted data (text files only)
- Apollo 11 Textract extracted data (text files only)
- Challenger transcribed audio data (text files only)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
import openai
from openai import OpenAI
import hashlib
import time
from datetime import datetime
import argparse
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chroma_embedding_text_only.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChromaEmbeddingPipelineTextOnly:
    """Pipeline for creating ChromaDB collections with OpenAI embeddings - Text files only"""
    
    def __init__(self, 
                 openai_api_key: str,
                 chroma_persist_directory: str = "./chroma_db",
                 collection_name: str = "nasa_space_missions_text",
                 embedding_model: str = "text-embedding-3-small",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the embedding pipeline
        
        Args:
            openai_api_key: OpenAI API key
            chroma_persist_directory: Directory to persist ChromaDB
            collection_name: Name of the ChromaDB collection
            embedding_model: OpenAI embedding model to use
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between chunks
        """
        # Initialize OpenAI client
        self.openai_client = OpenAI(
            api_key=openai_api_key,
            base_url=os.environ.get("OPENAI_API_BASE")
        )

        # Store configuration parameters
        self.openai_api_key = openai_api_key
        self.chroma_persist_directory = chroma_persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Create or get collection with OpenAI embedding function
        embedding_function = OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name=embedding_model,
            api_base=os.environ.get("OPENAI_API_BASE")
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Initialized collection '{collection_name}' with {self.collection.count()} documents")

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Split text into chunks with metadata
        
        Args:
            text: Text to chunk
            metadata: Base metadata for the text
            
        Returns:
            List of (chunk_text, chunk_metadata) tuples
        """
        # Handle short texts that don't need chunking
        if len(text) <= self.chunk_size:
            chunk_metadata = {**metadata, "chunk_index": 0, "total_chunks": 1}
            return [[text, chunk_metadata]]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size

            if end < len(text):
                # Try to break at sentence boundary within the overlap zone.
                # Only search in [start + chunk_overlap, end] so the chunk is
                # long enough that next_start = end - chunk_overlap > start.
                search_start = start + self.chunk_overlap
                for sep in [". ", ".\n", "! ", "!\n", "? ", "?\n", "\n\n"]:
                    pos = text.rfind(sep, search_start, end)
                    if pos > search_start:
                        end = pos + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunk_metadata = {
                    **metadata,
                    "chunk_index": chunk_index,
                    "chunk_start": start,
                    "chunk_end": end
                }
                chunks.append([chunk, chunk_metadata])
                chunk_index += 1

            # Next chunk starts exactly chunk_overlap characters before this
            # chunk's end, so consecutive chunks always share chunk_overlap chars.
            next_start = end - self.chunk_overlap
            if next_start <= start:
                # Safety: always advance at least 1 char to avoid infinite loop
                next_start = end
            start = next_start

        # Update total_chunks directly during construction (avoid a second pass)
        total = len(chunks)
        for i in range(total):
            chunks[i][1]["total_chunks"] = total

        return chunks

    def check_document_exists(self, doc_id: str) -> bool:
        """
        Check if a document with the given ID already exists in the collection.
        Uses the cached set populated by _preload_existing_ids for performance.
        Falls back to a direct DB query only if the cache is unavailable.
        """
        if hasattr(self, "_existing_ids_cache"):
            return doc_id in self._existing_ids_cache
        result = self.collection.get(ids=[doc_id])
        return len(result["ids"]) > 0

    def _preload_existing_ids(self) -> None:
        """
        Load all existing document IDs from ChromaDB into an in-memory set.
        Call once before processing a batch to avoid N individual DB round-trips.
        """
        result = self.collection.get(include=[])  # fetch IDs only, no docs/embeddings
        self._existing_ids_cache = set(result["ids"])
        logger.info(f"Preloaded {len(self._existing_ids_cache)} existing doc IDs into cache")

    def update_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """
        Update an existing document in the collection
        
        Args:
            doc_id: Document ID to update
            text: New text content
            metadata: New metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            embedding = self.get_embedding(text)
            self.collection.update(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding]
            )
            logger.debug(f"Updated document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False

    def delete_documents_by_source(self, source_pattern: str) -> int:
        """
        Delete all documents from a specific source using a metadata filter
        instead of fetching the entire collection.
        """
        try:
            result = self.collection.get(
                where={"source": {"$contains": source_pattern}},
                include=[]  # IDs only
            )
            ids_to_delete = result["ids"]
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} documents matching source pattern: {source_pattern}")
                return len(ids_to_delete)
            else:
                logger.info(f"No documents found matching source pattern: {source_pattern}")
                return 0
        except Exception as e:
            logger.error(f"Error deleting documents by source: {e}")
            return 0

    def get_file_documents(self, file_path: Path) -> List[str]:
        """
        Get all document IDs for a specific file using a metadata filter
        instead of loading the entire collection.
        """
        try:
            source = file_path.stem
            mission = self.extract_mission_from_path(file_path)
            result = self.collection.get(
                where={"$and": [{"source": {"$eq": source}}, {"mission": {"$eq": mission}}]},
                include=[]  # IDs only
            )
            return result["ids"]
        except Exception as e:
            logger.error(f"Error getting file documents: {e}")
            return []

    def get_embedding(self, text: str) -> List[float]:
        """Get OpenAI embedding for a single text string."""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def get_embeddings_batch(self, texts: List[str], api_batch_size: int = 20) -> List[List[float]]:
        """
        Get OpenAI embeddings for a list of texts sequentially in batches.
        """
        if not texts:
            return []

        all_embeddings = []
        for i in range(0, len(texts), api_batch_size):
            batch = texts[i:i + api_batch_size]
            try:
                response = self.openai_client.embeddings.create(
                    input=batch,
                    model=self.embedding_model
                )
                all_embeddings.extend([item.embedding for item in response.data])
                logger.info(f"Embedded batch {i // api_batch_size + 1}, {len(batch)} texts")
            except Exception as e:
                logger.error(f"Error generating batch embeddings: {e}")
                raise

        return all_embeddings

    def generate_document_id(self, file_path: Path, metadata: Dict[str, Any]) -> str:
        """
        Generate stable document ID based on file path and chunk position
        This allows for document updates without changing IDs
        """
        mission = metadata.get("mission", "unknown")
        source = metadata.get("source", file_path.stem)
        chunk_index = metadata.get("chunk_index", 0)
        # Format: mission_source_chunk_0001
        doc_id = f"{mission}_{source}_chunk_{chunk_index:04d}"
        return doc_id

    def process_text_file(self, file_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Process plain text files with enhanced metadata extraction
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of (text, metadata) tuples
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                return []
            
            metadata = {
                'source': file_path.stem,
                'file_path': str(file_path),
                'file_type': 'text',
                'content_type': 'full_text',
                'mission': self.extract_mission_from_path(file_path),
                'data_type': self.extract_data_type_from_path(file_path),
                'document_category': self.extract_document_category_from_filename(file_path.name),
                'file_size': len(content),
                'processed_timestamp': datetime.now().isoformat()
            }
            
            return self.chunk_text(content, metadata)
            
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return []

    def extract_mission_from_path(self, file_path: Path) -> str:
        """Extract mission name from file path"""
        path_str = str(file_path).lower()
        if 'apollo11' in path_str or 'apollo_11' in path_str:
            return 'apollo_11'
        elif 'apollo13' in path_str or 'apollo_13' in path_str:
            return 'apollo_13'
        elif 'challenger' in path_str:
            return 'challenger'
        else:
            return 'unknown'

    def extract_data_type_from_path(self, file_path: Path) -> str:
        """Extract data type from file path"""
        path_str = str(file_path).lower()
        if 'transcript' in path_str:
            return 'transcript'
        elif 'textract' in path_str:
            return 'textract_extracted'
        elif 'audio' in path_str:
            return 'audio_transcript'
        elif 'flight_plan' in path_str:
            return 'flight_plan'
        else:
            return 'document'

    def extract_document_category_from_filename(self, filename: str) -> str:
        """Extract document category from filename for better organization"""
        filename_lower = filename.lower()
        
        if 'pao' in filename_lower:
            return 'public_affairs_officer'
        elif 'cm' in filename_lower:
            return 'command_module'
        elif 'tec' in filename_lower:
            return 'technical'
        elif 'flight_plan' in filename_lower:
            return 'flight_plan'
        elif 'mission_audio' in filename_lower:
            return 'mission_audio'
        elif 'ntrs' in filename_lower:
            return 'nasa_archive'
        elif '19900066485' in filename_lower:
            return 'technical_report'
        elif '19710015566' in filename_lower:
            return 'mission_report'
        elif 'full_text' in filename_lower:
            return 'complete_document'
        else:
            return 'general_document'

    def scan_text_files_only(self, base_path: str) -> List[Path]:
        """
        Scan data directories for text files only (avoiding JSON duplicates)
        
        Args:
            base_path: Base directory path
            
        Returns:
            List of text file paths to process
        """
        base_path = Path(base_path)
        files_to_process = []
        
        data_dirs = ['apollo11', 'apollo13', 'challenger']
        
        for data_dir in data_dirs:
            dir_path = base_path / data_dir
            if dir_path.exists():
                logger.info(f"Scanning directory: {dir_path}")
                text_files = list(dir_path.glob('**/*.txt'))
                files_to_process.extend(text_files)
                logger.info(f"Found {len(text_files)} text files in {data_dir}")
        
        filtered_files = []
        for file_path in files_to_process:
            if (file_path.name.startswith('.') or 
                'summary' in file_path.name.lower() or
                file_path.suffix.lower() != '.txt'):
                continue
            filtered_files.append(file_path)
        
        logger.info(f"Total text files to process: {len(filtered_files)}")
        
        mission_counts = {}
        for file_path in filtered_files:
            mission = self.extract_mission_from_path(file_path)
            mission_counts[mission] = mission_counts.get(mission, 0) + 1
        
        logger.info("Files by mission:")
        for mission, count in mission_counts.items():
            logger.info(f"  {mission}: {count} files")
        
        return filtered_files

    def add_documents_to_collection(self, documents: List, 
                                   file_path: Path, batch_size: int = 100, 
                                   update_mode: str = 'skip') -> Dict[str, int]:
        """
        Add documents to ChromaDB collection in batches.

        Key performance improvements vs. original:
        - Existence checks use an in-memory set (_existing_ids_cache) instead of
          one DB round-trip per document.
        - Embeddings are generated in a single batched API call per batch instead
          of one call per document.
        - get_file_documents uses a metadata WHERE filter instead of full fetch.
        """
        if not documents:
            return {'added': 0, 'updated': 0, 'skipped': 0}

        stats = {'added': 0, 'updated': 0, 'skipped': 0}

        # Preload all existing IDs once (skip/update modes) to avoid per-doc queries
        if update_mode in ('skip', 'update') and not hasattr(self, '_existing_ids_cache'):
            self._preload_existing_ids()

        # Handle 'replace' mode: delete all existing docs for this file first
        if update_mode == 'replace':
            existing_ids = self.get_file_documents(file_path)
            if existing_ids:
                self.collection.delete(ids=existing_ids)
                logger.info(f"Deleted {len(existing_ids)} existing documents for {file_path.name}")
            # Invalidate cache since we just removed docs
            if hasattr(self, '_existing_ids_cache'):
                for eid in existing_ids:
                    self._existing_ids_cache.discard(eid)

        # Process in batches
        for batch_start in range(0, len(documents), batch_size):
            batch = documents[batch_start:batch_start + batch_size]

            ids_to_add = []
            texts_to_add = []
            metadatas_to_add = []

            for text, metadata in batch:
                doc_id = self.generate_document_id(file_path, metadata)

                if update_mode == 'skip' and doc_id in self._existing_ids_cache:
                    stats['skipped'] += 1
                    continue

                if update_mode == 'update' and doc_id in self._existing_ids_cache:
                    # update_document still calls get_embedding individually;
                    # updates are rare so this is acceptable
                    self.update_document(doc_id, text, metadata)
                    stats['updated'] += 1
                    continue

                # New document (or replace mode)
                ids_to_add.append(doc_id)
                texts_to_add.append(text)
                metadatas_to_add.append(metadata)

            if ids_to_add:
                try:
                    # One API call for all texts in this batch
                    embeddings = self.get_embeddings_batch(texts_to_add)
                    self.collection.add(
                        ids=ids_to_add,
                        documents=texts_to_add,
                        metadatas=metadatas_to_add,
                        embeddings=embeddings
                    )
                    stats['added'] += len(ids_to_add)
                    # Keep cache in sync
                    if hasattr(self, '_existing_ids_cache'):
                        self._existing_ids_cache.update(ids_to_add)
                    logger.debug(f"Added batch of {len(ids_to_add)} documents")
                except Exception as e:
                    logger.error(f"Error adding batch to collection: {e}")

        return stats

    def process_all_text_data(self, base_path: str, update_mode: str = 'skip') -> Dict[str, int]:
        """
        Process all text files and add to ChromaDB
        
        Args:
            base_path: Base directory containing data folders
            update_mode: How to handle existing documents
            
        Returns:
            Statistics about processed files
        """
        stats = {
            'files_processed': 0,
            'documents_added': 0,
            'documents_updated': 0,
            'documents_skipped': 0,
            'errors': 0,
            'total_chunks': 0,
            'missions': {}
        }

        files = self.scan_text_files_only(base_path)

        # Preload existing IDs once for the entire run (avoids N*chunks DB queries)
        if update_mode in ('skip', 'update'):
            self._preload_existing_ids()

        for file_path in files:
            mission = self.extract_mission_from_path(file_path)
            if mission not in stats['missions']:
                stats['missions'][mission] = {'files': 0, 'chunks': 0, 'added': 0, 'updated': 0, 'skipped': 0}

            try:
                documents = self.process_text_file(file_path)
                if not documents:
                    logger.warning(f"No content extracted from {file_path}")
                    continue

                logger.info(f"Processing {file_path.name}: {len(documents)} chunks to embed...")
                file_stats = self.add_documents_to_collection(
                    documents, file_path, batch_size=20, update_mode=update_mode
                )

                stats['files_processed'] += 1
                stats['documents_added'] += file_stats['added']
                stats['documents_updated'] += file_stats['updated']
                stats['documents_skipped'] += file_stats['skipped']
                stats['total_chunks'] += len(documents)

                stats['missions'][mission]['files'] += 1
                stats['missions'][mission]['chunks'] += len(documents)
                stats['missions'][mission]['added'] += file_stats['added']
                stats['missions'][mission]['updated'] += file_stats['updated']
                stats['missions'][mission]['skipped'] += file_stats['skipped']

                logger.info(f"Processed {file_path.name}: {len(documents)} chunks, "
                            f"added={file_stats['added']}, updated={file_stats['updated']}, "
                            f"skipped={file_stats['skipped']}")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                stats['errors'] += 1

        return stats

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the ChromaDB collection"""
        return {
            'collection_name': self.collection_name,
            'document_count': self.collection.count(),
            'persist_directory': self.chroma_persist_directory,
            'embedding_model': self.embedding_model
        }

    def query_collection(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Query the collection for testing
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            
        Returns:
            Query results
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the collection"""
        try:
            all_docs = self.collection.get()
            
            if not all_docs['metadatas']:
                return {'error': 'No documents in collection'}
            
            stats = {
                'total_documents': len(all_docs['metadatas']),
                'missions': {},
                'data_types': {},
                'document_categories': {},
                'file_types': {}
            }
            
            for metadata in all_docs['metadatas']:
                mission = metadata.get('mission', 'unknown')
                data_type = metadata.get('data_type', 'unknown')
                doc_category = metadata.get('document_category', 'unknown')
                file_type = metadata.get('file_type', 'unknown')
                
                stats['missions'][mission] = stats['missions'].get(mission, 0) + 1
                stats['data_types'][data_type] = stats['data_types'].get(data_type, 0) + 1
                stats['document_categories'][doc_category] = stats['document_categories'].get(doc_category, 0) + 1
                stats['file_types'][file_type] = stats['file_types'].get(file_type, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='ChromaDB Embedding Pipeline for NASA Data')
    parser.add_argument('--data-path', default='./data_text', help='Path to data directories')
    parser.add_argument('--openai-key', default=os.environ.get("OPENAI_API_KEY"), required=not os.environ.get("OPENAI_API_KEY"), help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--chroma-dir', default='./chroma_db_openai', help='ChromaDB persist directory')
    parser.add_argument('--collection-name', default='nasa_space_missions_text', help='Collection name')
    parser.add_argument('--embedding-model', default='text-embedding-3-small', help='OpenAI embedding model')
    parser.add_argument('--chunk-size', type=int, default=500, help='Text chunk size')
    parser.add_argument('--chunk-overlap', type=int, default=100, help='Chunk overlap size')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--update-mode', choices=['skip', 'update', 'replace'], default='skip',
                       help='How to handle existing documents: skip, update, or replace')
    parser.add_argument('--test-query', help='Test query after processing')
    parser.add_argument('--stats-only', action='store_true', help='Only show collection statistics')
    parser.add_argument('--delete-source', help='Delete all documents from a specific source pattern')
    
    args = parser.parse_args()
    
    logger.info("Initializing ChromaDB Embedding Pipeline...")
    pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key=args.openai_key,
        chroma_persist_directory=args.chroma_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    if args.delete_source:
        deleted_count = pipeline.delete_documents_by_source(args.delete_source)
        logger.info(f"Deleted {deleted_count} documents matching source pattern: {args.delete_source}")
        return
    
    if args.stats_only:
        logger.info("Collection Statistics:")
        stats = pipeline.get_collection_stats()
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        return
    
    logger.info(f"Starting text data processing with update mode: {args.update_mode}")
    start_time = time.time()
    
    stats = pipeline.process_all_text_data(args.data_path, update_mode=args.update_mode)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Total chunks created: {stats['total_chunks']}")
    logger.info(f"Documents added to collection: {stats['documents_added']}")
    logger.info(f"Documents updated in collection: {stats['documents_updated']}")
    logger.info(f"Documents skipped (already exist): {stats['documents_skipped']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Processing time: {processing_time:.2f} seconds")
    
    logger.info("\nMission breakdown:")
    for mission, mission_stats in stats['missions'].items():
        logger.info(f"  {mission}: {mission_stats['files']} files, {mission_stats['chunks']} chunks")
        logger.info(f"    Added: {mission_stats['added']}, Updated: {mission_stats['updated']}, Skipped: {mission_stats['skipped']}")
    
    collection_info = pipeline.get_collection_info()
    logger.info(f"\nCollection: {collection_info.get('collection_name', 'N/A')}")
    logger.info(f"Total documents in collection: {collection_info.get('document_count', 'N/A')}")
    
    if args.test_query:
        logger.info(f"\nTesting query: '{args.test_query}'")
        results = pipeline.query_collection(args.test_query)
        if results and 'documents' in results:
            logger.info(f"Found {len(results['documents'][0])} results:")
            for i, doc in enumerate(results['documents'][0][:3]):
                logger.info(f"Result {i+1}: {doc[:200]}...")
    
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()