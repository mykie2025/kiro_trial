"""LangChain-based vector persistence implementation for context storage."""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from langchain_community.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from ..config.config_manager import ConfigManager
from .memory_document import MemoryDocument, MemoryType


logger = logging.getLogger(__name__)


class LangChainVectorPersistence:
    """
    LangChain-based vector persistence system for semantic similarity search.
    
    This class provides functionality to store and retrieve memories using
    vector embeddings and semantic similarity search, with user-based filtering.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the vector persistence system.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        
        # Initialize OpenAI embeddings
        openai_config = config_manager.get_openai_client_config()
        self.embeddings = OpenAIEmbeddings(
            model=self.config.embedding_model,
            openai_api_key=openai_config['api_key'],
            openai_api_base=openai_config['base_url']
        )
        
        # Initialize in-memory vector store
        self.vector_store = InMemoryVectorStore(self.embeddings)
        
        logger.info("LangChain vector persistence initialized successfully")
    
    def save_memory(
        self, 
        content: str, 
        user_id: str, 
        memory_type: MemoryType = MemoryType.CONVERSATION,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a memory to the vector store.
        
        Args:
            content: The memory content to store
            user_id: User identifier for filtering
            memory_type: Type of memory being stored
            metadata: Additional metadata to store with the memory
            
        Returns:
            Document ID of the stored memory
            
        Raises:
            ValueError: If content or user_id is empty
            Exception: If storage operation fails
        """
        try:
            # Create memory document
            memory_doc = MemoryDocument(
                content=content,
                user_id=user_id,
                memory_type=memory_type,
                metadata=metadata or {}
            )
            
            # Create LangChain document
            langchain_doc = Document(
                page_content=content,
                metadata=memory_doc.get_langchain_metadata()
            )
            
            # Add to vector store
            doc_ids = self.vector_store.add_documents([langchain_doc])
            doc_id = doc_ids[0] if doc_ids else None
            
            if not doc_id:
                raise Exception("Failed to generate document ID")
            
            logger.info(f"Memory saved successfully for user {user_id} with ID {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to save memory for user {user_id}: {str(e)}")
            raise
    
    def search_memories(
        self, 
        query: str, 
        user_id: str, 
        k: int = 3,
        memory_type: Optional[MemoryType] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for memories using semantic similarity.
        
        Args:
            query: Search query text
            user_id: User identifier for filtering
            k: Number of results to return (default: 3)
            memory_type: Optional memory type filter
            
        Returns:
            List of memory results with content and metadata
            
        Raises:
            ValueError: If query or user_id is empty
            Exception: If search operation fails
        """
        try:
            if not query or not query.strip():
                raise ValueError("Search query cannot be empty")
            
            if not user_id or not user_id.strip():
                raise ValueError("User ID cannot be empty")
            
            # Build metadata filter function
            def metadata_filter(doc):
                """Filter function for metadata matching."""
                doc_metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                
                # Check user_id match
                if doc_metadata.get('user_id') != user_id:
                    return False
                
                # Check memory_type match if specified
                if memory_type and doc_metadata.get('memory_type') != memory_type.value:
                    return False
                
                return True
            
            # Perform similarity search with metadata filtering
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=metadata_filter
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                result = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': score,
                    'user_id': doc.metadata.get('user_id'),
                    'memory_type': doc.metadata.get('memory_type'),
                    'timestamp': doc.metadata.get('timestamp')
                }
                formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} memories for user {user_id}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search memories for user {user_id}: {str(e)}")
            raise
    
    def get_all_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all memories for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of all memories for the user
        """
        try:
            if not user_id or not user_id.strip():
                raise ValueError("User ID cannot be empty")
            
            # Build metadata filter function for user
            def user_filter(doc):
                """Filter function for user matching."""
                doc_metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                return doc_metadata.get('user_id') == user_id
            
            # Get all documents and filter by user
            all_docs = []
            for doc_id, doc_data in self.vector_store.store.items():
                doc = Document(
                    page_content=doc_data['text'],
                    metadata=doc_data['metadata']
                )
                if user_filter(doc):
                    result = {
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'similarity_score': 1.0,  # No similarity score for direct retrieval
                        'user_id': doc.metadata.get('user_id'),
                        'memory_type': doc.metadata.get('memory_type'),
                        'timestamp': doc.metadata.get('timestamp')
                    }
                    all_docs.append(result)
            
            logger.info(f"Retrieved {len(all_docs)} memories for user {user_id}")
            return all_docs
            
        except Exception as e:
            logger.error(f"Failed to get all memories for user {user_id}: {str(e)}")
            return []
    
    def clear_memories(self, user_id: Optional[str] = None) -> bool:
        """
        Clear memories from the vector store.
        
        Args:
            user_id: Optional user ID to clear memories for specific user.
                    If None, clears all memories.
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if user_id is None:
                # Clear all memories by reinitializing the vector store
                self.vector_store = InMemoryVectorStore(self.embeddings)
                logger.info("All memories cleared successfully")
            else:
                # For user-specific clearing, we would need to implement
                # a more sophisticated approach since InMemoryVectorStore
                # doesn't support selective deletion by metadata
                logger.warning(f"User-specific memory clearing not fully implemented for user {user_id}")
                # This is a limitation of the current InMemoryVectorStore implementation
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear memories: {str(e)}")
            return False
    
    def get_memory_count(self, user_id: Optional[str] = None) -> int:
        """
        Get the count of stored memories.
        
        Args:
            user_id: Optional user ID to count memories for specific user
            
        Returns:
            Number of stored memories
        """
        try:
            if user_id:
                memories = self.get_all_memories(user_id)
                return len(memories)
            else:
                # For total count, we use the vector store's internal storage
                # This is an approximation since we can't directly access the count
                return len(self.vector_store.store)
        except Exception as e:
            logger.error(f"Failed to get memory count: {str(e)}")
            return 0
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the vector persistence system.
        
        Returns:
            Dictionary with health status information
        """
        try:
            # Test embedding generation
            test_embedding = self.embeddings.embed_query("test")
            
            # Test vector store operations
            test_doc = Document(
                page_content="health check test",
                metadata={"test": True, "user_id": "health_check"}
            )
            
            # Add and immediately search for the test document
            doc_ids = self.vector_store.add_documents([test_doc])
            search_results = self.vector_store.similarity_search("health check test", k=1)
            
            return {
                'status': 'healthy',
                'embedding_model': self.config.embedding_model,
                'embedding_dimension': len(test_embedding),
                'vector_store_type': type(self.vector_store).__name__,
                'test_document_added': len(doc_ids) > 0,
                'test_search_successful': len(search_results) > 0,
                'total_documents': len(self.vector_store.store)
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'embedding_model': self.config.embedding_model,
                'vector_store_type': type(self.vector_store).__name__
            }