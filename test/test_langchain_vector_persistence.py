"""Tests for LangChain vector persistence implementation."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.config.config_manager import ConfigManager
from src.persistence.langchain_vector_persistence import LangChainVectorPersistence
from src.persistence.memory_document import MemoryDocument, MemoryType


class TestLangChainVectorPersistence:
    """Test cases for LangChain vector persistence."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock configuration manager."""
        config_manager = Mock(spec=ConfigManager)
        
        # Mock config object
        mock_config = Mock()
        mock_config.embedding_model = "text-embedding-3-small"
        mock_config.llm_model = "gpt-3.5-turbo"
        mock_config.openai_api_key = "test-key"
        mock_config.openai_base_url = "https://api.openai.com/v1"
        
        config_manager.get_config.return_value = mock_config
        config_manager.get_openai_client_config.return_value = {
            'api_key': 'test-key',
            'base_url': 'https://api.openai.com/v1'
        }
        
        return config_manager
    
    @patch('src.persistence.langchain_vector_persistence.ChatOpenAI')
    @patch('src.persistence.langchain_vector_persistence.OpenAIEmbeddings')
    @patch('src.persistence.langchain_vector_persistence.InMemoryVectorStore')
    def test_initialization(self, mock_vector_store, mock_embeddings, mock_chat_openai, mock_config_manager):
        """Test vector persistence initialization."""
        # Create instance
        persistence = LangChainVectorPersistence(mock_config_manager)
        
        # Verify initialization
        assert persistence.config_manager == mock_config_manager
        mock_embeddings.assert_called_once()
        mock_vector_store.assert_called_once()
    
    @patch('src.persistence.langchain_vector_persistence.ChatOpenAI')
    @patch('src.persistence.langchain_vector_persistence.OpenAIEmbeddings')
    @patch('src.persistence.langchain_vector_persistence.InMemoryVectorStore')
    def test_save_memory_success(self, mock_vector_store_class, mock_embeddings, mock_chat_openai, mock_config_manager):
        """Test successful memory saving."""
        # Setup mocks
        mock_vector_store = Mock()
        mock_vector_store.add_documents.return_value = ['doc_123']
        mock_vector_store_class.return_value = mock_vector_store
        
        # Create instance
        persistence = LangChainVectorPersistence(mock_config_manager)
        
        # Test save memory
        doc_id = persistence.save_memory(
            content="Test memory content",
            user_id="user123",
            memory_type=MemoryType.FACT,
            metadata={"source": "test"}
        )
        
        # Verify results
        assert doc_id == 'doc_123'
        mock_vector_store.add_documents.assert_called_once()
        
        # Verify document structure
        call_args = mock_vector_store.add_documents.call_args[0][0]
        doc = call_args[0]
        assert doc.page_content == "Test memory content"
        assert doc.metadata['user_id'] == "user123"
        assert doc.metadata['memory_type'] == "fact"
        assert 'timestamp' in doc.metadata
        assert doc.metadata['source'] == "test"
    
    @patch('src.persistence.langchain_vector_persistence.ChatOpenAI')
    @patch('src.persistence.langchain_vector_persistence.OpenAIEmbeddings')
    @patch('src.persistence.langchain_vector_persistence.InMemoryVectorStore')
    def test_save_memory_validation_error(self, mock_vector_store_class, mock_embeddings, mock_chat_openai, mock_config_manager):
        """Test memory saving with validation errors."""
        persistence = LangChainVectorPersistence(mock_config_manager)
        
        # Test empty content
        with pytest.raises(ValueError, match="Memory content cannot be empty"):
            persistence.save_memory("", "user123")
        
        # Test empty user_id
        with pytest.raises(ValueError, match="User ID cannot be empty"):
            persistence.save_memory("content", "")
    
    @patch('src.persistence.langchain_vector_persistence.ChatOpenAI')
    @patch('src.persistence.langchain_vector_persistence.OpenAIEmbeddings')
    @patch('src.persistence.langchain_vector_persistence.InMemoryVectorStore')
    def test_search_memories_success(self, mock_vector_store_class, mock_embeddings, mock_chat_openai, mock_config_manager):
        """Test successful memory search."""
        # Setup mocks
        mock_vector_store = Mock()
        mock_doc = Mock()
        mock_doc.page_content = "Test memory content"
        mock_doc.metadata = {
            'user_id': 'user123',
            'memory_type': 'fact',
            'timestamp': '2024-01-01T12:00:00'
        }
        
        mock_vector_store.similarity_search_with_score.return_value = [
            (mock_doc, 0.95)
        ]
        mock_vector_store_class.return_value = mock_vector_store
        
        # Create instance
        persistence = LangChainVectorPersistence(mock_config_manager)
        
        # Test search
        results = persistence.search_memories(
            query="test query",
            user_id="user123",
            k=3,
            memory_type=MemoryType.FACT
        )
        
        # Verify results
        assert len(results) == 1
        result = results[0]
        assert result['content'] == "Test memory content"
        assert result['user_id'] == 'user123'
        assert result['memory_type'] == 'fact'
        assert result['similarity_score'] == 0.95
        
        # Verify search call - check that it was called with correct parameters
        call_args = mock_vector_store.similarity_search_with_score.call_args
        assert call_args[1]['query'] == "test query"
        assert call_args[1]['k'] == 3
        assert callable(call_args[1]['filter'])  # Filter should be a function
    
    @patch('src.persistence.langchain_vector_persistence.ChatOpenAI')
    @patch('src.persistence.langchain_vector_persistence.OpenAIEmbeddings')
    @patch('src.persistence.langchain_vector_persistence.InMemoryVectorStore')
    def test_search_memories_validation_error(self, mock_vector_store_class, mock_embeddings, mock_chat_openai, mock_config_manager):
        """Test memory search with validation errors."""
        persistence = LangChainVectorPersistence(mock_config_manager)
        
        # Test empty query
        with pytest.raises(ValueError, match="Search query cannot be empty"):
            persistence.search_memories("", "user123")
        
        # Test empty user_id
        with pytest.raises(ValueError, match="User ID cannot be empty"):
            persistence.search_memories("query", "")
    
    @patch('src.persistence.langchain_vector_persistence.ChatOpenAI')
    @patch('src.persistence.langchain_vector_persistence.OpenAIEmbeddings')
    @patch('src.persistence.langchain_vector_persistence.InMemoryVectorStore')
    def test_health_check_success(self, mock_vector_store_class, mock_embeddings_class, mock_chat_openai, mock_config_manager):
        """Test successful health check."""
        # Setup mocks
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embeddings_class.return_value = mock_embeddings
        
        mock_vector_store = Mock()
        mock_vector_store.add_documents.return_value = ['test_doc']
        mock_vector_store.similarity_search.return_value = [Mock()]
        mock_vector_store.store = {'doc1': 'data1', 'doc2': 'data2'}
        mock_vector_store_class.return_value = mock_vector_store
        
        # Create instance
        persistence = LangChainVectorPersistence(mock_config_manager)
        
        # Test health check
        health_status = persistence.health_check()
        
        # Verify results
        assert health_status['status'] == 'healthy'
        assert health_status['embedding_dimension'] == 3
        assert health_status['test_document_added'] is True
        assert health_status['test_search_successful'] is True
        assert health_status['total_documents'] == 2
    
    @patch('src.persistence.langchain_vector_persistence.ChatOpenAI')
    @patch('src.persistence.langchain_vector_persistence.OpenAIEmbeddings')
    @patch('src.persistence.langchain_vector_persistence.InMemoryVectorStore')
    def test_clear_memories(self, mock_vector_store_class, mock_embeddings, mock_chat_openai, mock_config_manager):
        """Test memory clearing functionality."""
        persistence = LangChainVectorPersistence(mock_config_manager)
        
        # Test clear all memories
        result = persistence.clear_memories()
        assert result is True
        
        # Test clear user-specific memories (should return False due to limitation)
        result = persistence.clear_memories(user_id="user123")
        assert result is False