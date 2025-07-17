"""LangChain-based vector persistence implementation for context storage."""

from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import logging

from langchain_community.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from ..config.config_manager import ConfigManager
from .memory_document import MemoryDocument, MemoryType


logger = logging.getLogger(__name__)


class InMemoryChatMessageHistory(BaseChatMessageHistory):
    """
    In-memory chat message history implementation for conversation persistence.
    
    This class stores conversation history in memory with user and session isolation.
    """
    
    def __init__(self, session_id: str, user_id: str):
        """
        Initialize chat message history.
        
        Args:
            session_id: Unique session identifier
            user_id: User identifier for isolation
        """
        self.session_id = session_id
        self.user_id = user_id
        self.messages: List[BaseMessage] = []
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
    
    @property
    def messages(self) -> List[BaseMessage]:
        """Get all messages in the conversation."""
        self.last_accessed = datetime.now()
        return self._messages
    
    @messages.setter
    def messages(self, value: List[BaseMessage]):
        """Set messages list."""
        self._messages = value
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the conversation history."""
        self._messages.append(message)
        self.last_accessed = datetime.now()
    
    def clear(self) -> None:
        """Clear all messages from the conversation history."""
        self._messages.clear()
        self.last_accessed = datetime.now()
    
    def get_message_count(self) -> int:
        """Get the number of messages in the conversation."""
        return len(self._messages)
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get session information."""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'message_count': self.get_message_count(),
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat()
        }


class ConversationHistoryManager:
    """
    Manages conversation histories across multiple users and sessions.
    
    This class provides session management and user context isolation
    for conversation history persistence.
    """
    
    def __init__(self):
        """Initialize the conversation history manager."""
        self._sessions: Dict[str, InMemoryChatMessageHistory] = {}
        self._user_sessions: Dict[str, List[str]] = {}
    
    def get_session_history(self, session_id: str, user_id: str) -> InMemoryChatMessageHistory:
        """
        Get or create a chat message history for a session.
        
        Args:
            session_id: Unique session identifier
            user_id: User identifier for isolation
            
        Returns:
            Chat message history instance
        """
        # Create composite key for user isolation
        composite_key = f"{user_id}:{session_id}"
        
        if composite_key not in self._sessions:
            # Create new session history
            history = InMemoryChatMessageHistory(session_id, user_id)
            self._sessions[composite_key] = history
            
            # Track user sessions
            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = []
            self._user_sessions[user_id].append(session_id)
            
            logger.info(f"Created new conversation session {session_id} for user {user_id}")
        
        return self._sessions[composite_key]
    
    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all sessions for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of session information dictionaries
        """
        if user_id not in self._user_sessions:
            return []
        
        sessions = []
        for session_id in self._user_sessions[user_id]:
            composite_key = f"{user_id}:{session_id}"
            if composite_key in self._sessions:
                session_info = self._sessions[composite_key].get_session_info()
                sessions.append(session_info)
        
        return sessions
    
    def clear_user_sessions(self, user_id: str) -> bool:
        """
        Clear all sessions for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if user_id not in self._user_sessions:
                return True
            
            # Clear all sessions for the user
            for session_id in self._user_sessions[user_id]:
                composite_key = f"{user_id}:{session_id}"
                if composite_key in self._sessions:
                    del self._sessions[composite_key]
            
            # Clear user session tracking
            del self._user_sessions[user_id]
            
            logger.info(f"Cleared all sessions for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear sessions for user {user_id}: {str(e)}")
            return False
    
    def clear_session(self, session_id: str, user_id: str) -> bool:
        """
        Clear a specific session.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            composite_key = f"{user_id}:{session_id}"
            
            if composite_key in self._sessions:
                del self._sessions[composite_key]
            
            # Remove from user session tracking
            if user_id in self._user_sessions:
                if session_id in self._user_sessions[user_id]:
                    self._user_sessions[user_id].remove(session_id)
                
                # Clean up empty user session list
                if not self._user_sessions[user_id]:
                    del self._user_sessions[user_id]
            
            logger.info(f"Cleared session {session_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear session {session_id} for user {user_id}: {str(e)}")
            return False
    
    def get_session_count(self, user_id: Optional[str] = None) -> int:
        """
        Get the number of active sessions.
        
        Args:
            user_id: Optional user ID to count sessions for specific user
            
        Returns:
            Number of active sessions
        """
        if user_id:
            return len(self._user_sessions.get(user_id, []))
        else:
            return len(self._sessions)


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
        
        # Initialize conversation history manager
        self.conversation_manager = ConversationHistoryManager()
        
        # Initialize ChatOpenAI for conversation chains
        self.llm = ChatOpenAI(
            model=self.config.llm_model,
            openai_api_key=openai_config['api_key'],
            openai_api_base=openai_config['base_url']
        )
        
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
    
    def get_conversation_chain(
        self, 
        session_id: str, 
        user_id: str,
        system_prompt: Optional[str] = None,
        include_memory_context: bool = True
    ) -> RunnableWithMessageHistory:
        """
        Create a conversation chain with memory persistence across user sessions.
        
        Args:
            session_id: Unique session identifier
            user_id: User identifier for isolation
            system_prompt: Optional system prompt for the conversation
            include_memory_context: Whether to include relevant memories in context
            
        Returns:
            RunnableWithMessageHistory instance for conversation management
            
        Raises:
            ValueError: If session_id or user_id is empty
        """
        try:
            if not session_id or not session_id.strip():
                raise ValueError("Session ID cannot be empty")
            
            if not user_id or not user_id.strip():
                raise ValueError("User ID cannot be empty")
            
            # Create session history getter function
            def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
                return self.conversation_manager.get_session_history(session_id, user_id)
            
            # Create base prompt template
            if system_prompt is None:
                system_prompt = "You are a helpful AI assistant with access to conversation history and relevant memories."
            
            # Build prompt template with memory context if enabled
            if include_memory_context:
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt + "\n\nRelevant memories from previous conversations:\n{memory_context}"),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}")
                ])
            else:
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}")
                ])
            
            # Create the base chain
            base_chain = prompt_template | self.llm
            
            # Create memory-enhanced chain if memory context is enabled
            if include_memory_context:
                def memory_enhanced_chain(inputs: Dict[str, Any]) -> Dict[str, Any]:
                    """Enhanced chain that includes memory context."""
                    # Get relevant memories for the input
                    input_text = inputs.get("input", "")
                    if input_text:
                        try:
                            memories = self.search_memories(
                                query=input_text,
                                user_id=user_id,
                                k=3
                            )
                            memory_context = "\n".join([
                                f"- {memory['content']} (from {memory['timestamp']})"
                                for memory in memories
                            ])
                        except Exception as e:
                            logger.warning(f"Failed to retrieve memory context: {str(e)}")
                            memory_context = "No relevant memories found."
                    else:
                        memory_context = "No input provided for memory search."
                    
                    # Add memory context to inputs
                    enhanced_inputs = inputs.copy()
                    enhanced_inputs["memory_context"] = memory_context
                    
                    return base_chain.invoke(enhanced_inputs)
                
                # Wrap the enhanced chain
                chain = memory_enhanced_chain
            else:
                chain = base_chain
            
            # Create RunnableWithMessageHistory
            conversation_chain = RunnableWithMessageHistory(
                chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="history"
            )
            
            logger.info(f"Created conversation chain for session {session_id}, user {user_id}")
            return conversation_chain
            
        except Exception as e:
            logger.error(f"Failed to create conversation chain for session {session_id}, user {user_id}: {str(e)}")
            raise
    
    def save_conversation_memory(
        self, 
        session_id: str, 
        user_id: str, 
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save conversation content as a memory for future retrieval.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            content: Conversation content to save as memory
            metadata: Additional metadata
            
        Returns:
            Document ID of the saved memory
        """
        try:
            # Prepare metadata with session information
            memory_metadata = metadata or {}
            memory_metadata.update({
                'session_id': session_id,
                'source': 'conversation'
            })
            
            # Save as conversation memory
            return self.save_memory(
                content=content,
                user_id=user_id,
                memory_type=MemoryType.CONVERSATION,
                metadata=memory_metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to save conversation memory for session {session_id}, user {user_id}: {str(e)}")
            raise
    
    def get_session_info(self, session_id: str, user_id: str) -> Dict[str, Any]:
        """
        Get information about a conversation session.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            
        Returns:
            Dictionary with session information
        """
        try:
            history = self.conversation_manager.get_session_history(session_id, user_id)
            return history.get_session_info()
            
        except Exception as e:
            logger.error(f"Failed to get session info for session {session_id}, user {user_id}: {str(e)}")
            return {}
    
    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all conversation sessions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of session information dictionaries
        """
        try:
            return self.conversation_manager.get_user_sessions(user_id)
            
        except Exception as e:
            logger.error(f"Failed to get user sessions for user {user_id}: {str(e)}")
            return []
    
    def clear_conversation_history(
        self, 
        user_id: str, 
        session_id: Optional[str] = None
    ) -> bool:
        """
        Clear conversation history for a user or specific session.
        
        Args:
            user_id: User identifier
            session_id: Optional session ID to clear specific session
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if session_id:
                return self.conversation_manager.clear_session(session_id, user_id)
            else:
                return self.conversation_manager.clear_user_sessions(user_id)
                
        except Exception as e:
            logger.error(f"Failed to clear conversation history for user {user_id}: {str(e)}")
            return False
    
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
            
            # Test conversation chain creation
            try:
                test_chain = self.get_conversation_chain("test_session", "health_check")
                conversation_chain_healthy = True
            except Exception:
                conversation_chain_healthy = False
            
            return {
                'status': 'healthy',
                'embedding_model': self.config.embedding_model,
                'embedding_dimension': len(test_embedding),
                'vector_store_type': type(self.vector_store).__name__,
                'test_document_added': len(doc_ids) > 0,
                'test_search_successful': len(search_results) > 0,
                'conversation_chain_healthy': conversation_chain_healthy,
                'total_documents': len(self.vector_store.store),
                'active_sessions': self.conversation_manager.get_session_count()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'embedding_model': self.config.embedding_model,
                'vector_store_type': type(self.vector_store).__name__
            }