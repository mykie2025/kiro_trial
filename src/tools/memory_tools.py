"""Memory storage and retrieval tools for context persistence."""

from typing import List, Dict, Any, Optional
import logging

from langchain_core.runnables.history import RunnableWithMessageHistory

from ..persistence.langchain_vector_persistence import LangChainVectorPersistence
from ..persistence.memory_document import MemoryType
from ..config.config_manager import ConfigManager


logger = logging.getLogger(__name__)


class MemoryTools:
    """
    Tool functions for memory storage and retrieval operations.
    
    This class provides high-level tool functions that wrap the persistence
    layer for easy integration with external systems.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize memory tools with persistence backend.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.persistence = LangChainVectorPersistence(config_manager)
        logger.info("Memory tools initialized successfully")
    
    def save_recall_memory(
        self,
        content: str,
        user_id: str,
        memory_type: str = "conversation",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Save a memory with user ID filtering and structured metadata.
        
        Args:
            content: The memory content to store
            user_id: User identifier for filtering
            memory_type: Type of memory (conversation, fact, preference, context, event)
            metadata: Additional structured metadata to store with the memory
            
        Returns:
            Dictionary with operation result and memory ID
            
        Raises:
            ValueError: If content or user_id is empty or memory_type is invalid
            Exception: If storage operation fails
        """
        try:
            # Validate inputs
            if not content or not content.strip():
                raise ValueError("Memory content cannot be empty")
            
            if not user_id or not user_id.strip():
                raise ValueError("User ID cannot be empty")
            
            # Convert string memory type to enum
            try:
                memory_type_enum = MemoryType(memory_type.lower())
            except ValueError:
                raise ValueError(f"Invalid memory type: {memory_type}. Valid types: {[t.value for t in MemoryType]}")
            
            # Ensure metadata is a dictionary
            if metadata is None:
                metadata = {}
            elif not isinstance(metadata, dict):
                raise ValueError("Metadata must be a dictionary")
            
            # Add timestamp tracking to metadata
            metadata['tool_source'] = 'save_recall_memory'
            
            # Save memory using persistence layer
            memory_id = self.persistence.save_memory(
                content=content,
                user_id=user_id,
                memory_type=memory_type_enum,
                metadata=metadata
            )
            
            result = {
                'success': True,
                'memory_id': memory_id,
                'user_id': user_id,
                'memory_type': memory_type,
                'content_length': len(content),
                'metadata_keys': list(metadata.keys())
            }
            
            logger.info(f"Memory saved successfully: {memory_id} for user {user_id}")
            return result
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'user_id': user_id,
                'memory_type': memory_type
            }
            logger.error(f"Failed to save memory for user {user_id}: {str(e)}")
            return error_result
    
    def search_recall_memories(
        self,
        query: str,
        user_id: str,
        memory_type: Optional[str] = None,
        max_results: int = 3
    ) -> Dict[str, Any]:
        """
        Search for memories using semantic similarity search.
        
        Args:
            query: Search query text for semantic similarity
            user_id: User identifier for filtering
            memory_type: Optional memory type filter
            max_results: Maximum number of results to return (default: 3)
            
        Returns:
            Dictionary with search results and metadata
            
        Raises:
            ValueError: If query or user_id is empty or memory_type is invalid
            Exception: If search operation fails
        """
        try:
            # Validate inputs
            if not query or not query.strip():
                raise ValueError("Search query cannot be empty")
            
            if not user_id or not user_id.strip():
                raise ValueError("User ID cannot be empty")
            
            if max_results <= 0:
                raise ValueError("max_results must be greater than 0")
            
            # Convert string memory type to enum if provided
            memory_type_enum = None
            if memory_type:
                try:
                    memory_type_enum = MemoryType(memory_type.lower())
                except ValueError:
                    raise ValueError(f"Invalid memory type: {memory_type}. Valid types: {[t.value for t in MemoryType]}")
            
            # Search memories using persistence layer
            search_results = self.persistence.search_memories(
                query=query,
                user_id=user_id,
                k=max_results,
                memory_type=memory_type_enum
            )
            
            # Format results for tool output
            formatted_memories = []
            for result in search_results:
                formatted_memory = {
                    'content': result['content'],
                    'similarity_score': result['similarity_score'],
                    'memory_type': result['memory_type'],
                    'timestamp': result['timestamp'],
                    'metadata': result['metadata']
                }
                formatted_memories.append(formatted_memory)
            
            result = {
                'success': True,
                'query': query,
                'user_id': user_id,
                'memory_type_filter': memory_type,
                'results_count': len(formatted_memories),
                'max_results': max_results,
                'memories': formatted_memories
            }
            
            logger.info(f"Memory search completed: {len(formatted_memories)} results for user {user_id}")
            return result
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'query': query,
                'user_id': user_id,
                'memory_type_filter': memory_type,
                'results_count': 0,
                'memories': []
            }
            logger.error(f"Failed to search memories for user {user_id}: {str(e)}")
            return error_result
    
    def get_conversation_chain(
        self,
        session_id: str,
        user_id: str,
        system_prompt: Optional[str] = None,
        include_memory_context: bool = True
    ) -> Dict[str, Any]:
        """
        Create a conversation chain with memory persistence across user sessions.
        
        Args:
            session_id: Unique session identifier
            user_id: User identifier for isolation
            system_prompt: Optional system prompt for the conversation
            include_memory_context: Whether to include relevant memories in context
            
        Returns:
            Dictionary with operation result and chain information
        """
        try:
            # Validate inputs
            if not session_id or not session_id.strip():
                raise ValueError("Session ID cannot be empty")
            
            if not user_id or not user_id.strip():
                raise ValueError("User ID cannot be empty")
            
            # Create conversation chain using persistence layer
            chain = self.persistence.get_conversation_chain(
                session_id=session_id,
                user_id=user_id,
                system_prompt=system_prompt,
                include_memory_context=include_memory_context
            )
            
            # Get session info
            session_info = self.persistence.get_session_info(session_id, user_id)
            
            result = {
                'success': True,
                'session_id': session_id,
                'user_id': user_id,
                'chain_created': True,
                'include_memory_context': include_memory_context,
                'session_info': session_info,
                'chain': chain  # Return the actual chain for use
            }
            
            logger.info(f"Conversation chain created for session {session_id}, user {user_id}")
            return result
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'session_id': session_id,
                'user_id': user_id,
                'chain_created': False,
                'chain': None
            }
            logger.error(f"Failed to create conversation chain for session {session_id}, user {user_id}: {str(e)}")
            return error_result
    
    def save_conversation_memory(
        self,
        session_id: str,
        user_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Save conversation content as a memory for future retrieval.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            content: Conversation content to save as memory
            metadata: Additional metadata
            
        Returns:
            Dictionary with operation result and memory ID
        """
        try:
            # Validate inputs
            if not session_id or not session_id.strip():
                raise ValueError("Session ID cannot be empty")
            
            if not user_id or not user_id.strip():
                raise ValueError("User ID cannot be empty")
            
            if not content or not content.strip():
                raise ValueError("Content cannot be empty")
            
            # Save conversation memory using persistence layer
            memory_id = self.persistence.save_conversation_memory(
                session_id=session_id,
                user_id=user_id,
                content=content,
                metadata=metadata
            )
            
            result = {
                'success': True,
                'memory_id': memory_id,
                'session_id': session_id,
                'user_id': user_id,
                'content_length': len(content),
                'metadata_keys': list((metadata or {}).keys())
            }
            
            logger.info(f"Conversation memory saved: {memory_id} for session {session_id}, user {user_id}")
            return result
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'session_id': session_id,
                'user_id': user_id
            }
            logger.error(f"Failed to save conversation memory for session {session_id}, user {user_id}: {str(e)}")
            return error_result
    
    def get_user_sessions(self, user_id: str) -> Dict[str, Any]:
        """
        Get all conversation sessions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with user sessions information
        """
        try:
            if not user_id or not user_id.strip():
                raise ValueError("User ID cannot be empty")
            
            # Get user sessions from persistence layer
            sessions = self.persistence.get_user_sessions(user_id)
            
            result = {
                'success': True,
                'user_id': user_id,
                'session_count': len(sessions),
                'sessions': sessions
            }
            
            logger.info(f"Retrieved {len(sessions)} sessions for user {user_id}")
            return result
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'user_id': user_id,
                'session_count': 0,
                'sessions': []
            }
            logger.error(f"Failed to get user sessions for user {user_id}: {str(e)}")
            return error_result
    
    def clear_conversation_history(
        self,
        user_id: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Clear conversation history for a user or specific session.
        
        Args:
            user_id: User identifier
            session_id: Optional session ID to clear specific session
            
        Returns:
            Dictionary with operation result
        """
        try:
            if not user_id or not user_id.strip():
                raise ValueError("User ID cannot be empty")
            
            # Clear conversation history using persistence layer
            success = self.persistence.clear_conversation_history(user_id, session_id)
            
            if session_id:
                action = f"session {session_id}"
            else:
                action = "all sessions"
            
            result = {
                'success': success,
                'user_id': user_id,
                'session_id': session_id,
                'action': f"cleared {action}",
                'cleared': success
            }
            
            if success:
                logger.info(f"Cleared conversation history for user {user_id}, {action}")
            else:
                logger.warning(f"Failed to clear conversation history for user {user_id}, {action}")
            
            return result
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'user_id': user_id,
                'session_id': session_id,
                'cleared': False
            }
            logger.error(f"Failed to clear conversation history for user {user_id}: {str(e)}")
            return error_result


# Convenience functions for direct tool usage
def save_recall_memory(
    content: str,
    user_id: str,
    memory_type: str = "conversation",
    metadata: Optional[Dict[str, Any]] = None,
    config_manager: Optional[ConfigManager] = None
) -> Dict[str, Any]:
    """
    Convenience function to save a memory.
    
    Args:
        content: The memory content to store
        user_id: User identifier for filtering
        memory_type: Type of memory (conversation, fact, preference, context, event)
        metadata: Additional structured metadata to store with the memory
        config_manager: Optional config manager (will create default if not provided)
        
    Returns:
        Dictionary with operation result and memory ID
    """
    if config_manager is None:
        config_manager = ConfigManager()
    
    tools = MemoryTools(config_manager)
    return tools.save_recall_memory(content, user_id, memory_type, metadata)


def search_recall_memories(
    query: str,
    user_id: str,
    memory_type: Optional[str] = None,
    max_results: int = 3,
    config_manager: Optional[ConfigManager] = None
) -> Dict[str, Any]:
    """
    Convenience function to search for memories.
    
    Args:
        query: Search query text for semantic similarity
        user_id: User identifier for filtering
        memory_type: Optional memory type filter
        max_results: Maximum number of results to return (default: 3)
        config_manager: Optional config manager (will create default if not provided)
        
    Returns:
        Dictionary with search results and metadata
    """
    if config_manager is None:
        config_manager = ConfigManager()
    
    tools = MemoryTools(config_manager)
    return tools.search_recall_memories(query, user_id, memory_type, max_results)


def get_conversation_chain(
    session_id: str,
    user_id: str,
    system_prompt: Optional[str] = None,
    include_memory_context: bool = True,
    config_manager: Optional[ConfigManager] = None
) -> Dict[str, Any]:
    """
    Convenience function to create a conversation chain.
    
    Args:
        session_id: Unique session identifier
        user_id: User identifier for isolation
        system_prompt: Optional system prompt for the conversation
        include_memory_context: Whether to include relevant memories in context
        config_manager: Optional config manager (will create default if not provided)
        
    Returns:
        Dictionary with operation result and chain information
    """
    if config_manager is None:
        config_manager = ConfigManager()
    
    tools = MemoryTools(config_manager)
    return tools.get_conversation_chain(session_id, user_id, system_prompt, include_memory_context)


def save_conversation_memory(
    session_id: str,
    user_id: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    config_manager: Optional[ConfigManager] = None
) -> Dict[str, Any]:
    """
    Convenience function to save conversation memory.
    
    Args:
        session_id: Session identifier
        user_id: User identifier
        content: Conversation content to save as memory
        metadata: Additional metadata
        config_manager: Optional config manager (will create default if not provided)
        
    Returns:
        Dictionary with operation result and memory ID
    """
    if config_manager is None:
        config_manager = ConfigManager()
    
    tools = MemoryTools(config_manager)
    return tools.save_conversation_memory(session_id, user_id, content, metadata)


def get_user_sessions(
    user_id: str,
    config_manager: Optional[ConfigManager] = None
) -> Dict[str, Any]:
    """
    Convenience function to get user sessions.
    
    Args:
        user_id: User identifier
        config_manager: Optional config manager (will create default if not provided)
        
    Returns:
        Dictionary with user sessions information
    """
    if config_manager is None:
        config_manager = ConfigManager()
    
    tools = MemoryTools(config_manager)
    return tools.get_user_sessions(user_id)


def clear_conversation_history(
    user_id: str,
    session_id: Optional[str] = None,
    config_manager: Optional[ConfigManager] = None
) -> Dict[str, Any]:
    """
    Convenience function to clear conversation history.
    
    Args:
        user_id: User identifier
        session_id: Optional session ID to clear specific session
        config_manager: Optional config manager (will create default if not provided)
        
    Returns:
        Dictionary with operation result
    """
    if config_manager is None:
        config_manager = ConfigManager()
    
    tools = MemoryTools(config_manager)
    return tools.clear_conversation_history(user_id, session_id)