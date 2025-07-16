"""Memory document data structures for context persistence."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum


class MemoryType(Enum):
    """Enumeration of memory types for classification."""
    CONVERSATION = "conversation"
    FACT = "fact"
    PREFERENCE = "preference"
    CONTEXT = "context"
    EVENT = "event"


@dataclass
class MemoryDocument:
    """
    Data structure for storing memory documents with metadata.
    
    This class represents a single memory entry that can be stored
    in the vector database with associated metadata for filtering
    and retrieval.
    """
    content: str
    user_id: str
    memory_type: MemoryType = MemoryType.CONVERSATION
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Validate memory document after initialization."""
        if not self.content or not self.content.strip():
            raise ValueError("Memory content cannot be empty")
        
        if not self.user_id or not self.user_id.strip():
            raise ValueError("User ID cannot be empty")
        
        if not isinstance(self.memory_type, MemoryType):
            if isinstance(self.memory_type, str):
                try:
                    self.memory_type = MemoryType(self.memory_type)
                except ValueError:
                    raise ValueError(f"Invalid memory type: {self.memory_type}")
            else:
                raise ValueError("Memory type must be a MemoryType enum or valid string")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory document to dictionary format."""
        return {
            'content': self.content,
            'user_id': self.user_id,
            'memory_type': self.memory_type.value,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'embedding': self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryDocument':
        """Create memory document from dictionary."""
        # Parse timestamp
        timestamp = data.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()
        
        # Parse memory type
        memory_type = data.get('memory_type', MemoryType.CONVERSATION)
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)
        
        return cls(
            content=data['content'],
            user_id=data['user_id'],
            memory_type=memory_type,
            timestamp=timestamp,
            metadata=data.get('metadata', {}),
            embedding=data.get('embedding')
        )
    
    def get_langchain_metadata(self) -> Dict[str, Any]:
        """Get metadata formatted for LangChain vector store."""
        return {
            'user_id': self.user_id,
            'memory_type': self.memory_type.value,
            'timestamp': self.timestamp.isoformat(),
            **self.metadata
        }