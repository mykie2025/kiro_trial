"""Example usage of LangChain vector persistence system."""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config.config_manager import ConfigManager
from src.persistence.langchain_vector_persistence import LangChainVectorPersistence
from src.persistence.memory_document import MemoryType


def main():
    """Demonstrate LangChain vector persistence functionality."""
    try:
        # Initialize configuration and persistence
        print("Initializing LangChain vector persistence...")
        config_manager = ConfigManager()
        persistence = LangChainVectorPersistence(config_manager)
        
        # Perform health check
        print("\nPerforming health check...")
        health_status = persistence.health_check()
        print(f"Health status: {health_status['status']}")
        if health_status['status'] == 'healthy':
            print(f"Embedding model: {health_status['embedding_model']}")
            print(f"Embedding dimension: {health_status['embedding_dimension']}")
        else:
            print(f"Error: {health_status.get('error', 'Unknown error')}")
            return
        
        # Save some example memories
        print("\nSaving example memories...")
        user_id = "example_user"
        
        memories = [
            {
                'content': "User prefers dark mode for the interface",
                'memory_type': MemoryType.PREFERENCE,
                'metadata': {'category': 'ui_preference'}
            },
            {
                'content': "User is working on a Python machine learning project",
                'memory_type': MemoryType.CONTEXT,
                'metadata': {'category': 'project_context'}
            },
            {
                'content': "User mentioned they use PyTorch for deep learning",
                'memory_type': MemoryType.FACT,
                'metadata': {'category': 'technical_preference'}
            },
            {
                'content': "User asked about vector databases and embeddings",
                'memory_type': MemoryType.CONVERSATION,
                'metadata': {'category': 'conversation_history'}
            }
        ]
        
        saved_ids = []
        for memory in memories:
            doc_id = persistence.save_memory(
                content=memory['content'],
                user_id=user_id,
                memory_type=memory['memory_type'],
                metadata=memory['metadata']
            )
            saved_ids.append(doc_id)
            print(f"Saved memory: {memory['content'][:50]}... (ID: {doc_id})")
        
        # Search for memories
        print(f"\nSearching for memories related to 'Python'...")
        search_results = persistence.search_memories(
            query="Python programming",
            user_id=user_id,
            k=3
        )
        
        print(f"Found {len(search_results)} relevant memories:")
        for i, result in enumerate(search_results, 1):
            print(f"{i}. Content: {result['content']}")
            print(f"   Type: {result['memory_type']}")
            print(f"   Similarity: {result['similarity_score']:.3f}")
            print(f"   Metadata: {result['metadata']}")
            print()
        
        # Search for specific memory type
        print("Searching for preference memories...")
        preference_results = persistence.search_memories(
            query="interface settings",
            user_id=user_id,
            k=2,
            memory_type=MemoryType.PREFERENCE
        )
        
        print(f"Found {len(preference_results)} preference memories:")
        for result in preference_results:
            print(f"- {result['content']}")
        
        # Get memory count
        total_memories = persistence.get_memory_count(user_id)
        print(f"\nTotal memories for user: {total_memories}")
        
        # Get all memories for user
        all_memories = persistence.get_all_memories(user_id)
        print(f"Retrieved all memories: {len(all_memories)} items")
        
        print("\nExample completed successfully!")
        
    except Exception as e:
        print(f"Error running example: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()