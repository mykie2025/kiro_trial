"""Example usage of Neo4j graph persistence system."""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config.config_manager import ConfigManager
from src.persistence.neo4j_docker_manager import Neo4jDockerManager
from src.persistence.neo4j_graph_persistence import Neo4jGraphPersistence
from src.persistence.memory_document import MemoryDocument, MemoryType


def main():
    """Demonstrate Neo4j graph persistence functionality."""
    try:
        # Initialize configuration
        print("🚀 Neo4j Graph Persistence System Demo")
        print("=" * 50)
        
        config_manager = ConfigManager()
        
        # Initialize Docker manager
        print("\n🐳 Initializing Neo4j Docker Manager...")
        docker_manager = Neo4jDockerManager(config_manager)
        
        # Check if container is already running
        print("\n📊 Checking container status...")
        status = docker_manager.get_container_status()
        print(f"Container exists: {status['exists']}")
        print(f"Container status: {status['status']}")
        print(f"Container health: {status['health']}")
        
        # Start container if needed
        if not docker_manager.is_container_running():
            print("\n🚀 Starting Neo4j container...")
            print("This may take a few minutes for first-time setup...")
            
            if docker_manager.start_container():
                print("✅ Neo4j container started successfully!")
            else:
                print("❌ Failed to start Neo4j container")
                return
        else:
            print("✅ Neo4j container is already running")
        
        # Initialize graph persistence
        print("\n🕸️ Initializing Neo4j Graph Persistence...")
        graph_persistence = Neo4jGraphPersistence(config_manager)
        
        # Perform health check
        print("\n🔍 Performing health check...")
        health = graph_persistence.health_check()
        print(f"Database status: {health['status']}")
        
        if health['status'] != 'healthy':
            print(f"❌ Database not healthy: {health.get('error', 'Unknown error')}")
            return
        
        print(f"✅ Database is healthy!")
        print(f"   • Database: {health['database']}")
        print(f"   • Nodes: {health['node_count']}")
        print(f"   • Relationships: {health['relationship_count']}")
        
        # Demonstrate graph operations
        print("\n📝 Demonstrating Graph Operations")
        print("-" * 40)
        
        user_id = "demo_user"
        
        # Clear any existing demo data
        print("🧹 Clearing existing demo data...")
        graph_persistence.clear_graph(user_id=user_id)
        
        # 1. Create entity nodes
        print("\n1. Creating entity nodes...")
        
        entities_to_create = [
            {"name": "Python", "type": "Programming Language", "properties": {"paradigm": "multi-paradigm"}},
            {"name": "Machine Learning", "type": "Field", "properties": {"complexity": "high"}},
            {"name": "Neural Networks", "type": "Concept", "properties": {"category": "deep learning"}},
            {"name": "TensorFlow", "type": "Framework", "properties": {"language": "Python"}},
            {"name": "Data Science", "type": "Field", "properties": {"tools": "Python, R"}},
        ]
        
        created_nodes = {}
        for entity_info in entities_to_create:
            node_id = graph_persistence.create_entity_node(
                entity=entity_info["name"],
                entity_type=entity_info["type"],
                properties=entity_info["properties"],
                user_id=user_id
            )
            created_nodes[entity_info["name"]] = node_id
            print(f"   ✅ Created: {entity_info['name']} ({entity_info['type']})")
        
        # 2. Create relationships
        print("\n2. Creating relationships...")
        
        relationships_to_create = [
            {"from": "Python", "to": "Machine Learning", "type": "USED_IN"},
            {"from": "Python", "to": "Data Science", "type": "USED_IN"},
            {"from": "Machine Learning", "to": "Neural Networks", "type": "INCLUDES"},
            {"from": "TensorFlow", "to": "Neural Networks", "type": "IMPLEMENTS"},
            {"from": "TensorFlow", "to": "Python", "type": "BUILT_WITH"},
            {"from": "Data Science", "to": "Machine Learning", "type": "APPLIES"},
        ]
        
        for rel_info in relationships_to_create:
            success = graph_persistence.create_relationship(
                from_entity=rel_info["from"],
                to_entity=rel_info["to"],
                relationship_type=rel_info["type"],
                user_id=user_id
            )
            if success:
                print(f"   ✅ {rel_info['from']} -[{rel_info['type']}]-> {rel_info['to']}")
            else:
                print(f"   ❌ Failed to create relationship: {rel_info['from']} -> {rel_info['to']}")
        
        # 3. Store memory as graph
        print("\n3. Storing memory documents as graph...")
        
        memories = [
            MemoryDocument(
                content="Python is excellent for Machine Learning because of libraries like TensorFlow and scikit-learn",
                user_id=user_id,
                memory_type=MemoryType.FACT
            ),
            MemoryDocument(
                content="Neural Networks are a subset of Machine Learning inspired by biological neurons",
                user_id=user_id,
                memory_type=MemoryType.CONTEXT
            )
        ]
        
        for memory in memories:
            result = graph_persistence.store_memory_as_graph(memory)
            print(f"   ✅ Stored memory: {memory.content[:50]}...")
            print(f"      • Memory node: {result['memory_name']}")
            print(f"      • Extracted entities: {result['entity_count']}")
        
        # 4. Query context using graph traversal
        print("\n4. Querying context using graph traversal...")
        
        query_entities = ["Python", "Machine Learning"]
        
        for entity in query_entities:
            print(f"\n   🔍 Querying context for: {entity}")
            contexts = graph_persistence.query_context(
                entity=entity,
                user_id=user_id,
                max_depth=2
            )
            
            print(f"   Found {len(contexts)} connected entities:")
            for context in contexts[:5]:  # Show first 5
                print(f"      • {context['connected_entity']} ({context['entity_type']}) - distance: {context['path_length']}")
        
        # 5. Get directly connected entities
        print("\n5. Getting directly connected entities...")
        
        connections = graph_persistence.get_connected_entities(
            entity="Python",
            user_id=user_id,
            direction="outgoing"
        )
        
        print(f"   Python is directly connected to {len(connections)} entities:")
        for conn in connections:
            print(f"      • {conn['connected_entity']} via {conn['relationship_type']}")
        
        # 6. Search entities
        print("\n6. Searching entities...")
        
        search_results = graph_persistence.search_entities(
            search_term="Machine",
            user_id=user_id,
            limit=5
        )
        
        print(f"   Found {len(search_results)} entities matching 'Machine':")
        for result in search_results:
            print(f"      • {result['entity_name']} ({result['entity_type']})")
        
        # 7. Get graph statistics
        print("\n7. Graph statistics...")
        
        stats = graph_persistence.get_graph_statistics(user_id=user_id)
        print(f"   📊 Statistics for user '{user_id}':")
        print(f"      • Total nodes: {stats['node_count']}")
        print(f"      • Total relationships: {stats['relationship_count']}")
        print(f"      • Entity types:")
        for entity_type in stats['entity_types'][:5]:
            print(f"        - {entity_type['type']}: {entity_type['count']}")
        
        print("\n🎉 Neo4j Graph Persistence Demo Completed Successfully!")
        print("\n💡 Key Features Demonstrated:")
        print("   • Docker container management with health checks")
        print("   • Entity node creation with properties and metadata")
        print("   • Relationship creation between entities")
        print("   • Memory document storage as graph structures")
        print("   • Context querying with graph traversal")
        print("   • Entity search and filtering")
        print("   • Graph statistics and analytics")
        print("\n🔧 Next Steps:")
        print("   • Try modifying the entities and relationships")
        print("   • Experiment with different query patterns")
        print("   • Compare with LangChain vector similarity results")
        print("   • Use the evaluation framework to compare approaches")
        
        # Clean up
        print(f"\n🧹 Cleaning up...")
        graph_persistence.close()
        print("   ✅ Graph connection closed")
        
        # Note: We don't stop the container as it might be used by other processes
        print("   ℹ️  Neo4j container left running for potential reuse")
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 