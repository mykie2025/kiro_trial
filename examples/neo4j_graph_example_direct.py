"""Neo4j Graph Persistence Example - Direct Connection (Bypasses Docker Manager)."""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config.config_manager import ConfigManager
from src.persistence.neo4j_graph_persistence import Neo4jGraphPersistence
from src.persistence.memory_document import MemoryDocument, MemoryType


def main():
    """Demonstrate Neo4j graph persistence functionality with direct connection."""
    try:
        # Initialize configuration
        print("🚀 Neo4j Graph Persistence System Demo - Direct Connection")
        print("=" * 65)
        print("📝 Connecting directly to running Neo4j instance (bypasses Docker manager)")
        print()
        
        config_manager = ConfigManager()
        
        # Initialize graph persistence directly (skip Docker manager)
        print("🕸️ Initializing Neo4j Graph Persistence...")
        graph_persistence = Neo4jGraphPersistence(config_manager)
        
        # Perform health check
        print("\n🔍 Performing health check...")
        health = graph_persistence.health_check()
        print(f"Database status: {health['status']}")
        
        if health['status'] != 'healthy':
            print(f"❌ Database not healthy: {health.get('error', 'Unknown error')}")
            print("\n💡 Make sure Neo4j is running with:")
            print("   docker run --name neo4j-context-persistence -p 7474:7474 -p 7687:7687 \\")
            print("              -e NEO4J_AUTH=neo4j/password -d neo4j:latest")
            return
        
        print(f"✅ Database is healthy!")
        print(f"   • Database: {health['database']}")
        print(f"   • Nodes: {health['node_count']}")
        print(f"   • Relationships: {health['relationship_count']}")
        
        # Demonstrate graph operations
        print("\n📝 Demonstrating Full Neo4j Graph Operations")
        print("-" * 50)
        
        user_id = "demo_user"
        
        # Clear any existing demo data
        print("🧹 Clearing existing demo data...")
        graph_persistence.clear_graph(user_id=user_id)
        
        # 1. Create entity nodes
        print("\n1. 📊 Creating Entity Nodes")
        print("-" * 30)
        
        entities_to_create = [
            {"name": "Python", "type": "Programming Language", "properties": {"paradigm": "multi-paradigm", "year": 1991}},
            {"name": "Machine Learning", "type": "Field", "properties": {"complexity": "high", "applications": "many"}},
            {"name": "Neural Networks", "type": "Concept", "properties": {"category": "deep learning", "inspired_by": "biology"}},
            {"name": "TensorFlow", "type": "Framework", "properties": {"language": "Python", "company": "Google"}},
            {"name": "Data Science", "type": "Field", "properties": {"tools": "Python, R", "domains": "business, research"}},
            {"name": "scikit-learn", "type": "Library", "properties": {"language": "Python", "focus": "traditional ML"}},
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
            print(f"   ✅ Created: {entity_info['name']} ({entity_info['type']}) -> ID: {node_id}")
        
        # 2. Create relationships
        print("\n2. 🔗 Creating Relationships")
        print("-" * 30)
        
        relationships_to_create = [
            {"from": "Python", "to": "Machine Learning", "type": "USED_IN", "properties": {"since": "1990s"}},
            {"from": "Python", "to": "Data Science", "type": "USED_IN", "properties": {"primary": True}},
            {"from": "Machine Learning", "to": "Neural Networks", "type": "INCLUDES", "properties": {"subset": True}},
            {"from": "TensorFlow", "to": "Neural Networks", "type": "IMPLEMENTS", "properties": {"level": "high"}},
            {"from": "TensorFlow", "to": "Python", "type": "BUILT_WITH", "properties": {"API": "primary"}},
            {"from": "Data Science", "to": "Machine Learning", "type": "APPLIES", "properties": {"extensively": True}},
            {"from": "scikit-learn", "to": "Machine Learning", "type": "IMPLEMENTS", "properties": {"traditional": True}},
            {"from": "scikit-learn", "to": "Python", "type": "BUILT_WITH", "properties": {"pure_python": True}},
        ]
        
        for rel_info in relationships_to_create:
            success = graph_persistence.create_relationship(
                from_entity=rel_info["from"],
                to_entity=rel_info["to"],
                relationship_type=rel_info["type"],
                user_id=user_id,
                properties=rel_info.get("properties", {})
            )
            if success:
                props_str = f" {rel_info.get('properties', {})}" if rel_info.get('properties') else ""
                print(f"   ✅ {rel_info['from']} -[{rel_info['type']}]-> {rel_info['to']}{props_str}")
            else:
                print(f"   ❌ Failed to create relationship: {rel_info['from']} -> {rel_info['to']}")
        
        # 3. Store memory as graph
        print("\n3. 🧠 Storing Memory Documents as Graph")
        print("-" * 40)
        
        memories = [
            MemoryDocument(
                content="Python is excellent for Machine Learning because of libraries like TensorFlow and scikit-learn that provide powerful algorithms",
                user_id=user_id,
                memory_type=MemoryType.FACT
            ),
            MemoryDocument(
                content="Neural Networks are a subset of Machine Learning inspired by biological neurons and can model complex patterns",
                user_id=user_id,
                memory_type=MemoryType.CONTEXT
            ),
            MemoryDocument(
                content="Data Science combines Python programming with statistical analysis for prediction models and business insights",
                user_id=user_id,
                memory_type=MemoryType.FACT
            ),
            MemoryDocument(
                content="TensorFlow provides both high-level APIs for beginners and low-level operations for research in deep learning",
                user_id=user_id,
                memory_type=MemoryType.CONTEXT
            )
        ]
        
        for memory in memories:
            result = graph_persistence.store_memory_as_graph(memory)
            print(f"   ✅ Stored: {memory.content[:60]}...")
            print(f"      • Memory node: {result['memory_name']}")
            print(f"      • Extracted entities: {result['entity_count']}")
            print(f"      • Created relationships: {len(result['created_relationships'])}")
        
        # 4. Query context using graph traversal
        print("\n4. 🔍 Querying Context using Graph Traversal")
        print("-" * 45)
        
        query_entities = ["Python", "Machine Learning", "TensorFlow"]
        
        for entity in query_entities:
            print(f"\n   🎯 Querying context for: {entity}")
            contexts = graph_persistence.query_context(
                entity=entity,
                user_id=user_id,
                max_depth=3
            )
            
            print(f"   Found {len(contexts)} connected entities:")
            for context in contexts[:8]:  # Show first 8
                print(f"      • {context['connected_entity']} ({context['entity_type']}) - distance: {context['path_length']}")
        
        # 5. Get directly connected entities
        print("\n5. 🔗 Getting Directly Connected Entities")
        print("-" * 40)
        
        test_entities = ["Python", "Machine Learning"]
        for entity in test_entities:
            print(f"\n   🎯 Direct connections for: {entity}")
            
            # Outgoing connections
            outgoing = graph_persistence.get_connected_entities(
                entity=entity,
                user_id=user_id,
                direction="outgoing"
            )
            print(f"   Outgoing ({len(outgoing)}):")
            for conn in outgoing[:5]:
                print(f"      • {entity} -[{conn['relationship_type']}]-> {conn['connected_entity']}")
            
            # Incoming connections
            incoming = graph_persistence.get_connected_entities(
                entity=entity,
                user_id=user_id,
                direction="incoming"
            )
            print(f"   Incoming ({len(incoming)}):")
            for conn in incoming[:5]:
                print(f"      • {conn['connected_entity']} -[{conn['relationship_type']}]-> {entity}")
        
        # 6. Search entities
        print("\n6. 🔎 Searching Entities")
        print("-" * 25)
        
        search_terms = ["Machine", "Python", "Network"]
        for term in search_terms:
            results = graph_persistence.search_entities(
                search_term=term,
                user_id=user_id,
                limit=5
            )
            print(f"   Found {len(results)} entities matching '{term}':")
            for result in results:
                print(f"      • {result['entity_name']} ({result['entity_type']})")
        
        # 7. Get graph statistics
        print("\n7. 📈 Final Graph Statistics")
        print("-" * 30)
        
        stats = graph_persistence.get_graph_statistics(user_id=user_id)
        print(f"   📊 Statistics for user '{user_id}':")
        print(f"      • Total nodes: {stats['node_count']}")
        print(f"      • Total relationships: {stats['relationship_count']}")
        print(f"      • Entity types:")
        for entity_type in stats['entity_types']:
            print(f"        - {entity_type['type']}: {entity_type['count']}")
        
        # 8. Query memories
        print("\n8. 🧠 Querying Stored Memories")
        print("-" * 35)
        
        # Find all memory nodes
        memory_results = graph_persistence.search_entities(
            search_term="memory_",
            user_id=user_id,
            limit=10
        )
        print(f"   Found {len(memory_results)} stored memories:")
        for memory in memory_results:
            print(f"      • {memory['entity_name']}")
            
            # Get entities connected to this memory
            connected = graph_persistence.get_connected_entities(
                entity=memory['entity_name'],
                user_id=user_id,
                direction="outgoing"
            )
            if connected:
                entities = [conn['connected_entity'] for conn in connected]
                print(f"        Mentions: {', '.join(entities[:5])}")
        
        print("\n🎉 Neo4j Graph Persistence Demo Completed Successfully!")
        print("\n💡 Full Features Demonstrated:")
        print("   ✅ Real Neo4j database connection and health monitoring")
        print("   ✅ Entity node creation with rich properties and metadata")
        print("   ✅ Relationship creation with types and properties")
        print("   ✅ Memory document storage as interconnected graph structures")
        print("   ✅ Context querying with multi-hop graph traversal")
        print("   ✅ Directional relationship queries (incoming/outgoing)")
        print("   ✅ Entity search with pattern matching")
        print("   ✅ Graph statistics and analytics")
        print("   ✅ Memory querying and entity extraction visualization")
        print("\n🔧 Production Features Available:")
        print("   • Transaction support and ACID properties")
        print("   • Complex Cypher query execution")
        print("   • Multi-tenant user isolation")
        print("   • Backup and restore capabilities")
        print("   • Performance monitoring and optimization")
        print("   • Scalable graph operations")
        
        # Clean up
        print(f"\n🧹 Cleaning up...")
        graph_persistence.close()
        print("   ✅ Graph connection closed")
        
        print("\n🌐 You can also explore the graph visually at: http://localhost:7474")
        print("   Username: neo4j, Password: password")
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 