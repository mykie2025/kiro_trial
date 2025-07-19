"""Simplified Neo4j Graph Persistence Demo - Key Concepts without Docker."""

import sys
import os
from datetime import datetime
from typing import Dict, Any, List

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.persistence.memory_document import MemoryDocument, MemoryType


class MockGraphNode:
    """Mock representation of a graph node for demonstration."""
    
    def __init__(self, entity: str, entity_type: str, properties: Dict[str, Any], user_id: str):
        self.entity = entity
        self.entity_type = entity_type
        self.properties = properties
        self.user_id = user_id
        self.created_at = datetime.now()
        self.id = f"{entity}_{hash(entity + user_id) % 10000}"
    
    def __repr__(self):
        return f"Node({self.entity}, {self.entity_type}, {self.properties})"


class MockGraphRelationship:
    """Mock representation of a graph relationship for demonstration."""
    
    def __init__(self, from_entity: str, to_entity: str, relationship_type: str, user_id: str):
        self.from_entity = from_entity
        self.to_entity = to_entity
        self.relationship_type = relationship_type
        self.user_id = user_id
        self.created_at = datetime.now()
    
    def __repr__(self):
        return f"Rel({self.from_entity} -[{self.relationship_type}]-> {self.to_entity})"


class Neo4jGraphDemo:
    """Simplified demonstration of Neo4j graph persistence concepts."""
    
    def __init__(self):
        self.nodes: List[MockGraphNode] = []
        self.relationships: List[MockGraphRelationship] = []
        self.memories: List[Dict[str, Any]] = []
    
    def create_entity_node(self, entity: str, entity_type: str, properties: Dict[str, Any], user_id: str) -> str:
        """Create a mock entity node."""
        node = MockGraphNode(entity, entity_type, properties, user_id)
        self.nodes.append(node)
        return node.id
    
    def create_relationship(self, from_entity: str, to_entity: str, relationship_type: str, user_id: str) -> bool:
        """Create a mock relationship between entities."""
        relationship = MockGraphRelationship(from_entity, to_entity, relationship_type, user_id)
        self.relationships.append(relationship)
        return True
    
    def store_memory_as_graph(self, memory_doc: MemoryDocument) -> Dict[str, Any]:
        """Store a memory document as graph structure."""
        memory_name = f"memory_{len(self.memories) + 1}"
        
        # Simple entity extraction (keywords)
        entities = self._extract_entities_from_content(memory_doc.content)
        
        # Store memory
        memory_info = {
            'memory_name': memory_name,
            'content': memory_doc.content,
            'memory_type': memory_doc.memory_type.value,
            'user_id': memory_doc.user_id,
            'timestamp': memory_doc.timestamp.isoformat(),
            'extracted_entities': entities
        }
        self.memories.append(memory_info)
        
        # Create nodes for extracted entities
        created_entities = []
        for entity in entities:
            node_id = self.create_entity_node(entity, "Concept", {}, memory_doc.user_id)
            created_entities.append(node_id)
            
            # Create relationship from memory to entity
            self.create_relationship(memory_name, entity, "MENTIONS", memory_doc.user_id)
        
        return {
            'memory_name': memory_name,
            'created_entities': created_entities,
            'entity_count': len(created_entities)
        }
    
    def _extract_entities_from_content(self, content: str) -> List[str]:
        """Simple entity extraction from content."""
        # Simple keyword extraction - in real implementation would use NLP
        keywords = [
            "Python", "Machine Learning", "Neural Networks", "TensorFlow", 
            "Data Science", "algorithm", "model", "training", "prediction"
        ]
        
        entities = []
        content_lower = content.lower()
        for keyword in keywords:
            if keyword.lower() in content_lower:
                entities.append(keyword)
        
        return entities
    
    def query_context(self, entity: str, user_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Query connected entities (mock traversal)."""
        connected = []
        
        # Find direct relationships
        for rel in self.relationships:
            if rel.user_id == user_id:
                if rel.from_entity == entity:
                    connected.append({
                        'connected_entity': rel.to_entity,
                        'relationship_type': rel.relationship_type,
                        'path_length': 1,
                        'entity_type': self._get_entity_type(rel.to_entity, user_id)
                    })
                elif rel.to_entity == entity:
                    connected.append({
                        'connected_entity': rel.from_entity,
                        'relationship_type': rel.relationship_type,
                        'path_length': 1,
                        'entity_type': self._get_entity_type(rel.from_entity, user_id)
                    })
        
        # Find indirect relationships (depth 2)
        if max_depth > 1:
            for conn in connected[:]:  # Copy to avoid modification during iteration
                for rel in self.relationships:
                    if rel.user_id == user_id:
                        if rel.from_entity == conn['connected_entity']:
                            if rel.to_entity != entity:  # Avoid cycles
                                connected.append({
                                    'connected_entity': rel.to_entity,
                                    'relationship_type': f"{conn['relationship_type']} -> {rel.relationship_type}",
                                    'path_length': 2,
                                    'entity_type': self._get_entity_type(rel.to_entity, user_id)
                                })
        
        return connected
    
    def _get_entity_type(self, entity: str, user_id: str) -> str:
        """Get entity type for a given entity."""
        for node in self.nodes:
            if node.entity == entity and node.user_id == user_id:
                return node.entity_type
        return "Unknown"
    
    def search_entities(self, search_term: str, user_id: str) -> List[Dict[str, Any]]:
        """Search for entities by name."""
        results = []
        for node in self.nodes:
            if node.user_id == user_id and search_term.lower() in node.entity.lower():
                results.append({
                    'entity_name': node.entity,
                    'entity_type': node.entity_type,
                    'entity_id': node.id,
                    'properties': node.properties
                })
        return results
    
    def get_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get graph statistics."""
        user_nodes = [n for n in self.nodes if n.user_id == user_id]
        user_relationships = [r for r in self.relationships if r.user_id == user_id]
        user_memories = [m for m in self.memories if m['user_id'] == user_id]
        
        entity_types = {}
        for node in user_nodes:
            entity_types[node.entity_type] = entity_types.get(node.entity_type, 0) + 1
        
        return {
            'node_count': len(user_nodes),
            'relationship_count': len(user_relationships),
            'memory_count': len(user_memories),
            'entity_types': [{'type': k, 'count': v} for k, v in entity_types.items()]
        }


def main():
    """Demonstrate Neo4j graph persistence concepts."""
    print("🚀 Neo4j Graph Persistence Demo - Simplified Version")
    print("=" * 60)
    print("📝 This demo shows the key concepts without requiring Docker/Neo4j")
    print()
    
    # Initialize demo
    graph_demo = Neo4jGraphDemo()
    user_id = "demo_user"
    
    # 1. Create entity nodes
    print("1. 📊 Creating Entity Nodes")
    print("-" * 30)
    
    entities_to_create = [
        {"name": "Python", "type": "Programming Language", "properties": {"paradigm": "multi-paradigm"}},
        {"name": "Machine Learning", "type": "Field", "properties": {"complexity": "high"}},
        {"name": "Neural Networks", "type": "Concept", "properties": {"category": "deep learning"}},
        {"name": "TensorFlow", "type": "Framework", "properties": {"language": "Python"}},
        {"name": "Data Science", "type": "Field", "properties": {"tools": "Python, R"}},
    ]
    
    for entity_info in entities_to_create:
        node_id = graph_demo.create_entity_node(
            entity=entity_info["name"],
            entity_type=entity_info["type"],
            properties=entity_info["properties"],
            user_id=user_id
        )
        print(f"   ✅ Created: {entity_info['name']} ({entity_info['type']}) -> ID: {node_id}")
    
    # 2. Create relationships
    print("\n2. 🔗 Creating Relationships")
    print("-" * 30)
    
    relationships_to_create = [
        {"from": "Python", "to": "Machine Learning", "type": "USED_IN"},
        {"from": "Python", "to": "Data Science", "type": "USED_IN"},
        {"from": "Machine Learning", "to": "Neural Networks", "type": "INCLUDES"},
        {"from": "TensorFlow", "to": "Neural Networks", "type": "IMPLEMENTS"},
        {"from": "TensorFlow", "to": "Python", "type": "BUILT_WITH"},
        {"from": "Data Science", "to": "Machine Learning", "type": "APPLIES"},
    ]
    
    for rel_info in relationships_to_create:
        success = graph_demo.create_relationship(
            from_entity=rel_info["from"],
            to_entity=rel_info["to"],
            relationship_type=rel_info["type"],
            user_id=user_id
        )
        if success:
            print(f"   ✅ {rel_info['from']} -[{rel_info['type']}]-> {rel_info['to']}")
    
    # 3. Store memories as graph
    print("\n3. 🧠 Storing Memory Documents as Graph")
    print("-" * 40)
    
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
        ),
        MemoryDocument(
            content="Data Science combines Python programming with statistical analysis for prediction models",
            user_id=user_id,
            memory_type=MemoryType.FACT
        )
    ]
    
    for memory in memories:
        result = graph_demo.store_memory_as_graph(memory)
        print(f"   ✅ Stored: {memory.content[:50]}...")
        print(f"      Memory: {result['memory_name']}")
        print(f"      Extracted entities: {result['entity_count']}")
    
    # 4. Query context using graph traversal
    print("\n4. 🔍 Querying Context using Graph Traversal")
    print("-" * 45)
    
    query_entities = ["Python", "Machine Learning"]
    
    for entity in query_entities:
        print(f"\n   🎯 Context for: {entity}")
        contexts = graph_demo.query_context(entity, user_id, max_depth=2)
        
        print(f"   Found {len(contexts)} connected entities:")
        for context in contexts:
            print(f"      • {context['connected_entity']} ({context['entity_type']})")
            print(f"        via: {context['relationship_type']} (distance: {context['path_length']})")
    
    # 5. Search entities
    print("\n5. 🔎 Searching Entities")
    print("-" * 25)
    
    search_results = graph_demo.search_entities("Machine", user_id)
    print(f"   Found {len(search_results)} entities matching 'Machine':")
    for result in search_results:
        print(f"      • {result['entity_name']} ({result['entity_type']})")
    
    # 6. Graph statistics
    print("\n6. 📈 Graph Statistics")
    print("-" * 20)
    
    stats = graph_demo.get_statistics(user_id)
    print(f"   📊 Statistics for user '{user_id}':")
    print(f"      • Nodes: {stats['node_count']}")
    print(f"      • Relationships: {stats['relationship_count']}")
    print(f"      • Memories: {stats['memory_count']}")
    print(f"      • Entity types:")
    for entity_type in stats['entity_types']:
        print(f"        - {entity_type['type']}: {entity_type['count']}")
    
    # 7. Show internal state
    print("\n7. 🔧 Internal Graph Structure")
    print("-" * 35)
    
    print("   📊 Nodes:")
    for node in graph_demo.nodes[:5]:  # Show first 5 nodes
        print(f"      • {node}")
    
    print("\n   🔗 Relationships:")
    for rel in graph_demo.relationships[:5]:  # Show first 5 relationships
        print(f"      • {rel}")
    
    print("\n   🧠 Stored Memories:")
    for memory in graph_demo.memories:
        print(f"      • {memory['memory_name']}: {memory['content'][:50]}...")
    
    print("\n🎉 Neo4j Graph Persistence Demo Completed!")
    print("\n💡 Key Concepts Demonstrated:")
    print("   • Entity node creation with properties and metadata")
    print("   • Relationship creation between entities with types")
    print("   • Memory document storage as graph structures")
    print("   • Context querying with graph traversal simulation")
    print("   • Entity search and filtering capabilities")
    print("   • Graph statistics and analytics")
    print("\n🔧 Real Implementation Features:")
    print("   • Docker container management for Neo4j")
    print("   • Cypher query execution for complex traversals")
    print("   • Persistent storage with transaction support")
    print("   • Advanced NLP for entity extraction")
    print("   • Multi-tenant user isolation")
    print("   • Backup and restore capabilities")


if __name__ == "__main__":
    main() 