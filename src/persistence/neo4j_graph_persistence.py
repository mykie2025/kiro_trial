"""Neo4j graph-based context persistence implementation."""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, TransientError, ClientError

from ..config.config_manager import ConfigManager
from .memory_document import MemoryDocument, MemoryType


logger = logging.getLogger(__name__)


class Neo4jGraphError(Exception):
    """Custom exception for Neo4j graph operation errors."""
    pass


@dataclass
class GraphNode:
    """Structure for graph nodes."""
    id: str
    label: str
    properties: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class GraphRelationship:
    """Structure for graph relationships."""
    from_node: str
    to_node: str
    relationship_type: str
    properties: Dict[str, Any]
    created_at: datetime


@dataclass
class GraphPath:
    """Structure for graph traversal paths."""
    nodes: List[GraphNode]
    relationships: List[GraphRelationship]
    path_length: int


class Neo4jGraphPersistence:
    """
    Neo4j-based graph persistence system for contextual relationship storage.
    
    This class provides functionality to store and query context using graph
    relationships, enabling complex traversal and connection-based retrieval.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the Neo4j graph persistence system.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        
        # Initialize Neo4j driver
        try:
            driver_config = config_manager.get_neo4j_driver_config()
            self.driver = GraphDatabase.driver(**driver_config)
            self.database = driver_config['database']
            
            # Test connection
            self._test_connection()
            logger.info("Neo4j graph persistence initialized successfully")
            
        except Exception as e:
            raise Neo4jGraphError(f"Failed to initialize Neo4j driver: {e}")
    
    def _test_connection(self) -> None:
        """Test Neo4j database connection."""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if not record or record['test'] != 1:
                    raise Neo4jGraphError("Connection test failed")
                    
        except Exception as e:
            raise Neo4jGraphError(f"Database connection failed: {e}")
    
    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j driver connection closed")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the Neo4j database.
        
        Returns:
            Health status information
        """
        try:
            with self.driver.session(database=self.database) as session:
                # Test basic connectivity
                result = session.run("RETURN 1 as test")
                record = result.single()
                
                if not record or record['test'] != 1:
                    return {
                        'status': 'unhealthy',
                        'error': 'Connection test failed'
                    }
                
                # Get database info
                db_info = session.run("CALL dbms.components() YIELD name, versions, edition")
                components = [dict(record) for record in db_info]
                
                # Get node and relationship counts
                node_count_result = session.run("MATCH (n) RETURN count(n) as node_count")
                node_count_record = node_count_result.single()
                node_count = node_count_record['node_count'] if node_count_record else 0
                
                rel_count_result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                rel_count_record = rel_count_result.single()
                rel_count = rel_count_record['rel_count'] if rel_count_record else 0
                
                return {
                    'status': 'healthy',
                    'database': self.database,
                    'components': components,
                    'node_count': node_count,
                    'relationship_count': rel_count,
                    'driver_config': {
                        'uri': self.config.neo4j_uri,
                        'database': self.database
                    }
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def create_entity_node(self, 
                          entity: str, 
                          entity_type: str, 
                          properties: Dict[str, Any],
                          user_id: str) -> str:
        """
        Create or update an entity node in the graph.
        
        Args:
            entity: Entity name/identifier
            entity_type: Type of entity (e.g., 'Person', 'Concept', 'Event')
            properties: Additional properties for the node
            user_id: User identifier for multi-tenant support
            
        Returns:
            Node ID in the graph
        """
        node_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Merge entity properties with metadata
        node_properties = {
            'id': node_id,
            'name': entity,
            'entity_type': entity_type,
            'user_id': user_id,
            'created_at': timestamp.isoformat(),
            'updated_at': timestamp.isoformat(),
            **properties
        }
        
        try:
            with self.driver.session(database=self.database) as session:
                # Use MERGE to create or update existing entity
                query = """
                MERGE (n:Entity {name: $entity, user_id: $user_id})
                SET n += $properties
                SET n.updated_at = $timestamp
                RETURN n.id as node_id
                """
                
                result = session.run(query, {
                    'entity': entity,
                    'user_id': user_id,
                    'properties': node_properties,
                    'timestamp': timestamp.isoformat()
                })
                
                record = result.single()
                if record:
                    actual_node_id = record['node_id']
                    logger.info(f"Created/updated entity node: {entity} (ID: {actual_node_id})")
                    return actual_node_id
                else:
                    raise Neo4jGraphError(f"Failed to create entity node: {entity}")
                    
        except Exception as e:
            logger.error(f"Error creating entity node {entity}: {e}")
            raise Neo4jGraphError(f"Failed to create entity node: {e}")
    
    def create_relationship(self, 
                           from_entity: str, 
                           to_entity: str, 
                           relationship_type: str,
                           user_id: str,
                           properties: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a relationship between two entities.
        
        Args:
            from_entity: Source entity name
            to_entity: Target entity name
            relationship_type: Type of relationship (e.g., 'RELATES_TO', 'CAUSED_BY')
            user_id: User identifier for multi-tenant support
            properties: Additional properties for the relationship
            
        Returns:
            True if relationship created successfully
        """
        if properties is None:
            properties = {}
        
        timestamp = datetime.now()
        rel_properties = {
            'created_at': timestamp.isoformat(),
            'user_id': user_id,
            **properties
        }
        
        try:
            with self.driver.session(database=self.database) as session:
                # Create relationship between existing entities - use parameterized query
                query = """
                MATCH (from:Entity {name: $from_entity, user_id: $user_id})
                MATCH (to:Entity {name: $to_entity, user_id: $user_id})
                CALL apoc.create.relationship(from, $relationship_type, $properties, to) YIELD rel
                RETURN rel
                """
                
                result = session.run(query, {
                    'from_entity': from_entity,
                    'to_entity': to_entity,
                    'user_id': user_id,
                    'relationship_type': relationship_type,
                    'properties': rel_properties
                })
                
                record = result.single()
                if record:
                    logger.info(f"Created relationship: {from_entity} -{relationship_type}-> {to_entity}")
                    return True
                else:
                    logger.warning(f"No relationship created between {from_entity} and {to_entity}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error creating relationship {from_entity} -> {to_entity}: {e}")
            raise Neo4jGraphError(f"Failed to create relationship: {e}")
    
    def query_context(self, 
                     entity: str, 
                     user_id: str,
                     max_depth: int = 3,
                     relationship_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Query context using graph traversal from a starting entity.
        
        Args:
            entity: Starting entity name
            user_id: User identifier for filtering
            max_depth: Maximum traversal depth
            relationship_types: Optional list of relationship types to follow
            
        Returns:
            List of connected entities with their relationships
        """
        try:
            with self.driver.session(database=self.database) as session:
                # Build relationship type filter
                if relationship_types:
                    rel_types = "|".join(relationship_types)
                    rel_filter = f":{rel_types}"
                else:
                    rel_filter = ""
                
                # Create safe query based on max_depth (Neo4j doesn't allow parameters in variable-length patterns)
                if max_depth == 1:
                    base_query = """
                    MATCH path = (start:Entity {name: $entity, user_id: $user_id})
                    -[r*1..1]-(connected:Entity)
                    WHERE connected.user_id = $user_id
                    RETURN path, 
                           length(path) as path_length,
                           connected.name as connected_entity,
                           connected.entity_type as entity_type,
                           connected.id as entity_id
                    ORDER BY path_length ASC
                    LIMIT 50
                    """
                elif max_depth == 2:
                    base_query = """
                    MATCH path = (start:Entity {name: $entity, user_id: $user_id})
                    -[r*1..2]-(connected:Entity)
                    WHERE connected.user_id = $user_id
                    RETURN path, 
                           length(path) as path_length,
                           connected.name as connected_entity,
                           connected.entity_type as entity_type,
                           connected.id as entity_id
                    ORDER BY path_length ASC
                    LIMIT 50
                    """
                else:  # max_depth >= 3
                    base_query = """
                    MATCH path = (start:Entity {name: $entity, user_id: $user_id})
                    -[r*1..5]-(connected:Entity)
                    WHERE connected.user_id = $user_id
                    RETURN path, 
                           length(path) as path_length,
                           connected.name as connected_entity,
                           connected.entity_type as entity_type,
                           connected.id as entity_id
                    ORDER BY path_length ASC
                    LIMIT 50
                    """
                
                result = session.run(base_query, {
                    'entity': entity,
                    'user_id': user_id
                })
                
                contexts = []
                for record in result:
                    path_info = {
                        'connected_entity': record['connected_entity'],
                        'entity_type': record['entity_type'],
                        'entity_id': record['entity_id'],
                        'path_length': record['path_length'],
                        'path': self._extract_path_info(record['path'])
                    }
                    contexts.append(path_info)
                
                logger.info(f"Found {len(contexts)} connected entities for {entity}")
                return contexts
                
        except Exception as e:
            logger.error(f"Error querying context for {entity}: {e}")
            raise Neo4jGraphError(f"Failed to query context: {e}")
    
    def get_connected_entities(self, 
                              entity: str, 
                              user_id: str,
                              relationship_types: Optional[List[str]] = None,
                              direction: str = 'both') -> List[Dict[str, Any]]:
        """
        Get directly connected entities with specified relationship types.
        
        Args:
            entity: Source entity name
            user_id: User identifier for filtering
            relationship_types: List of relationship types to follow
            direction: 'incoming', 'outgoing', or 'both'
            
        Returns:
            List of connected entities with relationship information
        """
        try:
            with self.driver.session(database=self.database) as session:
                # Build direction pattern
                if direction == 'incoming':
                    direction_pattern = "<-[r]-"
                elif direction == 'outgoing':
                    direction_pattern = "-[r]->"
                else:  # both
                    direction_pattern = "-[r]-"
                
                # Create base query
                base_query = f"""
                MATCH (start:Entity {{name: $entity, user_id: $user_id}})
                {direction_pattern}(connected:Entity)
                WHERE connected.user_id = $user_id
                RETURN connected.name as connected_entity,
                       connected.entity_type as entity_type,
                       connected.id as entity_id,
                       type(r) as relationship_type,
                       r.created_at as relationship_created,
                       properties(r) as relationship_properties
                ORDER BY connected.name
                """
                
                result = session.run(base_query, {
                    'entity': entity,
                    'user_id': user_id
                })
                
                connections = []
                for record in result:
                    # Filter by relationship types if specified
                    if relationship_types and record['relationship_type'] not in relationship_types:
                        continue
                    
                    connection_info = {
                        'connected_entity': record['connected_entity'],
                        'entity_type': record['entity_type'],
                        'entity_id': record['entity_id'],
                        'relationship_type': record['relationship_type'],
                        'relationship_created': record['relationship_created'],
                        'relationship_properties': record['relationship_properties']
                    }
                    connections.append(connection_info)
                
                logger.info(f"Found {len(connections)} direct connections for {entity}")
                return connections
                
        except Exception as e:
            logger.error(f"Error getting connected entities for {entity}: {e}")
            raise Neo4jGraphError(f"Failed to get connected entities: {e}")
    
    def search_entities(self, 
                       search_term: str, 
                       user_id: str,
                       entity_types: Optional[List[str]] = None,
                       limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for entities by name or properties.
        
        Args:
            search_term: Search term to match against entity names
            user_id: User identifier for filtering
            entity_types: Optional list of entity types to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of matching entities
        """
        try:
            with self.driver.session(database=self.database) as session:
                # Create base query with parameterized type filtering
                base_query = """
                MATCH (n:Entity)
                WHERE n.user_id = $user_id
                AND (n.name CONTAINS $search_term OR n.name =~ $regex_term)
                RETURN n.name as entity_name,
                       n.entity_type as entity_type,
                       n.id as entity_id,
                       n.created_at as created_at,
                       properties(n) as properties
                ORDER BY n.name
                LIMIT $limit
                """
                
                # Create case-insensitive regex pattern
                regex_term = f"(?i).*{search_term}.*"
                
                result = session.run(base_query, {
                    'search_term': search_term,
                    'regex_term': regex_term,
                    'user_id': user_id,
                    'limit': limit
                })
                
                entities = []
                for record in result:
                    # Filter by entity types if specified
                    if entity_types and record['entity_type'] not in entity_types:
                        continue
                    
                    entity_info = {
                        'entity_name': record['entity_name'],
                        'entity_type': record['entity_type'],
                        'entity_id': record['entity_id'],
                        'created_at': record['created_at'],
                        'properties': record['properties']
                    }
                    entities.append(entity_info)
                
                logger.info(f"Found {len(entities)} entities matching '{search_term}'")
                return entities
                
        except Exception as e:
            logger.error(f"Error searching entities with term '{search_term}': {e}")
            raise Neo4jGraphError(f"Failed to search entities: {e}")
    
    def store_memory_as_graph(self, memory_doc: MemoryDocument) -> Dict[str, Any]:
        """
        Store a memory document as graph nodes and relationships.
        
        Args:
            memory_doc: Memory document to store
            
        Returns:
            Dictionary with created nodes and relationships info
        """
        try:
            # Generate a unique ID for this memory
            memory_id = str(uuid.uuid4())
            memory_name = f"memory_{memory_id}"
            
            with self.driver.session(database=self.database) as session:
                # Create main memory node
                # Convert metadata to JSON string for Neo4j compatibility
                import json
                metadata_str = json.dumps(memory_doc.metadata) if memory_doc.metadata else "{}"
                
                memory_node_id = self.create_entity_node(
                    entity=memory_name,
                    entity_type="Memory",
                    properties={
                        'content': memory_doc.content,
                        'memory_type': memory_doc.memory_type.value,
                        'timestamp': memory_doc.timestamp.isoformat(),
                        'metadata': metadata_str
                    },
                    user_id=memory_doc.user_id
                )
                
                # Extract entities from memory content (simple keyword extraction)
                entities = self._extract_entities_from_content(memory_doc.content)
                created_entities = []
                created_relationships = []
                
                for entity in entities:
                    # Create entity node
                    entity_id = self.create_entity_node(
                        entity=entity,
                        entity_type="Concept",
                        properties={},
                        user_id=memory_doc.user_id
                    )
                    created_entities.append(entity_id)
                    
                    # Create relationship to memory
                    self.create_relationship(
                        from_entity=memory_name,
                        to_entity=entity,
                        relationship_type="MENTIONS",
                        user_id=memory_doc.user_id
                    )
                    created_relationships.append(f"{memory_name} -MENTIONS-> {entity}")
                
                return {
                    'memory_node_id': memory_node_id,
                    'memory_name': memory_name,
                    'created_entities': created_entities,
                    'created_relationships': created_relationships,
                    'entity_count': len(created_entities)
                }
                
        except Exception as e:
            logger.error(f"Error storing memory as graph: {e}")
            raise Neo4jGraphError(f"Failed to store memory as graph: {e}")
    
    def _extract_entities_from_content(self, content: str) -> List[str]:
        """
        Simple entity extraction from content (can be enhanced with NLP).
        
        Args:
            content: Text content to extract entities from
            
        Returns:
            List of extracted entity names
        """
        # Simple keyword extraction - in production this would use NLP
        import re
        
        # Extract capitalized words as potential entities
        words = re.findall(r'\b[A-Z][a-z]+\b', content)
        
        # Filter common stop words
        stop_words = {'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'And', 'But', 'Or', 'So'}
        entities = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity.lower() not in seen:
                seen.add(entity.lower())
                unique_entities.append(entity)
        
        return unique_entities[:10]  # Limit to 10 entities per memory
    
    def _extract_path_info(self, path) -> Dict[str, Any]:
        """
        Extract information from a Neo4j path object.
        
        Args:
            path: Neo4j path object
            
        Returns:
            Dictionary with path information
        """
        try:
            nodes = []
            relationships = []
            
            # Extract node information
            for node in path.nodes:
                node_info = {
                    'id': node.get('id', ''),
                    'name': node.get('name', ''),
                    'entity_type': node.get('entity_type', ''),
                    'properties': dict(node)
                }
                nodes.append(node_info)
            
            # Extract relationship information
            for rel in path.relationships:
                rel_info = {
                    'type': rel.type,
                    'properties': dict(rel)
                }
                relationships.append(rel_info)
            
            return {
                'nodes': nodes,
                'relationships': relationships,
                'length': len(path)
            }
            
        except Exception as e:
            logger.error(f"Error extracting path info: {e}")
            return {'nodes': [], 'relationships': [], 'length': 0}
    
    def clear_graph(self, user_id: Optional[str] = None) -> bool:
        """
        Clear graph data (optionally filtered by user).
        
        Args:
            user_id: Optional user ID to filter deletion
            
        Returns:
            True if clearing successful
        """
        try:
            with self.driver.session(database=self.database) as session:
                if user_id:
                    # Clear only user's data
                    query = "MATCH (n:Entity {user_id: $user_id}) DETACH DELETE n"
                    session.run(query, {'user_id': user_id})
                    logger.info(f"Cleared graph data for user: {user_id}")
                else:
                    # Clear all data
                    query = "MATCH (n) DETACH DELETE n"
                    session.run(query)
                    logger.info("Cleared all graph data")
                
                return True
                
        except Exception as e:
            logger.error(f"Error clearing graph: {e}")
            return False
    
    def get_graph_statistics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get graph statistics.
        
        Args:
            user_id: Optional user ID to filter statistics
            
        Returns:
            Dictionary with graph statistics
        """
        try:
            with self.driver.session(database=self.database) as session:
                if user_id:
                    # User-specific statistics
                    node_query = "MATCH (n:Entity {user_id: $user_id}) RETURN count(n) as count"
                    rel_query = "MATCH (n:Entity {user_id: $user_id})-[r]->() RETURN count(r) as count"
                    type_query = """
                    MATCH (n:Entity {user_id: $user_id}) 
                    RETURN n.entity_type as type, count(n) as count 
                    ORDER BY count DESC
                    """
                    params = {'user_id': user_id}
                else:
                    # Global statistics
                    node_query = "MATCH (n:Entity) RETURN count(n) as count"
                    rel_query = "MATCH ()-[r]->() RETURN count(r) as count"
                    type_query = """
                    MATCH (n:Entity) 
                    RETURN n.entity_type as type, count(n) as count 
                    ORDER BY count DESC
                    """
                    params = {}
                
                # Get counts
                node_count_result = session.run(node_query, params)
                node_count_record = node_count_result.single()
                node_count = node_count_record['count'] if node_count_record else 0
                
                rel_count_result = session.run(rel_query, params)
                rel_count_record = rel_count_result.single()
                rel_count = rel_count_record['count'] if rel_count_record else 0
                
                # Get entity type distribution
                type_result = session.run(type_query, params)
                entity_types = [{'type': record['type'], 'count': record['count']} 
                              for record in type_result]
                
                stats = {
                    'node_count': node_count,
                    'relationship_count': rel_count,
                    'entity_types': entity_types,
                    'user_id': user_id
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting graph statistics: {e}")
            return {'error': str(e)} 