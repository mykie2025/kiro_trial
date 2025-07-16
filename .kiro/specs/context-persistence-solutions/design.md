# Design Document

## Overview

This design implements two complementary context persistence solutions for AI applications: a LangChain-based vector storage system for semantic similarity search and a Neo4j graph database for relationship-based queries. The system follows an MVP approach, prioritizing the LangChain solution first, then adding Neo4j as a secondary persistence layer.

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Context Persistence System               │
├─────────────────────────────────────────────────────────────┤
│  Configuration Layer (.env)                                │
│  ├── OpenAI API (gpt-4o-mini, text-embedding-3-small)     │
│  └── Neo4j Connection (localhost:7687)                     │
├─────────────────────────────────────────────────────────────┤
│  LangChain Vector Solution (Primary - MVP)                 │
│  ├── InMemoryVectorStore + OpenAI Embeddings              │
│  ├── Memory Tools (save_recall_memory, search_memories)    │
│  └── RunnableWithMessageHistory                           │
├─────────────────────────────────────────────────────────────┤
│  Neo4j Graph Solution (Secondary)                          │
│  ├── Docker Container Management                           │
│  ├── Graph Node/Relationship Management                    │
│  └── Cypher Query Interface                               │
├─────────────────────────────────────────────────────────────┤
│  Evaluation Framework                                       │
│  ├── LangChain Evaluators (context_recall, relevance)     │
│  ├── Performance Metrics Collection                        │
│  └── Comparative Analysis Reports                          │
└─────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Configuration Manager

**Purpose**: Centralized configuration management using environment variables

**Interface**:
```python
class ConfigManager:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL")
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.neo4j_uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        self.neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        self.neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
    
    def validate_config(self) -> bool
    def get_openai_client(self) -> OpenAI
    def get_neo4j_driver(self) -> neo4j.Driver
```

### 2. LangChain Vector Persistence (MVP Priority)

**Purpose**: Semantic similarity-based context storage and retrieval

**Interface**:
```python
class LangChainVectorPersistence:
    def __init__(self, config: ConfigManager):
        self.vector_store = InMemoryVectorStore(OpenAIEmbeddings())
        self.config = config
    
    def save_memory(self, memory: str, user_id: str, metadata: dict) -> str
    def search_memories(self, query: str, user_id: str, k: int = 3) -> List[str]
    def get_conversation_chain(self, session_id: str) -> RunnableWithMessageHistory
    def clear_memories(self, user_id: str = None) -> bool
```

**Tools Implementation**:
- `save_recall_memory`: Stores text memories with user ID and metadata
- `search_recall_memories`: Performs semantic search filtered by user ID
- Memory includes structured metadata: user_id, timestamp, memory_type

### 3. Neo4j Graph Persistence (Secondary)

**Purpose**: Relationship-based context storage using graph traversal

**Interface**:
```python
class Neo4jGraphPersistence:
    def __init__(self, config: ConfigManager):
        self.driver = config.get_neo4j_driver()
        self.database = config.neo4j_database
    
    def create_entity_node(self, entity: str, entity_type: str, properties: dict) -> str
    def create_relationship(self, from_entity: str, to_entity: str, relationship_type: str) -> bool
    def query_context(self, query: str, max_depth: int = 3) -> List[dict]
    def get_connected_entities(self, entity: str, relationship_types: List[str]) -> List[dict]
    def clear_graph(self) -> bool
```

**Docker Management**:
```python
class Neo4jDockerManager:
    def start_container(self) -> bool
    def stop_container(self) -> bool
    def is_container_running(self) -> bool
    def setup_persistent_volumes(self) -> bool
```

### 4. Evaluation Framework

**Purpose**: Compare effectiveness of both persistence solutions using LangChain's evaluation tools

**Interface**:
```python
class PersistenceEvaluator:
    def __init__(self, vector_persistence: LangChainVectorPersistence, 
                 graph_persistence: Neo4jGraphPersistence):
        self.vector_persistence = vector_persistence
        self.graph_persistence = graph_persistence
        self.evaluators = self._setup_langchain_evaluators()
    
    def evaluate_context_recall(self, test_queries: List[str], ground_truths: List[str]) -> dict
    def evaluate_relevance(self, queries: List[str], responses: List[str]) -> dict
    def compare_solutions(self, test_dataset: List[dict]) -> dict
    def generate_evaluation_report(self, results: dict) -> str
```

## Data Models

### Memory Document Structure
```python
@dataclass
class MemoryDocument:
    content: str
    user_id: str
    timestamp: datetime
    memory_type: str  # "conversation", "fact", "preference", etc.
    metadata: dict
    embedding: Optional[List[float]] = None
```

### Graph Node Structure
```python
@dataclass
class GraphNode:
    id: str
    label: str  # "Entity", "Concept", "Event", etc.
    properties: dict
    created_at: datetime
    updated_at: datetime
```

### Graph Relationship Structure
```python
@dataclass
class GraphRelationship:
    from_node: str
    to_node: str
    relationship_type: str  # "RELATES_TO", "CAUSED_BY", "MENTIONED_IN", etc.
    properties: dict
    created_at: datetime
```

## Error Handling

### Configuration Validation
- Validate all required environment variables on startup
- Provide clear error messages for missing or invalid configurations
- Graceful fallback for optional configurations

### Persistence Layer Errors
- Handle vector store initialization failures
- Manage Neo4j connection timeouts and retries
- Implement circuit breaker pattern for external dependencies

### Docker Container Management
- Handle container startup failures
- Implement health checks for Neo4j container
- Automatic container restart on failure

## Testing Strategy

### Unit Tests
- Configuration manager validation
- Memory storage and retrieval functions
- Graph node and relationship creation
- Evaluation metric calculations

### Integration Tests
- End-to-end memory persistence workflows
- Cross-system query comparisons
- Docker container lifecycle management
- LangChain evaluation framework integration

### Performance Tests
- Query response time measurements
- Memory usage monitoring
- Concurrent access handling
- Large dataset performance evaluation

## MVP Implementation Plan

### Phase 1: Core LangChain Implementation
1. Configuration management setup
2. InMemoryVectorStore with OpenAI embeddings
3. Basic memory save/search tools
4. Simple conversation history management

### Phase 2: Neo4j Integration
1. Docker container management
2. Basic graph node/relationship operations
3. Simple Cypher query interface
4. Data persistence validation

### Phase 3: Evaluation Framework
1. LangChain evaluator integration
2. Test dataset preparation
3. Comparative analysis implementation
4. Report generation system

## Security Considerations

- Environment variable validation and sanitization
- Neo4j authentication and authorization
- OpenAI API key protection
- Docker container security best practices
- Input validation for all user-provided data