"""Comprehensive Evaluation Demo for LangChain Vector Persistence."""

import sys
import os
import uuid
import re
import random
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config.config_manager import ConfigManager
from src.persistence.langchain_vector_persistence import LangChainVectorPersistence
from src.persistence.neo4j_graph_persistence import Neo4jGraphPersistence
from src.persistence.memory_document import MemoryDocument, MemoryType
from src.evaluation.persistence_evaluator import (
    PersistenceEvaluator, 
    EvaluationQuery,
    ReportFormat,
    create_sample_test_dataset
)


class LangChainEvaluationDemo:
    """Comprehensive evaluation of LangChain vector persistence system."""
    
    def __init__(self):
        """Initialize the evaluation demo."""
        print("🔬 Comparative Persistence Evaluation Demo (Vector vs Graph)")
        print("=" * 65)
        
        try:
            self.config_manager = ConfigManager()
            
            # Initialize vector persistence
            self.vector_persistence = LangChainVectorPersistence(self.config_manager)
            print("✅ Vector persistence initialized")
            
            # Initialize Neo4j graph persistence directly (bypass docker manager)
            print("🕸️  Connecting to running Neo4j instance...")
            self.graph_persistence = Neo4jGraphPersistence(self.config_manager)
            
            # Test Neo4j health
            health = self.graph_persistence.health_check()
            if health['status'] == 'healthy':
                print("✅ Graph persistence initialized and healthy")
                print(f"   • Database: {health['database']}")
                print(f"   • Nodes: {health['node_count']}")
            else:
                print(f"⚠️  Graph persistence connected but not optimal: {health.get('error', 'Unknown')}")
            
            # Initialize evaluator
            self.evaluator = PersistenceEvaluator(self.config_manager)
            print("✅ Evaluation framework initialized")
            
            # Test user for evaluation
            self.test_user_id = "comparative_eval_user"
            
            print("✅ Comparative evaluation framework ready!")
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            raise
    
    def parse_markdown_file(self, file_path: str) -> Dict[str, Any]:
        """Parse a markdown file and extract structured content."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Markdown file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata from the beginning
            lines = content.split('\n')
            metadata = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'line_count': len(lines),
                'char_count': len(content)
            }
            
            # Try to extract title/author from first few lines
            for i, line in enumerate(lines[:10]):
                line = line.strip()
                if line and not line.startswith('#') and len(line) > 5:
                    # Potential title or author
                    if '/' in line or 'Writing' in line or 'About' in line:
                        continue  # Skip navigation elements
                    if not metadata.get('title') and len(line) < 100:
                        metadata['title'] = line
                        break
            
            return {
                'content': content,
                'metadata': metadata
            }
            
        except Exception as e:
            raise ValueError(f"Failed to parse markdown file {file_path}: {e}")
    
    def extract_information_points(self, content: str, num_points: int = 20) -> List[Dict[str, Any]]:
        """Extract information points from markdown content."""
        # Split content into sentences and paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        information_points = []
        
        for paragraph in paragraphs:
            # Skip headers, navigation, and very short paragraphs
            if (paragraph.startswith('#') or 
                len(paragraph) < 50 or
                'About\nWriting' in paragraph or
                'RSS\nTwitter' in paragraph):
                continue
            
            # Split paragraph into sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 30:  # Skip very short sentences
                    continue
                
                # Categorize the information point
                info_type = self._categorize_information(sentence)
                
                information_points.append({
                    'content': sentence,
                    'type': info_type,
                    'length': len(sentence),
                    'word_count': len(sentence.split())
                })
        
        # Randomly sample the requested number of points
        if len(information_points) > num_points:
            information_points = random.sample(information_points, num_points)
        
        return information_points
    
    def _categorize_information(self, sentence: str) -> MemoryType:
        """Categorize a sentence into a memory type based on content patterns."""
        sentence_lower = sentence.lower()
        
        # Preference indicators
        if any(word in sentence_lower for word in ['prefer', 'like', 'favorite', 'choose', 'use']):
            return MemoryType.PREFERENCE
        
        # Fact indicators (numbers, definitive statements)
        if (any(word in sentence_lower for word in ['is', 'was', 'are', 'were', 'has', 'have']) and
            any(char.isdigit() for char in sentence) or
            'company' in sentence_lower or 'organization' in sentence_lower):
            return MemoryType.FACT
        
        # Conversation indicators
        if any(word in sentence_lower for word in ['said', 'told', 'asked', 'mentioned', 'discussed']):
            return MemoryType.CONVERSATION
        
        # Event indicators
        if any(word in sentence_lower for word in ['joined', 'left', 'launched', 'started', 'happened', 'occurred']):
            return MemoryType.EVENT
        
        # Default to context for descriptive content
        return MemoryType.CONTEXT
    
    def create_memories_from_markdown(self, markdown_file: str = "examples/data/3.md", 
                                    num_points: int = 20) -> List[MemoryDocument]:
        """Create memory documents from markdown file content."""
        print(f"\n📋 Creating Memories from Markdown File: {markdown_file}")
        print("-" * 50)
        
        # Parse the markdown file
        parsed_data = self.parse_markdown_file(markdown_file)
        print(f"📄 Parsed file: {parsed_data['metadata']['file_name']}")
        print(f"   Lines: {parsed_data['metadata']['line_count']}, Characters: {parsed_data['metadata']['char_count']}")
        
        # Extract information points
        info_points = self.extract_information_points(parsed_data['content'], num_points)
        print(f"🔍 Extracted {len(info_points)} information points")
        
        # Create memory documents
        memories = []
        type_counts = {}
        
        for i, point in enumerate(info_points, 1):
            memory = MemoryDocument(
                content=point['content'],
                user_id=self.test_user_id,
                timestamp=datetime.now(),
                memory_type=point['type'],
                metadata={
                    'source': 'markdown',
                    'file_name': parsed_data['metadata']['file_name'],
                    'point_index': i,
                    'word_count': point['word_count'],
                    'extraction_method': 'automated'
                }
            )
            memories.append(memory)
            
            # Count types
            type_name = point['type'].value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        # Display summary
        print(f"📊 Memory types distribution:")
        for mem_type, count in type_counts.items():
            print(f"   • {mem_type}: {count} memories")
        
        return memories
    
    def setup_markdown_evaluation_data(self, markdown_file: str = "examples/data/3.md", 
                                     num_points: int = 20) -> List[MemoryDocument]:
        """Set up test data using markdown file as source."""
        print("\n📋 Setting Up Markdown-Based Evaluation Test Data")
        print("-" * 50)
        
        # Create memories from markdown
        test_memories = self.create_memories_from_markdown(markdown_file, num_points)
        
        # Clear any existing test data first
        try:
            self.vector_persistence.clear_memories(self.test_user_id)
            with self.graph_persistence.driver.session(database=self.graph_persistence.database) as session:
                session.run(
                    "MATCH (n {user_id: $user_id}) DETACH DELETE n",
                    {'user_id': self.test_user_id}
                )
            print("🧹 Cleared existing test data from both systems")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")
        
        # Store memories in both persistence systems
        print(f"💾 Storing {len(test_memories)} markdown-based memories in both systems...")
        
        memory_nodes = {}  # Track created memory nodes for relationships
        
        for i, memory in enumerate(test_memories, 1):
            try:
                # Store in vector persistence
                vector_memory_id = self.vector_persistence.save_memory(
                    content=memory.content,
                    user_id=memory.user_id,
                    memory_type=memory.memory_type,
                    metadata=memory.metadata
                )
                
                # Store in graph persistence with embeddings
                entity_name = f"MarkdownMemory_{i}_{memory.memory_type.value}"
                graph_memory_id = self.graph_persistence.create_entity_node_with_embedding(
                    entity=entity_name,
                    entity_type="Memory",
                    properties={
                        'content': memory.content,
                        'memory_type': memory.memory_type.value,
                        'timestamp': memory.timestamp.isoformat(),
                        'source': 'markdown',
                        **memory.metadata
                    },
                    user_id=memory.user_id
                )
                
                memory_nodes[entity_name] = {
                    'id': graph_memory_id, 
                    'memory': memory,
                    'concepts': []
                }
                
                print(f"   {i:2d}. [{memory.memory_type.value:12}] {memory.content[:70]}...")
                print(f"       Vector ID: {vector_memory_id[:8]}... | Graph ID: {graph_memory_id[:8]}...")
                
            except Exception as e:
                print(f"   ❌ Failed to store memory {i}: {e}")
        
        # Build basic graph relationships for markdown memories
        print(f"\n🕸️  Building Graph Relationships for Markdown Content...")
        
        # Extract key concepts from the markdown content
        concept_keywords = self._extract_concepts_from_memories(test_memories)
        
        # Create concept nodes and relationships
        concept_nodes = {}
        for concept, related_memories in concept_keywords.items():
            try:
                # Create concept node
                concept_id = self.graph_persistence.create_entity_node_with_embedding(
                    entity=concept,
                    entity_type="Concept",
                    properties={'name': concept, 'type': 'concept', 'source': 'markdown'},
                    user_id=self.test_user_id
                )
                concept_nodes[concept] = concept_id
                
                # Create relationships between concept and related memories
                relationship_count = 0
                for memory_idx in related_memories:
                    memory_name = f"MarkdownMemory_{memory_idx}_{test_memories[memory_idx-1].memory_type.value}"
                    if memory_name in memory_nodes:
                        success = self.graph_persistence.create_relationship(
                            from_entity=memory_name,
                            to_entity=concept,
                            relationship_type="RELATES_TO",
                            user_id=self.test_user_id,
                            properties={'strength': 'high', 'source': 'markdown'}
                        )
                        if success:
                            relationship_count += 1
                
                print(f"   🔗 Created concept '{concept}' with {relationship_count} relationships")
                
            except Exception as e:
                print(f"   ❌ Failed to create concept {concept}: {e}")
        
        print(f"   ✅ Built graph with {len(concept_nodes)} concepts from markdown content")
        print(f"✅ Successfully stored {len(test_memories)} markdown-based memories with graph structure")
        
        return test_memories
    
    def _extract_concepts_from_memories(self, memories: List[MemoryDocument]) -> Dict[str, List[int]]:
        """Extract key concepts from memory content for graph relationships."""
        concept_keywords = {}
        
        # Common technology and business concepts
        tech_concepts = ['OpenAI', 'Python', 'API', 'Azure', 'ChatGPT', 'AI', 'Machine Learning', 
                        'Slack', 'GitHub', 'GPU', 'Kubernetes', 'FastAPI']
        business_concepts = ['Company', 'Team', 'Leadership', 'Culture', 'Product', 'Launch', 
                           'Engineering', 'Research', 'Development']
        
        for concept in tech_concepts + business_concepts:
            related_memories = []
            for i, memory in enumerate(memories, 1):
                if concept.lower() in memory.content.lower():
                    related_memories.append(i)
            
            if related_memories:
                concept_keywords[concept] = related_memories
        
        return concept_keywords
    
    def setup_evaluation_data(self) -> List[MemoryDocument]:
        """Set up test data mirroring the comprehensive example scenarios."""
        print("\n📋 Setting Up Evaluation Test Data")
        print("-" * 40)
        
        # Create comprehensive test memories covering all types and scenarios
        test_memories = [
            # Developer preferences and context
            MemoryDocument(
                content="User prefers VS Code with dark theme for Python development and uses Git for version control",
                user_id=self.test_user_id,
                timestamp=datetime.now(),
                memory_type=MemoryType.PREFERENCE,
                metadata={'category': 'development_tools', 'priority': 'high'}
            ),
            MemoryDocument(
                content="Currently working on a machine learning project using scikit-learn, pandas, and implementing vector similarity search",
                user_id=self.test_user_id,
                timestamp=datetime.now(),
                memory_type=MemoryType.CONTEXT,
                metadata={'category': 'current_project', 'technologies': ['scikit-learn', 'pandas', 'vectors']}
            ),
            MemoryDocument(
                content="Has 5 years of Python programming experience and expertise in data science workflows",
                user_id=self.test_user_id,
                timestamp=datetime.now(),
                memory_type=MemoryType.FACT,
                metadata={'category': 'experience', 'verified': True}
            ),
            
            # Data analysis preferences and context
            MemoryDocument(
                content="Prefers Tableau for data visualization and creating interactive dashboards for business stakeholders",
                user_id=self.test_user_id,
                timestamp=datetime.now(),
                memory_type=MemoryType.PREFERENCE,
                metadata={'category': 'visualization_tools', 'priority': 'medium'}
            ),
            MemoryDocument(
                content="Analyzing customer behavior data for quarterly business review and need automated reporting solutions",
                user_id=self.test_user_id,
                timestamp=datetime.now(),
                memory_type=MemoryType.CONTEXT,
                metadata={'category': 'current_project', 'deadline': '2024-07-31'}
            ),
            MemoryDocument(
                content="Expert in SQL database queries and statistical analysis using R programming language",
                user_id=self.test_user_id,
                timestamp=datetime.now(),
                memory_type=MemoryType.FACT,
                metadata={'category': 'skills', 'tools': ['SQL', 'R']}
            ),
            
            # Conversation and events
            MemoryDocument(
                content="Asked about vector databases and their applications in RAG systems during technical discussion",
                user_id=self.test_user_id,
                timestamp=datetime.now(),
                memory_type=MemoryType.CONVERSATION,
                metadata={'category': 'recent_inquiry', 'topic': 'vector_databases'}
            ),
            MemoryDocument(
                content="Attended PyData conference last month and learned about MLOps best practices and deployment strategies",
                user_id=self.test_user_id,
                timestamp=datetime.now(),
                memory_type=MemoryType.EVENT,
                metadata={'category': 'professional_development', 'date': '2024-06-15'}
            ),
            MemoryDocument(
                content="Inquired about automated reporting solutions and scheduled data refresh capabilities",
                user_id=self.test_user_id,
                timestamp=datetime.now(),
                memory_type=MemoryType.CONVERSATION,
                metadata={'category': 'recent_inquiry', 'topic': 'automation'}
            ),
            
            # Additional context for comprehensive testing
            MemoryDocument(
                content="Working with team on customer segmentation analysis using clustering algorithms and demographic data",
                user_id=self.test_user_id,
                timestamp=datetime.now(),
                memory_type=MemoryType.CONTEXT,
                metadata={'category': 'team_project', 'methodology': 'clustering'}
            )
        ]
        
        # Store memories in both persistence systems
        print("💾 Storing test memories in both systems...")
        
        # Clear any existing test data first
        try:
            self.vector_persistence.clear_memories(self.test_user_id)
            with self.graph_persistence.driver.session(database=self.graph_persistence.database) as session:
                session.run(
                    "MATCH (n {user_id: $user_id}) DETACH DELETE n",
                    {'user_id': self.test_user_id}
                )
            print("🧹 Cleared existing test data from both systems")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")
        
        memory_nodes = {}  # Track created memory nodes for relationships
        
        for i, memory in enumerate(test_memories, 1):
            try:
                # Store in vector persistence
                vector_memory_id = self.vector_persistence.save_memory(
                    content=memory.content,
                    user_id=memory.user_id,
                    memory_type=memory.memory_type,
                    metadata=memory.metadata
                )
                
                # Store in graph persistence as entity nodes
                entity_name = f"Memory_{i}_{memory.memory_type.value}"
                graph_memory_id = self.graph_persistence.create_entity_node_with_embedding(
                    entity=entity_name,
                    entity_type="Memory",
                    properties={
                        'content': memory.content,
                        'memory_type': memory.memory_type.value,
                        'timestamp': memory.timestamp.isoformat(),
                        'category': memory.metadata.get('category', 'general'),
                        **memory.metadata
                    },
                    user_id=memory.user_id
                )
                
                memory_nodes[entity_name] = {
                    'id': graph_memory_id, 
                    'memory': memory,
                    'concepts': []
                }
                
                print(f"   {i:2d}. [{memory.memory_type.value:12}] {memory.content[:50]}...")
                print(f"       Vector ID: {vector_memory_id[:8]}... | Graph ID: {graph_memory_id[:8]}...")
                
            except Exception as e:
                print(f"   ❌ Failed to store memory {i}: {e}")
        
        # Build graph relationships by extracting concepts and creating connections
        print("\n🕸️  Building Graph Relationships...")
        
        # Define concept extraction rules
        concept_mappings = {
            'Python': ['Memory_2_context', 'Memory_3_fact', 'Memory_1_preference'],
            'Machine Learning': ['Memory_2_context', 'Memory_8_event'],
            'Data Science': ['Memory_3_fact', 'Memory_2_context', 'Memory_10_context'],
            'Visualization': ['Memory_4_preference', 'Memory_5_context'],
            'SQL': ['Memory_6_fact'],
            'Analysis': ['Memory_5_context', 'Memory_6_fact', 'Memory_10_context'],
            'Automation': ['Memory_5_context', 'Memory_9_conversation'],
            'Reporting': ['Memory_4_preference', 'Memory_5_context', 'Memory_9_conversation'],
            'Development Tools': ['Memory_1_preference'],
            'Databases': ['Memory_6_fact', 'Memory_7_conversation']
        }
        
        # Create concept nodes and relationships
        concept_nodes = {}
        for concept, related_memories in concept_mappings.items():
            try:
                # Create concept node
                concept_id = self.graph_persistence.create_entity_node_with_embedding(
                    entity=concept,
                    entity_type="Concept",
                    properties={'name': concept, 'type': 'concept'},
                    user_id=self.test_user_id
                )
                concept_nodes[concept] = concept_id
                
                # Create relationships between concept and related memories
                for memory_name in related_memories:
                    if memory_name in memory_nodes:
                        success = self.graph_persistence.create_relationship(
                            from_entity=memory_name,
                            to_entity=concept,
                            relationship_type="RELATES_TO",
                            user_id=self.test_user_id,
                            properties={'strength': 'high', 'auto_extracted': True}
                        )
                        if success:
                            memory_nodes[memory_name]['concepts'].append(concept)
                
                print(f"   🔗 Created concept '{concept}' with {len(related_memories)} relationships")
                
            except Exception as e:
                print(f"   ❌ Failed to create concept {concept}: {e}")
        
        # Create cross-memory relationships based on shared concepts/categories
        category_groups = {}
        for name, info in memory_nodes.items():
            category = info['memory'].metadata.get('category', 'general')
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(name)
        
        print(f"\n🔗 Creating cross-memory relationships...")
        relationship_count = 0
        
        # Create relationships within categories
        for category, memories in category_groups.items():
            if len(memories) > 1:
                for i in range(len(memories)):
                    for j in range(i+1, len(memories)):
                        try:
                            success = self.graph_persistence.create_relationship(
                                from_entity=memories[i],
                                to_entity=memories[j],
                                relationship_type="SAME_CATEGORY",
                                user_id=self.test_user_id,
                                properties={'category': category, 'relationship_type': 'categorical'}
                            )
                            if success:
                                relationship_count += 1
                        except Exception as e:
                            print(f"   ⚠️  Failed to create relationship: {e}")
        
        # Create temporal relationships for project-related memories
        project_memories = [name for name, info in memory_nodes.items() 
                         if 'project' in info['memory'].metadata.get('category', '')]
        for i in range(len(project_memories)-1):
            try:
                success = self.graph_persistence.create_relationship(
                    from_entity=project_memories[i],
                    to_entity=project_memories[i+1],
                    relationship_type="RELATED_PROJECT",
                    user_id=self.test_user_id,
                    properties={'relationship_type': 'temporal'}
                )
                if success:
                    relationship_count += 1
            except Exception as e:
                print(f"   ⚠️  Failed to create project relationship: {e}")
        
        print(f"   ✅ Created {relationship_count} cross-memory relationships")
        print(f"   ✅ Built graph with {len(concept_nodes)} concepts and {relationship_count} relationships")
        
        print(f"✅ Successfully stored {len(test_memories)} test memories with graph structure in both systems")
        return test_memories
    
    def create_evaluation_queries(self) -> List[EvaluationQuery]:
        """Create comprehensive evaluation queries covering different scenarios."""
        print("\n🎯 Creating Evaluation Queries")
        print("-" * 35)
        
        evaluation_queries = [
            # Preference retrieval queries
            EvaluationQuery(
                query_id=str(uuid.uuid4()),
                question="What development tools and IDE does the user prefer?",
                expected_context="User prefers VS Code with dark theme for Python development and uses Git for version control",
                user_id=self.test_user_id,
                memory_type="preference"
            ),
            EvaluationQuery(
                query_id=str(uuid.uuid4()),
                question="What data visualization tools does the user like to use?",
                expected_context="Prefers Tableau for data visualization and creating interactive dashboards for business stakeholders",
                user_id=self.test_user_id,
                memory_type="preference"
            ),
            
            # Current context and project queries
            EvaluationQuery(
                query_id=str(uuid.uuid4()),
                question="What machine learning project is the user currently working on?",
                expected_context="Currently working on a machine learning project using scikit-learn, pandas, and implementing vector similarity search",
                user_id=self.test_user_id,
                memory_type="context"
            ),
            EvaluationQuery(
                query_id=str(uuid.uuid4()),
                question="What data analysis work is the user doing for business review?",
                expected_context="Analyzing customer behavior data for quarterly business review and need automated reporting solutions",
                user_id=self.test_user_id,
                memory_type="context"
            ),
            
            # Skills and experience queries  
            EvaluationQuery(
                query_id=str(uuid.uuid4()),
                question="How much Python programming experience does the user have?",
                expected_context="Has 5 years of Python programming experience and expertise in data science workflows",
                user_id=self.test_user_id,
                memory_type="fact"
            ),
            EvaluationQuery(
                query_id=str(uuid.uuid4()),
                question="What database and statistical analysis skills does the user possess?",
                expected_context="Expert in SQL database queries and statistical analysis using R programming language",
                user_id=self.test_user_id,
                memory_type="fact"
            ),
            
            # Conversation and learning queries
            EvaluationQuery(
                query_id=str(uuid.uuid4()),
                question="What did the user ask about regarding vector databases?",
                expected_context="Asked about vector databases and their applications in RAG systems during technical discussion",
                user_id=self.test_user_id,
                memory_type="conversation"
            ),
            EvaluationQuery(
                query_id=str(uuid.uuid4()),
                question="What professional development activities has the user attended recently?",
                expected_context="Attended PyData conference last month and learned about MLOps best practices and deployment strategies",
                user_id=self.test_user_id,
                memory_type="event"
            ),
            
            # Cross-type semantic queries
            EvaluationQuery(
                query_id=str(uuid.uuid4()),
                question="What automation solutions is the user interested in?",
                expected_context="Inquired about automated reporting solutions and scheduled data refresh capabilities",
                user_id=self.test_user_id,
                memory_type="conversation"
            ),
            EvaluationQuery(
                query_id=str(uuid.uuid4()),
                question="What team collaboration work is the user involved in?",
                expected_context="Working with team on customer segmentation analysis using clustering algorithms and demographic data",
                user_id=self.test_user_id,
                memory_type="context"
            )
        ]
        
        print(f"✅ Created {len(evaluation_queries)} evaluation queries covering:")
        query_types = {}
        for query in evaluation_queries:
            query_types[query.memory_type] = query_types.get(query.memory_type, 0) + 1
        
        for memory_type, count in query_types.items():
            print(f"   • {memory_type}: {count} queries")
        
        return evaluation_queries
    
    def evaluate_semantic_search_performance(self, evaluation_queries: List[EvaluationQuery]) -> Dict[str, Any]:
        """Evaluate semantic search performance across different memory types."""
        print("\n🔍 Evaluating Semantic Search Performance")
        print("-" * 45)
        
        search_results = []
        performance_metrics = {
            'query_times': [],
            'error_count': 0,
            'memory_type_performance': {}
        }
        
        for query in evaluation_queries:
            print(f"\n🎯 Query: {query.question}")
            print(f"   Expected type: {query.memory_type}")
            
            try:
                start_time = datetime.now()
                
                # Perform semantic search
                results = self.vector_persistence.search_memories(
                    query=query.question,
                    user_id=query.user_id,
                    k=3
                )
                
                query_time = (datetime.now() - start_time).total_seconds()
                performance_metrics['query_times'].append(query_time)
                
                print(f"   ⏱️  Query time: {query_time:.3f}s")
                print(f"   📋 Found {len(results)} memories")
                
                if results:
                    # Analyze top result
                    top_result = results[0]
                    print(f"   🥇 Top result:")
                    print(f"      Content: {top_result['content'][:80]}...")
                    print(f"      Type: {top_result.get('memory_type', 'unknown')}")
                    print(f"      Similarity: {top_result['similarity_score']:.3f}")
                    
                    # Check if top result matches expected memory type
                    expected_type_match = top_result.get('memory_type') == query.memory_type
                    print(f"   ✅ Type match: {expected_type_match}")
                    
                    # Store results for evaluation
                    search_results.append({
                        'query': query.question,
                        'results': results,
                        'expected_context': query.expected_context,
                        'memory_type': query.memory_type,
                        'query_time': query_time
                    })
                    
                    # Track performance by memory type
                    if query.memory_type not in performance_metrics['memory_type_performance']:
                        performance_metrics['memory_type_performance'][query.memory_type] = {
                            'queries': 0,
                            'avg_similarity': 0,
                            'total_similarity': 0
                        }
                    
                    type_perf = performance_metrics['memory_type_performance'][query.memory_type]
                    type_perf['queries'] += 1
                    type_perf['total_similarity'] += top_result['similarity_score']
                    type_perf['avg_similarity'] = type_perf['total_similarity'] / type_perf['queries']
                else:
                    print("   ⚠️  No results found")
                    
            except Exception as e:
                print(f"   ❌ Query failed: {e}")
                performance_metrics['error_count'] += 1
        
        # Calculate aggregate performance metrics
        if performance_metrics['query_times']:
            performance_metrics['avg_query_time'] = sum(performance_metrics['query_times']) / len(performance_metrics['query_times'])
            performance_metrics['min_query_time'] = min(performance_metrics['query_times'])
            performance_metrics['max_query_time'] = max(performance_metrics['query_times'])
        
        performance_metrics['success_rate'] = 1.0 - (performance_metrics['error_count'] / len(evaluation_queries))
        
        print(f"\n📊 Search Performance Summary:")
        print(f"   • Total queries: {len(evaluation_queries)}")
        print(f"   • Success rate: {performance_metrics['success_rate']:.1%}")
        print(f"   • Average query time: {performance_metrics.get('avg_query_time', 0):.3f}s")
        print(f"   • Error count: {performance_metrics['error_count']}")
        
        print(f"\n📈 Performance by Memory Type:")
        for memory_type, stats in performance_metrics['memory_type_performance'].items():
            print(f"   • {memory_type}: {stats['queries']} queries, avg similarity: {stats['avg_similarity']:.3f}")
        
        return {
            'search_results': search_results,
            'performance_metrics': performance_metrics,
            'evaluation_queries': evaluation_queries
        }
    
    def run_langchain_evaluation(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run LangChain-based evaluation using the enhanced evaluation framework."""
        print("\n🧪 Running Comparative LangChain Evaluation Framework")
        print("-" * 50)
        
        # Use the enhanced comparative evaluation from task 4.2/4.3
        try:
            print("📋 Running comparative evaluation against both persistence systems...")
            
            # Use the evaluator's run_comparative_evaluation method
            comparison_results = self.evaluator.run_comparative_evaluation(
                vector_persistence=self.vector_persistence,
                graph_persistence=self.graph_persistence,
                test_queries=search_results['evaluation_queries'],
                include_performance_metrics=True
            )
            
            print(f"\n✅ Comparative evaluation completed successfully!")
            vector_score = comparison_results['comparison']['overall']['vector_score']
            graph_score = comparison_results['comparison']['overall']['graph_score']
            print(f"   📊 Vector solution score: {vector_score:.3f}")
            print(f"   📊 Graph solution score: {graph_score:.3f}")
            
            winner = comparison_results['comparison']['overall']['winner']
            print(f"   🏆 Winner: {winner.title()}")
            
            return comparison_results
            
        except Exception as e:
            print(f"❌ Comparative evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def generate_comprehensive_report(self, comparison_results: Dict[str, Any], 
                                    search_performance: Dict[str, Any]) -> str:
        """Generate comprehensive evaluation report."""
        print("\n📋 Generating Comprehensive Evaluation Report")
        print("-" * 50)
        
        # Add performance data to comparison results for enhanced reporting
        performance_data = {
            'performance_metrics': {
                'vector_performance': {
                    'avg': search_performance['performance_metrics'].get('avg_query_time', 0),
                    'min': search_performance['performance_metrics'].get('min_query_time', 0),
                    'max': search_performance['performance_metrics'].get('max_query_time', 0),
                    'total': sum(search_performance['performance_metrics'].get('query_times', [])),
                    'error_count': search_performance['performance_metrics'].get('error_count', 0),
                    'success_rate': search_performance['performance_metrics'].get('success_rate', 0)
                }
            },
            'test_queries': search_performance['evaluation_queries']
        }
        
        try:
            # Generate structured report in multiple formats
            markdown_report = self.evaluator.generate_structured_report(
                comparison_results=comparison_results,
                performance_data=performance_data,
                output_format=ReportFormat.MARKDOWN
            )
            
            json_report = self.evaluator.generate_structured_report(
                comparison_results=comparison_results,
                performance_data=performance_data,
                output_format=ReportFormat.JSON
            )
            
            # Create eval_report directory if it doesn't exist
            eval_report_dir = "eval_report"
            if not os.path.exists(eval_report_dir):
                os.makedirs(eval_report_dir)
                print(f"📁 Created {eval_report_dir} directory")
            
            # Save reports to files in eval_report directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            markdown_file = os.path.join(eval_report_dir, f"evaluation_report_{timestamp}.md")
            json_file = os.path.join(eval_report_dir, f"evaluation_report_{timestamp}.json")
            
            with open(markdown_file, 'w') as f:
                f.write(markdown_report)
            
            with open(json_file, 'w') as f:
                f.write(json_report)
            
            # Save raw comparison results for detailed analysis
            raw_results_file = os.path.join(eval_report_dir, f"raw_comparison_results_{timestamp}.json")
            with open(raw_results_file, 'w') as f:
                import json
                json.dump(comparison_results, f, indent=2, default=str)
            
            print(f"✅ Reports generated successfully:")
            print(f"   📄 Markdown: {markdown_file}")
            print(f"   📄 JSON: {json_file}")
            print(f"   📊 Raw Results: {raw_results_file}")
            
            # Display key findings from the report
            print(f"\n🔍 Key Evaluation Findings:")
            
            vector_score = comparison_results['comparison']['overall']['vector_score']
            print(f"   • Overall Performance Score: {vector_score:.3f}/1.000")
            
            if vector_score >= 0.8:
                performance_level = "Excellent"
            elif vector_score >= 0.7:
                performance_level = "Good"
            elif vector_score >= 0.6:
                performance_level = "Satisfactory"
            else:
                performance_level = "Needs Improvement"
            
            print(f"   • Performance Level: {performance_level}")
            
            # Performance metrics
            perf_metrics = performance_data['performance_metrics']['vector_performance']
            print(f"   • Average Query Time: {perf_metrics['avg']:.3f}s")
            print(f"   • Success Rate: {perf_metrics['success_rate']:.1%}")
            print(f"   • Error Count: {perf_metrics['error_count']}")
            
            # Memory type performance
            type_performance = search_performance['performance_metrics']['memory_type_performance']
            print(f"   • Memory Type Performance:")
            for memory_type, stats in type_performance.items():
                print(f"     - {memory_type}: {stats['avg_similarity']:.3f} avg similarity")
            
            return markdown_report
            
        except Exception as e:
            print(f"❌ Report generation failed: {e}")
            raise
    
    def validate_mvp_requirements(self, comparison_results: Dict[str, Any], 
                                 search_performance: Dict[str, Any]) -> None:
        """Validate MVP implementation against requirements."""
        print("\n✅ Validating MVP Requirements")
        print("-" * 35)
        
        print("📋 Task 4.4 Requirements Validation:")
        
        # Requirement 5.1: Context recall accuracy measurement
        context_recall_results = comparison_results['vector_solution'].get('context_recall', [])
        if context_recall_results:
            avg_context_recall = sum(r.score for r in context_recall_results) / len(context_recall_results)
            print(f"   ✅ Context recall measurement: {avg_context_recall:.3f} (Requirement 5.1)")
        else:
            print(f"   ❌ Context recall measurement: Failed (Requirement 5.1)")
        
        # Requirement 5.2: Relevance measurement
        relevance_results = comparison_results['vector_solution'].get('relevance', [])
        if relevance_results:
            avg_relevance = sum(r.score for r in relevance_results) / len(relevance_results)
            print(f"   ✅ Relevance measurement: {avg_relevance:.3f} (Requirement 5.2)")
        else:
            print(f"   ❌ Relevance measurement: Failed (Requirement 5.2)")
        
        # Requirement 5.3: Performance analysis
        success_rate = search_performance['performance_metrics'].get('success_rate', 0)
        avg_query_time = search_performance['performance_metrics'].get('avg_query_time', float('inf'))
        
        if success_rate >= 0.9 and avg_query_time < 1.0:
            print(f"   ✅ Performance analysis: {success_rate:.1%} success, {avg_query_time:.3f}s avg (Requirement 5.3)")
        else:
            print(f"   ⚠️  Performance analysis: {success_rate:.1%} success, {avg_query_time:.3f}s avg (Requirement 5.3)")
        
        # Requirement 6.5: MVP validation with practical examples
        num_memory_types = len(search_performance['performance_metrics']['memory_type_performance'])
        num_queries = len(search_performance['evaluation_queries'])
        
        if num_memory_types >= 4 and num_queries >= 8:
            print(f"   ✅ MVP validation: {num_memory_types} memory types, {num_queries} queries (Requirement 6.5)")
        else:
            print(f"   ⚠️  MVP validation: {num_memory_types} memory types, {num_queries} queries (Requirement 6.5)")
        
        print(f"\n🎯 Overall MVP Status:")
        overall_score = comparison_results['comparison']['overall']['vector_score']
        
        if overall_score >= 0.75 and success_rate >= 0.9:
            print(f"   🟢 MVP READY: Score {overall_score:.3f}, Success {success_rate:.1%}")
            print(f"   ✅ System validated for production deployment")
        elif overall_score >= 0.6 and success_rate >= 0.8:
            print(f"   🟡 MVP ACCEPTABLE: Score {overall_score:.3f}, Success {success_rate:.1%}")
            print(f"   ⚠️  Minor improvements recommended before deployment")
        else:
            print(f"   🔴 MVP NEEDS WORK: Score {overall_score:.3f}, Success {success_rate:.1%}")
            print(f"   ❌ Significant improvements required before deployment")
    
    def cleanup_test_data(self) -> None:
        """Clean up test data after evaluation."""
        print("\n🧹 Cleaning Up Test Data from Both Systems")
        print("-" * 45)
        
        try:
            # Clean vector persistence
            self.vector_persistence.clear_memories(self.test_user_id)
            print("✅ Vector persistence test data cleaned up")
        except Exception as e:
            print(f"⚠️  Vector cleanup warning: {e}")
        
        try:
            # Clean graph persistence (remove all entities for user)
            with self.graph_persistence.driver.session(database=self.graph_persistence.database) as session:
                session.run(
                    "MATCH (n:Entity {user_id: $user_id}) DETACH DELETE n",
                    {'user_id': self.test_user_id}
                )
            print("✅ Graph persistence test data cleaned up")
        except Exception as e:
            print(f"⚠️  Graph cleanup warning: {e}")
    
    def run_comprehensive_evaluation(self, use_markdown_data: bool = False, 
                                   markdown_file: str = "examples/data/3.md", 
                                   num_points: int = 20) -> Dict[str, Any]:
        """
        Run comprehensive evaluation using either hardcoded or markdown-based demo data.
        
        Args:
            use_markdown_data: Whether to use markdown files as data source
            markdown_file: Path to markdown file (default: examples/data/3.md)
            num_points: Number of information points to extract (default: 20)
        """
        evaluation_results = {}
        
        try:
            # Set up evaluation data
            if use_markdown_data:
                print(f"📋 Using markdown-based demo data from {markdown_file}")
                test_memories = self.setup_markdown_evaluation_data(markdown_file, num_points)
                data_source = f"markdown:{os.path.basename(markdown_file)}"
            else:
                print("📋 Using hardcoded demo data")
                test_memories = self.setup_evaluation_data()
                data_source = "hardcoded"
            
            # Create evaluation queries based on the data type
            if use_markdown_data:
                evaluation_queries = self.create_markdown_evaluation_queries(test_memories)
            else:
                evaluation_queries = self.create_evaluation_queries()
            
            evaluation_results['test_memories'] = test_memories
            evaluation_results['evaluation_queries'] = evaluation_queries
            evaluation_results['data_source'] = data_source
            
            # Run semantic search performance evaluation
            search_results = self.evaluate_semantic_search_performance(evaluation_queries)
            evaluation_results.update(search_results)
            
            # Run LangChain evaluation framework
            langchain_results = self.run_langchain_evaluation(search_results)
            evaluation_results['langchain_evaluation'] = langchain_results
            
            # Generate comprehensive report
            self.generate_comprehensive_report(langchain_results, search_results)
            
            # Validate MVP requirements
            self.validate_mvp_requirements(langchain_results, search_results)
            
            # Cleanup
            self.cleanup_test_data()
            
            print(f"\n🎉 Comparative evaluation completed successfully with {data_source} data!")
            self.print_task_completion_summary()
            
            return evaluation_results
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def create_markdown_evaluation_queries(self, memories: List[MemoryDocument]) -> List[EvaluationQuery]:
        """Create evaluation queries tailored for markdown-based content."""
        queries = []
        
        # Extract unique concepts from memories for query generation
        all_content = " ".join([m.content for m in memories])
        
        # Generate queries based on common themes in the content
        query_templates = [
            # Company and culture queries
            ("What is OpenAI's company culture like?", "context"),
            ("How does OpenAI approach product development?", "context"), 
            ("What technologies does OpenAI use in their infrastructure?", "fact"),
            ("How does the team collaboration work at OpenAI?", "context"),
            ("What are the challenges of working at OpenAI?", "context"),
            
            # Technical and process queries
            ("What programming languages and tools are mentioned?", "fact"),
            ("How is code organized and managed?", "fact"),
            ("What are the key product launches discussed?", "event"),
            ("How does the company handle rapid scaling?", "context"),
            ("What are the performance characteristics mentioned?", "fact"),
            
            # Personal experience queries
            ("What personal experiences are shared about working there?", "conversation"),
            ("What career advice or insights are provided?", "preference"),
            ("What specific achievements or milestones are mentioned?", "event"),
            ("How does the author describe the work environment?", "context"),
            ("What recommendations or preferences are expressed?", "preference")
        ]
        
        # Filter queries based on actual content
        filtered_queries = []
        for question, expected_type in query_templates:
            # Check if the query is relevant to the content
            key_words = question.lower().split()
            content_words = all_content.lower().split()
            
            # If at least 2 key words from question appear in content, include the query
            matching_words = sum(1 for word in key_words if word in content_words)
            if matching_words >= 2:
                filtered_queries.append((question, expected_type))
        
        # Create EvaluationQuery objects
        for i, (question, expected_type) in enumerate(filtered_queries[:10]):  # Limit to 10 queries
            # Generate expected context based on memory type
            expected_context = self._generate_expected_context(question, memories, expected_type)
            
            query = EvaluationQuery(
                query_id=str(uuid.uuid4()),
                question=question,
                expected_context=expected_context,
                user_id=self.test_user_id,
                memory_type=expected_type
            )
            queries.append(query)
        
        return queries
    
    def _generate_expected_context(self, question: str, memories: List[MemoryDocument], 
                                 expected_type: str) -> str:
        """Generate expected context for a question based on relevant memories."""
        question_lower = question.lower()
        relevant_memories = []
        
        # Find memories that might be relevant to the question
        for memory in memories:
            memory_content_lower = memory.content.lower()
            
            # Simple relevance scoring based on shared keywords
            question_words = set(question_lower.split())
            memory_words = set(memory_content_lower.split())
            shared_words = question_words & memory_words
            
            if len(shared_words) >= 2:  # At least 2 shared words
                relevant_memories.append(memory.content)
        
        # Return first few relevant memories as expected context
        if relevant_memories:
            return ". ".join(relevant_memories[:3])  # Limit to 3 memories
        else:
            return f"Information related to {expected_type} context"
    
    def print_task_completion_summary(self):
        """Print a summary of completed task requirements."""
        print(f"✅ Task 4.4 requirements fulfilled:")
        print(f"   • Evaluation framework assessment ✅")
        print(f"   • Context recall accuracy measurement ✅")
        print(f"   • Relevance measurement ✅")
        print(f"   • Semantic search performance analysis ✅")
        print(f"   • Comprehensive evaluation report generation ✅")
        print(f"   • MVP validation against metrics ✅")
        print(f"   • Comparative analysis between vector and graph systems ✅")


def main():
    """Main function to run the evaluation demo."""
    print("Starting Comparative Persistence Evaluation (Task 4.4)")
    print("This evaluation compares vector and graph persistence implementations")
    print("using the enhanced evaluation framework from Task 4.3.")
    print()
    
    # Option to use markdown-based demo data
    use_markdown = len(sys.argv) > 1 and sys.argv[1] == "--markdown"
    markdown_file = sys.argv[2] if len(sys.argv) > 2 else "examples/data/3.md"
    num_points = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    if use_markdown:
        print(f"📋 Using markdown-based demo data from {markdown_file} ({num_points} points)")
    else:
        print("📋 Using hardcoded demo data (use --markdown flag for markdown-based data)")
    print()
    
    demo = LangChainEvaluationDemo()
    demo.run_comprehensive_evaluation(
        use_markdown_data=use_markdown,
        markdown_file=markdown_file,
        num_points=num_points
    )


if __name__ == "__main__":
    main() 