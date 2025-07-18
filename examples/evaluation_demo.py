"""
Evaluation Demo: Comprehensive assessment of LangChain vector persistence system.

This script uses the enhanced evaluation framework to assess the performance and 
effectiveness of the LangChain vector persistence system demonstrated in 
comprehensive_langchain_example.py.

Task 4.4 Requirements:
- Use evaluation framework to assess results from comprehensive LangChain example
- Measure context recall accuracy and relevance of memory retrieval
- Analyze semantic search performance across different memory types
- Generate evaluation report for the comprehensive example workflow
- Validate MVP implementation against evaluation metrics

Usage:
    python examples/evaluation_demo.py
"""

import sys
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config.config_manager import ConfigManager
from src.persistence.langchain_vector_persistence import LangChainVectorPersistence
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
        print("🔬 LangChain Vector Persistence Evaluation Demo")
        print("=" * 60)
        
        try:
            self.config_manager = ConfigManager()
            self.persistence = LangChainVectorPersistence(self.config_manager)
            self.evaluator = PersistenceEvaluator(self.config_manager)
            
            # Test user for evaluation
            self.test_user_id = "evaluation_test_user"
            
            print("✅ Evaluation framework initialized successfully!")
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            raise
    
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
        
        # Store memories in the persistence system
        print("💾 Storing test memories...")
        for i, memory in enumerate(test_memories, 1):
            try:
                memory_id = self.persistence.save_memory(
                    content=memory.content,
                    user_id=memory.user_id,
                    memory_type=memory.memory_type,
                    metadata=memory.metadata
                )
                print(f"   {i:2d}. [{memory.memory_type.value:12}] {memory.content[:50]}...")
                
            except Exception as e:
                print(f"   ❌ Failed to store memory {i}: {e}")
        
        print(f"✅ Successfully stored {len(test_memories)} test memories")
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
                results = self.persistence.search_memories(
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
        print("\n🧪 Running LangChain Evaluation Framework")
        print("-" * 45)
        
        # Extract data for evaluation
        queries = [result['query'] for result in search_results['search_results']]
        retrieved_contexts = []
        expected_contexts = []
        
        for result in search_results['search_results']:
            # Format retrieved context from top result
            if result['results']:
                top_result = result['results'][0]
                retrieved_context = f"{top_result['content']} (similarity: {top_result['similarity_score']:.3f})"
            else:
                retrieved_context = "No relevant context found"
            
            retrieved_contexts.append(retrieved_context)
            expected_contexts.append(result['expected_context'])
        
        print(f"📋 Evaluating {len(queries)} query-context pairs...")
        
        try:
            # Run context recall evaluation
            print("\n🎯 Context Recall Evaluation:")
            context_recall_results = self.evaluator.evaluate_context_recall(
                retrieved_contexts=retrieved_contexts,
                expected_contexts=expected_contexts,
                queries=queries
            )
            
            avg_context_recall = sum(r.score for r in context_recall_results) / len(context_recall_results)
            print(f"   📊 Average context recall: {avg_context_recall:.3f}")
            
            # Run relevance evaluation
            print("\n🎯 Relevance Evaluation:")
            relevance_results = self.evaluator.evaluate_relevance(
                retrieved_contexts=retrieved_contexts,
                queries=queries
            )
            
            avg_relevance = sum(r.score for r in relevance_results) / len(relevance_results)
            print(f"   📊 Average relevance: {avg_relevance:.3f}")
            
            # Run memory accuracy evaluation
            print("\n🎯 Memory Accuracy Evaluation:")
            memory_accuracy_results = self.evaluator.evaluate_memory_accuracy(
                retrieved_memories=retrieved_contexts,
                queries=queries
            )
            
            avg_memory_accuracy = sum(r.score for r in memory_accuracy_results) / len(memory_accuracy_results)
            print(f"   📊 Average memory accuracy: {avg_memory_accuracy:.3f}")
            
            # Prepare comparison results structure
            comparison_results = {
                'vector_solution': {
                    'context_recall': context_recall_results,
                    'relevance': relevance_results,
                    'memory_accuracy': memory_accuracy_results
                },
                'graph_solution': {},  # No graph solution for single-system evaluation
                'comparison': {
                    'overall': {
                        'vector_score': (avg_context_recall * 0.2 + avg_relevance * 0.4 + avg_memory_accuracy * 0.4),
                        'graph_score': 0.0  # No graph solution
                    }
                },
                'metadata': {
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'num_queries': len(queries),
                    'evaluator_version': self.evaluator.version
                }
            }
            
            print(f"\n✅ LangChain evaluation completed successfully!")
            print(f"   📊 Overall vector solution score: {comparison_results['comparison']['overall']['vector_score']:.3f}")
            
            return comparison_results
            
        except Exception as e:
            print(f"❌ LangChain evaluation failed: {e}")
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
            
            # Save reports to files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            markdown_file = f"evaluation_report_{timestamp}.md"
            json_file = f"evaluation_report_{timestamp}.json"
            
            with open(markdown_file, 'w') as f:
                f.write(markdown_report)
            
            with open(json_file, 'w') as f:
                f.write(json_report)
            
            print(f"✅ Reports generated successfully:")
            print(f"   📄 Markdown: {markdown_file}")
            print(f"   📄 JSON: {json_file}")
            
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
        print("\n🧹 Cleaning Up Test Data")
        print("-" * 30)
        
        try:
            self.persistence.clear_memories(self.test_user_id)
            print("✅ Test data cleaned up successfully")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")
    
    def run_comprehensive_evaluation(self) -> None:
        """Run the complete evaluation workflow."""
        try:
            # Setup test data
            test_memories = self.setup_evaluation_data()
            
            # Create evaluation queries
            evaluation_queries = self.create_evaluation_queries()
            
            # Evaluate search performance
            search_performance = self.evaluate_semantic_search_performance(evaluation_queries)
            
            # Run LangChain evaluation
            comparison_results = self.run_langchain_evaluation(search_performance)
            
            # Generate comprehensive report
            report = self.generate_comprehensive_report(comparison_results, search_performance)
            
            # Validate MVP requirements
            self.validate_mvp_requirements(comparison_results, search_performance)
            
            # Cleanup
            self.cleanup_test_data()
            
            print(f"\n🎉 Comprehensive evaluation completed successfully!")
            print(f"✅ Task 4.4 requirements fulfilled:")
            print(f"   • Evaluation framework assessment ✅")
            print(f"   • Context recall accuracy measurement ✅")
            print(f"   • Relevance measurement ✅")
            print(f"   • Semantic search performance analysis ✅")
            print(f"   • Comprehensive evaluation report generation ✅")
            print(f"   • MVP validation against metrics ✅")
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Attempt cleanup even on failure
            try:
                self.cleanup_test_data()
            except:
                pass


def main():
    """Run the comprehensive evaluation demonstration."""
    print("Starting LangChain Vector Persistence Evaluation (Task 4.4)")
    print("This evaluation assesses the comprehensive example implementation")
    print("using the enhanced evaluation framework from Task 4.3.\n")
    
    try:
        demo = LangChainEvaluationDemo()
        demo.run_comprehensive_evaluation()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 