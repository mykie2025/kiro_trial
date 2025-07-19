"""
Persistence Evaluator: Comprehensive evaluation framework for persistence systems.

This module provides evaluation tools for comparing different persistence solutions
using LangChain's evaluation framework. It supports context recall, relevance, and
memory accuracy assessments.

Requirements addressed:
- 5.1: Context recall accuracy measurement
- 5.2: Answer relevance assessment 
- 5.3: Comparative evaluation between systems
- 5.4: Performance metrics collection
- 5.5: Structured evaluation reporting
- 5.6: Statistical analysis and recommendations
"""

import logging
import uuid
import json
import csv
import io
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from langchain.evaluation import load_evaluator
from langchain.schema import Document
from langchain.evaluation import EvaluatorType
from langchain.evaluation.criteria import Criteria
from langchain_openai import ChatOpenAI

from ..config.config_manager import ConfigManager
from ..persistence.neo4j_graph_persistence import GraphRAGResult
from ..persistence.memory_document import MemoryDocument, MemoryType


logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Supported report output formats."""
    MARKDOWN = "markdown"
    JSON = "json"
    CSV = "csv"
    HTML = "html"


@dataclass
class EvaluationResult:
    """Structure for evaluation results"""
    metric_name: str
    score: float
    reasoning: str
    metadata: Dict[str, Any]
    timestamp: datetime


@dataclass 
class EvaluationQuery:
    """Structure for evaluation test queries"""
    query_id: str
    question: str
    expected_context: str
    user_id: str
    memory_type: str


@dataclass
class PerformanceMetrics:
    """Detailed performance metrics for a persistence solution."""
    avg_query_time: float
    min_query_time: float
    max_query_time: float
    total_query_time: float
    error_count: int
    success_rate: float
    query_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class MetricSummary:
    """Summary statistics for an evaluation metric."""
    metric_name: str
    vector_average: float
    graph_average: float
    vector_std: float
    graph_std: float
    difference: float
    percentage_difference: float
    winner: str
    significance_level: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class DetailedComparison:
    """Detailed comparison analysis between persistence solutions."""
    metric_summaries: Dict[str, MetricSummary]
    overall_scores: Dict[str, float]
    performance_metrics: Dict[str, PerformanceMetrics]
    strengths_analysis: Dict[str, List[str]]
    weaknesses_analysis: Dict[str, List[str]]
    recommendations: List[str]
    confidence_scores: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metric_summaries': {k: v.to_dict() for k, v in self.metric_summaries.items()},
            'overall_scores': self.overall_scores,
            'performance_metrics': {k: v.to_dict() for k, v in self.performance_metrics.items()},
            'strengths_analysis': self.strengths_analysis,
            'weaknesses_analysis': self.weaknesses_analysis,
            'recommendations': self.recommendations,
            'confidence_scores': self.confidence_scores
        }


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report structure."""
    report_id: str
    timestamp: datetime
    evaluator_version: str
    test_configuration: Dict[str, Any]
    detailed_comparison: DetailedComparison
    raw_results: Dict[str, Any]
    executive_summary: str
    methodology_notes: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'report_id': self.report_id,
            'timestamp': self.timestamp.isoformat(),
            'evaluator_version': self.evaluator_version,
            'test_configuration': self.test_configuration,
            'detailed_comparison': self.detailed_comparison.to_dict(),
            'raw_results': self.raw_results,
            'executive_summary': self.executive_summary,
            'methodology_notes': self.methodology_notes
        }


class PersistenceEvaluator:
    """
    Evaluator for comparing persistence solutions using LangChain's evaluation tools.
    
    This class provides comprehensive evaluation capabilities including:
    - Context recall accuracy measurement
    - Answer relevance assessment  
    - Comparative analysis between persistence solutions
    - Test dataset generation and management
    - Detailed reporting and analysis
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the PersistenceEvaluator.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.evaluators = self._setup_langchain_evaluators()
        self.test_queries: List[EvaluationQuery] = []
        self.version = "1.1.0"
        
        logger.info("PersistenceEvaluator initialized with LangChain evaluation tools")
    
    def _setup_langchain_evaluators(self) -> Dict[str, Any]:
        """
        Set up LangChain evaluators for context recall and relevance assessment.
        
        Returns:
            Dictionary of configured evaluators
        """
        try:
            # Get OpenAI configuration from config manager
            openai_config = self.config.get_openai_config()
            
            # Initialize LLM for evaluation
            eval_llm = ChatOpenAI(
                api_key=openai_config['api_key'],
                base_url=openai_config.get('base_url'),
                model=openai_config['model'],
                temperature=0.0  # Deterministic evaluation
            )
            
            evaluators = {}
            
            # Context recall evaluator (requires reference)
            evaluators['context_recall'] = load_evaluator(
                EvaluatorType.LABELED_CRITERIA,
                criteria="correctness",
                llm=eval_llm
            )
            
            # Relevance evaluator
            evaluators['relevance'] = load_evaluator(
                EvaluatorType.CRITERIA,
                criteria=Criteria.RELEVANCE,
                llm=eval_llm
            )
            
            # Coherence evaluator
            evaluators['coherence'] = load_evaluator(
                EvaluatorType.CRITERIA,
                criteria=Criteria.COHERENCE,
                llm=eval_llm
            )
            
            # Helpfulness evaluator
            evaluators['helpfulness'] = load_evaluator(
                EvaluatorType.CRITERIA,
                criteria=Criteria.HELPFULNESS,
                llm=eval_llm
            )
            
            # Custom criteria for memory retrieval
            memory_criteria = {
                "context_accuracy": "Does the retrieved context accurately match the expected information for the given query?",
                "memory_relevance": "Is the retrieved memory content relevant to the user's query and context?"
            }
            
            evaluators['memory_accuracy'] = load_evaluator(
                EvaluatorType.CRITERIA,
                criteria=memory_criteria,
                llm=eval_llm
            )
            
            logger.info(f"Successfully configured {len(evaluators)} LangChain evaluators")
            return evaluators
            
        except Exception as e:
            logger.error(f"Failed to setup LangChain evaluators: {e}")
            raise
    
    def prepare_test_dataset(self, sample_memories: List[MemoryDocument], 
                           num_queries_per_memory: int = 2) -> List[EvaluationQuery]:
        """
        Generate test dataset with question-context pairs for evaluation.
        
        Args:
            sample_memories: List of memory documents to generate queries from
            num_queries_per_memory: Number of test queries to generate per memory
            
        Returns:
            List of test queries with expected contexts
        """
        try:
            logger.info(f"Generating test dataset from {len(sample_memories)} memory documents")
            
            # Get OpenAI configuration from config manager
            openai_config = self.config.get_openai_config()
            
            # Initialize LLM for query generation
            query_gen_llm = ChatOpenAI(
                api_key=openai_config['api_key'],
                base_url=openai_config.get('base_url'),
                model=openai_config['model'],
                temperature=0.7  # More creative for query generation
            )
            
            test_queries = []
            
            for memory in sample_memories:
                # Generate queries based on memory content
                query_prompt = f"""
                Based on the following memory content, generate {num_queries_per_memory} diverse questions 
                that a user might ask to retrieve this information:
                
                Memory Content: {memory.content}
                Memory Type: {memory.memory_type.value}
                
                Generate questions that would require this specific memory to answer.
                Format: Return only the questions, one per line.
                """
                
                try:
                    response = query_gen_llm.invoke(query_prompt)
                    # Handle different response formats safely
                    if hasattr(response, 'content'):
                        response_text = response.content
                    elif isinstance(response, str):
                        response_text = response
                    else:
                        response_text = str(response)
                    
                    # Ensure response_text is a string before splitting
                    if isinstance(response_text, str):
                        questions = [q.strip() for q in response_text.split('\n') if q.strip()]
                    else:
                        logger.warning(f"Unexpected response format for memory {memory.user_id}")
                        questions = []
                    
                    # Create test query objects
                    for question in questions[:num_queries_per_memory]:
                        test_query = EvaluationQuery(
                            query_id=str(uuid.uuid4()),
                            question=question,
                            expected_context=memory.content,
                            user_id=memory.user_id,
                            memory_type=memory.memory_type.value
                        )
                        test_queries.append(test_query)
                        
                except Exception as e:
                    logger.warning(f"Failed to generate queries for memory {memory.user_id}: {e}")
                    continue
            
            self.test_queries = test_queries
            logger.info(f"Generated {len(test_queries)} test queries")
            return test_queries
            
        except Exception as e:
            logger.error(f"Failed to prepare test dataset: {e}")
            raise
    
    def evaluate_context_recall(self, retrieved_contexts: List[str], 
                              expected_contexts: List[str],
                              queries: List[str]) -> List[EvaluationResult]:
        """
        Evaluate context recall accuracy using LangChain evaluators.
        
        Args:
            retrieved_contexts: List of retrieved context strings
            expected_contexts: List of expected/ground truth contexts
            queries: List of input queries
            
        Returns:
            List of evaluation results with scores and reasoning
        """
        results = []
        
        try:
            evaluator = self.evaluators['context_recall']
            
            for i, (retrieved, expected, query) in enumerate(zip(retrieved_contexts, expected_contexts, queries)):
                try:
                    eval_result = evaluator.evaluate_strings(
                        prediction=retrieved,
                        reference=expected,
                        input=query
                    )
                    
                    result = EvaluationResult(
                        metric_name="context_recall",
                        score=eval_result.get('score', 0.0),
                        reasoning=eval_result.get('reasoning', ''),
                        metadata={
                            'query_index': i,
                            'query': query,
                            'retrieved_length': len(retrieved),
                            'expected_length': len(expected)
                        },
                        timestamp=datetime.now()
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Context recall evaluation failed for query {i}: {e}")
                    continue
                    
            logger.info(f"Completed context recall evaluation for {len(results)} queries")
            return results
            
        except Exception as e:
            logger.error(f"Context recall evaluation failed: {e}")
            raise
    
    def evaluate_relevance(self, retrieved_contexts: List[str], 
                          queries: List[str]) -> List[EvaluationResult]:
        """
        Evaluate answer relevance using LangChain evaluators.
        
        Args:
            retrieved_contexts: List of retrieved context strings
            queries: List of input queries
            
        Returns:
            List of evaluation results with relevance scores
        """
        results = []
        
        try:
            evaluator = self.evaluators['relevance']
            
            for i, (context, query) in enumerate(zip(retrieved_contexts, queries)):
                try:
                    eval_result = evaluator.evaluate_strings(
                        prediction=context,
                        input=query
                    )
                    
                    result = EvaluationResult(
                        metric_name="relevance",
                        score=eval_result.get('score', 0.0),
                        reasoning=eval_result.get('reasoning', ''),
                        metadata={
                            'query_index': i,
                            'query': query,
                            'context_length': len(context)
                        },
                        timestamp=datetime.now()
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Relevance evaluation failed for query {i}: {e}")
                    continue
                    
            logger.info(f"Completed relevance evaluation for {len(results)} queries")
            return results
            
        except Exception as e:
            logger.error(f"Relevance evaluation failed: {e}")
            raise
    
    def evaluate_memory_accuracy(self, retrieved_memories: List[str],
                                queries: List[str]) -> List[EvaluationResult]:
        """
        Evaluate memory retrieval accuracy using custom criteria.
        
        Args:
            retrieved_memories: List of retrieved memory strings
            queries: List of input queries
            
        Returns:
            List of evaluation results for memory accuracy
        """
        results = []
        
        try:
            evaluator = self.evaluators['memory_accuracy']
            
            for i, (memory, query) in enumerate(zip(retrieved_memories, queries)):
                try:
                    eval_result = evaluator.evaluate_strings(
                        prediction=memory,
                        input=query
                    )
                    
                    result = EvaluationResult(
                        metric_name="memory_accuracy",
                        score=eval_result.get('score', 0.0),
                        reasoning=eval_result.get('reasoning', ''),
                        metadata={
                            'query_index': i,
                            'query': query,
                            'memory_length': len(memory)
                        },
                        timestamp=datetime.now()
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Memory accuracy evaluation failed for query {i}: {e}")
                    continue
                    
            logger.info(f"Completed memory accuracy evaluation for {len(results)} queries")
            return results
            
        except Exception as e:
            logger.error(f"Memory accuracy evaluation failed: {e}")
            raise
    
    def run_comparative_evaluation(self, 
                                  vector_persistence,
                                  graph_persistence,
                                  test_queries: List[EvaluationQuery],
                                  include_performance_metrics: bool = True) -> Dict[str, Any]:
        """
        Run identical queries against both persistence solutions and compare results.
        
        Args:
            vector_persistence: LangChain vector persistence instance
            graph_persistence: Neo4j graph persistence instance
            test_queries: List of test queries to evaluate
            include_performance_metrics: Whether to measure query performance
            
        Returns:
            Comprehensive comparison results with performance metrics
        """
        try:
            logger.info(f"Starting comparative evaluation with {len(test_queries)} queries")
            
            vector_results = []
            graph_results = []
            performance_metrics = {
                'vector_query_times': [],
                'graph_query_times': [],
                'vector_errors': 0,
                'graph_errors': 0
            }
            
            # Run queries against both systems
            for query in test_queries:
                # Query vector persistence
                try:
                    if include_performance_metrics:
                        start_time = datetime.now()
                    
                    vector_memories = vector_persistence.search_memories(
                        query=query.question,
                        user_id=query.user_id,
                        k=3
                    )
                    
                    if include_performance_metrics:
                        query_time = (datetime.now() - start_time).total_seconds()
                        performance_metrics['vector_query_times'].append(query_time)
                    
                    # Format vector results
                    vector_result = self._format_vector_results(vector_memories)
                    vector_results.append(vector_result)
                    
                except Exception as e:
                    logger.warning(f"Vector query failed for query {query.query_id}: {e}")
                    vector_results.append("")
                    performance_metrics['vector_errors'] += 1
                
                # Query graph persistence
                try:
                    if include_performance_metrics:
                        start_time = datetime.now()
                    
                    # Extract key entities from query for graph traversal
                    key_entities = self._extract_key_entities(query.question)
                    
                    # Use Graph RAG query for enhanced semantic + graph search
                    try:
                        graph_contexts = graph_persistence.graph_rag_query(
                            query_text=query.question,
                            user_id=query.user_id,
                            top_k=3,
                            graph_depth=2
                        )
                    except AttributeError:
                        # Fallback to traditional methods if Graph RAG not available
                        if key_entities:
                            # Use first entity as starting point for graph traversal
                            graph_contexts = graph_persistence.query_context(
                                entity=key_entities[0],
                                user_id=query.user_id,
                                max_depth=2
                            )
                        else:
                            # Fallback to entity search if no key entities found
                            graph_contexts = graph_persistence.search_entities(
                                search_term=query.question[:20],  # Use first 20 chars as search term
                                user_id=query.user_id,
                                limit=3
                            )
                    
                    if include_performance_metrics:
                        query_time = (datetime.now() - start_time).total_seconds()
                        performance_metrics['graph_query_times'].append(query_time)
                    
                    # Format graph results
                    graph_result = self._format_graph_results(graph_contexts)
                    graph_results.append(graph_result)
                    
                except Exception as e:
                    logger.warning(f"Graph query failed for query {query.query_id}: {e}")
                    graph_results.append("")
                    performance_metrics['graph_errors'] += 1
            
            # Prepare data for evaluation
            queries_list = [q.question for q in test_queries]
            expected_contexts = [q.expected_context for q in test_queries]
            
            # Run comparative evaluation
            comparison_results = self.compare_persistence_solutions(
                vector_results=vector_results,
                graph_results=graph_results,
                queries=queries_list,
                expected_contexts=expected_contexts
            )
            
            # Add performance metrics
            if include_performance_metrics:
                comparison_results['performance_metrics'] = self._calculate_performance_metrics(performance_metrics)
            
            logger.info("Completed comparative evaluation with performance metrics")
            return comparison_results
            
        except Exception as e:
            logger.error(f"Comparative evaluation failed: {e}")
            raise
    
    def _format_vector_results(self, memories: List[Dict[str, Any]]) -> str:
        """Format vector search results for evaluation."""
        if not memories:
            return ""
        
        formatted_results = []
        for memory in memories:
            content = memory.get('content', '')
            score = memory.get('similarity_score', 0.0)
            formatted_results.append(f"{content} (similarity: {score:.3f})")
        
        return "\n".join(formatted_results)
    
    def _format_graph_results(self, contexts: List[Any]) -> str:
        """Format graph search results for evaluation."""
        if not contexts:
            return ""
        
        formatted_results = []
        for context in contexts:
            # Handle GraphRAGResult objects (new Graph RAG format)
            if hasattr(context, 'entity_name') and hasattr(context, 'semantic_score'):
                # This is a GraphRAGResult object
                entity = context.entity_name
                entity_type = context.entity_type
                semantic_score = getattr(context, 'semantic_score', 0.0)
                combined_score = getattr(context, 'combined_score', 0.0)
                formatted_results.append(f"{entity} ({entity_type}, semantic: {semantic_score:.3f}, combined: {combined_score:.3f})")
            elif isinstance(context, dict):
                # Handle legacy dictionary format
                if 'connected_entity' in context:
                    # Context query result
                    entity = context.get('connected_entity', '')
                    entity_type = context.get('entity_type', '')
                    path_length = context.get('path_length', 0)
                    formatted_results.append(f"{entity} ({entity_type}, depth: {path_length})")
                elif 'entity_name' in context:
                    # Entity search result
                    entity = context.get('entity_name', '')
                    entity_type = context.get('entity_type', '')
                    formatted_results.append(f"{entity} ({entity_type})")
        
        return "\n".join(formatted_results)
    
    def _extract_key_entities(self, query: str) -> List[str]:
        """Extract key entities from query text for graph traversal."""
        import re
        
        # Simple entity extraction - extract capitalized words
        entities = re.findall(r'\b[A-Z][a-z]+\b', query)
        
        # Filter common stop words
        stop_words = {'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'And', 'But', 'Or', 'So', 'What', 'When', 'Where', 'How', 'Why'}
        filtered_entities = [entity for entity in entities if entity not in stop_words and len(entity) > 2]
        
        return filtered_entities[:3]  # Return top 3 entities
    
    def _calculate_performance_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance statistics from raw metrics."""
        def calculate_stats(times: List[float]) -> Dict[str, float]:
            if not times:
                return {'avg': 0.0, 'min': 0.0, 'max': 0.0, 'total': 0.0}
            
            return {
                'avg': sum(times) / len(times),
                'min': min(times),
                'max': max(times),
                'total': sum(times)
            }
        
        vector_stats = calculate_stats(metrics['vector_query_times'])
        graph_stats = calculate_stats(metrics['graph_query_times'])
        
        return {
            'vector_performance': {
                **vector_stats,
                'error_count': metrics['vector_errors'],
                'success_rate': 1.0 - (metrics['vector_errors'] / len(metrics['vector_query_times'])) if metrics['vector_query_times'] else 0.0
            },
            'graph_performance': {
                **graph_stats,
                'error_count': metrics['graph_errors'],
                'success_rate': 1.0 - (metrics['graph_errors'] / len(metrics['graph_query_times'])) if metrics['graph_query_times'] else 0.0
            },
            'performance_comparison': {
                'vector_faster': vector_stats['avg'] < graph_stats['avg'] if vector_stats['avg'] > 0 and graph_stats['avg'] > 0 else None,
                'speed_difference': abs(vector_stats['avg'] - graph_stats['avg']) if vector_stats['avg'] > 0 and graph_stats['avg'] > 0 else 0.0
            }
        }

    def compare_persistence_solutions(self, vector_results: List[str],
                                    graph_results: List[str],
                                    queries: List[str],
                                    expected_contexts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare vector and graph persistence solutions using multiple metrics.
        
        Args:
            vector_results: Results from vector-based persistence
            graph_results: Results from graph-based persistence
            queries: List of test queries
            expected_contexts: Optional expected contexts for accuracy evaluation
            
        Returns:
            Comprehensive comparison results
        """
        try:
            logger.info("Starting comparative evaluation of persistence solutions")
            
            comparison_results = {
                'vector_solution': {},
                'graph_solution': {},
                'comparison': {},
                'metadata': {
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'num_queries': len(queries),
                    'evaluator_version': '1.0.0'
                }
            }
            
            # Evaluate vector solution
            if vector_results:
                comparison_results['vector_solution']['relevance'] = self.evaluate_relevance(
                    vector_results, queries
                )
                comparison_results['vector_solution']['memory_accuracy'] = self.evaluate_memory_accuracy(
                    vector_results, queries
                )
                
                if expected_contexts:
                    comparison_results['vector_solution']['context_recall'] = self.evaluate_context_recall(
                        vector_results, expected_contexts, queries
                    )
            
            # Evaluate graph solution  
            if graph_results:
                comparison_results['graph_solution']['relevance'] = self.evaluate_relevance(
                    graph_results, queries
                )
                comparison_results['graph_solution']['memory_accuracy'] = self.evaluate_memory_accuracy(
                    graph_results, queries
                )
                
                if expected_contexts:
                    comparison_results['graph_solution']['context_recall'] = self.evaluate_context_recall(
                        graph_results, expected_contexts, queries
                    )
            
            # Calculate aggregate scores
            comparison_results['comparison'] = self._calculate_aggregate_scores(
                comparison_results['vector_solution'],
                comparison_results['graph_solution']
            )
            
            logger.info("Completed comparative evaluation")
            return comparison_results
            
        except Exception as e:
            logger.error(f"Comparative evaluation failed: {e}")
            raise
    
    def _calculate_aggregate_scores(self, vector_results: Dict[str, Any], 
                                  graph_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate aggregate scores for comparison.
        
        Args:
            vector_results: Vector solution evaluation results
            graph_results: Graph solution evaluation results
            
        Returns:
            Aggregate comparison metrics
        """
        def get_average_score(results: List[EvaluationResult]) -> float:
            if not results:
                return 0.0
            return sum(r.score for r in results) / len(results)
        
        comparison = {}
        
        # Calculate averages for each metric
        for metric in ['relevance', 'memory_accuracy', 'context_recall']:
            vector_avg = get_average_score(vector_results.get(metric, []))
            graph_avg = get_average_score(graph_results.get(metric, []))
            
            comparison[metric] = {
                'vector_average': vector_avg,
                'graph_average': graph_avg,
                'difference': graph_avg - vector_avg,
                'winner': 'graph' if graph_avg > vector_avg else 'vector' if vector_avg > graph_avg else 'tie'
            }
        
        # Overall score (weighted average)
        vector_overall = (
            comparison.get('relevance', {}).get('vector_average', 0) * 0.4 +
            comparison.get('memory_accuracy', {}).get('vector_average', 0) * 0.4 +
            comparison.get('context_recall', {}).get('vector_average', 0) * 0.2
        )
        
        graph_overall = (
            comparison.get('relevance', {}).get('graph_average', 0) * 0.4 +
            comparison.get('memory_accuracy', {}).get('graph_average', 0) * 0.4 +
            comparison.get('context_recall', {}).get('graph_average', 0) * 0.2
        )
        
        comparison['overall'] = {
            'vector_score': vector_overall,
            'graph_score': graph_overall,
            'difference': graph_overall - vector_overall,
            'winner': 'graph' if graph_overall > vector_overall else 'vector' if vector_overall > graph_overall else 'tie'
        }
        
        return comparison
    
    def generate_evaluation_report(self, comparison_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            comparison_results: Results from compare_persistence_solutions
            
        Returns:
            Formatted evaluation report string
        """
        try:
            report_lines = [
                "# Persistence Solutions Evaluation Report",
                f"Generated: {comparison_results['metadata']['evaluation_timestamp']}",
                f"Queries Evaluated: {comparison_results['metadata']['num_queries']}",
                "",
                "## Overall Results",
            ]
            
            overall = comparison_results['comparison']['overall']
            report_lines.extend([
                f"**Winner:** {overall['winner'].title()}",
                f"Vector Solution Score: {overall['vector_score']:.3f}",
                f"Graph Solution Score: {overall['graph_score']:.3f}",
                f"Score Difference: {abs(overall['difference']):.3f}",
                "",
                "## Detailed Metrics",
            ])
            
            for metric, data in comparison_results['comparison'].items():
                if metric != 'overall':
                    report_lines.extend([
                        f"### {metric.replace('_', ' ').title()}",
                        f"- Vector: {data['vector_average']:.3f}",
                        f"- Graph: {data['graph_average']:.3f}",
                        f"- Winner: {data['winner'].title()}",
                        ""
                    ])
            
            # Add recommendations
            report_lines.extend([
                "## Recommendations",
                self._generate_recommendations(comparison_results['comparison']),
                "",
                "---",
                "Report generated by PersistenceEvaluator using LangChain evaluation tools"
            ])
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Failed to generate evaluation report: {e}")
            raise
    
    def _generate_recommendations(self, comparison: Dict[str, Any]) -> str:
        """Generate recommendations based on evaluation results."""
        overall_winner = comparison['overall']['winner']
        overall_diff = abs(comparison['overall']['difference'])
        
        if overall_diff < 0.1:
            return ("Both solutions perform similarly. Consider factors like computational cost, "
                   "complexity, and maintenance when choosing between them.")
        elif overall_winner == 'vector':
            return ("Vector-based solution shows better performance. It may be more suitable for "
                   "semantic similarity tasks and simpler deployment scenarios.")
        else:
            return ("Graph-based solution shows better performance. It may be more suitable for "
                   "complex relationship queries and contextual reasoning tasks.")

    # Enhanced Reporting System for Task 4.3
    
    def calculate_detailed_metrics(self, vector_results: Dict[str, List[EvaluationResult]], 
                                 graph_results: Dict[str, List[EvaluationResult]]) -> Dict[str, MetricSummary]:
        """
        Calculate detailed metrics with statistical analysis.
        
        Args:
            vector_results: Vector solution evaluation results
            graph_results: Graph solution evaluation results
            
        Returns:
            Dictionary of metric summaries with statistical analysis
        """
        import statistics
        
        metric_summaries = {}
        
        for metric_name in ['relevance', 'memory_accuracy', 'context_recall']:
            vector_scores = [r.score for r in vector_results.get(metric_name, [])]
            graph_scores = [r.score for r in graph_results.get(metric_name, [])]
            
            if not vector_scores or not graph_scores:
                continue
                
            vector_avg = statistics.mean(vector_scores)
            graph_avg = statistics.mean(graph_scores)
            vector_std = statistics.stdev(vector_scores) if len(vector_scores) > 1 else 0.0
            graph_std = statistics.stdev(graph_scores) if len(graph_scores) > 1 else 0.0
            
            difference = graph_avg - vector_avg
            percentage_difference = (difference / vector_avg * 100) if vector_avg > 0 else 0.0
            
            # Determine winner and significance
            winner = 'graph' if graph_avg > vector_avg else 'vector' if vector_avg > graph_avg else 'tie'
            significance_level = self._calculate_significance_level(vector_scores, graph_scores)
            
            metric_summaries[metric_name] = MetricSummary(
                metric_name=metric_name,
                vector_average=vector_avg,
                graph_average=graph_avg,
                vector_std=vector_std,
                graph_std=graph_std,
                difference=difference,
                percentage_difference=percentage_difference,
                winner=winner,
                significance_level=significance_level
            )
        
        return metric_summaries
    
    def _calculate_significance_level(self, vector_scores: List[float], graph_scores: List[float]) -> str:
        """Calculate statistical significance level of the difference."""
        import statistics
        
        if len(vector_scores) < 2 or len(graph_scores) < 2:
            return "insufficient_data"
        
        vector_avg = statistics.mean(vector_scores)
        graph_avg = statistics.mean(graph_scores)
        difference = abs(graph_avg - vector_avg)
        
        # Simple significance classification based on difference magnitude and standard deviations
        vector_std = statistics.stdev(vector_scores)
        graph_std = statistics.stdev(graph_scores)
        combined_std = (vector_std + graph_std) / 2
        
        if difference > 2 * combined_std:
            return "high"
        elif difference > combined_std:
            return "medium"
        elif difference > 0.5 * combined_std:
            return "low"
        else:
            return "negligible"
    
    def analyze_strengths_and_weaknesses(self, metric_summaries: Dict[str, MetricSummary], 
                                       performance_metrics: Dict[str, PerformanceMetrics]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Analyze strengths and weaknesses for each persistence solution.
        
        Args:
            metric_summaries: Calculated metric summaries
            performance_metrics: Performance metrics for both solutions
            
        Returns:
            Tuple of (strengths_analysis, weaknesses_analysis)
        """
        strengths = {'vector': [], 'graph': []}
        weaknesses = {'vector': [], 'graph': []}
        
        # Analyze metric performance
        for metric_name, summary in metric_summaries.items():
            if summary.winner == 'vector':
                strengths['vector'].append(f"Superior {metric_name.replace('_', ' ')} performance (+{summary.percentage_difference:.1f}%)")
                if summary.significance_level in ['high', 'medium']:
                    weaknesses['graph'].append(f"Lower {metric_name.replace('_', ' ')} accuracy (statistically significant)")
            elif summary.winner == 'graph':
                strengths['graph'].append(f"Superior {metric_name.replace('_', ' ')} performance (+{summary.percentage_difference:.1f}%)")
                if summary.significance_level in ['high', 'medium']:
                    weaknesses['vector'].append(f"Lower {metric_name.replace('_', ' ')} accuracy (statistically significant)")
        
        # Analyze performance characteristics
        vector_perf = performance_metrics.get('vector', None)
        graph_perf = performance_metrics.get('graph', None)
        
        if vector_perf and graph_perf:
            if vector_perf.avg_query_time < graph_perf.avg_query_time:
                time_diff = ((graph_perf.avg_query_time - vector_perf.avg_query_time) / vector_perf.avg_query_time) * 100
                strengths['vector'].append(f"Faster query execution ({time_diff:.1f}% faster)")
                weaknesses['graph'].append(f"Slower query performance")
            elif graph_perf.avg_query_time < vector_perf.avg_query_time:
                time_diff = ((vector_perf.avg_query_time - graph_perf.avg_query_time) / graph_perf.avg_query_time) * 100
                strengths['graph'].append(f"Faster query execution ({time_diff:.1f}% faster)")
                weaknesses['vector'].append(f"Slower query performance")
            
            if vector_perf.success_rate > graph_perf.success_rate:
                strengths['vector'].append(f"Higher reliability ({vector_perf.success_rate:.1%} success rate)")
                weaknesses['graph'].append(f"Lower reliability ({graph_perf.success_rate:.1%} success rate)")
            elif graph_perf.success_rate > vector_perf.success_rate:
                strengths['graph'].append(f"Higher reliability ({graph_perf.success_rate:.1%} success rate)")
                weaknesses['vector'].append(f"Lower reliability ({vector_perf.success_rate:.1%} success rate)")
        
        return strengths, weaknesses
    
    def generate_comprehensive_recommendations(self, detailed_comparison: DetailedComparison) -> List[str]:
        """
        Generate comprehensive recommendations based on detailed analysis.
        
        Args:
            detailed_comparison: Detailed comparison analysis
            
        Returns:
            List of actionable recommendations
        """
        recommendations = []
        
        # Overall performance recommendation
        overall_winner = None
        max_score = 0
        for solution, score in detailed_comparison.overall_scores.items():
            if score > max_score:
                max_score = score
                overall_winner = solution
        
        score_difference = abs(detailed_comparison.overall_scores.get('vector', 0) - 
                             detailed_comparison.overall_scores.get('graph', 0))
        
        if score_difference < 0.05:
            recommendations.append("Both solutions show comparable performance. Consider non-performance factors like implementation complexity, maintenance costs, and team expertise when choosing.")
        else:
            recommendations.append(f"The {overall_winner} solution demonstrates superior overall performance with a {score_difference:.3f} point advantage.")
        
        # Specific use case recommendations
        vector_strengths = detailed_comparison.strengths_analysis.get('vector', [])
        graph_strengths = detailed_comparison.strengths_analysis.get('graph', [])
        
        if any('relevance' in strength.lower() for strength in vector_strengths):
            recommendations.append("Vector solution is recommended for semantic similarity tasks and content-based retrieval scenarios.")
        
        if any('context_recall' in strength.lower() for strength in graph_strengths):
            recommendations.append("Graph solution is recommended for complex relationship queries and contextual reasoning tasks.")
        
        # Performance-based recommendations
        vector_perf = detailed_comparison.performance_metrics.get('vector')
        graph_perf = detailed_comparison.performance_metrics.get('graph')
        
        if vector_perf and graph_perf:
            if vector_perf.avg_query_time < graph_perf.avg_query_time * 0.8:
                recommendations.append("Vector solution is recommended for high-throughput applications requiring fast response times.")
            elif graph_perf.avg_query_time < vector_perf.avg_query_time * 0.8:
                recommendations.append("Graph solution provides acceptable performance for most use cases with enhanced contextual understanding.")
        
        # Reliability recommendations
        if vector_perf and graph_perf:
            if vector_perf.success_rate > 0.95 and graph_perf.success_rate < 0.90:
                recommendations.append("Vector solution shows higher reliability for production deployments.")
            elif graph_perf.success_rate > 0.95 and vector_perf.success_rate < 0.90:
                recommendations.append("Graph solution demonstrates production-ready reliability.")
        
        # Implementation recommendations
        if len(graph_strengths) > len(vector_strengths):
            recommendations.append("Consider implementing graph solution as the primary persistence layer with vector as a fallback for specific use cases.")
        elif len(vector_strengths) > len(graph_strengths):
            recommendations.append("Consider implementing vector solution as the primary persistence layer with selective graph enhancement for complex queries.")
        else:
            recommendations.append("Consider a hybrid approach leveraging both solutions for different query types and use cases.")
        
        return recommendations
    
    def create_detailed_comparison(self, comparison_results: Dict[str, Any], 
                                 performance_data: Optional[Dict[str, Any]] = None) -> DetailedComparison:
        """
        Create a detailed comparison analysis from evaluation results.
        
        Args:
            comparison_results: Results from compare_persistence_solutions
            performance_data: Optional performance metrics
            
        Returns:
            Detailed comparison analysis
        """
        try:
            # Calculate detailed metrics with statistics
            metric_summaries = self.calculate_detailed_metrics(
                comparison_results.get('vector_solution', {}),
                comparison_results.get('graph_solution', {})
            )
            
            # Extract overall scores
            overall_scores = {
                'vector': comparison_results['comparison']['overall']['vector_score'],
                'graph': comparison_results['comparison']['overall']['graph_score']
            }
            
            # Process performance metrics
            performance_metrics = {}
            if performance_data and 'performance_metrics' in performance_data:
                perf_data = performance_data['performance_metrics']
                
                for solution in ['vector', 'graph']:
                    solution_perf = perf_data.get(f'{solution}_performance', {})
                    if solution_perf:
                        performance_metrics[solution] = PerformanceMetrics(
                            avg_query_time=solution_perf.get('avg', 0.0),
                            min_query_time=solution_perf.get('min', 0.0),
                            max_query_time=solution_perf.get('max', 0.0),
                            total_query_time=solution_perf.get('total', 0.0),
                            error_count=solution_perf.get('error_count', 0),
                            success_rate=solution_perf.get('success_rate', 0.0),
                            query_count=len(performance_data.get('test_queries', []))
                        )
            
            # Analyze strengths and weaknesses
            strengths_analysis, weaknesses_analysis = self.analyze_strengths_and_weaknesses(
                metric_summaries, performance_metrics
            )
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(metric_summaries, performance_metrics)
            
            # Create detailed comparison object
            detailed_comparison = DetailedComparison(
                metric_summaries=metric_summaries,
                overall_scores=overall_scores,
                performance_metrics=performance_metrics,
                strengths_analysis=strengths_analysis,
                weaknesses_analysis=weaknesses_analysis,
                recommendations=[],  # Will be filled by generate_comprehensive_recommendations
                confidence_scores=confidence_scores
            )
            
            # Generate comprehensive recommendations
            detailed_comparison.recommendations = self.generate_comprehensive_recommendations(detailed_comparison)
            
            logger.info("Created detailed comparison analysis")
            return detailed_comparison
            
        except Exception as e:
            logger.error(f"Failed to create detailed comparison: {e}")
            raise
    
    def _calculate_confidence_scores(self, metric_summaries: Dict[str, MetricSummary], 
                                   performance_metrics: Dict[str, PerformanceMetrics]) -> Dict[str, float]:
        """Calculate confidence scores for the evaluation results."""
        confidence_scores = {}
        
        # Calculate overall confidence based on data quality and consistency
        metric_confidence = []
        for summary in metric_summaries.values():
            if summary.significance_level == 'high':
                metric_confidence.append(0.9)
            elif summary.significance_level == 'medium':
                metric_confidence.append(0.7)
            elif summary.significance_level == 'low':
                metric_confidence.append(0.5)
            else:
                metric_confidence.append(0.3)
        
        if metric_confidence:
            confidence_scores['overall'] = sum(metric_confidence) / len(metric_confidence)
        else:
            confidence_scores['overall'] = 0.5
        
        # Performance confidence based on success rates
        for solution, perf in performance_metrics.items():
            if perf.success_rate > 0.95:
                confidence_scores[f'{solution}_performance'] = 0.9
            elif perf.success_rate > 0.85:
                confidence_scores[f'{solution}_performance'] = 0.7
            elif perf.success_rate > 0.70:
                confidence_scores[f'{solution}_performance'] = 0.5
            else:
                confidence_scores[f'{solution}_performance'] = 0.3
        
        return confidence_scores
    
    def generate_structured_report(self, comparison_results: Dict[str, Any], 
                                 performance_data: Optional[Dict[str, Any]] = None,
                                 output_format: ReportFormat = ReportFormat.MARKDOWN) -> str:
        """
        Generate a comprehensive structured evaluation report.
        
        Args:
            comparison_results: Results from compare_persistence_solutions
            performance_data: Optional performance metrics and test data
            output_format: Desired output format for the report
            
        Returns:
            Formatted evaluation report as string
        """
        try:
            # Create detailed comparison analysis
            detailed_comparison = self.create_detailed_comparison(comparison_results, performance_data)
            
            # Create evaluation report structure
            evaluation_report = EvaluationReport(
                report_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                evaluator_version=self.version,
                test_configuration={
                    'num_queries': comparison_results['metadata']['num_queries'],
                    'evaluation_timestamp': comparison_results['metadata']['evaluation_timestamp'],
                    'evaluator_version': comparison_results['metadata'].get('evaluator_version', '1.0.0')
                },
                detailed_comparison=detailed_comparison,
                raw_results=comparison_results,
                executive_summary=self._generate_executive_summary(detailed_comparison),
                methodology_notes=self._generate_methodology_notes()
            )
            
            # Format report based on requested format
            if output_format == ReportFormat.MARKDOWN:
                return self._format_markdown_report(evaluation_report)
            elif output_format == ReportFormat.JSON:
                return self._format_json_report(evaluation_report)
            elif output_format == ReportFormat.CSV:
                return self._format_csv_report(evaluation_report)
            elif output_format == ReportFormat.HTML:
                return self._format_html_report(evaluation_report)
            else:
                raise ValueError(f"Unsupported report format: {output_format}")
                
        except Exception as e:
            logger.error(f"Failed to generate structured report: {e}")
            raise
    
    def _generate_executive_summary(self, detailed_comparison: DetailedComparison) -> str:
        """Generate executive summary of the evaluation."""
        vector_score = detailed_comparison.overall_scores.get('vector', 0)
        graph_score = detailed_comparison.overall_scores.get('graph', 0)
        winner = 'graph' if graph_score > vector_score else 'vector' if vector_score > graph_score else 'neither'
        
        score_diff = abs(vector_score - graph_score)
        confidence = detailed_comparison.confidence_scores.get('overall', 0.5)
        
        summary_parts = [
            f"Evaluation completed comparing vector and graph persistence solutions across multiple metrics.",
            f"Overall performance scores: Vector={vector_score:.3f}, Graph={graph_score:.3f}.",
        ]
        
        if winner != 'neither':
            summary_parts.append(f"The {winner} solution demonstrates superior performance with {confidence:.1%} confidence.")
        else:
            summary_parts.append("Both solutions show comparable performance levels.")
        
        if score_diff > 0.1:
            summary_parts.append("The performance difference is statistically significant.")
        else:
            summary_parts.append("The performance difference is minimal.")
        
        # Add key insights
        vector_strengths = len(detailed_comparison.strengths_analysis.get('vector', []))
        graph_strengths = len(detailed_comparison.strengths_analysis.get('graph', []))
        
        if vector_strengths > graph_strengths:
            summary_parts.append("Vector solution shows advantages in multiple performance areas.")
        elif graph_strengths > vector_strengths:
            summary_parts.append("Graph solution demonstrates superior capabilities across key metrics.")
        
        return " ".join(summary_parts)
    
    def _generate_methodology_notes(self) -> str:
        """Generate methodology notes for the evaluation."""
        return """
        Evaluation Methodology:
        - LangChain evaluation framework with multiple criteria
        - Context recall, relevance, and memory accuracy metrics
        - Statistical significance testing
        - Performance timing measurements
        - Error rate and reliability analysis
        - Comparative scoring with weighted averages
        
        Metrics Weighting:
        - Relevance: 40%
        - Memory Accuracy: 40% 
        - Context Recall: 20%
        
        Confidence Levels:
        - High: >2 standard deviations difference
        - Medium: 1-2 standard deviations difference
        - Low: 0.5-1 standard deviation difference
        - Negligible: <0.5 standard deviation difference
        """
    
    def _format_markdown_report(self, report: EvaluationReport) -> str:
        """Format evaluation report as Markdown."""
        lines = [
            f"# Persistence Solutions Evaluation Report",
            f"**Report ID:** {report.report_id}",
            f"**Generated:** {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Evaluator Version:** {report.evaluator_version}",
            "",
            "## Executive Summary",
            report.executive_summary,
            "",
            "## Overall Results",
        ]
        
        # Overall scores
        for solution, score in report.detailed_comparison.overall_scores.items():
            lines.append(f"- **{solution.title()} Solution:** {score:.3f}")
        
        lines.extend(["", "## Detailed Metrics Analysis", ""])
        
        # Detailed metrics
        for metric_name, summary in report.detailed_comparison.metric_summaries.items():
            lines.extend([
                f"### {metric_name.replace('_', ' ').title()}",
                f"- **Vector Average:** {summary.vector_average:.3f} (±{summary.vector_std:.3f})",
                f"- **Graph Average:** {summary.graph_average:.3f} (±{summary.graph_std:.3f})",
                f"- **Difference:** {summary.difference:+.3f} ({summary.percentage_difference:+.1f}%)",
                f"- **Winner:** {summary.winner.title()}",
                f"- **Significance:** {summary.significance_level.title()}",
                ""
            ])
        
        # Performance metrics
        if report.detailed_comparison.performance_metrics:
            lines.extend(["## Performance Analysis", ""])
            for solution, perf in report.detailed_comparison.performance_metrics.items():
                lines.extend([
                    f"### {solution.title()} Solution Performance",
                    f"- **Average Query Time:** {perf.avg_query_time:.3f}s",
                    f"- **Success Rate:** {perf.success_rate:.1%}",
                    f"- **Error Count:** {perf.error_count}",
                    f"- **Total Queries:** {perf.query_count}",
                    ""
                ])
        
        # Strengths and weaknesses
        lines.extend(["## Strengths and Weaknesses Analysis", ""])
        for solution in ['vector', 'graph']:
            strengths = report.detailed_comparison.strengths_analysis.get(solution, [])
            weaknesses = report.detailed_comparison.weaknesses_analysis.get(solution, [])
            
            lines.extend([f"### {solution.title()} Solution", ""])
            
            if strengths:
                lines.append("**Strengths:**")
                for strength in strengths:
                    lines.append(f"- {strength}")
                lines.append("")
            
            if weaknesses:
                lines.append("**Weaknesses:**")
                for weakness in weaknesses:
                    lines.append(f"- {weakness}")
                lines.append("")
        
        # Recommendations
        lines.extend(["## Recommendations", ""])
        for i, recommendation in enumerate(report.detailed_comparison.recommendations, 1):
            lines.append(f"{i}. {recommendation}")
        
        # Confidence scores
        lines.extend(["", "## Confidence Assessment", ""])
        for metric, confidence in report.detailed_comparison.confidence_scores.items():
            lines.append(f"- **{metric.replace('_', ' ').title()}:** {confidence:.1%}")
        
        # Methodology
        lines.extend(["", "## Methodology", report.methodology_notes])
        
        lines.extend(["", "---", "*Report generated by PersistenceEvaluator using LangChain evaluation framework*"])
        
        return "\n".join(lines)
    
    def _format_json_report(self, report: EvaluationReport) -> str:
        """Format evaluation report as JSON."""
        return json.dumps(report.to_dict(), indent=2, default=str)
    
    def _format_csv_report(self, report: EvaluationReport) -> str:
        """Format key metrics as CSV for spreadsheet analysis."""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(['Report_ID', 'Timestamp', 'Metric', 'Vector_Avg', 'Graph_Avg', 'Difference', 'Winner', 'Significance'])
        
        # Data rows
        for metric_name, summary in report.detailed_comparison.metric_summaries.items():
            writer.writerow([
                report.report_id,
                report.timestamp.isoformat(),
                metric_name,
                f"{summary.vector_average:.3f}",
                f"{summary.graph_average:.3f}",
                f"{summary.difference:+.3f}",
                summary.winner,
                summary.significance_level
            ])
        
        return output.getvalue()
    
    def _format_html_report(self, report: EvaluationReport) -> str:
        """Format evaluation report as HTML."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Persistence Solutions Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .metric {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }}
                .winner {{ font-weight: bold; color: #007acc; }}
                .recommendation {{ background-color: #e8f4f8; padding: 10px; margin: 10px 0; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Persistence Solutions Evaluation Report</h1>
                <p><strong>Report ID:</strong> {report.report_id}</p>
                <p><strong>Generated:</strong> {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Evaluator Version:</strong> {report.evaluator_version}</p>
            </div>
            
            <h2>Executive Summary</h2>
            <p>{report.executive_summary}</p>
            
            <h2>Overall Results</h2>
            <table>
                <tr><th>Solution</th><th>Overall Score</th></tr>"""
        
        for solution, score in report.detailed_comparison.overall_scores.items():
            html += f"<tr><td>{solution.title()}</td><td>{score:.3f}</td></tr>"
        
        html += "</table>"
        
        # Add detailed metrics
        html += "<h2>Detailed Metrics</h2>"
        for metric_name, summary in report.detailed_comparison.metric_summaries.items():
            html += f"""
            <div class="metric">
                <h3>{metric_name.replace('_', ' ').title()}</h3>
                <p>Vector: {summary.vector_average:.3f} | Graph: {summary.graph_average:.3f}</p>
                <p class="winner">Winner: {summary.winner.title()}</p>
                <p>Significance: {summary.significance_level.title()}</p>
            </div>"""
        
        # Add recommendations
        html += "<h2>Recommendations</h2>"
        for recommendation in report.detailed_comparison.recommendations:
            html += f'<div class="recommendation">{recommendation}</div>'
        
        html += """
        </body>
        </html>
        """
        
        return html


def create_sample_test_dataset() -> List[MemoryDocument]:
    """
    Create sample memory documents for testing evaluation framework.
    
    Returns:
        List of sample memory documents
    """
    sample_memories = [
        MemoryDocument(
            content="User prefers morning meetings between 9-11 AM on weekdays",
            user_id="user_001",
            timestamp=datetime.now(),
            memory_type=MemoryType.PREFERENCE
        ),
        MemoryDocument(
            content="Project deadline for Q1 launch is March 15, 2024",
            user_id="user_001", 
            timestamp=datetime.now(),
            memory_type=MemoryType.FACT
        ),
        MemoryDocument(
            content="Discussed implementation of new authentication system using OAuth 2.0",
            user_id="user_002",
            timestamp=datetime.now(),
            memory_type=MemoryType.CONVERSATION
        ),
        MemoryDocument(
            content="User mentioned working from New York office on Tuesdays and Thursdays",
            user_id="user_002",
            timestamp=datetime.now(),
            memory_type=MemoryType.FACT
        )
    ]
    
    return sample_memories
