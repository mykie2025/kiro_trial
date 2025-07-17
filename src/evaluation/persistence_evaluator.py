"""
Persistence Evaluator Module

This module implements evaluation tools for comparing persistence solutions using LangChain's 
evaluation framework. It provides metrics for context recall, relevance assessment, and 
comparative analysis between vector and graph-based persistence systems.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# LangChain evaluation imports
from langchain.evaluation import load_evaluator, EvaluatorType
from langchain.evaluation.criteria import Criteria
from langchain_openai import ChatOpenAI

# Local imports
try:
    from ..config.config_manager import ConfigManager
    from ..persistence.memory_document import MemoryDocument, MemoryType
except ImportError:
    # Fallback for when module is imported from outside the package
    from config.config_manager import ConfigManager
    from persistence.memory_document import MemoryDocument, MemoryType


logger = logging.getLogger(__name__)


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


class PersistenceEvaluator:
    """
    Evaluator for comparing persistence solutions using LangChain's evaluation tools.
    
    This class provides comprehensive evaluation capabilities including:
    - Context recall accuracy measurement
    - Answer relevance assessment  
    - Comparative analysis between persistence solutions
    - Test dataset generation and management
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
