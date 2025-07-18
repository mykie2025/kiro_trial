"""
Integration tests for comparative evaluation system.

This module tests the cross-system evaluation capabilities including
comparative analysis between vector and graph persistence solutions.
"""

import pytest
import tempfile
import os
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Test imports
from src.config.config_manager import ConfigManager
from src.evaluation.persistence_evaluator import (
    PersistenceEvaluator, 
    EvaluationResult, 
    EvaluationQuery,
    create_sample_test_dataset,
    PerformanceMetrics,
    MetricSummary,
    DetailedComparison,
    ReportFormat
)
from src.persistence.memory_document import MemoryDocument, MemoryType
from src.persistence.langchain_vector_persistence import LangChainVectorPersistence
from src.persistence.neo4j_graph_persistence import Neo4jGraphPersistence


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = Mock(spec=ConfigManager)
    config.get_openai_config.return_value = {
        'api_key': 'test-key',
        'model': 'gpt-3.5-turbo',
        'base_url': None
    }
    config.get_neo4j_config.return_value = {
        'uri': 'bolt://localhost:7687',
        'username': 'neo4j',
        'password': 'testpass'
    }
    return config


@pytest.fixture
def mock_vector_persistence():
    """Create a mock vector persistence instance."""
    mock_persistence = Mock(spec=LangChainVectorPersistence)
    
    # Mock search_memories method
    mock_persistence.search_memories.return_value = [
        {
            'content': 'User prefers morning meetings between 9-11 AM',
            'similarity_score': 0.85,
            'metadata': {'user_id': 'user_001', 'memory_type': 'preference'}
        },
        {
            'content': 'Project deadline for Q1 launch is March 15, 2024',
            'similarity_score': 0.72,
            'metadata': {'user_id': 'user_001', 'memory_type': 'fact'}
        }
    ]
    
    return mock_persistence


@pytest.fixture
def mock_graph_persistence():
    """Create a mock graph persistence instance."""
    mock_persistence = Mock(spec=Neo4jGraphPersistence)
    
    # Mock query_context method
    mock_persistence.query_context.return_value = [
        {
            'connected_entity': 'Morning Meeting',
            'entity_type': 'preference',
            'path_length': 1
        },
        {
            'connected_entity': 'Q1 Launch',
            'entity_type': 'project',
            'path_length': 2
        }
    ]
    
    # Mock search_entities method
    mock_persistence.search_entities.return_value = [
        {
            'entity_name': 'User Schedule',
            'entity_type': 'preference'
        }
    ]
    
    return mock_persistence


@pytest.fixture
def sample_test_queries():
    """Create sample test queries for evaluation."""
    return [
        EvaluationQuery(
            query_id="test_001",
            question="When does the user prefer to have meetings?",
            expected_context="User prefers morning meetings between 9-11 AM on weekdays",
            user_id="user_001",
            memory_type="preference"
        ),
        EvaluationQuery(
            query_id="test_002",
            question="What is the Q1 launch deadline?",
            expected_context="Project deadline for Q1 launch is March 15, 2024",
            user_id="user_001",
            memory_type="fact"
        )
    ]


class TestPersistenceEvaluatorIntegration:
    """Integration tests for PersistenceEvaluator."""
    
    @patch('src.evaluation.persistence_evaluator.ChatOpenAI')
    @patch('src.evaluation.persistence_evaluator.load_evaluator')
    def test_evaluator_initialization(self, mock_load_evaluator, mock_chat_openai, mock_config):
        """Test that evaluator initializes correctly with all components."""
        # Mock evaluator loading
        mock_evaluator = Mock()
        mock_load_evaluator.return_value = mock_evaluator
        
        evaluator = PersistenceEvaluator(mock_config)
        
        # Verify initialization
        assert evaluator.config == mock_config
        assert len(evaluator.evaluators) == 5  # All evaluators configured
        assert 'context_recall' in evaluator.evaluators
        assert 'relevance' in evaluator.evaluators
        assert 'memory_accuracy' in evaluator.evaluators
        
        # Verify OpenAI configuration was called
        mock_config.get_openai_config.assert_called()
        
        # Verify evaluators were loaded
        assert mock_load_evaluator.call_count == 5
    
    @patch('src.evaluation.persistence_evaluator.ChatOpenAI')
    @patch('src.evaluation.persistence_evaluator.load_evaluator')
    def test_test_dataset_preparation(self, mock_load_evaluator, mock_chat_openai, mock_config):
        """Test preparation of test dataset from sample memories."""
        # Mock LLM response for query generation
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "When do you prefer meetings?\nWhat are your schedule preferences?"
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        # Mock evaluator loading
        mock_evaluator = Mock()
        mock_load_evaluator.return_value = mock_evaluator
        
        evaluator = PersistenceEvaluator(mock_config)
        
        # Create sample memories
        sample_memories = create_sample_test_dataset()
        
        # Generate test queries
        test_queries = evaluator.prepare_test_dataset(sample_memories, num_queries_per_memory=2)
        
        # Verify test queries were generated
        assert len(test_queries) > 0
        assert all(isinstance(q, EvaluationQuery) for q in test_queries)
        assert all(hasattr(q, 'query_id') for q in test_queries)
        assert all(hasattr(q, 'question') for q in test_queries)
        assert all(hasattr(q, 'expected_context') for q in test_queries)
    
    @patch('src.evaluation.persistence_evaluator.load_evaluator')
    def test_context_recall_evaluation(self, mock_load_evaluator, mock_config):
        """Test context recall evaluation functionality."""
        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.evaluate_strings.return_value = {
            'score': 0.85,
            'reasoning': 'Good context match'
        }
        mock_load_evaluator.return_value = mock_evaluator
        
        evaluator = PersistenceEvaluator(mock_config)
        
        # Test data
        retrieved_contexts = ["User prefers morning meetings"]
        expected_contexts = ["User prefers morning meetings between 9-11 AM"]
        queries = ["When does user prefer meetings?"]
        
        # Run evaluation
        results = evaluator.evaluate_context_recall(retrieved_contexts, expected_contexts, queries)
        
        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], EvaluationResult)
        assert results[0].metric_name == "context_recall"
        assert results[0].score == 0.85
        assert "Good context match" in results[0].reasoning
    
    @patch('src.evaluation.persistence_evaluator.load_evaluator')
    def test_relevance_evaluation(self, mock_load_evaluator, mock_config):
        """Test relevance evaluation functionality."""
        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.evaluate_strings.return_value = {
            'score': 0.90,
            'reasoning': 'Highly relevant response'
        }
        mock_load_evaluator.return_value = mock_evaluator
        
        evaluator = PersistenceEvaluator(mock_config)
        
        # Test data
        retrieved_contexts = ["User prefers morning meetings"]
        queries = ["When does user prefer meetings?"]
        
        # Run evaluation
        results = evaluator.evaluate_relevance(retrieved_contexts, queries)
        
        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], EvaluationResult)
        assert results[0].metric_name == "relevance"
        assert results[0].score == 0.90
    
    @patch('src.evaluation.persistence_evaluator.load_evaluator')
    def test_memory_accuracy_evaluation(self, mock_load_evaluator, mock_config):
        """Test memory accuracy evaluation with custom criteria."""
        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.evaluate_strings.return_value = {
            'score': 0.88,
            'reasoning': 'Accurate memory retrieval'
        }
        mock_load_evaluator.return_value = mock_evaluator
        
        evaluator = PersistenceEvaluator(mock_config)
        
        # Test data
        retrieved_memories = ["User prefers morning meetings between 9-11 AM"]
        queries = ["What are the user's meeting preferences?"]
        
        # Run evaluation
        results = evaluator.evaluate_memory_accuracy(retrieved_memories, queries)
        
        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], EvaluationResult)
        assert results[0].metric_name == "memory_accuracy"
        assert results[0].score == 0.88
    
    @patch('src.evaluation.persistence_evaluator.load_evaluator')
    def test_comparative_evaluation_integration(self, mock_load_evaluator, mock_config, 
                                              mock_vector_persistence, mock_graph_persistence,
                                              sample_test_queries):
        """Test full comparative evaluation integration between both persistence solutions."""
        # Mock evaluators
        mock_evaluator = Mock()
        mock_evaluator.evaluate_strings.return_value = {
            'score': 0.85,
            'reasoning': 'Good performance'
        }
        mock_load_evaluator.return_value = mock_evaluator
        
        evaluator = PersistenceEvaluator(mock_config)
        
        # Mock entity extraction to ensure query_context is called
        with patch.object(evaluator, '_extract_key_entities', return_value=['Meeting', 'Schedule']):
            # Run comparative evaluation
            results = evaluator.run_comparative_evaluation(
                vector_persistence=mock_vector_persistence,
                graph_persistence=mock_graph_persistence,
                test_queries=sample_test_queries,
                include_performance_metrics=True
            )
            
            # Verify structure of results
            assert 'vector_solution' in results
            assert 'graph_solution' in results
            assert 'comparison' in results
            assert 'performance_metrics' in results
            
            # Verify persistence solutions were called
            mock_vector_persistence.search_memories.assert_called()
            mock_graph_persistence.query_context.assert_called()
            
            # Verify performance metrics
            perf_metrics = results['performance_metrics']
            assert 'vector_performance' in perf_metrics
            assert 'graph_performance' in perf_metrics
            assert 'performance_comparison' in perf_metrics
    
    @patch('src.evaluation.persistence_evaluator.load_evaluator')
    def test_performance_metrics_calculation(self, mock_load_evaluator, mock_config,
                                           mock_vector_persistence, mock_graph_persistence,
                                           sample_test_queries):
        """Test that performance metrics are calculated correctly."""
        # Mock evaluators
        mock_evaluator = Mock()
        mock_evaluator.evaluate_strings.return_value = {
            'score': 0.80,
            'reasoning': 'Acceptable performance'
        }
        mock_load_evaluator.return_value = mock_evaluator
        
        evaluator = PersistenceEvaluator(mock_config)
        
        # Run evaluation with performance metrics
        results = evaluator.run_comparative_evaluation(
            vector_persistence=mock_vector_persistence,
            graph_persistence=mock_graph_persistence,
            test_queries=sample_test_queries,
            include_performance_metrics=True
        )
        
        # Verify performance metrics structure
        perf_metrics = results['performance_metrics']
        
        # Check vector performance metrics
        vector_perf = perf_metrics['vector_performance']
        assert 'avg' in vector_perf
        assert 'min' in vector_perf
        assert 'max' in vector_perf
        assert 'error_count' in vector_perf
        assert 'success_rate' in vector_perf
        
        # Check graph performance metrics
        graph_perf = perf_metrics['graph_performance']
        assert 'avg' in graph_perf
        assert 'min' in graph_perf
        assert 'max' in graph_perf
        assert 'error_count' in graph_perf
        assert 'success_rate' in graph_perf
        
        # Check comparison metrics
        comparison = perf_metrics['performance_comparison']
        assert 'vector_faster' in comparison
        assert 'speed_difference' in comparison
    
    @patch('src.evaluation.persistence_evaluator.load_evaluator')
    def test_error_handling_in_comparative_evaluation(self, mock_load_evaluator, mock_config):
        """Test error handling during comparative evaluation."""
        # Mock evaluators
        mock_evaluator = Mock()
        mock_evaluator.evaluate_strings.return_value = {
            'score': 0.75,
            'reasoning': 'Partial success'
        }
        mock_load_evaluator.return_value = mock_evaluator
        
        evaluator = PersistenceEvaluator(mock_config)
        
        # Create failing persistence mocks
        failing_vector = Mock()
        failing_vector.search_memories.side_effect = Exception("Vector search failed")
        
        failing_graph = Mock()
        failing_graph.query_context.side_effect = Exception("Graph query failed")
        failing_graph.search_entities.side_effect = Exception("Graph search failed")
        
        sample_queries = [
            EvaluationQuery(
                query_id="test_001",
                question="Test query",
                expected_context="Test context",
                user_id="user_001",
                memory_type="test"
            )
        ]
        
        # Run evaluation with failing systems
        results = evaluator.run_comparative_evaluation(
            vector_persistence=failing_vector,
            graph_persistence=failing_graph,
            test_queries=sample_queries,
            include_performance_metrics=True
        )
        
        # Verify error handling
        perf_metrics = results['performance_metrics']
        assert perf_metrics['vector_performance']['error_count'] == 1
        assert perf_metrics['graph_performance']['error_count'] == 1
        assert perf_metrics['vector_performance']['success_rate'] == 0.0
        assert perf_metrics['graph_performance']['success_rate'] == 0.0
    
    @patch('src.evaluation.persistence_evaluator.load_evaluator')
    def test_aggregate_score_calculation(self, mock_load_evaluator, mock_config):
        """Test calculation of aggregate scores for comparison."""
        # Mock evaluators
        mock_evaluator = Mock()
        mock_evaluator.evaluate_strings.return_value = {
            'score': 0.80,
            'reasoning': 'Good performance'
        }
        mock_load_evaluator.return_value = mock_evaluator
        
        evaluator = PersistenceEvaluator(mock_config)
        
        # Create mock evaluation results
        vector_results = {
            'relevance': [EvaluationResult('relevance', 0.85, 'Good', {}, datetime.now())],
            'memory_accuracy': [EvaluationResult('memory_accuracy', 0.80, 'Good', {}, datetime.now())],
            'context_recall': [EvaluationResult('context_recall', 0.75, 'Fair', {}, datetime.now())]
        }
        
        graph_results = {
            'relevance': [EvaluationResult('relevance', 0.78, 'Good', {}, datetime.now())],
            'memory_accuracy': [EvaluationResult('memory_accuracy', 0.82, 'Good', {}, datetime.now())],
            'context_recall': [EvaluationResult('context_recall', 0.88, 'Excellent', {}, datetime.now())]
        }
        
        # Calculate aggregate scores
        comparison = evaluator._calculate_aggregate_scores(vector_results, graph_results)
        
        # Verify comparison structure
        assert 'relevance' in comparison
        assert 'memory_accuracy' in comparison
        assert 'context_recall' in comparison
        assert 'overall' in comparison
        
        # Verify overall scoring
        overall = comparison['overall']
        assert 'vector_score' in overall
        assert 'graph_score' in overall
        assert 'difference' in overall
        assert 'winner' in overall
        
        # Verify winner determination
        assert overall['winner'] in ['vector', 'graph', 'tie']
    
    @patch('src.evaluation.persistence_evaluator.load_evaluator')
    def test_evaluation_report_generation(self, mock_load_evaluator, mock_config):
        """Test generation of comprehensive evaluation reports."""
        # Mock evaluators
        mock_evaluator = Mock()
        mock_load_evaluator.return_value = mock_evaluator
        
        evaluator = PersistenceEvaluator(mock_config)
        
        # Create mock comparison results
        comparison_results = {
            'metadata': {
                'evaluation_timestamp': '2024-01-01T12:00:00',
                'num_queries': 5
            },
            'comparison': {
                'overall': {
                    'vector_score': 0.82,
                    'graph_score': 0.85,
                    'difference': 0.03,
                    'winner': 'graph'
                },
                'relevance': {
                    'vector_average': 0.80,
                    'graph_average': 0.85,
                    'winner': 'graph'
                },
                'memory_accuracy': {
                    'vector_average': 0.85,
                    'graph_average': 0.82,
                    'winner': 'vector'
                }
            }
        }
        
        # Generate report
        report = evaluator.generate_evaluation_report(comparison_results)
        
        # Verify report content
        assert "Persistence Solutions Evaluation Report" in report
        assert "2024-01-01T12:00:00" in report
        assert "Queries Evaluated: 5" in report
        assert "Winner: Graph" in report
        assert "Vector Solution Score: 0.820" in report
        assert "Graph Solution Score: 0.850" in report
        assert "Relevance" in report
        assert "Memory Accuracy" in report
        assert "Recommendations" in report


class TestEvaluationComponents:
    """Test individual evaluation components."""
    
    def test_evaluation_result_structure(self):
        """Test EvaluationResult dataclass structure."""
        result = EvaluationResult(
            metric_name="test_metric",
            score=0.85,
            reasoning="Test reasoning",
            metadata={"test": "data"},
            timestamp=datetime.now()
        )
        
        assert result.metric_name == "test_metric"
        assert result.score == 0.85
        assert result.reasoning == "Test reasoning"
        assert result.metadata["test"] == "data"
        assert isinstance(result.timestamp, datetime)
    
    def test_evaluation_query_structure(self):
        """Test EvaluationQuery dataclass structure."""
        query = EvaluationQuery(
            query_id="test_001",
            question="Test question?",
            expected_context="Test context",
            user_id="user_001",
            memory_type="test"
        )
        
        assert query.query_id == "test_001"
        assert query.question == "Test question?"
        assert query.expected_context == "Test context"
        assert query.user_id == "user_001"
        assert query.memory_type == "test"
    
    def test_sample_test_dataset_creation(self):
        """Test creation of sample test dataset."""
        sample_memories = create_sample_test_dataset()
        
        assert len(sample_memories) == 4
        assert all(isinstance(memory, MemoryDocument) for memory in sample_memories)
        assert all(hasattr(memory, 'content') for memory in sample_memories)
        assert all(hasattr(memory, 'user_id') for memory in sample_memories)
        assert all(hasattr(memory, 'memory_type') for memory in sample_memories)
        
        # Verify different memory types are included
        memory_types = {memory.memory_type for memory in sample_memories}
        assert MemoryType.PREFERENCE in memory_types
        assert MemoryType.FACT in memory_types
        assert MemoryType.CONVERSATION in memory_types


class TestEnhancedReportingSystem:
    """Test enhanced reporting system functionality for task 4.3."""
    
    @patch('src.evaluation.persistence_evaluator.load_evaluator')
    def test_detailed_metrics_calculation(self, mock_load_evaluator, mock_config):
        """Test calculation of detailed metrics with statistical analysis."""
        # Mock evaluators
        mock_evaluator = Mock()
        mock_load_evaluator.return_value = mock_evaluator
        
        evaluator = PersistenceEvaluator(mock_config)
        
        # Create mock evaluation results
        vector_results = {
            'relevance': [
                EvaluationResult('relevance', 0.85, 'Good', {}, datetime.now()),
                EvaluationResult('relevance', 0.82, 'Good', {}, datetime.now()),
                EvaluationResult('relevance', 0.88, 'Excellent', {}, datetime.now())
            ],
            'memory_accuracy': [
                EvaluationResult('memory_accuracy', 0.80, 'Good', {}, datetime.now()),
                EvaluationResult('memory_accuracy', 0.83, 'Good', {}, datetime.now())
            ]
        }
        
        graph_results = {
            'relevance': [
                EvaluationResult('relevance', 0.78, 'Good', {}, datetime.now()),
                EvaluationResult('relevance', 0.81, 'Good', {}, datetime.now()),
                EvaluationResult('relevance', 0.84, 'Good', {}, datetime.now())
            ],
            'memory_accuracy': [
                EvaluationResult('memory_accuracy', 0.87, 'Excellent', {}, datetime.now()),
                EvaluationResult('memory_accuracy', 0.85, 'Good', {}, datetime.now())
            ]
        }
        
        # Calculate detailed metrics
        metric_summaries = evaluator.calculate_detailed_metrics(vector_results, graph_results)
        
        # Verify structure and content
        assert 'relevance' in metric_summaries
        assert 'memory_accuracy' in metric_summaries
        
        relevance_summary = metric_summaries['relevance']
        assert relevance_summary.metric_name == 'relevance'
        assert relevance_summary.vector_average > 0.8
        assert relevance_summary.graph_average > 0.7
        assert relevance_summary.winner in ['vector', 'graph', 'tie']
        assert relevance_summary.significance_level in ['high', 'medium', 'low', 'negligible', 'insufficient_data']
        assert hasattr(relevance_summary, 'percentage_difference')
    
    @patch('src.evaluation.persistence_evaluator.load_evaluator')
    def test_performance_metrics_structure(self, mock_load_evaluator, mock_config):
        """Test PerformanceMetrics dataclass functionality."""
        mock_evaluator = Mock()
        mock_load_evaluator.return_value = mock_evaluator
        
        # Create performance metrics
        perf_metrics = PerformanceMetrics(
            avg_query_time=0.15,
            min_query_time=0.10,
            max_query_time=0.25,
            total_query_time=1.50,
            error_count=2,
            success_rate=0.85,
            query_count=10
        )
        
        # Test structure
        assert perf_metrics.avg_query_time == 0.15
        assert perf_metrics.min_query_time == 0.10
        assert perf_metrics.max_query_time == 0.25
        assert perf_metrics.error_count == 2
        assert perf_metrics.success_rate == 0.85
        assert perf_metrics.query_count == 10
        
        # Test serialization
        perf_dict = perf_metrics.to_dict()
        assert isinstance(perf_dict, dict)
        assert perf_dict['avg_query_time'] == 0.15
        assert perf_dict['success_rate'] == 0.85
    
    @patch('src.evaluation.persistence_evaluator.load_evaluator')
    def test_strengths_weaknesses_analysis(self, mock_load_evaluator, mock_config):
        """Test analysis of strengths and weaknesses."""
        # Mock evaluators
        mock_evaluator = Mock()
        mock_load_evaluator.return_value = mock_evaluator
        
        evaluator = PersistenceEvaluator(mock_config)
        
        # Create mock metric summaries
        metric_summaries = {
            'relevance': MetricSummary(
                metric_name='relevance',
                vector_average=0.85,
                graph_average=0.80,
                vector_std=0.03,
                graph_std=0.04,
                difference=-0.05,
                percentage_difference=-6.25,
                winner='vector',
                significance_level='medium'
            ),
            'memory_accuracy': MetricSummary(
                metric_name='memory_accuracy',
                vector_average=0.78,
                graph_average=0.86,
                vector_std=0.02,
                graph_std=0.03,
                difference=0.08,
                percentage_difference=10.26,
                winner='graph',
                significance_level='high'
            )
        }
        
        # Create mock performance metrics
        performance_metrics = {
            'vector': PerformanceMetrics(0.12, 0.08, 0.18, 1.2, 1, 0.90, 10),
            'graph': PerformanceMetrics(0.18, 0.15, 0.25, 1.8, 2, 0.80, 10)
        }
        
        # Analyze strengths and weaknesses
        strengths, weaknesses = evaluator.analyze_strengths_and_weaknesses(metric_summaries, performance_metrics)
        
        # Verify analysis structure
        assert 'vector' in strengths
        assert 'graph' in strengths
        assert 'vector' in weaknesses
        assert 'graph' in weaknesses
        
        # Verify content based on test data
        vector_strengths = strengths['vector']
        graph_strengths = strengths['graph']
        
        # Vector should be stronger in relevance and speed
        assert any('relevance' in strength.lower() for strength in vector_strengths)
        assert any('faster' in strength.lower() for strength in vector_strengths)
        
        # Graph should be stronger in memory accuracy
        assert any('memory accuracy' in strength.lower() for strength in graph_strengths)
    
    @patch('src.evaluation.persistence_evaluator.load_evaluator')
    def test_comprehensive_recommendations_generation(self, mock_load_evaluator, mock_config):
        """Test generation of comprehensive recommendations."""
        # Mock evaluators
        mock_evaluator = Mock()
        mock_load_evaluator.return_value = mock_evaluator
        
        evaluator = PersistenceEvaluator(mock_config)
        
        # Create mock detailed comparison
        detailed_comparison = DetailedComparison(
            metric_summaries={},
            overall_scores={'vector': 0.75, 'graph': 0.90},  # Larger difference to ensure clear winner
            performance_metrics={
                'vector': PerformanceMetrics(0.10, 0.08, 0.15, 1.0, 0, 1.0, 10),
                'graph': PerformanceMetrics(0.15, 0.12, 0.20, 1.5, 1, 0.90, 10)
            },
            strengths_analysis={
                'vector': ['Faster query execution', 'Higher reliability'],
                'graph': ['Superior context recall performance', 'Better memory accuracy']
            },
            weaknesses_analysis={
                'vector': ['Lower context recall accuracy'],
                'graph': ['Slower query performance', 'Lower reliability']
            },
            recommendations=[],
            confidence_scores={'overall': 0.8}
        )
        
        # Generate recommendations
        recommendations = evaluator.generate_comprehensive_recommendations(detailed_comparison)
        
        # Verify recommendations structure and content
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should have overall performance recommendation (graph has higher score) or comparable performance
        recommendation_text = ' '.join(recommendations).lower()
        has_winner_declaration = any('graph' in rec.lower() and ('superior' in rec.lower() or 'advantage' in rec.lower()) for rec in recommendations)
        has_comparable_performance = any('comparable performance' in rec.lower() for rec in recommendations)
        
        # Either should declare graph as winner (large difference) or mention comparable performance (small difference)
        assert has_winner_declaration or has_comparable_performance
        
        # Should have specific use case recommendations
        assert 'vector' in recommendation_text and 'graph' in recommendation_text
    
    @patch('src.evaluation.persistence_evaluator.load_evaluator')
    def test_detailed_comparison_creation(self, mock_load_evaluator, mock_config):
        """Test creation of detailed comparison analysis."""
        # Mock evaluators
        mock_evaluator = Mock()
        mock_load_evaluator.return_value = mock_evaluator
        
        evaluator = PersistenceEvaluator(mock_config)
        
        # Create mock comparison results
        comparison_results = {
            'vector_solution': {
                'relevance': [EvaluationResult('relevance', 0.85, 'Good', {}, datetime.now())],
                'memory_accuracy': [EvaluationResult('memory_accuracy', 0.80, 'Good', {}, datetime.now())]
            },
            'graph_solution': {
                'relevance': [EvaluationResult('relevance', 0.83, 'Good', {}, datetime.now())],
                'memory_accuracy': [EvaluationResult('memory_accuracy', 0.87, 'Good', {}, datetime.now())]
            },
            'comparison': {
                'overall': {
                    'vector_score': 0.82,
                    'graph_score': 0.85
                }
            }
        }
        
        performance_data = {
            'performance_metrics': {
                'vector_performance': {'avg': 0.12, 'min': 0.10, 'max': 0.15, 'total': 1.2, 'error_count': 0, 'success_rate': 1.0},
                'graph_performance': {'avg': 0.18, 'min': 0.15, 'max': 0.22, 'total': 1.8, 'error_count': 1, 'success_rate': 0.9}
            },
            'test_queries': [Mock() for _ in range(10)]
        }
        
        # Create detailed comparison
        detailed_comparison = evaluator.create_detailed_comparison(comparison_results, performance_data)
        
        # Verify structure
        assert isinstance(detailed_comparison, DetailedComparison)
        assert detailed_comparison.overall_scores['vector'] == 0.82
        assert detailed_comparison.overall_scores['graph'] == 0.85
        assert len(detailed_comparison.performance_metrics) == 2
        assert 'vector' in detailed_comparison.strengths_analysis
        assert 'graph' in detailed_comparison.strengths_analysis
        assert len(detailed_comparison.recommendations) > 0
        assert 'overall' in detailed_comparison.confidence_scores
    
    @patch('src.evaluation.persistence_evaluator.load_evaluator')
    def test_structured_report_generation_markdown(self, mock_load_evaluator, mock_config):
        """Test generation of structured reports in Markdown format."""
        # Mock evaluators
        mock_evaluator = Mock()
        mock_load_evaluator.return_value = mock_evaluator
        
        evaluator = PersistenceEvaluator(mock_config)
        
        # Create minimal comparison results
        comparison_results = {
            'vector_solution': {},
            'graph_solution': {},
            'comparison': {
                'overall': {
                    'vector_score': 0.80,
                    'graph_score': 0.85
                }
            },
            'metadata': {
                'num_queries': 5,
                'evaluation_timestamp': '2024-01-01T12:00:00',
                'evaluator_version': '1.0.0'
            }
        }
        
        # Generate report
        from src.evaluation.persistence_evaluator import ReportFormat
        report = evaluator.generate_structured_report(
            comparison_results, 
            output_format=ReportFormat.MARKDOWN
        )
        
        # Verify report content
        assert isinstance(report, str)
        assert "# Persistence Solutions Evaluation Report" in report
        assert "## Executive Summary" in report
        assert "## Overall Results" in report
        assert "## Recommendations" in report
        assert "## Confidence Assessment" in report
        assert "## Methodology" in report
        assert "Vector Solution" in report
        assert "Graph Solution" in report
    
    @patch('src.evaluation.persistence_evaluator.load_evaluator')
    def test_structured_report_generation_json(self, mock_load_evaluator, mock_config):
        """Test generation of structured reports in JSON format."""
        # Mock evaluators
        mock_evaluator = Mock()
        mock_load_evaluator.return_value = mock_evaluator
        
        evaluator = PersistenceEvaluator(mock_config)
        
        # Create minimal comparison results
        comparison_results = {
            'vector_solution': {},
            'graph_solution': {},
            'comparison': {
                'overall': {
                    'vector_score': 0.80,
                    'graph_score': 0.85
                }
            },
            'metadata': {
                'num_queries': 5,
                'evaluation_timestamp': '2024-01-01T12:00:00',
                'evaluator_version': '1.0.0'
            }
        }
        
        # Generate report
        from src.evaluation.persistence_evaluator import ReportFormat
        report = evaluator.generate_structured_report(
            comparison_results, 
            output_format=ReportFormat.JSON
        )
        
        # Verify JSON structure
        import json
        report_data = json.loads(report)
        assert 'report_id' in report_data
        assert 'timestamp' in report_data
        assert 'evaluator_version' in report_data
        assert 'detailed_comparison' in report_data
        assert 'executive_summary' in report_data
        assert 'methodology_notes' in report_data
    
    @patch('src.evaluation.persistence_evaluator.load_evaluator')
    def test_structured_report_generation_csv(self, mock_load_evaluator, mock_config):
        """Test generation of structured reports in CSV format."""
        # Mock evaluators
        mock_evaluator = Mock()
        mock_load_evaluator.return_value = mock_evaluator
        
        evaluator = PersistenceEvaluator(mock_config)
        
        # Create comparison results with metrics
        comparison_results = {
            'vector_solution': {
                'relevance': [EvaluationResult('relevance', 0.85, 'Good', {}, datetime.now())]
            },
            'graph_solution': {
                'relevance': [EvaluationResult('relevance', 0.82, 'Good', {}, datetime.now())]
            },
            'comparison': {
                'overall': {
                    'vector_score': 0.85,
                    'graph_score': 0.82
                }
            },
            'metadata': {
                'num_queries': 2,
                'evaluation_timestamp': '2024-01-01T12:00:00',
                'evaluator_version': '1.0.0'
            }
        }
        
        # Generate report
        from src.evaluation.persistence_evaluator import ReportFormat
        report = evaluator.generate_structured_report(
            comparison_results, 
            output_format=ReportFormat.CSV
        )
        
        # Verify CSV structure
        lines = report.strip().split('\n')
        assert len(lines) >= 2  # Header + at least one data row
        header = lines[0]
        assert 'Report_ID' in header
        assert 'Metric' in header
        assert 'Vector_Avg' in header
        assert 'Graph_Avg' in header
        assert 'Winner' in header
    
    @patch('src.evaluation.persistence_evaluator.load_evaluator')
    def test_structured_report_generation_html(self, mock_load_evaluator, mock_config):
        """Test generation of structured reports in HTML format."""
        # Mock evaluators
        mock_evaluator = Mock()
        mock_load_evaluator.return_value = mock_evaluator
        
        evaluator = PersistenceEvaluator(mock_config)
        
        # Create minimal comparison results
        comparison_results = {
            'vector_solution': {},
            'graph_solution': {},
            'comparison': {
                'overall': {
                    'vector_score': 0.80,
                    'graph_score': 0.85
                }
            },
            'metadata': {
                'num_queries': 5,
                'evaluation_timestamp': '2024-01-01T12:00:00',
                'evaluator_version': '1.0.0'
            }
        }
        
        # Generate report
        from src.evaluation.persistence_evaluator import ReportFormat
        report = evaluator.generate_structured_report(
            comparison_results, 
            output_format=ReportFormat.HTML
        )
        
        # Verify HTML structure
        assert isinstance(report, str)
        assert "<!DOCTYPE html>" in report
        assert "<title>Persistence Solutions Evaluation Report</title>" in report
        assert "<h1>Persistence Solutions Evaluation Report</h1>" in report
        assert "<h2>Executive Summary</h2>" in report
        assert "<h2>Overall Results</h2>" in report
        assert "<table>" in report
        assert "</html>" in report
    
    @patch('src.evaluation.persistence_evaluator.load_evaluator')
    def test_confidence_scores_calculation(self, mock_load_evaluator, mock_config):
        """Test calculation of confidence scores."""
        # Mock evaluators
        mock_evaluator = Mock()
        mock_load_evaluator.return_value = mock_evaluator
        
        evaluator = PersistenceEvaluator(mock_config)
        
        # Create mock metric summaries
        metric_summaries = {
            'relevance': MetricSummary(
                metric_name='relevance',
                vector_average=0.85,
                graph_average=0.80,
                vector_std=0.03,
                graph_std=0.04,
                difference=-0.05,
                percentage_difference=-6.25,
                winner='vector',
                significance_level='high'  # Should result in high confidence
            ),
            'memory_accuracy': MetricSummary(
                metric_name='memory_accuracy',
                vector_average=0.78,
                graph_average=0.82,
                vector_std=0.02,
                graph_std=0.03,
                difference=0.04,
                percentage_difference=5.13,
                winner='graph',
                significance_level='low'  # Should result in lower confidence
            )
        }
        
        # Create mock performance metrics
        performance_metrics = {
            'vector': PerformanceMetrics(0.12, 0.08, 0.18, 1.2, 0, 0.98, 10),  # High success rate
            'graph': PerformanceMetrics(0.18, 0.15, 0.25, 1.8, 2, 0.80, 10)   # Lower success rate
        }
        
        # Calculate confidence scores
        confidence_scores = evaluator._calculate_confidence_scores(metric_summaries, performance_metrics)
        
        # Verify confidence score structure and logic
        assert 'overall' in confidence_scores
        assert 'vector_performance' in confidence_scores
        assert 'graph_performance' in confidence_scores
        
        # Overall confidence should be between high and low significance levels
        assert 0.3 <= confidence_scores['overall'] <= 0.9
        
        # Vector performance confidence should be higher due to better success rate
        assert confidence_scores['vector_performance'] > confidence_scores['graph_performance']
        
        # All confidence scores should be between 0 and 1
        for score in confidence_scores.values():
            assert 0.0 <= score <= 1.0
    
    @patch('src.evaluation.persistence_evaluator.load_evaluator')
    def test_significance_level_calculation(self, mock_load_evaluator, mock_config):
        """Test statistical significance level calculation."""
        # Mock evaluators
        mock_evaluator = Mock()
        mock_load_evaluator.return_value = mock_evaluator
        
        evaluator = PersistenceEvaluator(mock_config)
        
        # Test different significance scenarios
        
        # High significance: large difference, small standard deviations
        vector_scores_high = [0.80, 0.82, 0.81, 0.83, 0.79]  # Mean ~0.81, low std
        graph_scores_high = [0.90, 0.92, 0.91, 0.93, 0.89]   # Mean ~0.91, low std
        significance_high = evaluator._calculate_significance_level(vector_scores_high, graph_scores_high)
        assert significance_high == 'high'
        
        # Low significance: small difference
        vector_scores_low = [0.80, 0.82, 0.81, 0.83, 0.79]   # Mean ~0.81
        graph_scores_low = [0.81, 0.83, 0.82, 0.84, 0.80]    # Mean ~0.82
        significance_low = evaluator._calculate_significance_level(vector_scores_low, graph_scores_low)
        assert significance_low in ['low', 'negligible']
        
        # Insufficient data
        vector_scores_insufficient = [0.80]
        graph_scores_insufficient = [0.85]
        significance_insufficient = evaluator._calculate_significance_level(vector_scores_insufficient, graph_scores_insufficient)
        assert significance_insufficient == 'insufficient_data'
    
    def test_report_format_enum(self):
        """Test ReportFormat enum functionality."""
        from src.evaluation.persistence_evaluator import ReportFormat
        
        # Test enum values
        assert ReportFormat.MARKDOWN.value == "markdown"
        assert ReportFormat.JSON.value == "json"
        assert ReportFormat.CSV.value == "csv"
        assert ReportFormat.HTML.value == "html"
        
        # Test enum iteration
        formats = list(ReportFormat)
        assert len(formats) == 4
        assert ReportFormat.MARKDOWN in formats
        assert ReportFormat.JSON in formats
        assert ReportFormat.CSV in formats
        assert ReportFormat.HTML in formats


if __name__ == "__main__":
    pytest.main([__file__])
