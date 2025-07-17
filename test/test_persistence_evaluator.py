"""
Unit tests for PersistenceEvaluator class and evaluation framework.

These tests verify the LangChain evaluation tools integration, test dataset preparation,
and comparative analysis functionality for persistence solutions.
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.config.config_manager import ConfigManager
from src.evaluation.persistence_evaluator import (
    PersistenceEvaluator, 
    EvaluationResult, 
    EvaluationQuery,
    create_sample_test_dataset
)
from src.persistence.memory_document import MemoryDocument, MemoryType


class TestPersistenceEvaluator:
    """Test cases for PersistenceEvaluator class."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration manager for testing."""
        config = Mock(spec=ConfigManager)
        config.get_openai_config.return_value = {
            'api_key': 'test-key',
            'base_url': 'https://api.openai.com/v1',
            'model': 'gpt-4o-mini'
        }
        return config
    
    @pytest.fixture
    def sample_memories(self):
        """Sample memory documents for testing."""
        return [
            MemoryDocument(
                content="User prefers morning meetings between 9-11 AM",
                user_id="user_001",
                memory_type=MemoryType.PREFERENCE,
                timestamp=datetime.now()
            ),
            MemoryDocument(
                content="Project deadline is March 15, 2024",
                user_id="user_001",
                memory_type=MemoryType.FACT,
                timestamp=datetime.now()
            )
        ]
    
    @pytest.fixture
    def mock_evaluators(self):
        """Mock LangChain evaluators for testing."""
        evaluators = {}
        
        # Mock context recall evaluator
        context_recall = Mock()
        context_recall.evaluate_strings.return_value = {
            'score': 0.85,
            'reasoning': 'Good context match',
            'value': 'Y'
        }
        evaluators['context_recall'] = context_recall
        
        # Mock relevance evaluator
        relevance = Mock()
        relevance.evaluate_strings.return_value = {
            'score': 0.90,
            'reasoning': 'Highly relevant',
            'value': 'Y'
        }
        evaluators['relevance'] = relevance
        
        # Mock memory accuracy evaluator
        memory_accuracy = Mock()
        memory_accuracy.evaluate_strings.return_value = {
            'score': 0.80,
            'reasoning': 'Accurate memory retrieval',
            'value': 'Y'
        }
        evaluators['memory_accuracy'] = memory_accuracy
        
        return evaluators
    
    @patch('src.evaluation.persistence_evaluator.ChatOpenAI')
    @patch('src.evaluation.persistence_evaluator.load_evaluator')
    def test_setup_langchain_evaluators(self, mock_load_evaluator, mock_chat_openai, mock_config):
        """Test LangChain evaluators setup."""
        # Setup mocks
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        mock_evaluator = Mock()
        mock_load_evaluator.return_value = mock_evaluator
        
        # Create evaluator
        evaluator = PersistenceEvaluator(mock_config)
        
        # Verify LLM initialization
        mock_chat_openai.assert_called_once()
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs['api_key'] == 'test-key'
        assert call_kwargs['model'] == 'gpt-4o-mini'
        assert call_kwargs['temperature'] == 0.0
        
        # Verify evaluators were loaded
        assert mock_load_evaluator.call_count >= 5  # At least 5 evaluators
        assert len(evaluator.evaluators) >= 5
    
    def test_persistence_evaluator_initialization(self, mock_config):
        """Test PersistenceEvaluator initialization."""
        with patch('src.evaluation.persistence_evaluator.ChatOpenAI'), \
             patch('src.evaluation.persistence_evaluator.load_evaluator'):
            
            evaluator = PersistenceEvaluator(mock_config)
            
            assert evaluator.config == mock_config
            assert isinstance(evaluator.evaluators, dict)
            assert isinstance(evaluator.test_queries, list)
    
    @patch('src.evaluation.persistence_evaluator.ChatOpenAI')
    def test_prepare_test_dataset(self, mock_chat_openai, mock_config, sample_memories):
        """Test test dataset preparation with query generation."""
        # Setup mock LLM response
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "What are the user's meeting preferences?\nWhen does the user prefer meetings?"
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        # Create evaluator with mocked setup
        with patch('src.evaluation.persistence_evaluator.load_evaluator'):
            evaluator = PersistenceEvaluator(mock_config)
        
        # Prepare test dataset
        test_queries = evaluator.prepare_test_dataset(sample_memories, num_queries_per_memory=2)
        
        # Verify results
        assert len(test_queries) <= len(sample_memories) * 2  # May be fewer due to processing
        assert all(isinstance(query, EvaluationQuery) for query in test_queries)
        
        # Verify LLM was called for each memory
        assert mock_llm.invoke.call_count == len(sample_memories)
        
        # Check test query structure
        if test_queries:
            test_query = test_queries[0]
            assert test_query.query_id
            assert test_query.question
            assert test_query.expected_context
            assert test_query.user_id
            assert test_query.memory_type
    
    def test_evaluate_context_recall(self, mock_config, mock_evaluators):
        """Test context recall evaluation."""
        with patch('src.evaluation.persistence_evaluator.ChatOpenAI'), \
             patch('src.evaluation.persistence_evaluator.load_evaluator'):
            
            evaluator = PersistenceEvaluator(mock_config)
            evaluator.evaluators = mock_evaluators
        
        # Test data
        retrieved_contexts = ["User prefers morning meetings"]
        expected_contexts = ["User prefers morning meetings between 9-11 AM"]
        queries = ["What are user's meeting preferences?"]
        
        # Evaluate context recall
        results = evaluator.evaluate_context_recall(retrieved_contexts, expected_contexts, queries)
        
        # Verify results
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, EvaluationResult)
        assert result.metric_name == "context_recall"
        assert result.score == 0.85
        assert result.reasoning == "Good context match"
        assert result.metadata['query_index'] == 0
        assert isinstance(result.timestamp, datetime)
        
        # Verify evaluator was called correctly
        mock_evaluators['context_recall'].evaluate_strings.assert_called_once_with(
            prediction="User prefers morning meetings",
            reference="User prefers morning meetings between 9-11 AM",
            input="What are user's meeting preferences?"
        )
    
    def test_evaluate_relevance(self, mock_config, mock_evaluators):
        """Test relevance evaluation."""
        with patch('src.evaluation.persistence_evaluator.ChatOpenAI'), \
             patch('src.evaluation.persistence_evaluator.load_evaluator'):
            
            evaluator = PersistenceEvaluator(mock_config)
            evaluator.evaluators = mock_evaluators
        
        # Test data
        retrieved_contexts = ["User prefers morning meetings"]
        queries = ["What are user's meeting preferences?"]
        
        # Evaluate relevance
        results = evaluator.evaluate_relevance(retrieved_contexts, queries)
        
        # Verify results
        assert len(results) == 1
        result = results[0]
        assert result.metric_name == "relevance"
        assert result.score == 0.90
        assert result.reasoning == "Highly relevant"
        
        # Verify evaluator was called correctly
        mock_evaluators['relevance'].evaluate_strings.assert_called_once_with(
            prediction="User prefers morning meetings",
            input="What are user's meeting preferences?"
        )
    
    def test_evaluate_memory_accuracy(self, mock_config, mock_evaluators):
        """Test memory accuracy evaluation."""
        with patch('src.evaluation.persistence_evaluator.ChatOpenAI'), \
             patch('src.evaluation.persistence_evaluator.load_evaluator'):
            
            evaluator = PersistenceEvaluator(mock_config)
            evaluator.evaluators = mock_evaluators
        
        # Test data
        retrieved_memories = ["User prefers morning meetings"]
        queries = ["What are user's meeting preferences?"]
        
        # Evaluate memory accuracy
        results = evaluator.evaluate_memory_accuracy(retrieved_memories, queries)
        
        # Verify results
        assert len(results) == 1
        result = results[0]
        assert result.metric_name == "memory_accuracy"
        assert result.score == 0.80
        assert result.reasoning == "Accurate memory retrieval"
        
        # Verify evaluator was called correctly
        mock_evaluators['memory_accuracy'].evaluate_strings.assert_called_once_with(
            prediction="User prefers morning meetings",
            input="What are user's meeting preferences?"
        )
    
    def test_compare_persistence_solutions(self, mock_config, mock_evaluators):
        """Test comparative evaluation of persistence solutions."""
        with patch('src.evaluation.persistence_evaluator.ChatOpenAI'), \
             patch('src.evaluation.persistence_evaluator.load_evaluator'):
            
            evaluator = PersistenceEvaluator(mock_config)
            evaluator.evaluators = mock_evaluators
        
        # Test data
        vector_results = ["Vector result"]
        graph_results = ["Graph result"]
        queries = ["Test query"]
        expected_contexts = ["Expected context"]
        
        # Compare solutions
        comparison_results = evaluator.compare_persistence_solutions(
            vector_results, graph_results, queries, expected_contexts
        )
        
        # Verify structure
        assert 'vector_solution' in comparison_results
        assert 'graph_solution' in comparison_results
        assert 'comparison' in comparison_results
        assert 'metadata' in comparison_results
        
        # Check metadata
        metadata = comparison_results['metadata']
        assert 'evaluation_timestamp' in metadata
        assert metadata['num_queries'] == 1
        assert 'evaluator_version' in metadata
        
        # Check comparison results
        comparison = comparison_results['comparison']
        assert 'overall' in comparison
        assert 'relevance' in comparison
        assert 'memory_accuracy' in comparison
        assert 'context_recall' in comparison
        
        # Verify overall scores
        overall = comparison['overall']
        assert 'vector_score' in overall
        assert 'graph_score' in overall
        assert 'winner' in overall
    
    def test_calculate_aggregate_scores(self, mock_config):
        """Test aggregate score calculation."""
        with patch('src.evaluation.persistence_evaluator.ChatOpenAI'), \
             patch('src.evaluation.persistence_evaluator.load_evaluator'):
            
            evaluator = PersistenceEvaluator(mock_config)
        
        # Create mock evaluation results
        vector_results = {
            'relevance': [
                EvaluationResult("relevance", 0.8, "Good", {}, datetime.now()),
                EvaluationResult("relevance", 0.9, "Great", {}, datetime.now())
            ],
            'memory_accuracy': [
                EvaluationResult("memory_accuracy", 0.7, "OK", {}, datetime.now())
            ]
        }
        
        graph_results = {
            'relevance': [
                EvaluationResult("relevance", 0.85, "Good", {}, datetime.now())
            ],
            'memory_accuracy': [
                EvaluationResult("memory_accuracy", 0.75, "OK", {}, datetime.now())
            ]
        }
        
        # Calculate aggregate scores
        comparison = evaluator._calculate_aggregate_scores(vector_results, graph_results)
        
        # Verify calculations (using approximate equality for floating point)
        assert abs(comparison['relevance']['vector_average'] - 0.85) < 0.001  # (0.8 + 0.9) / 2
        assert abs(comparison['relevance']['graph_average'] - 0.85) < 0.001
        assert abs(comparison['memory_accuracy']['vector_average'] - 0.7) < 0.001
        assert abs(comparison['memory_accuracy']['graph_average'] - 0.75) < 0.001
        
        # Check overall score calculation
        assert 'overall' in comparison
        assert 'vector_score' in comparison['overall']
        assert 'graph_score' in comparison['overall']
        assert 'winner' in comparison['overall']
    
    def test_generate_evaluation_report(self, mock_config):
        """Test evaluation report generation."""
        with patch('src.evaluation.persistence_evaluator.ChatOpenAI'), \
             patch('src.evaluation.persistence_evaluator.load_evaluator'):
            
            evaluator = PersistenceEvaluator(mock_config)
        
        # Mock comparison results
        comparison_results = {
            'metadata': {
                'evaluation_timestamp': '2024-01-01T12:00:00',
                'num_queries': 5
            },
            'comparison': {
                'overall': {
                    'winner': 'vector',
                    'vector_score': 0.85,
                    'graph_score': 0.80,
                    'difference': -0.05
                },
                'relevance': {
                    'vector_average': 0.9,
                    'graph_average': 0.85,
                    'winner': 'vector'
                }
            }
        }
        
        # Generate report
        report = evaluator.generate_evaluation_report(comparison_results)
        
        # Verify report content
        assert "# Persistence Solutions Evaluation Report" in report
        assert "Generated: 2024-01-01T12:00:00" in report
        assert "Queries Evaluated: 5" in report
        assert "**Winner:** Vector" in report
        assert "Vector Solution Score: 0.850" in report
        assert "Graph Solution Score: 0.800" in report
        assert "### Relevance" in report
        assert "## Recommendations" in report
    
    def test_error_handling_in_evaluation(self, mock_config):
        """Test error handling in evaluation methods."""
        with patch('src.evaluation.persistence_evaluator.ChatOpenAI'), \
             patch('src.evaluation.persistence_evaluator.load_evaluator'):
            
            evaluator = PersistenceEvaluator(mock_config)
            
            # Mock evaluator that raises exception
            mock_evaluator = Mock()
            mock_evaluator.evaluate_strings.side_effect = Exception("Evaluation failed")
            evaluator.evaluators = {'relevance': mock_evaluator}
        
        # Test that evaluation continues despite individual failures
        results = evaluator.evaluate_relevance(["test"], ["query"])
        
        # Should return empty results but not crash
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_evaluation_query_dataclass(self):
        """Test EvaluationQuery dataclass."""
        query = EvaluationQuery(
            query_id="test-id",
            question="Test question?",
            expected_context="Test context",
            user_id="user_001",
            memory_type="preference"
        )
        
        assert query.query_id == "test-id"
        assert query.question == "Test question?"
        assert query.expected_context == "Test context"
        assert query.user_id == "user_001"
        assert query.memory_type == "preference"
    
    def test_evaluation_result_dataclass(self):
        """Test EvaluationResult dataclass."""
        timestamp = datetime.now()
        result = EvaluationResult(
            metric_name="test_metric",
            score=0.85,
            reasoning="Test reasoning",
            metadata={"key": "value"},
            timestamp=timestamp
        )
        
        assert result.metric_name == "test_metric"
        assert result.score == 0.85
        assert result.reasoning == "Test reasoning"
        assert result.metadata == {"key": "value"}
        assert result.timestamp == timestamp


class TestSampleDatasetCreation:
    """Test cases for sample dataset creation utilities."""
    
    def test_create_sample_test_dataset(self):
        """Test creation of sample memory documents."""
        sample_memories = create_sample_test_dataset()
        
        # Verify dataset structure
        assert isinstance(sample_memories, list)
        assert len(sample_memories) > 0
        assert all(isinstance(memory, MemoryDocument) for memory in sample_memories)
        
        # Verify memory types
        memory_types = [memory.memory_type for memory in sample_memories]
        assert MemoryType.PREFERENCE in memory_types
        assert MemoryType.FACT in memory_types
        assert MemoryType.CONVERSATION in memory_types
        
        # Verify all memories have required fields
        for memory in sample_memories:
            assert memory.content
            assert memory.user_id
            assert isinstance(memory.timestamp, datetime)
            assert isinstance(memory.memory_type, MemoryType)


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI response for testing."""
    response = Mock()
    response.content = "What are the user's preferences?\nWhen does the user work?"
    return response


def test_response_handling_edge_cases(mock_openai_response):
    """Test edge cases in response handling."""
    # Test string response
    assert isinstance(mock_openai_response.content, str)
    
    # Test response splitting
    questions = [q.strip() for q in mock_openai_response.content.split('\n') if q.strip()]
    assert len(questions) == 2
    assert "What are the user's preferences?" in questions
    assert "When does the user work?" in questions


if __name__ == "__main__":
    pytest.main([__file__]) 