"""Tests for Neo4j Docker container management."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import docker
from docker.errors import DockerException, NotFound

from src.config.config_manager import ConfigManager
from src.persistence.neo4j_docker_manager import Neo4jDockerManager, Neo4jDockerManagerError


class TestNeo4jDockerManager:
    """Test cases for Neo4j Docker manager."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock configuration manager."""
        config_manager = Mock(spec=ConfigManager)
        
        # Mock config object
        mock_config = Mock()
        mock_config.neo4j_uri = "neo4j://localhost:7687"
        mock_config.neo4j_username = "neo4j"
        mock_config.neo4j_password = "password"
        mock_config.neo4j_database = "neo4j"
        
        config_manager.get_config.return_value = mock_config
        config_manager.get_neo4j_config.return_value = {
            'uri': 'neo4j://localhost:7687',
            'username': 'neo4j',
            'password': 'password',
            'database': 'neo4j'
        }
        
        return config_manager
    
    @patch('src.persistence.neo4j_docker_manager.docker')
    def test_initialization_success(self, mock_docker, mock_config_manager):
        """Test successful Docker manager initialization."""
        # Mock Docker client
        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client
        
        # Create instance
        manager = Neo4jDockerManager(mock_config_manager)
        
        # Verify initialization
        assert manager.config_manager == mock_config_manager
        assert manager.container_name == "neo4j-context-persistence"
        assert manager.image_name == "neo4j:latest"
        assert manager.neo4j_port == 7687
        assert manager.http_port == 7474
        assert manager.https_port == 7473
        assert isinstance(manager.data_dir, Path)
        assert isinstance(manager.logs_dir, Path)
        assert isinstance(manager.import_dir, Path)
        
        mock_docker.from_env.assert_called_once()
    
    @patch('src.persistence.neo4j_docker_manager.docker')
    def test_initialization_docker_error(self, mock_docker, mock_config_manager):
        """Test Docker manager initialization with Docker error."""
        # Mock Docker client error
        mock_docker.from_env.side_effect = DockerException("Docker not available")
        # Test initialization failure
        with pytest.raises(Neo4jDockerManagerError) as exc_info:
            Neo4jDockerManager(mock_config_manager)
        assert "Failed to initialize Docker client" in str(exc_info.value)
    @patch('src.persistence.neo4j_docker_manager.docker')
    def test_ensure_directories(self, mock_docker, mock_config_manager, tmp_path):
        """Test directory creation for persistent volumes."""
        # Mock Docker client
        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client
        
        # Create manager with temporary directories
        manager = Neo4jDockerManager(mock_config_manager)
        manager.data_dir = tmp_path / "data"
        manager.logs_dir = tmp_path / "logs"
        manager.import_dir = tmp_path / "import"
        
        # Call private method
        manager._ensure_directories()
        
        # Verify directories exist
        assert manager.data_dir.exists()
        assert manager.logs_dir.exists()
        assert manager.import_dir.exists()
    
    @patch('src.persistence.neo4j_docker_manager.docker')
    def test_get_container_config(self, mock_docker, mock_config_manager):
        """Test container configuration generation."""
        # Mock Docker client
        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client
        
        # Create manager
        manager = Neo4jDockerManager(mock_config_manager)
        
        # Get container config
        config = manager._get_container_config()
        
        # Verify configuration
        assert config['image'] == "neo4j:latest"
        assert config['name'] == "neo4j-context-persistence"
        assert config['detach'] is True
        assert '7687/tcp' in config['ports']
        assert '7474/tcp' in config['ports']
        assert '7473/tcp' in config['ports']
        assert 'NEO4J_AUTH' in config['environment']
        assert config['environment']['NEO4J_AUTH'] == "neo4j/password"
        assert len(config['volumes']) == 3
    
    @patch('src.persistence.neo4j_docker_manager.docker')
    def test_start_container_new(self, mock_docker, mock_config_manager, tmp_path):
        """Test starting a new container."""
        # Mock Docker client and operations
        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client
        
        # Mock container operations
        mock_client.containers.get.side_effect = docker.errors.NotFound("Container not found")
        mock_client.images.get.return_value = Mock()  # Image exists
        mock_container = Mock()
        mock_client.containers.run.return_value = mock_container
        
        # Create manager with temporary directories
        manager = Neo4jDockerManager(mock_config_manager)
        manager.data_dir = tmp_path / "data"
        manager.logs_dir = tmp_path / "logs"
        manager.import_dir = tmp_path / "import"
        
        # Mock health check
        manager._wait_for_container_ready = Mock(return_value=True)
        
        # Start container
        result = manager.start_container()
        
        # Verify result
        assert result is True
        mock_client.containers.run.assert_called_once()
        manager._wait_for_container_ready.assert_called_once()
    
    @patch('src.persistence.neo4j_docker_manager.docker')
    def test_start_container_existing_running(self, mock_docker, mock_config_manager):
        """Test starting an already running container."""
        # Mock Docker client
        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client
        
        # Mock existing running container
        mock_container = Mock()
        mock_container.status = 'running'
        mock_client.containers.get.return_value = mock_container
        
        # Create manager
        manager = Neo4jDockerManager(mock_config_manager)
        
        # Start container
        result = manager.start_container()
        
        # Verify result
        assert result is True
        mock_client.containers.run.assert_not_called()
    
    @patch('src.persistence.neo4j_docker_manager.docker')
    def test_start_container_existing_stopped(self, mock_docker, mock_config_manager):
        """Test starting an existing stopped container."""
        # Mock Docker client
        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client
        
        # Mock existing stopped container
        mock_container = Mock()
        mock_container.status = 'exited'
        mock_client.containers.get.return_value = mock_container
        
        # Create manager
        manager = Neo4jDockerManager(mock_config_manager)
        manager._wait_for_container_ready = Mock(return_value=True)
        
        # Start container
        result = manager.start_container()
        
        # Verify result
        assert result is True
        mock_container.start.assert_called_once()
        manager._wait_for_container_ready.assert_called_once()
    
    @patch('src.persistence.neo4j_docker_manager.docker')
    def test_stop_container_success(self, mock_docker, mock_config_manager):
        """Test stopping a running container."""
        # Mock Docker client
        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client
        
        # Mock running container
        mock_container = Mock()
        mock_container.status = 'running'
        mock_client.containers.get.return_value = mock_container
        
        # Mock container stop and reload
        def mock_reload():
            mock_container.status = 'exited'
        mock_container.reload = mock_reload
        
        # Create manager
        manager = Neo4jDockerManager(mock_config_manager)
        
        # Stop container
        result = manager.stop_container()
        
        # Verify result
        assert result is True
        mock_container.stop.assert_called_once_with(timeout=30)
    
    @patch('src.persistence.neo4j_docker_manager.docker')
    def test_stop_container_not_found(self, mock_docker, mock_config_manager):
        """Test stopping a non-existent container."""
        # Mock Docker client
        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client
        
        # Mock container not found
        mock_client.containers.get.side_effect = docker.errors.NotFound("Container not found")
        
        # Create manager
        manager = Neo4jDockerManager(mock_config_manager)
        
        # Stop container
        result = manager.stop_container()
        
        # Verify result
        assert result is True  # Not found is considered success
    
    @patch('src.persistence.neo4j_docker_manager.docker')
    def test_remove_container_success(self, mock_docker, mock_config_manager):
        """Test removing a container."""
        # Mock Docker client
        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client
        
        # Mock container
        mock_container = Mock()
        mock_container.status = 'exited'
        mock_client.containers.get.return_value = mock_container
        
        # Create manager
        manager = Neo4jDockerManager(mock_config_manager)
        
        # Remove container
        result = manager.remove_container()
        
        # Verify result
        assert result is True
        mock_container.remove.assert_called_once()
    
    @patch('src.persistence.neo4j_docker_manager.docker')
    def test_remove_container_running(self, mock_docker, mock_config_manager):
        """Test removing a running container (should stop first)."""
        # Mock Docker client
        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client
        
        # Mock running container
        mock_container = Mock()
        mock_container.status = 'running'
        mock_client.containers.get.return_value = mock_container
        
        # Create manager
        manager = Neo4jDockerManager(mock_config_manager)
        manager.stop_container = Mock(return_value=True)
        
        # Remove container
        result = manager.remove_container()
        
        # Verify result
        assert result is True
        manager.stop_container.assert_called_once()
        mock_container.remove.assert_called_once()
    
    @patch('src.persistence.neo4j_docker_manager.docker')
    def test_is_container_running_true(self, mock_docker, mock_config_manager):
        """Test checking if container is running (true case)."""
        # Mock Docker client
        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client
        
        # Mock running container
        mock_container = Mock()
        mock_container.status = 'running'
        mock_client.containers.get.return_value = mock_container
        
        # Create manager
        manager = Neo4jDockerManager(mock_config_manager)
        
        # Check if running
        result = manager.is_container_running()
        
        # Verify result
        assert result is True
    
    @patch('src.persistence.neo4j_docker_manager.docker')
    def test_is_container_running_false(self, mock_docker, mock_config_manager):
        """Test checking if container is running (false case)."""
        # Mock Docker client
        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client
        
        # Mock stopped container
        mock_container = Mock()
        mock_container.status = 'exited'
        mock_client.containers.get.return_value = mock_container
        
        # Create manager
        manager = Neo4jDockerManager(mock_config_manager)
        
        # Check if running
        result = manager.is_container_running()
        
        # Verify result
        assert result is False
    
    @patch('src.persistence.neo4j_docker_manager.docker')
    def test_is_container_running_not_found(self, mock_docker, mock_config_manager):
        """Test checking if container is running (not found case)."""
        # Mock Docker client
        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client
        
        # Mock container not found
        mock_client.containers.get.side_effect = docker.errors.NotFound("Container not found")
        
        # Create manager
        manager = Neo4jDockerManager(mock_config_manager)
        
        # Check if running
        result = manager.is_container_running()
        
        # Verify result
        assert result is False
    
    @patch('src.persistence.neo4j_docker_manager.docker')
    def test_get_container_status_running(self, mock_docker, mock_config_manager):
        """Test getting container status for running container."""
        # Mock Docker client
        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client
        
        # Mock running container
        mock_container = Mock()
        mock_container.status = 'running'
        mock_container.attrs = {
            'Created': '2023-01-01T00:00:00Z',
            'State': {'StartedAt': '2023-01-01T00:00:01Z'},
            'NetworkSettings': {'Ports': {'7687/tcp': [{'HostPort': '7687'}]}},
            'Config': {'Image': 'neo4j:latest'}
        }
        mock_client.containers.get.return_value = mock_container
        
        # Create manager
        manager = Neo4jDockerManager(mock_config_manager)
        manager._check_container_health = Mock(return_value='healthy')
        
        # Get status
        status = manager.get_container_status()
        
        # Verify status
        assert status['exists'] is True
        assert status['status'] == 'running'
        assert status['health'] == 'healthy'
        assert 'created' in status
        assert 'started' in status
        assert 'ports' in status
        assert 'image' in status
    
    @patch('src.persistence.neo4j_docker_manager.docker')
    def test_get_container_status_not_found(self, mock_docker, mock_config_manager):
        """Test getting container status for non-existent container."""
        # Mock Docker client
        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client
        
        # Mock container not found
        mock_client.containers.get.side_effect = docker.errors.NotFound("Container not found")
        
        # Create manager
        manager = Neo4jDockerManager(mock_config_manager)
        
        # Get status
        status = manager.get_container_status()
        
        # Verify status
        assert status['exists'] is False
        assert status['status'] == 'not_found'
        assert status['health'] == 'unknown'
    
    @patch('src.persistence.neo4j_docker_manager.docker')
    @patch('neo4j.GraphDatabase')
    def test_check_container_health_healthy(self, mock_graph_db, mock_docker, mock_config_manager):
        """Test container health check (healthy case)."""
        # Mock Docker client
        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client
        
        # Mock Neo4j driver and session
        mock_driver = Mock()
        # Mock Neo4j driver configuration
        mock_config_manager.get_neo4j_driver_config.return_value = {
            "uri": "neo4j://localhost:7687",
            "auth": ("neo4j", "password"),
            "database": "neo4j"
        }
        mock_session = Mock()
        mock_result = Mock()
        mock_record = Mock()
        mock_record.__getitem__ = Mock(return_value=1)
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        # Set up context manager properly
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_session)
        mock_context.__exit__ = Mock(return_value=None)
        mock_driver.session.return_value = mock_context
        mock_graph_db.driver.return_value = mock_driver
        
        # Create manager
        manager = Neo4jDockerManager(mock_config_manager)
        manager.is_container_running = Mock(return_value=True)
        
        # Check health
        health = manager._check_container_health()
        
        # Verify health
        assert health == 'healthy'
        mock_session.run.assert_called_once_with("RETURN 1 as test")
    
    @patch('src.persistence.neo4j_docker_manager.docker')
    def test_check_container_health_not_running(self, mock_docker, mock_config_manager):
        """Test container health check (not running case)."""
        # Mock Docker client
        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client
        
        # Create manager
        manager = Neo4jDockerManager(mock_config_manager)
        manager.is_container_running = Mock(return_value=False)
        
        # Check health
        health = manager._check_container_health()
        
        # Verify health
        assert health == 'unhealthy'
    
    @patch('src.persistence.neo4j_docker_manager.docker')
    @patch('neo4j.GraphDatabase')
    def test_check_container_health_connection_error(self, mock_graph_db, mock_docker, mock_config_manager):
        """Test container health check (connection error case)."""
        # Mock Docker client
        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client
        
        # Mock Neo4j connection error
        mock_graph_db.driver.side_effect = Exception("Connection failed")
        
        # Create manager
        manager = Neo4jDockerManager(mock_config_manager)
        manager.is_container_running = Mock(return_value=True)
        
        # Check health
        health = manager._check_container_health()
        
        # Verify health
        assert health == 'starting'
    
    @patch('src.persistence.neo4j_docker_manager.docker')
    def test_get_logs_success(self, mock_docker, mock_config_manager):
        """Test getting container logs."""
        # Mock Docker client
        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client
        
        # Mock container with logs
        mock_container = Mock()
        mock_container.logs.return_value = b"Test log output\nAnother line"
        mock_client.containers.get.return_value = mock_container
        
        # Create manager
        manager = Neo4jDockerManager(mock_config_manager)
        
        # Get logs
        logs = manager.get_logs(tail=50)
        
        # Verify logs
        assert logs == "Test log output\nAnother line"
        mock_container.logs.assert_called_once_with(tail=50, timestamps=True)
    
    @patch('src.persistence.neo4j_docker_manager.docker')
    def test_get_logs_container_not_found(self, mock_docker, mock_config_manager):
        """Test getting logs from non-existent container."""
        # Mock Docker client
        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client
        
        # Mock container not found
        mock_client.containers.get.side_effect = docker.errors.NotFound("Container not found")
        
        # Create manager
        manager = Neo4jDockerManager(mock_config_manager)
        
        # Get logs
        logs = manager.get_logs()
        
        # Verify logs
        assert logs == "Container not found"
    
    @patch('src.persistence.neo4j_docker_manager.docker')
    def test_restart_container_success(self, mock_docker, mock_config_manager):
        """Test restarting container."""
        # Mock Docker client
        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client
        
        # Create manager
        manager = Neo4jDockerManager(mock_config_manager)
        manager.stop_container = Mock(return_value=True)
        manager.start_container = Mock(return_value=True)
        
        # Restart container
        result = manager.restart_container()
        
        # Verify result
        assert result is True
        manager.stop_container.assert_called_once()
        manager.start_container.assert_called_once()
    
    @patch('src.persistence.neo4j_docker_manager.docker')
    def test_restart_container_stop_failure(self, mock_docker, mock_config_manager):
        """Test restarting container with stop failure."""
        # Mock Docker client
        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client
        
        # Create manager
        manager = Neo4jDockerManager(mock_config_manager)
        manager.stop_container = Mock(return_value=False)
        manager.start_container = Mock(return_value=True)
        
        # Restart container
        result = manager.restart_container()
        
        # Verify result
        assert result is False
        manager.stop_container.assert_called_once()
        manager.start_container.assert_not_called()
    
    @patch('src.persistence.neo4j_docker_manager.docker')
    def test_cleanup_success(self, mock_docker, mock_config_manager):
        """Test cleanup operation."""
        # Mock Docker client
        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client
        
        # Create manager
        manager = Neo4jDockerManager(mock_config_manager)
        manager.remove_container = Mock(return_value=True)
        
        # Cleanup
        result = manager.cleanup()
        
        # Verify result
        assert result is True
        manager.remove_container.assert_called_once()
        mock_client.close.assert_called_once()
    
    @patch('src.persistence.neo4j_docker_manager.docker')
    def test_setup_persistent_volumes(self, mock_docker, mock_config_manager, tmp_path):
        """Test setting up persistent volumes."""
        # Mock Docker client
        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client
        
        # Create manager with temporary directories
        manager = Neo4jDockerManager(mock_config_manager)
        manager.data_dir = tmp_path / "data"
        manager.logs_dir = tmp_path / "logs"
        manager.import_dir = tmp_path / "import"
        
        # Setup volumes
        result = manager.setup_persistent_volumes()
        
        # Verify result
        assert result is True
        assert manager.data_dir.exists()
        assert manager.logs_dir.exists()
        assert manager.import_dir.exists() 
