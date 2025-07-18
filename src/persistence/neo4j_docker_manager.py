"""Neo4j Docker container management for context persistence."""

import logging
import os
import time
import json
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

import docker
from docker.models.containers import Container
from docker.errors import DockerException, NotFound, APIError

from ..config.config_manager import ConfigManager


logger = logging.getLogger(__name__)


class Neo4jDockerManagerError(Exception):
    """Custom exception for Neo4j Docker manager errors."""
    pass


class Neo4jDockerManager:
    """
    Manages Neo4j Docker container lifecycle with persistent volume mounting.
    
    This class provides functionality to start, stop, and monitor Neo4j containers
    with proper health checks and data persistence.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the Neo4j Docker manager.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        self.neo4j_config = config_manager.get_neo4j_config()
        
        # Container configuration
        self.container_name = "neo4j-context-persistence"
        self.image_name = "neo4j:latest"
        self.neo4j_port = 7687
        self.http_port = 7474
        self.https_port = 7473
        
        # Volume configuration
        self.data_dir = Path("./neo4j_data")
        self.logs_dir = Path("./neo4j_logs")
        self.import_dir = Path("./neo4j_import")
        self.backup_dir = Path("./neo4j_backups")
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized successfully")
        except DockerException as e:
            raise Neo4jDockerManagerError(f"Failed to initialize Docker client: {e}")
    
    def _ensure_directories(self) -> None:
        """Create necessary directories for persistent volumes."""
        directories = [self.data_dir, self.logs_dir, self.import_dir, self.backup_dir]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {directory}")
            except PermissionError as e:
                raise Neo4jDockerManagerError(f"Permission denied creating directory {directory}: {e}")
    
    def _get_container_config(self) -> Dict[str, Any]:
        """
        Get Docker container configuration.
        
        Returns:
            Container configuration dictionary
        """
        return {
            'image': self.image_name,
            'name': self.container_name,
            'ports': {
                f'{self.neo4j_port}/tcp': self.neo4j_port,
                f'{self.http_port}/tcp': self.http_port,
                f'{self.https_port}/tcp': self.https_port
            },
            'environment': {
                'NEO4J_AUTH': f"{self.neo4j_config['username']}/{self.neo4j_config['password']}",
                'NEO4J_PLUGINS': '["apoc"]',
                'NEO4J_apoc_export_file_enabled': 'true',
                'NEO4J_apoc_import_file_enabled': 'true',
                'NEO4J_apoc_import_file_use__neo4j__config': 'true',
                'NEO4J_ACCEPT_LICENSE_AGREEMENT': 'yes'
            },
            'volumes': {
                str(self.data_dir.absolute()): {'bind': '/data', 'mode': 'rw'},
                str(self.logs_dir.absolute()): {'bind': '/logs', 'mode': 'rw'},
                str(self.import_dir.absolute()): {'bind': '/var/lib/neo4j/import', 'mode': 'rw'}
            },
            'detach': True,
            'restart_policy': {'Name': 'unless-stopped'}
        }
    
    def start_container(self) -> bool:
        """
        Start Neo4j Docker container with persistent volume mounting.
        
        Returns:
            True if container started successfully, False otherwise
        """
        try:
            # Check if container already exists
            existing_container = self.get_container()
            if existing_container:
                if existing_container.status == 'running':
                    logger.info(f"Container {self.container_name} is already running")
                    return True
                else:
                    logger.info(f"Starting existing container {self.container_name}")
                    existing_container.start()
                    return self._wait_for_container_ready()
            
            # Ensure directories exist
            self._ensure_directories()
            
            # Pull image if not exists
            try:
                self.docker_client.images.get(self.image_name)
            except NotFound:
                logger.info(f"Pulling Neo4j image: {self.image_name}")
                self.docker_client.images.pull(self.image_name)
            
            # Create and start container
            container_config = self._get_container_config()
            logger.info(f"Creating Neo4j container: {self.container_name}")
            
            container = self.docker_client.containers.run(**container_config)
            
            # Wait for container to be ready
            if self._wait_for_container_ready():
                logger.info(f"Neo4j container {self.container_name} started successfully")
                return True
            else:
                logger.error(f"Neo4j container {self.container_name} failed to become ready")
                return False
                
        except DockerException as e:
            logger.error(f"Failed to start Neo4j container: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error starting container: {e}")
            return False
    
    def stop_container(self) -> bool:
        """
        Stop Neo4j Docker container.
        
        Returns:
            True if container stopped successfully, False otherwise
        """
        try:
            container = self.get_container()
            if not container:
                logger.info(f"Container {self.container_name} not found")
                return True
            
            if container.status != 'running':
                logger.info(f"Container {self.container_name} is not running")
                return True
            
            logger.info(f"Stopping container {self.container_name}")
            container.stop(timeout=30)
            
            # Wait for container to stop
            for _ in range(10):
                container.reload()
                if container.status != 'running':
                    logger.info(f"Container {self.container_name} stopped successfully")
                    return True
                time.sleep(1)
            
            logger.error(f"Container {self.container_name} failed to stop within timeout")
            return False
            
        except DockerException as e:
            logger.error(f"Failed to stop Neo4j container: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error stopping container: {e}")
            return False
    
    def remove_container(self) -> bool:
        """
        Remove Neo4j Docker container (stops first if running).
        
        Returns:
            True if container removed successfully, False otherwise
        """
        try:
            container = self.get_container()
            if not container:
                logger.info(f"Container {self.container_name} not found")
                return True
            
            # Stop if running
            if container.status == 'running':
                if not self.stop_container():
                    logger.error("Failed to stop container before removal")
                    return False
            
            logger.info(f"Removing container {self.container_name}")
            container.remove()
            logger.info(f"Container {self.container_name} removed successfully")
            return True
            
        except DockerException as e:
            logger.error(f"Failed to remove Neo4j container: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error removing container: {e}")
            return False
    
    def get_container(self) -> Optional[Container]:
        """
        Get the Neo4j container if it exists.
        
        Returns:
            Container instance or None if not found
        """
        try:
            return self.docker_client.containers.get(self.container_name)
        except NotFound:
            return None
        except DockerException as e:
            logger.error(f"Error getting container: {e}")
            return None
    
    def is_container_running(self) -> bool:
        """
        Check if Neo4j container is running.
        
        Returns:
            True if container is running, False otherwise
        """
        container = self.get_container()
        return container is not None and container.status == 'running'
    
    def get_container_status(self) -> Dict[str, Any]:
        """
        Get detailed container status information.
        
        Returns:
            Dictionary with container status details
        """
        container = self.get_container()
        if not container:
            return {
                'exists': False,
                'status': 'not_found',
                'health': 'unknown'
            }
        
        container.reload()
        
        return {
            'exists': True,
            'status': container.status,
            'health': self._check_container_health(),
            'created': container.attrs.get('Created'),
            'started': container.attrs.get('State', {}).get('StartedAt'),
            'ports': container.attrs.get('NetworkSettings', {}).get('Ports', {}),
            'image': container.attrs.get('Config', {}).get('Image')
        }
    
    def _check_container_health(self) -> str:
        """
        Check container health by attempting connection.
        
        Returns:
            Health status string ('healthy', 'unhealthy', 'starting')
        """
        if not self.is_container_running():
            return 'unhealthy'
        
        try:
            # Test connection to Neo4j
            from neo4j import GraphDatabase
            
            driver_config = self.config_manager.get_neo4j_driver_config()
            driver = GraphDatabase.driver(**driver_config)
            
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record and record['test'] == 1:
                    driver.close()
                    return 'healthy'
                else:
                    driver.close()
                    return 'unhealthy'
                    
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return 'starting'
    
    def _wait_for_container_ready(self, timeout: int = 60) -> bool:
        """
        Wait for container to be ready and healthy.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if container becomes ready, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if not self.is_container_running():
                logger.debug("Container not running, waiting...")
                time.sleep(2)
                continue
            
            health = self._check_container_health()
            if health == 'healthy':
                return True
            elif health == 'starting':
                logger.debug("Container starting, waiting for health check...")
                time.sleep(3)
            else:
                logger.debug(f"Container unhealthy: {health}")
                time.sleep(2)
        
        logger.error(f"Container failed to become ready within {timeout} seconds")
        return False
    
    def setup_persistent_volumes(self) -> bool:
        """
        Setup persistent volume directories with proper permissions.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            self._ensure_directories()
            
            # Set appropriate permissions for Neo4j
            for directory in [self.data_dir, self.logs_dir, self.import_dir, self.backup_dir]:
                if os.name != 'nt':  # Not Windows
                    os.chmod(directory, 0o755)
            
            logger.info("Persistent volumes setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup persistent volumes: {e}")
            return False
    
    def get_logs(self, tail: int = 100) -> str:
        """
        Get container logs.
        
        Args:
            tail: Number of lines to return from the end
            
        Returns:
            Container logs as string
        """
        container = self.get_container()
        if not container:
            return "Container not found"
        
        try:
            logs = container.logs(tail=tail, timestamps=True)
            return logs.decode('utf-8') if isinstance(logs, bytes) else logs
        except DockerException as e:
            logger.error(f"Failed to get container logs: {e}")
            return f"Error getting logs: {e}"
    
    def restart_container(self) -> bool:
        """
        Restart the Neo4j container.
        
        Returns:
            True if restart successful, False otherwise
        """
        logger.info(f"Restarting container {self.container_name}")
        
        if not self.stop_container():
            logger.error("Failed to stop container for restart")
            return False
        
        return self.start_container()
    
    def cleanup(self) -> bool:
        """
        Clean up resources (stop and remove container).
        
        Returns:
            True if cleanup successful, False otherwise
        """
        logger.info("Cleaning up Neo4j Docker resources")
        
        success = True
        if not self.remove_container():
            success = False
        
        try:
            # Close Docker client
            self.docker_client.close()
        except Exception as e:
            logger.error(f"Error closing Docker client: {e}")
            success = False
        
        return success
    
    def create_graph_backup(self, backup_name: Optional[str] = None, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Create a complete backup of the Neo4j graph database.
        
        Args:
            backup_name: Optional custom name for the backup. If None, uses timestamp
            include_metadata: Whether to include backup metadata
            
        Returns:
            Dictionary with backup information
        """
        try:
            if not self.is_container_running():
                raise Neo4jDockerManagerError("Container must be running to create backup")
            
            # Generate backup name if not provided
            if not backup_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"neo4j_backup_{timestamp}"
            
            backup_path = self.backup_dir / backup_name
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Create backup using Neo4j's built-in backup functionality
            container = self.get_container()
            if not container:
                raise Neo4jDockerManagerError("Container not found")
            
            # Execute backup command in container
            backup_command = [
                "neo4j-admin", "backup", 
                "--database=neo4j",
                f"--to={backup_path}",
                "--verbose"
            ]
            
            logger.info(f"Creating Neo4j backup: {backup_name}")
            result = container.exec_run(backup_command)
            
            if result.exit_code != 0:
                raise Neo4jDockerManagerError(f"Backup failed: {result.output.decode()}")
            
            # Create metadata file
            metadata = {
                'backup_name': backup_name,
                'created_at': datetime.now().isoformat(),
                'container_id': container.id,
                'neo4j_version': self.image_name,
                'backup_path': str(backup_path),
                'backup_size': self._get_directory_size(backup_path),
                'command_output': result.output.decode() if include_metadata else None
            }
            
            if include_metadata:
                metadata_file = backup_path / "backup_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            logger.info(f"Backup created successfully: {backup_name}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise Neo4jDockerManagerError(f"Backup creation failed: {e}")
    
    def _get_directory_size(self, path: Path) -> int:
        """Get the size of a directory in bytes."""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            return total_size
        except Exception:
            return 0
    
    def _get_directory_creation_time(self, path: Path) -> str:
        """Get the creation time of a directory."""
        try:
            stat = path.stat()
            return datetime.fromtimestamp(stat.st_ctime).isoformat()
        except Exception:
            return datetime.now().isoformat()
    
    def create_graph_dump(self, dump_name: Optional[str] = None, format: str = "cypher") -> Dict[str, Any]:
        """
        Create a graph dump in various formats (Cypher, JSON, CSV).
        
        Args:
            dump_name: Optional custom name for the dump. If None, uses timestamp
            format: Output format ('cypher', 'json', 'csv')
            
        Returns:
            Dictionary with dump information
        """
        try:
            if not self.is_container_running():
                raise Neo4jDockerManagerError("Container must be running to create dump")
            
            # Generate dump name if not provided
            if not dump_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dump_name = f"neo4j_dump_{timestamp}"
            
            dump_path = self.backup_dir / dump_name
            dump_path.mkdir(parents=True, exist_ok=True)
            
            # Get Neo4j connection for data export
            from neo4j import GraphDatabase
            
            driver_config = self.config_manager.get_neo4j_driver_config()
            driver = GraphDatabase.driver(**driver_config)
            
            try:
                with driver.session() as session:
                    if format.lower() == "cypher":
                        dump_info = self._create_cypher_dump(session, dump_path, dump_name)
                    elif format.lower() == "json":
                        dump_info = self._create_json_dump(session, dump_path, dump_name)
                    elif format.lower() == "csv":
                        dump_info = self._create_csv_dump(session, dump_path, dump_name)
                    else:
                        raise Neo4jDockerManagerError(f"Unsupported dump format: {format}")
                
                logger.info(f"Graph dump created successfully: {dump_name} ({format})")
                return dump_info
                
            finally:
                driver.close()
                
        except Exception as e:
            logger.error(f"Failed to create graph dump: {e}")
            raise Neo4jDockerManagerError(f"Graph dump creation failed: {e}")
    
    def _create_cypher_dump(self, session, dump_path: Path, dump_name: str) -> Dict[str, Any]:
        """Create a Cypher dump of the graph."""
        # Export nodes
        nodes_file = dump_path / "nodes.cypher"
        nodes_query = """
        MATCH (n)
        RETURN 'CREATE (' + 
               id(n) + ':' + labels(n)[0] + ' ' + 
               apoc.text.join([k + ':' + toString(v) for k, v in properties(n)], ', ') + 
               ')'
        """
        
        with open(nodes_file, 'w') as f:
            f.write("// Neo4j Graph Dump - Nodes\n")
            f.write("// Generated: " + datetime.now().isoformat() + "\n\n")
            
            result = session.run(nodes_query)
            for record in result:
                f.write(record[0] + ";\n")
        
        # Export relationships
        rels_file = dump_path / "relationships.cypher"
        rels_query = """
        MATCH (a)-[r]->(b)
        RETURN 'CREATE (' + id(a) + ')-[:' + type(r) + ' ' + 
               apoc.text.join([k + ':' + toString(v) for k, v in properties(r)], ', ') + 
               ']->(' + id(b) + ')'
        """
        
        with open(rels_file, 'w') as f:
            f.write("// Neo4j Graph Dump - Relationships\n")
            f.write("// Generated: " + datetime.now().isoformat() + "\n\n")
            
            result = session.run(rels_query)
            for record in result:
                f.write(record[0] + ";\n")
        
        return {
            'dump_name': dump_name,
            'format': 'cypher',
            'created_at': datetime.now().isoformat(),
            'files': {
                'nodes': str(nodes_file),
                'relationships': str(rels_file)
            },
            'dump_size': self._get_directory_size(dump_path)
        }
    
    def _create_json_dump(self, session, dump_path: Path, dump_name: str) -> Dict[str, Any]:
        """Create a JSON dump of the graph."""
        dump_file = dump_path / "graph_dump.json"
        
        # Get all nodes and relationships
        nodes_query = "MATCH (n) RETURN n"
        rels_query = "MATCH (a)-[r]->(b) RETURN a, r, b"
        
        graph_data = {
            'metadata': {
                'dump_name': dump_name,
                'created_at': datetime.now().isoformat(),
                'format': 'json'
            },
            'nodes': [],
            'relationships': []
        }
        
        # Export nodes
        result = session.run(nodes_query)
        for record in result:
            node = record['n']
            graph_data['nodes'].append({
                'id': node.id,
                'labels': list(node.labels),
                'properties': dict(node)
            })
        
        # Export relationships
        result = session.run(rels_query)
        for record in result:
            rel = record['r']
            graph_data['relationships'].append({
                'start_node': record['a'].id,
                'end_node': record['b'].id,
                'type': rel.type,
                'properties': dict(rel)
            })
        
        with open(dump_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        return {
            'dump_name': dump_name,
            'format': 'json',
            'created_at': datetime.now().isoformat(),
            'files': {
                'graph_dump': str(dump_file)
            },
            'dump_size': self._get_directory_size(dump_path),
            'node_count': len(graph_data['nodes']),
            'relationship_count': len(graph_data['relationships'])
        }
    
    def _create_csv_dump(self, session, dump_path: Path, dump_name: str) -> Dict[str, Any]:
        """Create a CSV dump of the graph."""
        # Export nodes to CSV
        nodes_file = dump_path / "nodes.csv"
        nodes_query = """
        MATCH (n)
        RETURN id(n) as id, 
               labels(n)[0] as label,
               properties(n) as properties
        """
        
        with open(nodes_file, 'w', newline='') as f:
            f.write("id,label,properties\n")
            result = session.run(nodes_query)
            for record in result:
                f.write(f"{record['id']},{record['label']},\"{json.dumps(record['properties'])}\"\n")
        
        # Export relationships to CSV
        rels_file = dump_path / "relationships.csv"
        rels_query = """
        MATCH (a)-[r]->(b)
        RETURN id(a) as start_id, 
               id(b) as end_id,
               type(r) as type,
               properties(r) as properties
        """
        
        with open(rels_file, 'w', newline='') as f:
            f.write("start_id,end_id,type,properties\n")
            result = session.run(rels_query)
            for record in result:
                f.write(f"{record['start_id']},{record['end_id']},{record['type']},\"{json.dumps(record['properties'])}\"\n")
        
        return {
            'dump_name': dump_name,
            'format': 'csv',
            'created_at': datetime.now().isoformat(),
            'files': {
                'nodes': str(nodes_file),
                'relationships': str(rels_file)
            },
            'dump_size': self._get_directory_size(dump_path)
        } 