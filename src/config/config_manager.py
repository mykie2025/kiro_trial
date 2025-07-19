"""Configuration management for context persistence solutions."""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Config:
    """Configuration data class with validation."""
    
    # OpenAI Configuration
    openai_api_key: str
    openai_api_base: str
    openai_base_url: str
    llm_model: str
    
    # Neo4j Configuration
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str
    
    # Optional Settings
    embedding_model: str = "text-embedding-3-small"
    is_embedding: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_required_fields()
        self._validate_urls()
        self._validate_boolean_fields()
    
    def _validate_required_fields(self):
        """Validate that all required fields are present and non-empty."""
        required_fields = [
            'openai_api_key', 'openai_api_base', 'openai_base_url', 'llm_model',
            'neo4j_uri', 'neo4j_username', 'neo4j_password', 'neo4j_database'
        ]
        
        for field in required_fields:
            value = getattr(self, field)
            if not value or not isinstance(value, str) or not value.strip():
                raise ValueError(f"Required configuration field '{field}' is missing or empty")
    
    def _validate_urls(self):
        """Validate URL formats."""
        url_fields = ['openai_api_base', 'openai_base_url', 'neo4j_uri']
        
        for field in url_fields:
            value = getattr(self, field)
            if not value.startswith(('http://', 'https://', 'neo4j://', 'bolt://')):
                raise ValueError(f"Invalid URL format for '{field}': {value}")
    
    def _validate_boolean_fields(self):
        """Validate boolean field types."""
        if not isinstance(self.is_embedding, bool):
            raise ValueError("'is_embedding' must be a boolean value")


class ConfigManager:
    """Manages application configuration with validation and environment variable loading."""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize ConfigManager.
        
        Args:
            env_file: Optional path to .env file. Defaults to '.env' in current directory.
        """
        self._config: Optional[Config] = None
        self._env_file = env_file or '.env'
        self._load_environment()
    
    def _load_environment(self):
        """Load environment variables from .env file."""
        if os.path.exists(self._env_file):
            load_dotenv(self._env_file)
        else:
            raise FileNotFoundError(f"Environment file not found: {self._env_file}")
    
    def _get_env_var(self, key: str, default: Optional[str] = None, required: bool = True) -> str:
        """
        Get environment variable with validation.
        
        Args:
            key: Environment variable key
            default: Default value if not found
            required: Whether the variable is required
            
        Returns:
            Environment variable value
            
        Raises:
            ValueError: If required variable is missing
        """
        value = os.getenv(key, default)
        
        if required and (value is None or value.strip() == ''):
            raise ValueError(f"Required environment variable '{key}' is not set")
        
        return value or ''
    
    def _parse_boolean(self, value: str) -> bool:
        """Parse string value to boolean."""
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        
        return bool(value)
    
    def load_config(self) -> Config:
        """
        Load and validate configuration from environment variables.
        
        Returns:
            Validated Config instance
            
        Raises:
            ValueError: If configuration validation fails
        """
        try:
            config_data = {
                # OpenAI Configuration
                'openai_api_key': self._get_env_var('OPENAI_API_KEY'),
                'openai_api_base': self._get_env_var('OPENAI_API_BASE'),
                'openai_base_url': self._get_env_var('OPENAI_BASE_URL'),
                'llm_model': self._get_env_var('LLM_MODEL'),
                
                # Neo4j Configuration
                'neo4j_uri': self._get_env_var('NEO4J_URI'),
                'neo4j_username': self._get_env_var('NEO4J_USERNAME'),
                'neo4j_password': self._get_env_var('NEO4J_PASSWORD'),
                'neo4j_database': self._get_env_var('NEO4J_DATABASE'),
                
                # Optional Settings
                'embedding_model': self._get_env_var('EMBEDDING_MODEL', 'text-embedding-3-small', required=False),
                'is_embedding': self._parse_boolean(self._get_env_var('IS_EMBEDDING', 'true', required=False))
            }
            
            self._config = Config(**config_data)
            return self._config
            
        except Exception as e:
            raise ValueError(f"Configuration loading failed: {str(e)}")
    
    def get_config(self) -> Config:
        """
        Get current configuration, loading it if not already loaded.
        
        Returns:
            Current Config instance
        """
        if self._config is None:
            self._config = self.load_config()
        
        return self._config
    
    def reload_config(self) -> Config:
        """
        Reload configuration from environment variables.
        
        Returns:
            Newly loaded Config instance
        """
        self._load_environment()
        return self.load_config()
    
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI-specific configuration as dictionary."""
        config = self.get_config()
        return {
            'api_key': config.openai_api_key,
            'api_base': config.openai_api_base,
            'base_url': config.openai_base_url,
            'model': config.llm_model,
            'embedding_model': config.embedding_model
        }
    
    def get_neo4j_config(self) -> Dict[str, Any]:
        """Get Neo4j-specific configuration as dictionary."""
        config = self.get_config()
        return {
            'uri': config.neo4j_uri,
            'username': config.neo4j_username,
            'password': config.neo4j_password,
            'database': config.neo4j_database
        }
    
    def is_embedding_enabled(self) -> bool:
        """Check if embedding functionality is enabled."""
        return self.get_config().is_embedding
    
    def get_openai_client_config(self) -> Dict[str, Any]:
        """Get configuration formatted for OpenAI client initialization."""
        config = self.get_config()
        return {
            'api_key': config.openai_api_key,
            'base_url': config.openai_base_url
        }
    
    def get_neo4j_driver_config(self) -> Dict[str, Any]:
        """Get configuration formatted for Neo4j driver initialization."""
        config = self.get_config()
        return {
            'uri': config.neo4j_uri,
            'auth': (config.neo4j_username, config.neo4j_password),
            'database': config.neo4j_database
        }


# Global configuration manager instance
config_manager = ConfigManager()