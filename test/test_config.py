"""Test configuration management functionality."""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.config_manager import ConfigManager, Config


def test_config_loading():
    """Test configuration loading and validation."""
    try:
        # Initialize config manager
        config_manager = ConfigManager()
        
        # Load configuration
        config = config_manager.load_config()
        
        print("✅ Configuration loaded successfully!")
        print(f"OpenAI Model: {config.llm_model}")
        print(f"Neo4j URI: {config.neo4j_uri}")
        print(f"Embedding Model: {config.embedding_model}")
        print(f"Embedding Enabled: {config.is_embedding}")
        
        # Test specific config getters
        openai_config = config_manager.get_openai_config()
        neo4j_config = config_manager.get_neo4j_config()
        
        print("\n✅ OpenAI Config:")
        for key, value in openai_config.items():
            if 'key' in key.lower():
                print(f"  {key}: {'*' * 10}")  # Hide API key
            else:
                print(f"  {key}: {value}")
        
        print("\n✅ Neo4j Config:")
        for key, value in neo4j_config.items():
            if 'password' in key.lower():
                print(f"  {key}: {'*' * 8}")  # Hide password
            else:
                print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_config_validation():
    """Test configuration validation."""
    try:
        # Test with valid config
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        # Validate required fields are present
        assert config.openai_api_key, "OpenAI API key should be present"
        assert config.neo4j_uri, "Neo4j URI should be present"
        assert config.llm_model, "LLM model should be present"
        
        print("✅ Configuration validation passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


if __name__ == "__main__":
    print("Running configuration tests...\n")
    
    success = True
    success &= test_config_loading()
    success &= test_config_validation()
    
    if success:
        print("\n🎉 All configuration tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)