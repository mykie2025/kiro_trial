---
inclusion: always
---

# Project Standards and Guidelines

## Code Style & Standards
- Use Python type hints for all function parameters and return values
- Follow PEP 8 style guidelines
- Write comprehensive docstrings for all classes and functions
- Use dataclasses for data structures when appropriate
- Implement proper error handling with specific exception types

## Testing Requirements
- Write unit tests for all new functionality
- Aim for >80% test coverage
- Use pytest for testing framework
- Mock external dependencies in tests
- Include both positive and negative test cases

## Architecture Patterns
- Follow dependency injection patterns
- Use configuration management for environment variables
- Implement proper logging throughout the application
- Separate concerns with clear module boundaries
- Use factory patterns for complex object creation

## Documentation
- Include README files for each major component
- Document API endpoints and data models
- Provide usage examples for complex functionality
- Keep inline comments focused on "why" not "what"

## Security Best Practices
- Never commit API keys or secrets to version control
- Use environment variables for configuration
- Validate all input data
- Implement proper authentication and authorization
- Use secure communication protocols