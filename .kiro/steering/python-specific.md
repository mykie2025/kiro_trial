---
inclusion: fileMatch
fileMatchPattern: "*.py"
---

# Python-Specific Guidelines

## Import Organization
```python
# Standard library imports
import os
import sys
from typing import List, Dict, Optional

# Third-party imports
import pytest
import requests

# Local application imports
from .config import settings
from .models import User
```

## Error Handling Patterns
```python
# Prefer specific exceptions
try:
    result = risky_operation()
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    raise
except ConnectionError as e:
    logger.error(f"Connection failed: {e}")
    return None
```

## Logging Configuration
```python
import logging

logger = logging.getLogger(__name__)

# Use structured logging
logger.info("Operation completed", extra={
    'user_id': user_id,
    'operation': 'save_memory',
    'duration': elapsed_time
})
```

## Type Hints Best Practices
- Always use type hints for function signatures
- Use Union types sparingly, prefer Optional for nullable values
- Use Protocol for duck typing when appropriate
- Import types from typing module consistently