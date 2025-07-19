---
inclusion: always
---

# Workspace Organization Best Practices

## Directory Structure
```
project/
├── .kiro/
│   ├── specs/           # Feature specifications
│   ├── steering/        # AI behavior guidelines
│   └── settings/        # Kiro configuration
├── src/                 # Source code
│   ├── config/         # Configuration management
│   ├── models/         # Data models
│   ├── services/       # Business logic
│   └── utils/          # Utility functions
├── test/               # Test files
├── docs/               # Documentation
├── examples/           # Usage examples
└── scripts/            # Automation scripts
```

## File Naming Conventions
- Use snake_case for Python files
- Use kebab-case for spec directories
- Prefix test files with `test_`
- Use descriptive, specific names

## Context Usage Tips
- Use #File to reference specific files in chat
- Use #Folder to include entire directories
- Use #Codebase for project-wide searches
- Use #Problems to see current issues
- Use #Terminal to check command output
- Use #Git to see current changes

## Spec Management
- Create specs for complex features (>1 day of work)
- Break large specs into smaller, focused ones
- Use clear, actionable task descriptions
- Reference requirements explicitly in tasks
- Review and update specs regularly