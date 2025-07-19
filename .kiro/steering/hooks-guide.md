---
inclusion: manual
---

# Useful Agent Hooks for Development

## Recommended Hooks to Create

### 1. **Test Runner Hook**
- **Trigger**: On file save (*.py files)
- **Action**: Run relevant tests automatically
- **Benefits**: Immediate feedback on code changes

### 2. **Code Quality Hook** 
- **Trigger**: Manual button
- **Action**: Run linting, formatting, and type checking
- **Benefits**: Maintain code quality standards

### 3. **Documentation Update Hook**
- **Trigger**: On file save (README.md, docs/*.md)
- **Action**: Update table of contents, check links
- **Benefits**: Keep documentation current

### 4. **Dependency Check Hook**
- **Trigger**: On requirements.txt change
- **Action**: Check for security vulnerabilities, update compatibility
- **Benefits**: Maintain secure dependencies

### 5. **Spec Validation Hook**
- **Trigger**: On spec file changes
- **Action**: Validate spec format and completeness
- **Benefits**: Ensure spec quality

## Hook Creation Tips
- Use the Command Palette: "Open Kiro Hook UI"
- Start with simple hooks and gradually add complexity
- Test hooks thoroughly before relying on them
- Use conditional logic to avoid unnecessary executions