# Cursor Prompts - Context Persistence Solutions

## 📋 Spec-Driven Development Prompts

### 🎯 Requirements Analysis
```
Analyze the current requirement in `.kiro/specs/context-persistence-solutions/requirements.md` and help me understand:
1. The user story and acceptance criteria
2. How it relates to the overall system design
3. What implementation approach would best satisfy the requirements
4. Any dependencies or constraints I should consider
```

### 🏗️ Design Implementation
```
Based on the design document in `.kiro/specs/context-persistence-solutions/design.md`, help me implement:
1. Follow the established architecture patterns
2. Use the defined interfaces and data structures
3. Maintain consistency with existing components
4. Include proper error handling and logging
```

### ✅ Task Execution
```
Check the task status in `.kiro/specs/context-persistence-solutions/tasks.md` and help me:
1. Identify the next logical task to work on
2. Understand the task requirements and acceptance criteria
3. Implement the task following established patterns
4. Update the task status when complete
```

## 🔧 Implementation Prompts

### 🧠 LangChain Memory Implementation
```
Help me implement a LangChain memory component that:
- Uses InMemoryVectorStore with OpenAI embeddings
- Includes user_id filtering for multi-tenant scenarios
- Follows the MemoryDocument structure from the design
- Includes comprehensive error handling and logging
- Has full test coverage with mocked dependencies
```

### 🕸️ Neo4j Graph Implementation
```
Help me implement a Neo4j graph component that:
- Uses parameterized Cypher queries for security
- Implements proper connection management
- Creates meaningful node/relationship structures
- Includes comprehensive error handling
- Has full test coverage with mocked database
```

### 🧪 Testing Implementation
```
Create comprehensive tests for this component that:
- Mock all external dependencies (OpenAI, Neo4j)
- Test both success and failure scenarios
- Use pytest fixtures appropriately
- Follow the existing test patterns in the project
- Achieve >80% test coverage
```

## 🔍 Code Review Prompts

### 📝 Code Quality Check
```
Review this code against the project standards:
1. Type hints and docstrings compliance
2. Error handling and edge cases
3. Test coverage and quality
4. Consistency with existing patterns
5. Performance and security considerations
```

### 🎯 Requirement Validation
```
Validate this implementation against:
1. The acceptance criteria in the requirements document
2. The design specifications and interfaces
3. The task completion criteria
4. Existing code patterns and standards
```

## 🚀 Development Workflow Prompts

### 🆕 New Feature Start
```
I want to start working on a new feature. Help me:
1. Review the relevant requirements and design specs
2. Identify the specific task and its dependencies
3. Plan the implementation approach
4. Set up the basic structure and tests
```

### 🔄 Feature Completion
```
I've completed a feature implementation. Help me:
1. Verify it meets all acceptance criteria
2. Ensure comprehensive test coverage
3. Update documentation and docstrings
4. Mark the task as complete in tasks.md
5. Prepare a proper git commit message
```

### 🐛 Debug and Fix
```
Help me debug this issue by:
1. Understanding the expected behavior from specs
2. Analyzing the current implementation
3. Identifying the root cause
4. Implementing a proper fix
5. Adding tests to prevent regression
```

## 🔧 Git & Version Control Prompts

### 📊 Repository Status Check
```
Check the current repository status and help me understand:
1. Use mcp_git_git_status to see current changes
2. Show me what files are modified, staged, or untracked
3. Recommend next steps for committing changes
4. Check if there are any conflicts or issues to resolve
```

### 💾 Smart Commit Workflow
```
Help me create a proper commit with these steps:
1. Use mcp_git_git_status to review current changes
2. Use mcp_git_git_diff to see specific modifications
3. Stage appropriate files with mcp_git_git_add
4. Create a meaningful commit message following conventional commits
5. Use mcp_git_git_commit with the proper message
```

### 🔍 Code History Analysis
```
Analyze the code history for this feature:
1. Use mcp_git_git_log to see recent commits
2. Use mcp_git_git_show to examine specific commits
3. Understand the evolution of this component
4. Identify patterns and previous decisions
```

### 🌿 Branch Management
```
Help me manage branches effectively:
1. Use mcp_git_git_branch to see current branches
2. Check the status of current branch work
3. Recommend branch strategy for this feature
4. Ensure clean commit history before merging
```

### 🔄 Change Review
```
Review my changes before committing:
1. Use mcp_git_git_diff to show unstaged changes
2. Use mcp_git_git_diff_staged to show staged changes
3. Validate changes against requirements and design
4. Suggest improvements or missing pieces
5. Ensure tests and documentation are updated
```

## 🧠 Memory & Context Prompts

### 🔍 Query Project Memory
```
Search the project memory for relevant context about:
1. Use mcp_memory_search_nodes to find related information
2. Look for previous decisions and implementations
3. Find connections between requirements and code
4. Understand the evolution of specific features
```

### 📝 Record Development Activity
```
Record this development activity in memory:
1. Create entities for the files, features, and decisions involved
2. Add observations about the implementation approach
3. Create relationships to related requirements and tests
4. Document any architectural decisions or trade-offs made
```

### 🕸️ Explore Project Relationships
```
Help me understand the project context by:
1. Opening relevant memory nodes about this feature
2. Showing connections between requirements, design, and implementation
3. Identifying related components and dependencies
4. Highlighting previous decisions that might impact this work
```

### 💭 Capture Technical Decisions
```
Record this technical decision in project memory:
1. Create an entity for the decision with full context
2. Link it to the relevant requirements and design components
3. Document the rationale and alternatives considered
4. Connect it to the affected code and test files
```

## 📚 Documentation Prompts

### 📖 Documentation Update
```
Update the documentation for this component:
1. Add comprehensive docstrings
2. Update any relevant design documents
3. Include usage examples
4. Document any breaking changes
5. Update the README if needed
6. Record documentation changes in memory MCP
```

### 💡 Example Creation
```
Create a practical example that demonstrates:
1. How to use this component
2. Common use cases and patterns
3. Error handling scenarios
4. Integration with other components
5. Best practices and tips
6. Document the example in project memory for future reference
``` 