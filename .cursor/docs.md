# Cursor + Kiro: Hybrid Development Approach

This project uses a **hybrid approach** combining Kiro's structured spec-driven development with Cursor's excellent coding experience.

## 🚀 Quick Start

### 1. Open in Cursor
```bash
cursor .
```

### 2. Try a Spec-Driven Chat
Open Cursor chat and try this prompt:
```
Check the task status in #File:.kiro/specs/context-persistence-solutions/tasks.md 
and help me understand what needs to be implemented next. Then review the 
requirements in #File:.kiro/specs/context-persistence-solutions/requirements.md 
for context.
```

### 3. Use Predefined Prompts
Copy prompts from `.cursor/prompts.md` for common development tasks.

## 🎯 How This Works

### Kiro Structure (Already Set Up)
Your repository already has the complete Kiro spec-driven structure:

```
.kiro/
├── specs/context-persistence-solutions/    # Project specifications
│   ├── requirements.md                     # User stories & acceptance criteria
│   ├── design.md                          # Technical architecture & design
│   └── tasks.md                           # Implementation tasks & progress
├── hooks/                                 # Automation (git, testing, etc.)
├── steering/                              # AI behavior guidelines
└── settings/                              # MCP server configuration
```

### Cursor Enhancement (Just Added)
Now you have Cursor-specific configuration that leverages the Kiro structure:

```
.cursorrules                               # Main Cursor AI rules
.cursor/
├── prompts.md                            # Pre-built prompts for common tasks
└── docs.md                               # This documentation file
```

## 🚀 Development Workflow

### 1. **Start with Specs** (Kiro Way)
Before coding, always check the specs:

```bash
# Check what you're building
cat .kiro/specs/context-persistence-solutions/requirements.md

# Understand the architecture  
cat .kiro/specs/context-persistence-solutions/design.md

# See what's next to implement
cat .kiro/specs/context-persistence-solutions/tasks.md
```

### 2. **Code with Cursor** (Enhanced Experience)
Use Cursor's chat with context-aware prompts:

- **For new features**: Use the "New Feature Start" prompt from `.cursor/prompts.md`
- **For implementation**: Reference the specs in your Cursor prompts
- **For testing**: Use the "Testing Implementation" prompt

### 3. **Leverage Automation** (Kiro Hooks)
The repository has smart hooks that automatically:
- Handle git operations when files change
- Run tests when appropriate
- Track memory and performance
- Maintain project standards
- **Record development activities in memory graph** for persistent context

### 4. **Use Memory Context** (Memory MCP)
The memory system helps you understand and track:
- Previous development decisions and their rationale
- Connections between requirements, design, and implementation
- Evolution of features and components over time
- Related work and dependencies across the project

### 5. **Smart Git Operations** (Git MCP)
Integrated git operations provide:
- Automated status checking and change analysis
- Smart commit workflows with proper messaging
- Code history analysis and pattern recognition
- Branch management and conflict resolution

## 📋 Common Development Patterns

### Starting a New Task
1. **Check task status**: Look at `.kiro/specs/context-persistence-solutions/tasks.md`
2. **Understand requirements**: Read the specific requirement in `requirements.md`
3. **Use Cursor prompt**: Copy the "New Feature Start" prompt from `.cursor/prompts.md`
4. **Implement following patterns**: Reference existing code in `src/`

### Code Review & Quality
1. **Use quality prompts**: Reference "Code Quality Check" from `.cursor/prompts.md`
2. **Validate against specs**: Use "Requirement Validation" prompt
3. **Follow project standards**: See `.kiro/steering/project-standards.md`

### Debugging Issues
1. **Understand expected behavior**: Check the requirements and design
2. **Query memory context**: Use `mcp_memory_search_nodes` to find related decisions
3. **Use debug prompts**: Reference "Debug and Fix" from `.cursor/prompts.md`
4. **Add regression tests**: Prevent similar issues in the future
5. **Record the fix**: Document the solution in memory for future reference

### Working with Memory Context
1. **Query before coding**: Search memory for related work and decisions
2. **Record as you go**: Document important decisions and their rationale
3. **Build relationships**: Connect new work to existing components and requirements
4. **Learn from history**: Review past decisions when making similar choices

### Git Workflow with MCP
1. **Check status**: Use `mcp_git_git_status` to see current changes
2. **Review changes**: Use `mcp_git_git_diff` to understand modifications
3. **Stage strategically**: Use `mcp_git_git_add` for logical groupings
4. **Commit meaningfully**: Use `mcp_git_git_commit` with conventional messages
5. **Track history**: Use `mcp_git_git_log` to understand evolution

## 🎯 Key File References

### For Requirements & Design
- **User Stories**: `.kiro/specs/context-persistence-solutions/requirements.md`
- **Architecture**: `.kiro/specs/context-persistence-solutions/design.md`  
- **Task Status**: `.kiro/specs/context-persistence-solutions/tasks.md`

### For Coding Standards
- **Project Standards**: `.kiro/steering/project-standards.md`
- **Python Guidelines**: `.kiro/steering/python-specific.md`
- **Workflow Tips**: `.kiro/steering/workflow-optimization.md`

### For Development Tools
- **MCP Configuration**: `.kiro/settings/mcp.json`
- **Git Automation**: `.kiro/hooks/git-auto-commit-hook.kiro.hook`
- **Testing Hooks**: `.kiro/hooks/test-runner-hook.kiro.hook`

## 💡 Pro Tips

### Cursor Chat Best Practices
1. **Include file context**: Use `#File` to reference specific files
2. **Reference specs**: Mention relevant requirements or design sections
3. **Use predefined prompts**: Copy from `.cursor/prompts.md` for consistency
4. **Be specific**: Include acceptance criteria and constraints

### Example Cursor Prompts
```
# Good prompt example:
Based on requirement 1.2 in #File:.kiro/specs/context-persistence-solutions/requirements.md, 
help me implement the memory storage functionality following the MemoryDocument 
structure defined in #File:.kiro/specs/context-persistence-solutions/design.md

# Include tests that follow the patterns in #File:test/test_langchain_vector_persistence.py
```

### Leveraging Existing Code
- **Configuration**: Reference `src/config/config_manager.py`
- **Memory Tools**: Look at `src/tools/memory_tools.py`
- **Testing Patterns**: Check files in `test/` directory
- **Persistence Layer**: Examine `src/persistence/langchain_vector_persistence.py`

## 🔄 Integration Points

### Kiro Hooks + Cursor + MCP
The automation hooks work seamlessly with Cursor development:
- **Git operations** are handled automatically when you save files (plus manual MCP control)
- **Tests run** when appropriate changes are detected
- **Code quality** is maintained through steering guidelines
- **Memory tracking** records development activities automatically
- **Git MCP tools** provide full repository control from chat

### Specs + Chat + Memory
Use all three systems together for maximum context:
- Reference specific requirements when asking for help
- Include design patterns when implementing features
- Query memory for related work and previous decisions
- Check task completion criteria when finishing work
- Record new decisions and implementations in memory

## 🎉 Benefits of This Approach

### From Kiro
- ✅ **Structured development**: Clear requirements → design → tasks flow
- ✅ **Living documentation**: Specs stay in sync with code
- ✅ **Smart automation**: Hooks handle repetitive tasks
- ✅ **Quality standards**: Consistent coding patterns

### From Cursor  
- ✅ **Excellent autocomplete**: Superior coding experience
- ✅ **Context-aware chat**: Understands your entire codebase
- ✅ **Fast iteration**: Quick code generation and refinement
- ✅ **Flexible interaction**: Natural language programming

### Combined
- 🚀 **Best of both worlds**: Structured planning + excellent execution
- 🎯 **Context-driven development**: AI understands specs, code, and project history
- ⚡ **Efficient workflow**: Automated tasks + smart assistance + persistent memory
- 📈 **High quality output**: Standards + patterns + assistance + historical context
- 🧠 **Intelligent context**: Memory graph tracks decisions, relationships, and evolution 