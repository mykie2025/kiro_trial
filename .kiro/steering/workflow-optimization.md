---
inclusion: always
---

# Development Workflow Optimization

## Kiro Chat Best Practices

### Context Management
- **Start conversations with clear context**: Use #File, #Folder, or #Codebase
- **Be specific about requirements**: Include acceptance criteria and constraints
- **Reference existing patterns**: Point to similar implementations in your codebase
- **Use incremental requests**: Break complex tasks into smaller steps

### Effective Prompting
```
Good: "Update the LangChain persistence class to add batch operations, following the existing error handling patterns in #File:src/persistence/langchain_vector_persistence.py"

Better: "Add batch_save_memories() method to LangChainVectorPersistence class that:
- Accepts List[MemoryDocument] 
- Returns List[str] (document IDs)
- Uses transaction-like behavior
- Follows existing error handling patterns
- Includes comprehensive logging
- Has unit tests with mocked dependencies"
```

### Code Review Process
1. **Use Kiro for initial review**: Ask Kiro to review code for common issues
2. **Focus on architecture**: Have Kiro check design patterns and structure
3. **Validate against specs**: Ensure implementation matches requirements
4. **Test coverage**: Ask Kiro to identify missing test cases

## Autopilot vs Supervised Mode
- **Use Autopilot for**: Routine tasks, refactoring, test writing
- **Use Supervised for**: Architecture changes, new features, critical fixes
- **Switch modes based on**: Risk level and your availability to review

## Spec-Driven Development
1. **Always start with specs** for features >4 hours of work
2. **Iterate on requirements** before design
3. **Get explicit approval** at each spec phase
4. **Execute tasks incrementally** - one at a time
5. **Review and adjust** specs based on implementation learnings