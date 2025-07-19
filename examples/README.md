# LangChain Vector Persistence Examples

This directory contains examples demonstrating the LangChain vector persistence system implementation.

## Examples

### 1. `langchain_vector_example.py`
Basic example showing fundamental operations:
- System initialization and health checks
- Memory storage with different types
- Semantic search capabilities
- Memory counting and retrieval

### 2. `comprehensive_langchain_example.py` ⭐
**Comprehensive demonstration** that fulfills Task 2.5 requirements:
- Complete LangChain vector persistence workflow
- Multiple user scenarios with realistic data
- All memory types (preference, context, fact, conversation, event)
- Conversation chain management
- User isolation testing
- Performance benchmarking
- Analytics and reporting

## Running the Examples

### Prerequisites

1. **Environment Setup**: Ensure you have a `.env` file with required configuration:
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

2. **Dependencies**: Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Comprehensive Example

From the project root directory:

```bash
python examples/comprehensive_langchain_example.py
```

## Expected Output

The comprehensive example produces detailed output showing:

### 🚀 **Initialization**
```
🚀 Initializing Comprehensive LangChain Vector Persistence Demo
======================================================================
✅ Initialization successful!
```

### 🔍 **Health Check**
```
🔍 Health Check
------------------------------
Status: healthy
✅ Embedding model: text-embedding-3-small
✅ Embedding dimension: 1536
```

### 💾 **Memory Storage Demonstration**
Shows storing memories for two user personas:
- **Alice (Developer)**: VS Code preferences, ML project context, Python experience, vector database inquiries, conference attendance
- **Bob (Analyst)**: Tableau preferences, customer analysis project, SQL/R skills, automation inquiries

```
💾 Memory Storage Demonstration
----------------------------------------

👤 Storing memories for user: alice_developer
   ✅ [preference] Alice prefers using VS Code with dark theme for Python deve...
      Memory ID: mem_123...
   ✅ [context] Alice is working on a machine learning project using sciki...
      Memory ID: mem_124...
   [continues...]
   📊 Total memories saved for alice_developer: 5

👤 Storing memories for user: bob_analyst
   ✅ [preference] Bob prefers Tableau for data visualization and analysis...
      Memory ID: mem_125...
   [continues...]
   📊 Total memories saved for bob_analyst: 4
```

### 🔍 **Memory Search Demonstration**
Semantic search across different queries:

```
🔍 Memory Search Demonstration
-----------------------------------

🎯 Search for Alice's development setup preferences
   User: alice_developer
   Query: 'Python development tools and preferences'
   📋 Found 2 relevant memories:
      1. [preference] Alice prefers using VS Code with dark theme for Python development
         Similarity: 0.856
         Metadata: {'category': 'development_tools', 'priority': 'high'}
      2. [fact] Alice mentioned she has 5 years of Python experience
         Similarity: 0.743
         Metadata: {'category': 'experience', 'verified': True}
```

### 🎛️ **Memory Type Filtering**
Shows filtering by specific memory types:

```
🎛️  Memory Type Filtering Demonstration
---------------------------------------------

🔖 Filtering Alice's preferences
   User: alice_developer
   Type: preference
   📋 Found 1 preference memories:
      • Alice prefers using VS Code with dark theme for Python development
```

### 💬 **Conversation Chain Demonstration**
Shows conversation session management:

```
💬 Conversation Chain Demonstration
----------------------------------------

💭 Session: alice_session_1 (User: alice_developer)
   🔗 Conversation chain created successfully
   📝 Simulating conversation flow:
      1. User: Hi! I'm working on a new ML project and need some guidance.
         💭 (Conversation stored with memory context)
      2. User: I'm thinking about using vector databases for similarity search.
         💭 (Conversation stored with memory context)
```

### 🔒 **User Isolation Testing**
Validates security and privacy:

```
🔒 User Isolation Demonstration
-----------------------------------
Testing that users can only access their own memories...

🧪 Test: Alice searching for Bob's Tableau preferences
   ✅ Correctly isolated - no relevant memories found

🧪 Test: Bob searching for Alice's Python experience  
   ✅ Correctly isolated - no relevant memories found

🧪 Test: Alice searching for her own VS Code preferences
   ✅ Correctly found 1 relevant memories
```

### 📊 **Memory Analytics**
Statistical analysis of stored memories:

```
📊 Memory Analytics Demonstration
--------------------------------------

👤 Analytics for alice_developer:
   📈 Total memories: 5
   📋 Conversation: 1 memories
   📋 Fact: 1 memories
   📋 Preference: 1 memories
   📋 Context: 1 memories
   📋 Event: 1 memories
   📏 Average memory length: 67.2 characters
```

### ⚡ **Performance Testing**
Benchmarks system performance:

```
⚡ Performance Test
-------------------------
Testing with 10 operations...

💾 Storage Performance:
   ⏱️  Stored 10 memories in 0.234s
   📊 Average: 23.40ms per memory

🔍 Search Performance:
   ⏱️  Performed 10 searches in 0.156s
   📊 Average: 15.60ms per search
   🧹 Test data cleaned up
```

### 📋 **Summary Report**
Final validation against requirements:

```
📋 Demonstration Summary Report
==================================================
👥 Total Users: 2
💾 Total Memories Stored: 9
   • Alice (Developer): 5 memories
   • Bob (Analyst): 4 memories

✅ Demonstrated Features:
   • Memory storage with multiple types
   • Semantic similarity search
   • User-specific memory isolation
   • Memory type filtering
   • Conversation chain management
   • Performance testing
   • Analytics and reporting

🎯 Requirements Satisfied:
   • 6.5: MVP testing with practical examples ✅
   • 6.6: Foundation for full feature set ✅

🚀 System Ready for:
   • Production deployment
   • Integration with applications
   • Scaling to more users
   • Adding Neo4j graph persistence
   • Implementing evaluation framework

🎉 Comprehensive demonstration completed successfully!

This example demonstrates the complete LangChain vector persistence
workflow and validates the MVP implementation against requirements 6.5 and 6.6.
```

## Features Demonstrated

### Core Functionality
- ✅ **Memory Storage**: Save memories with different types and metadata
- ✅ **Semantic Search**: Find relevant memories using natural language queries
- ✅ **User Isolation**: Ensure users can only access their own memories
- ✅ **Memory Filtering**: Filter by memory type (preference, fact, context, etc.)
- ✅ **Conversation Chains**: Manage conversation history with memory context

### Advanced Features
- ✅ **Multi-User Support**: Handle multiple users with isolated data
- ✅ **Performance Monitoring**: Measure storage and search performance
- ✅ **Analytics**: Generate statistics and insights about memory usage
- ✅ **Error Handling**: Graceful error handling and recovery
- ✅ **Health Checks**: System status validation

### Realistic Scenarios
- 👩‍💻 **Developer Persona**: Alice with development preferences, project context, and learning activities
- 👨‍💼 **Analyst Persona**: Bob with business tools, analysis projects, and automation needs

## Requirements Validation

This comprehensive example satisfies:

- **Requirement 6.5**: "Testing the MVP with practical examples"
  - ✅ Multiple realistic user scenarios
  - ✅ All memory types demonstrated
  - ✅ Performance validation included

- **Requirement 6.6**: "Foundation for extending to full feature set"  
  - ✅ Demonstrates scalability patterns
  - ✅ Shows integration points for future features
  - ✅ Validates architecture design decisions

## Next Steps

After running this example successfully, the system is ready for:

1. **Production Integration**: Use the demonstrated patterns in real applications
2. **Neo4j Integration**: Add graph-based persistence as secondary storage
3. **Evaluation Framework**: Implement comparative analysis tools
4. **Scaling**: Handle larger user bases and memory volumes
5. **Advanced Features**: Add memory summarization, relationship extraction, etc.

## Troubleshooting

### Common Issues

1. **OpenAI API Error**: Check your API key and base URL in `.env`
2. **Import Errors**: Ensure you're running from the project root directory
3. **Permission Errors**: Verify file permissions and environment setup

### Debug Mode

To run with additional debugging information:

```bash
export PYTHONPATH=$(pwd)
export LOG_LEVEL=DEBUG
python examples/comprehensive_langchain_example.py
```

## Files Structure

```
examples/
├── README.md                           # This documentation
├── langchain_vector_example.py         # Basic example
└── comprehensive_langchain_example.py  # Comprehensive demonstration (Task 2.5)
```

---

**Task 2.5 Status**: ✅ **COMPLETED**

This comprehensive example fulfills all requirements for demonstrating the complete LangChain vector persistence workflow with practical examples, multiple user scenarios, and validation against MVP requirements 6.5 and 6.6. 