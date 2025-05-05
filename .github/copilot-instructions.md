# Cortex Development Guidelines

## Project Overview
Cortex is an AI-powered project scaffolding tool that generates project structures and code files from natural language descriptions. It uses direct HTTP connections to LLM APIs (like Ollama) to process user requests and generate project structures.

## Architecture
- Direct HTTP connection to LLM APIs (default: Ollama at http://localhost:11434)
- Async HTTP client using aiohttp for LLM communication
- JSON-based request/response format
- Environment-based configuration

## Key Components
1. `MCPClient` class (in agent.py)
   - Handles direct HTTP communication with LLM APIs
   - Supports configuration via environment variables or command line
   - Implements async request/response handling

2. Message Processing
   - System messages define the scaffolding behavior
   - User messages contain project requirements
   - Responses are parsed as JSON for project generation

## Development Guidelines
1. Keep the LLM interaction simple and direct
2. Maintain clear error handling for API communication
3. Support both local (Ollama) and remote LLM endpoints
4. Use environment variables for configuration
5. Implement proper async/await patterns for HTTP requests

## Testing
- Test with both local and remote LLM endpoints
- Verify JSON response parsing
- Ensure proper error handling for API failures
- Test project generation with various input formats

## Configuration
The project supports configuration through:
- Environment variables
- .env file
- Command-line arguments

Key configuration parameters:
- MCP_API_URL (default: http://localhost:11434)
- MCP_API_KEY (optional)
- MCP_MODEL (default: llama3)
- MCP_TEMPERATURE (default: 0.7)
- MCP_MAX_TOKENS (default: 4096)
- MCP_TIMEOUT (default: 120)
- DEBUG (default: false) 