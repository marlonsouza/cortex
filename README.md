# Cortex

Cortex is an AI-powered project scaffolding tool that helps you generate project structures and code files from natural language descriptions. It uses the Model Context Protocol (MCP) to interact with language models and create projects based on your requirements.

## Features

- 🚀 Generate project structures and code files from text descriptions
- 🤖 Interacts with LLMs through the Model Context Protocol
- ⚙️ Configurable through command-line arguments or `.env` file
- 🛠️ Supports multiple programming languages and frameworks
- 🔍 Built-in error handling and debugging capabilities
- 📝 Clean and maintainable code structure

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/cortex.git
cd cortex
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Create a configuration file:
```bash
cp config.example.env .env
# Edit .env with your preferred settings
```

5. Install and set up Ollama (if using local models):
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model you want to use (e.g., llama3)
ollama pull llama3
```

## Usage

### Basic Usage

```bash
python -m cortex.agent
```

When prompted:
1. Enter a project name
2. Enter your project description (finish with Ctrl-D on Unix or Ctrl-Z+Enter on Windows)

### Using with Ollama

Cortex works seamlessly with Ollama for local model inference. To use Ollama:

1. Make sure Ollama is running:
```bash
ollama serve
```

2. Run Cortex with Ollama:
```bash
python -m cortex.agent --url http://localhost:11434 --model llama3
```

The default configuration is already set up for Ollama, so you can also just run:
```bash
python -m cortex.agent
```

### Command-line Options

```bash
python -m cortex.agent [OPTIONS]
```

Options:
- `--model MODEL` - The model to use (default: llama3)
- `--url URL` - The MCP API URL (default: http://localhost:11434)
- `--api-key KEY` - Your MCP API key
- `--temperature TEMP` - Model temperature (default: 0.7)
- `--max-tokens TOKENS` - Maximum tokens to generate (default: 4096)
- `--debug` - Enable debug mode

### Configuration via `.env` File

Create a `.env` file in the project directory with these variables:

```env
MCP_API_URL=http://your-mcp-server/v1
MCP_API_KEY=your_api_key_here
MCP_MODEL=your_model_name
MCP_TEMPERATURE=0.7
MCP_MAX_TOKENS=4096
MCP_TIMEOUT=120
DEBUG=false
```

## Project Structure

```
cortex/
├── __init__.py          # Package initialization
├── agent.py             # Main agent implementation
├── requirements.txt     # Project dependencies
├── config.example.env   # Example configuration
└── README.md           # Project documentation
```

## Development

### Setting Up Development Environment

1. Install development dependencies:
```bash
pip install -r requirements.txt
```

2. Install pre-commit hooks:
```bash
pre-commit install
```

### Running Tests

```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[MIT License](LICENSE)