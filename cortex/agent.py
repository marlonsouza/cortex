"""Cortex Agent - Main agent implementation for project scaffolding."""

import asyncio
import json
import logging
import os
import pathlib
import re
from typing import Any, Dict

import aiohttp

from cortex.utils import clean_content, clean_path

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)


class SystemMessage:
    """Represents a system message in the chat context."""

    def __init__(self, content: str):
        """Initialize a system message.

        Args:
            content: The content of the system message
        """
        self.role = "system"
        self.content = content


class UserMessage:
    """Represents a user message in the chat context."""

    def __init__(self, content: str):
        """Initialize a user message.

        Args:
            content: The content of the user message
        """
        self.role = "user"
        self.content = content


def load_env_file(file_path: str = ".env") -> Dict[str, str]:
    """Load environment variables from file.

    Args:
        file_path: Path to the .env file

    Returns:
        Dictionary of environment variables
    """
    env_vars = {}

    env_path = pathlib.Path(file_path)
    if not env_path.exists():
        logging.info(f"{file_path} not found, using defaults and environment variables")
        return env_vars

    try:
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                try:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()
                except ValueError:
                    logging.warning(f"Skipping invalid line in .env file: {line}")

        logging.info(f"Loaded configuration from {file_path}")
        return env_vars
    except Exception as e:
        logging.warning(f"Error loading {file_path}: {str(e)}")
        return env_vars


# Load .env file first
env_vars = load_env_file()


def get_config(key: str, default: str, env_vars: Dict[str, str] = None) -> str:
    """Get configuration value from environment variables.

    Args:
        key: Configuration key to look up
        default: Default value if key not found
        env_vars: Optional dictionary of environment variables

    Returns:
        Configuration value
    """
    if env_vars and key in env_vars:
        return env_vars[key]
    return os.environ.get(key, default)


# Model Context Protocol (MCP) Configuration
MCP_API_URL = get_config("MCP_API_URL", "http://localhost:11434", env_vars)
MCP_API_KEY = get_config("MCP_API_KEY", "", env_vars)  # Set your API key if needed
MCP_MODEL = get_config("MCP_MODEL", "llama3", env_vars)
MCP_TEMPERATURE = float(get_config("MCP_TEMPERATURE", "0.7", env_vars))
MCP_MAX_TOKENS = int(get_config("MCP_MAX_TOKENS", "4096", env_vars))
MCP_TIMEOUT = int(get_config("MCP_TIMEOUT", "120", env_vars))

# Context7 Specific Configuration
CONTEXT7_ENABLED = get_config("CONTEXT7_ENABLED", "false", env_vars).lower() in [
    "true",
    "1",
    "yes",
    "y",
]
CONTEXT7_PORT = get_config("CONTEXT7_PORT", "8123", env_vars)
CONTEXT7_HOST = get_config("CONTEXT7_HOST", "localhost", env_vars)
CONTEXT7_STARTUP_TIMEOUT = int(get_config("CONTEXT7_STARTUP_TIMEOUT", "30", env_vars))
CONTEXT7_MIN_TOKENS = int(get_config("CONTEXT7_MIN_TOKENS", "10000", env_vars))
CONTEXT7_LIBS = get_config(
    "CONTEXT7_LIBS", "", env_vars
)  # Comma-separated list of libraries

# Debug mode configuration
DEBUG = get_config("DEBUG", "false", env_vars).lower() in ["true", "1", "yes", "y"]

# Start Context7 MCP server if enabled
if CONTEXT7_ENABLED:
    import atexit
    import subprocess
    import time

    # Function to start Context7 MCP server
    def start_context7_server():
        """Start the Context7 MCP server.

        Returns:
            bool: True if server started successfully, False otherwise
        """
        global context7_process
        try:
            print("Starting Context7 MCP server...")
            # Use npx to start the server
            cmd = ["npx", "-y", "@upstash/context7-mcp@latest"]

            # Set environment variables for Context7
            env = os.environ.copy()
            env["PORT"] = CONTEXT7_PORT
            env["DEFAULT_MINIMUM_TOKENS"] = str(CONTEXT7_MIN_TOKENS)

            # Start the process
            context7_process = subprocess.Popen(
                cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # Register cleanup function
            atexit.register(stop_context7_server)

            # Wait for server to start
            start_time = time.time()
            server_url = f"http://{CONTEXT7_HOST}:{CONTEXT7_PORT}"

            print(f"Waiting for Context7 server to start at {server_url}...")
            while time.time() - start_time < CONTEXT7_STARTUP_TIMEOUT:
                try:
                    # Try to connect to server
                    import urllib.request
                    urllib.request.urlopen(server_url, timeout=1)
                    print("Context7 MCP server started successfully")
                    return True
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                    time.sleep(1)

            print("Timed out waiting for Context7 server to start")
            stop_context7_server()
            return False
        except Exception as e:
            print(f"Error starting Context7 server: {str(e)}")
            return False

    # Function to stop Context7 MCP server
    def stop_context7_server():
        """Stop the Context7 MCP server and clean up resources."""
        global context7_process
        if "context7_process" in globals() and context7_process:
            print("Stopping Context7 MCP server...")
            try:
                context7_process.terminate()
                context7_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                context7_process.kill()
            context7_process = None

    # Start the server
    if start_context7_server():
        # Update MCP URL to point to Context7
        MCP_API_URL = f"http://{CONTEXT7_HOST}:{CONTEXT7_PORT}"
    else:
        print(
            "Warning: Failed to start Context7 server, falling back to direct LLM connection"
        )
        CONTEXT7_ENABLED = False

# Show configuration info
print("Model Configuration:")
print(f"  Model: {MCP_MODEL}")
print(f"  Temperature: {MCP_TEMPERATURE}")
print(f"  Max Tokens: {MCP_MAX_TOKENS}")

if CONTEXT7_ENABLED:
    print("\nContext7 MCP Enabled:")
    print(f"  Host: {CONTEXT7_HOST}")
    print(f"  Port: {CONTEXT7_PORT}")
    print(f"  Token Limit: {CONTEXT7_MIN_TOKENS}")
    if CONTEXT7_LIBS:
        print(f"  Libraries: {CONTEXT7_LIBS}")

    # Update MCP URL to point to Context7
    MCP_API_URL = f"http://{CONTEXT7_HOST}:{CONTEXT7_PORT}"
else:
    print("\nDirect LLM Connection:")
    print(f"  API URL: {MCP_API_URL}")
    print(f"  API Key: {'Configured' if MCP_API_KEY else 'Not configured'}")


class MCPClient:
    """Client for interacting with the Model Context Protocol API."""

    def __init__(
        self,
        model: str,
        api_url: str = MCP_API_URL,
        api_key: str = MCP_API_KEY,
        temperature: float = MCP_TEMPERATURE,
        max_tokens: int = MCP_MAX_TOKENS,
        timeout: int = MCP_TIMEOUT,
    ):
        """Initialize the MCP client.

        Args:
            model: The model to use
            api_url: The API URL
            api_key: The API key
            temperature: The temperature setting
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.using_context7 = CONTEXT7_ENABLED
        self.context7_libs = []
        if self.using_context7 and CONTEXT7_LIBS:
            self.context7_libs = [
                lib.strip() for lib in CONTEXT7_LIBS.split(",") if lib.strip()
            ]
            print(f"Context7 libraries: {', '.join(self.context7_libs)}")

    async def create(self, messages: list) -> Any:
        """Create a completion using the MCP API.

        Args:
            messages: List of messages to send to the API

        Returns:
            The API response
        """
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            formatted_messages = []
            for msg in messages:
                if hasattr(msg, "role"):
                    role = msg.role
                else:
                    role = (
                        "system"
                        if (
                            getattr(msg, "__class__", None)
                            and msg.__class__.__name__ == "SystemMessage"
                        )
                        else "user"
                    )
                formatted_messages.append({
                    "role": role,
                    "content": msg.content
                })

            self.is_ollama = (
                "ollama" in self.api_url.lower()
                or "11434" in self.api_url
            )

            if self.is_ollama:
                payload = {
                    "model": self.model,
                    "messages": formatted_messages,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                    "stream": False,
                }
            else:
                payload = {
                    "model": self.model,
                    "messages": formatted_messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": False,
                }

            if self.is_ollama:
                base_url = self.api_url.rstrip("/")
                if "11434" in base_url:
                    if "/api" in base_url:
                        if base_url.endswith("/api"):
                            endpoint = f"{base_url}/chat"
                        else:
                            base = base_url.split('/api')[0]
                            endpoint = f"{base}/api/chat"
                    else:
                        endpoint = f"{base_url}/api/chat"
                else:
                    if "/api/chat" in base_url:
                        endpoint = base_url
                    elif "/api" in base_url:
                        base = base_url.rstrip('/api')
                        endpoint = f"{base}/api/chat"
                    else:
                        endpoint = f"{base_url}/api/chat"

                if "localhost" in base_url or "127.0.0.1" in base_url:
                    try:
                        url_parts = (
                            base_url.split("://")[1]
                            if "://" in base_url
                            else base_url
                        )
                        host_part = url_parts.split("/")[0]
                        self.backup_endpoint = (
                            f"http://{host_part}/api/chat"
                        )
                    except:
                        self.backup_endpoint = (
                            "http://localhost:11434/api/chat"
                        )
            else:
                base_url = self.api_url.rstrip("/")
                if "/v1" in base_url:
                    if base_url.endswith("/v1"):
                        endpoint = f"{base_url}/chat/completions"
                    else:
                        parts = base_url.split("/v1")
                        endpoint = f"{parts[0]}/v1/chat/completions"
                else:
                    endpoint = f"{base_url}/v1/chat/completions"

            logging.info(f"Sending request to MCP: {endpoint}")
            logging.info(
                f"Debug - API URL: {self.api_url}, Is Ollama: {self.is_ollama}"
            )
            logging.info(f"Debug - Payload: {json.dumps(payload, indent=2)}")

            max_retries = 2
            retry_count = 0
            error_messages = []
            endpoints_to_try = [endpoint]
            if self.is_ollama and hasattr(self, "backup_endpoint"):
                endpoints_to_try.append(self.backup_endpoint)
            if self.is_ollama and "localhost" in endpoint:
                endpoints_to_try.append("http://localhost:11434/api/chat")

            logging.info(f"Will try the following endpoints: {endpoints_to_try}")

            for current_endpoint in endpoints_to_try:
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        logging.info(
                            "Attempt {} with endpoint: {}".format(
                                retry_count + 1, current_endpoint
                            )
                        )
                        async with session.post(
                            current_endpoint,
                            json=payload,
                            headers=self.headers,
                            timeout=aiohttp.ClientTimeout(total=self.timeout),
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                logging.info(
                                    "✅ Success! Received response "
                                    "({} chars) from: {}".format(
                                        len(str(result)), current_endpoint
                                    )
                                )
                                self.last_successful_endpoint = current_endpoint
                                break
                            else:
                                error_text = await response.text()
                                error_msg = (
                                    "API Error: HTTP {} from {} - {}".format(
                                        response.status,
                                        current_endpoint,
                                        error_text
                                    )
                                )
                                logging.warning(error_msg)
                                error_messages.append(error_msg)
                                retry_count += 1
                    except aiohttp.ClientError as e:
                        error_msg = (
                            "Connection error with {}: {}".format(
                                current_endpoint, str(e)
                            )
                        )
                        logging.warning(error_msg)
                        error_messages.append(error_msg)
                        retry_count += 1
                    except asyncio.TimeoutError:
                        error_msg = (
                            "Timeout connecting to {} after {}s".format(
                                current_endpoint, self.timeout
                            )
                        )
                        logging.warning(error_msg)
                        error_messages.append(error_msg)
                        retry_count += 1
                    except Exception as e:
                        error_msg = (
                            "Unexpected error with {}: {}".format(
                                current_endpoint, str(e)
                            )
                        )
                        logging.warning(error_msg)
                        error_messages.append(error_msg)
                        retry_count += 1
                if "result" in locals():
                    break

            if "result" not in locals():
                all_errors = "\n".join(error_messages)
                error_msg = (
                    f"All API connection attempts failed after trying "
                    f"{len(endpoints_to_try)} endpoints.\nErrors:\n{all_errors}"
                )
                logging.error(error_msg)
                raise Exception(f"Failed to connect to LLM API: {error_msg}")

            logging.info(
                f"Received response from API with {len(str(result))} characters"
            )

            class MCPResponse:
                """Response from the MCP API."""

                def __init__(self, content: str):
                    """Initialize the response.

                    Args:
                        content: The response content
                    """
                    self.content = content

            content = ""
            if self.is_ollama:
                content = result.get("message", {}).get("content", "")
                if not content:
                    if "response" in result:
                        content = result.get("response", "")
            else:
                content = (
                    result.get("choices", [{}])[0].get("message", {}).get("content", "")
                )

            if not content:
                logging.warning("Empty content received from API")
                logging.warning(f"Response structure: {json.dumps(result, indent=2)}")
            else:
                logging.info(f"Successfully extracted content (length: {len(content)})")

            return MCPResponse(content)


# Create MCP client
client = MCPClient(model=MCP_MODEL)


def debug_print(*args, **kwargs) -> None:
    """Print debug messages if debug mode is enabled."""
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)


def extract_json(s: str) -> str:
    """Extract JSON from a string.

    Args:
        s: The string to extract JSON from

    Returns:
        The extracted JSON string
    """
    s = s.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        debug_print(f"Extracting JSON from position {start} to {end+1}")
        return s[start : end + 1]
    debug_print("Could not find valid JSON markers, returning original string")
    return s


def fix_escaping(json_str: str) -> str:
    """Fix common escaping issues in JSON strings from LLMs.

    Args:
        json_str: The JSON string to fix

    Returns:
        The fixed JSON string
    """
    original = json_str

    debug_print("Original JSON string length:", len(json_str))

    if "]" not in json_str and json_str.count("{") > json_str.count("}"):
        json_str += "}]}"
        debug_print("Added missing brackets at the end")

    if "{{" in json_str or "}}" in json_str:
        json_str = json_str.replace("{{", "{").replace("}}", "}")
        debug_print("Fixed nested braces in code")

    if "HandleFunc=" in json_str:
        json_str = json_str.replace("HandleFunc=", "HandleFunc(")
        debug_print("Fixed HandleFunc syntax")

    try:
        content_match = re.search(
            r'"content"\s*:\s*"(.*?)(?:"\s*}|\s*$)', json_str, re.DOTALL
        )
        if content_match:
            content = content_match.group(1)
            debug_print("Found content field with length:", len(content))

            content = content.replace('\\"', '"').replace("\\n", "\n")

            if ".go" in json_str:
                content = re.sub(
                    r'import\s*\(\\"([^"]+)\\"\)', r'import ("\1")', content
                )
                content = re.sub(r'(\w+)\(\\"([^"]+)\\"\)', r'\1("\2")', content)
                content = re.sub(
                    r"map\[string\]string\{([^}]+)\}", r"map[string]string{\1}", content
                )

            json_str = re.sub(
                r'"content"\s*:\s*"(.*?)(?:"\s*}|\s*$)',
                lambda m: m.group(0).replace(m.group(1), content),
                json_str,
                flags=re.DOTALL,
            )
            debug_print("Fixed content field escaping")
    except Exception as e:
        debug_print(f"Error while trying to fix content field: {str(e)}")

    if "content" in json_str:
        if "\\\\n" in json_str:
            pass
        elif "\\n" in json_str:
            json_str = json_str.replace("\\n", "\\\\n")
            debug_print("Re-escaped newlines for JSON")

    content_match = re.search(
        r'"content"\s*:\s*"(.*?)(?:"\s*}|\s*$)', json_str, re.DOTALL
    )
    if content_match:
        content = content_match.group(1)
        debug_print("Found content field with length:", len(content))

        count_escaped = content.count('\\"')
        count_unescaped = content.count('"') - count_escaped

        debug_print(
            f"Quote counts: escaped={count_escaped}, " f"unescaped={count_unescaped}"
        )

        if count_unescaped % 2 != 0:
            debug_print("Detected unbalanced quotes, attempting to fix")
            last_idx = json_str.rfind('"', 0, json_str.rfind("}"))
            if last_idx > 0:
                json_str = json_str[:last_idx] + '"' + json_str[last_idx:]
                debug_print(f"Added missing quote at position {last_idx}")

    if json_str != original:
        debug_print("JSON string was modified for parsing")

    return json_str


if __name__ == "__main__":
    import argparse

    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description="Cortex - AI assistant to scaffold projects using MCP"
    )
    parser.add_argument(
        "--model", type=str, help=f"Model to use (default: {MCP_MODEL})"
    )
    parser.add_argument("--url", type=str, help=f"MCP API URL (default: {MCP_API_URL})")
    parser.add_argument("--api-key", type=str, help="MCP API key")
    parser.add_argument(
        "--temperature", type=float, help=f"Temperature (default: {MCP_TEMPERATURE})"
    )
    parser.add_argument(
        "--max-tokens", type=int, help=f"Max tokens (default: {MCP_MAX_TOKENS})"
    )
    parser.add_argument("--project", type=str, help="Project name (skips the prompt)")
    parser.add_argument(
        "--prompt-file",
        type=str,
        help="File containing the prompt (skips the prompt input)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Update configuration from command line args
    if args.model:
        MCP_MODEL = args.model
        print(f"Using model: {MCP_MODEL}")

    if args.url:
        MCP_API_URL = args.url
        print(f"Using MCP API URL: {MCP_API_URL}")

    if args.api_key:
        MCP_API_KEY = args.api_key
        print("API key provided via command line")

    if args.temperature is not None:
        MCP_TEMPERATURE = args.temperature
        print(f"Using temperature: {MCP_TEMPERATURE}")

    if args.max_tokens is not None:
        MCP_MAX_TOKENS = args.max_tokens
        print(f"Using max tokens: {MCP_MAX_TOKENS}")

    if args.debug:
        DEBUG = True
        print("Debug mode enabled")

    # Recreate client with updated parameters
    client = MCPClient(
        model=MCP_MODEL,
        api_url=MCP_API_URL,
        api_key=MCP_API_KEY,
        temperature=MCP_TEMPERATURE,
        max_tokens=MCP_MAX_TOKENS,
    )

    # Get project name
    if args.project:
        project_name = args.project
        print(f"Using project name: {project_name}")
    else:
        project_name = input("Enter new project name: ")

    os.makedirs(project_name, exist_ok=True)
    os.chdir(project_name)

    # Get prompt from file or stdin
    if args.prompt_file:
        try:
            with open(args.prompt_file, "r") as f:
                prompt = f.read()
            print(f"Read prompt from file: {args.prompt_file}")
        except Exception as e:
            print(f"Error reading prompt file: {str(e)}")
            exit(1)
    else:
        print(
            "Enter your prompt for the assistant. "
            "Finish with EOF (Ctrl-D on Unix, Ctrl-Z+Enter on Windows):"
        )
        import sys

        prompt = sys.stdin.read()

    if not prompt.strip():
        print("Error: Empty prompt received")
        exit(1)

    print(f"Prompt length: {len(prompt)} characters")

    # Prepare messages for LLM
    system_msg = SystemMessage(
        content=(
            "You are an AI assistant that generates code and project "
            "structures based on user requirements. "
            "IMPORTANT: You must ALWAYS reply with a valid JSON object. "
            "DO NOT provide explanations or any text outside the JSON object.\n\n"
            "The JSON object must have this EXACT structure:\n"
            "{\n"
            '  "tool_calls": [\n'
            '    {"name": "make_directory", "arguments": {"path": "directory_name"}},\n'
            '    {"name": "write_file", "arguments": {"path": "file_path", '
            '"content": "file content here"}}\n'
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "1. Generate the actual code based on the user's requirements\n"
            "2. Do not include any explanations or markdown in the response\n"
            "3. The content field should contain the actual code to be written\n"
            "4. Use proper file extensions and directory structure\n"
            "5. Include all necessary imports and dependencies\n"
            "6. Follow best practices for the target language\n"
            "7. Make sure the code is complete and runnable\n"
            "8. Do not include any placeholder comments or TODOs\n"
            "9. Do not include any example code or pseudocode\n"
            "10. Generate the actual implementation that matches the user's "
            "requirements exactly"
        )
    )

    # Create user message for LLM
    user_msg = UserMessage(content=prompt, source="user")

    # Call the LLM asynchronously
    async def _run_plan():
        try:
            logging.info("💬 Sending request to language model...")
            res = await client.create([system_msg, user_msg])
            logging.info("✅ Successfully received response from language model")
            return res.content
        except Exception as e:
            logging.error(f"❌ Error communicating with language model: {str(e)}")
            logging.error(
                "Please check your model configuration and network connection"
            )
            if DEBUG:
                import traceback

                logging.error("Detailed error traceback:")
                traceback.print_exc()
            raise

    try:
        print("\n🚀 Processing your request. This may take a moment...\n")
        text = asyncio.run(_run_plan())
        print("\n✅ Successfully processed your request!\n")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print(
            "\nPlease try again with different settings or check the logs "
            "for details.\n"
        )
        if not DEBUG:
            print("Set DEBUG=true in .env file for more detailed error information.")
        exit(1)
