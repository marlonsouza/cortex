"""Utility functions for Cortex."""

import re


def clean_path(path: str) -> str:
    """Clean and normalize a file path.

    Args:
        path: The path to clean

    Returns:
        The cleaned and normalized path
    """
    path = re.sub(r"[^\w\-./\\]", "", path)
    path = path.replace("\\", "/")
    path = re.sub(r"/+", "/", path)
    path = path.strip("/")
    if ".." in path:
        parts = path.split("/")
        stack = []
        for part in parts:
            if part == "..":
                if stack:
                    stack.pop()
            else:
                stack.append(part)
        path = "/".join(stack)
    return path


def clean_content(content: str) -> str:
    """Clean content for file writing.

    Args:
        content: The content to clean

    Returns:
        The cleaned content
    """
    content = content.replace('\\"', '"')
    content = content.replace("\\n", "\n")
    if ".go" in content:
        content = re.sub(r'import\s*\(\\"([^"]+)\\"\)', r'import ("\1")', content)
        content = re.sub(r'(\w+)\(\\"([^"]+)\\"\)', r'\1("\2")', content)
        content = re.sub(
            r"map\[string\]string\{([^}]+)\}", r"map[string]string{\1}", content
        )
    return content
