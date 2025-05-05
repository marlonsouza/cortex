"""Test cases for the Cortex agent."""

from cortex.utils import clean_path, clean_content


def test_clean_path():
    """Test path cleaning functionality."""
    assert clean_path("test/dir") == "test/dir"
    assert clean_path("test\\dir") == "test/dir"
    assert clean_path("test/dir/") == "test/dir"
    assert clean_path("test/../dir") == "dir"


def test_clean_content():
    """Test content cleaning functionality."""
    assert clean_content('hello world') == 'hello world'
    assert clean_content('hello \\"world\\"') == 'hello "world"'
    assert clean_content('hello\\nworld') == 'hello\nworld'
    assert clean_content('import (\\"fmt\\")') == 'import ("fmt")'
