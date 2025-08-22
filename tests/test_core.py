"""
Basic unit test for the core module.
Run with: pytest
"""

from app.core import hello

def test_hello():
    assert hello() == "Hello, World!"