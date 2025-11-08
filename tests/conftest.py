"""Pytest configuration and fixtures"""
import sys
from pathlib import Path

# Add src to path to match project structure
# This allows tests to import modules like 'classification' and 'gmail'
# without needing to install the package or use relative imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

