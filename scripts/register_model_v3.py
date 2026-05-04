#!/usr/bin/env python3
"""Register model_v3 in the database."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import register_current_model_from_metadata

if __name__ == "__main__":
    try:
        register_current_model_from_metadata("v3")
        print("✓ Model v3 registered successfully in database")
    except Exception as e:
        print(f"✗ Failed to register model v3: {e}")
        sys.exit(1)
