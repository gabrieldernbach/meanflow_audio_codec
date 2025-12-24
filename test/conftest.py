import os
import sys
from pathlib import Path

# Set JAX platform before any JAX imports
os.environ["JAX_PLATFORMS"] = "cpu"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
