#!/usr/bin/env python
import os
import sys
import uvicorn
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from server import app  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
