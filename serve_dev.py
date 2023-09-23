#!/usr/bin/env python
import os
import sys
import uvicorn
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    uvicorn.run('server:app', host="0.0.0.0", port=8000, reload=True)
