"""
Settings
"""
import logging
import os
from pathlib import Path
import sys

import yaml
from pydantic_yaml import parse_yaml_file_as

from configuration import Configuration, get_logging_level

BASE_DIR: str = Path(os.path.abspath(os.path.dirname(__file__)))
STATIC_DIR: str = BASE_DIR / "static"
DOCS_DIR: str = BASE_DIR / ".." / "docs"

config_path = os.path.join(os.path.curdir, "config", "default.yaml")

CONFIGURATION: Configuration
try:
    with open(config_path, "r", encoding="utf-8") as config_file:
        CONFIGURATION = parse_yaml_file_as(Configuration, config_file)
except (yaml.YAMLError, IOError) as e:
    sys.stderr.write(f"Could not open configuration file at {config_path}. {e}\n")
    sys.exit(1)

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=get_logging_level(CONFIGURATION.loglevel))

__all__ = [
    "DOCS_DIR",
    "STATIC_DIR",
    "CONFIGURATION"
]
