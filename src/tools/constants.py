import os
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(Path(__file__).parent))

PROJECT_ROOT_DIR = os.path.join(ROOT_DIR, '..')
REPORT_IMAGE_DIR = os.path.join(PROJECT_ROOT_DIR, 'images')
