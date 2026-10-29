import sys
import os

# Ensure current directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from visualizer_gui import main

if __name__ == '__main__':
    main()
