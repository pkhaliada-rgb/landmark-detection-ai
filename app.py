"""
Convenience entry point so you can run detection from the project root
without remembering the src/ path.

USAGE:
    python app.py --image path/to/photo.jpg
    python app.py --image path/to/photo.jpg --show-heatmap
"""

from src.detect import main

if __name__ == "__main__":
    main()
