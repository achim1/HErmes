"""
A set of figure sizes for publication-ready figures on A4 paper based on the
golden ratio. For the use with pylabe figure, e.g. fig =pylab.figure(figsize=FIGSIZE_A4)

Available layouts:

- FIGSIZE_A4: Figure on full A4 width, portrait mode, golden ratio
- FIGSIZE_A4_LANDSCAPE: Figure on full A4 width, landscape mode, golden ratio
- FIGSIZE_A4_LANDSCAPE_HALF_HEIGHT: Figure on full A4 width, landscape mode, height half of golden ratio
- FIGSIZE_A4_SQUARE: Figure on full A4 width, square
"""

import numpy as np

GOLDEN_RATIO = (1 + np.sqrt(5))/2.

WIDTH_A4 = 5.78851 # inch
FIGSIZE_A4_LANDSCAPE = (WIDTH_A4, WIDTH_A4/GOLDEN_RATIO)
FIGSIZE_A4_LANDSCAPE_HALF_HEIGHT = (WIDTH_A4, (WIDTH_A4/(2*GOLDEN_RATIO)))

FIGSIZE_A4 = (WIDTH_A4, (WIDTH_A4/GOLDEN_RATIO) + WIDTH_A4)
FIGSIZE_A4_SQUARE = (WIDTH_A4, WIDTH_A4)



