import os
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
from matplotlib import colors
import numpy as np
from matplotlib.ticker import LogFormatter
import seaborn as sns

# Plot sizes in inches
col_wid = 3.3258 * 1.3
col_sep = 0.3486 * 1.3
col_mar = 0.5990 * 1.3

plt.style.use('default')
# sns.set_style("white")  # optional

# Full path to your local font directory
font_dir = os.path.expanduser('~/.local/share/matplotlib/mpl-data/fonts/ttf')

# Register each font with Matplotlib
for font_file in os.listdir(font_dir):
    if font_file.lower().endswith(('.otf', '.ttf')):
        font_path = os.path.join(font_dir, font_file)
        font_manager.fontManager.addfont(font_path)

# # Confirm fonts are detected (optional)
# print([f.name for f in font_manager.fontManager.ttflist if 'Termes' in f.name])

# Set TeX Gyre Termes as your serif font
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['TeX Gyre Termes']
rcParams['font.size'] = 14
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{txfonts}'

# Sets default figure size to a square one column figure
rcParams['figure.figsize'] = [col_wid, col_wid]


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Takes a Matplotlib colour map and returns a truncated colour map between minval and maxval.
    This is useful for chopping off the ends of colour maps where the colours can be a bit garish
    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def correct_tick_marks(axis):
    """
    Takes an axis object from e.g. fig, ax = plt.subplots() and sets the correct tick marks, i.e.,
    ticks on all four axes facing inwards, and minor ticks for log scale plots.
    """

    axis.tick_params(axis='y', which='major', direction='in', color='k',
                     length=5.0, width=1.0, right=True)
    axis.tick_params(axis='y', which='minor', direction='in', color='k',
                     length=3.0, width=1.0, right=True)
    axis.tick_params(axis='x', which='major', direction='in', color='k',
                     length=5.0, width=1.0, top=True)
    axis.tick_params(axis='x', which='minor', direction='in', color='k',
                     length=3.0, width=1.0, top=True)
