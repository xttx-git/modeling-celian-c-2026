import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _utils.plot_utils import setup_style, save_fig, PALETTE, COLORS
setup_style("nature")
from figures.plot_common import generate

generate("fig_q2_constraint_margins_lollipop")
