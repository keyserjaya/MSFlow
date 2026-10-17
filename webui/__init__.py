"""MSFlow webui package. Pins matplotlib to a headless backend early, because
several repo modules (utils.plot_script via raw_inpaint_common) import
matplotlib.pyplot at module scope, and auto-detection picks TkAgg in WSL
(which breaks on uv's statically-linked _tkinter).
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")
