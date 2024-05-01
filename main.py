import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from numba import cuda
import time
from concurrent.futures import ProcessPoolExecutor
import os
import tkinter as tk

from mandelbrot_plotting import  ZoomableMandelbrot
from mandelbrot_core import processCount
 

# if __name__ == '__main__':
    
#     plot_mandelbrot_gpu(max_iter=10000, width=1024, height=1024)
#     plot_zoomable_mandelbrot(1000)

def on_resolution_selected(selected_resolution):
    width, height = map(int, selected_resolution.split('x'))
    for processors in processCount:
        mandelbrot_display = ZoomableMandelbrot(root, max_iter=100, regions="auto", processors=processors, width=width, height=height)
        # Do further operations with mandelbrot_display if needed

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Mandelbrot Set Viewer")
    
    resolutions = ["800x600", "1024x768", "1280x720", "1920x1080"]
    selected_resolution = tk.StringVar(root)
    selected_resolution.set(resolutions[0])  # Default resolution
    
    # Dropdown menu for selecting resolution
    resolution_dropdown = tk.OptionMenu(root, selected_resolution, *resolutions, command=on_resolution_selected)
    resolution_dropdown.pack(pady=20)
    
    root.mainloop()