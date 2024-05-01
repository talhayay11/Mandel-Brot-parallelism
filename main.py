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


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Mandelbrot Set Viewer")
    button = tk.Button(root, text="Click Me")
    button.pack(pady=20)
    #mandelbrot_display = ZoomableMandelbrot(ax, max_iter=100,regions="auto")
    for processors in processCount:
        mandelbrot_display =  ZoomableMandelbrot(root, max_iter=100,regions="auto", processors=processors)
        #plt.show()
    
    root.mainloop()