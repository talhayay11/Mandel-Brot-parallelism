import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from numba import cuda
import time
from concurrent.futures import ProcessPoolExecutor
import os
import tkinter as tk

from mandelbrot_plotting import  ZoomableMandelbrot, Efficiency, Speedup
from mandelbrot_gpu import plot_mandelbrot_gpu
from mandelbrot_core import processCount

maximumPhysicalCores = os.cpu_count() // 2

minToMaxProcessors = list(range(1,maximumPhysicalCores+1))

 

# if __name__ == '__main__':
    
#     plot_mandelbrot_gpu(max_iter=10000, width=1024, height=1024)
#     plot_zoomable_mandelbrot(1000)

def on_resolution_selected():
    # width, height = map(int, selected_resolution.split('x'))
    # print(width, height)
    print(f"Testing for {minToMaxProcessors} processors")
    for processors in minToMaxProcessors:
        mandelbrot = ZoomableMandelbrot(root, max_iter=100, regions="auto", processors=processors, width=800, height=800, ComputeOnce=False)
        # Do further operations with mandelbrot_display if needed
    
    print(f"Efficiency: {Efficiency}")
        
    print(f"Average Efficiency: {np.mean(Efficiency)}")

    print(f"Speedup: {Speedup}")

    print(f"Average Speedup: {np.mean(Speedup)}")

    Efficiency.clear()
    Speedup.clear()

def ComputeOnce():

    ZoomableMandelbrot(root, max_iter=100, regions="auto", processors=maximumPhysicalCores, width=800, height=800, ComputeOnce=True)
        # Do further operations with mandelbrot_display if needed

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Mandelbrot Set Viewer")
    
    resolutions = ["800x800", "1024x1024", "1280x720", "1920x1080"]
    selected_resolution = tk.StringVar(root)
    selected_resolution.set(resolutions[0])  # Default resolution

    button = tk.Button(root, text="Tüm Çekirdeklerle Hesapla", command=ComputeOnce)
    button.pack(pady=20)  # Add some vertical padding
    button = tk.Button(root, text="Sırayla Tüm Çekirdekler", command=on_resolution_selected)
    button.pack(pady=20)  # Add some vertical padding

    # Dropdown menu for selecting resolution
    resolution_dropdown = tk.OptionMenu(root, selected_resolution, *resolutions, command=on_resolution_selected)
    resolution_dropdown.pack(pady=20)
    
    root.mainloop()
    #plot_mandelbrot_gpu(max_iter=10000, width=1024, height=1024)