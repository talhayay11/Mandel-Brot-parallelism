from tkinter import messagebox
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from numba import cuda
import time
from concurrent.futures import ProcessPoolExecutor
import os
import tkinter as tk

from mandelbrot_plotting import  ZoomableMandelbrot, Efficiency, Speedup
#from mandelbrot_gpu import plot_mandelbrot_gpu
from mandelbrot_core import processCount

maximumPhysicalCores = os.cpu_count() // 2

minToMaxProcessors = list(range(1,maximumPhysicalCores+1))


# if __name__ == '__main__':
    
#     plot_mandelbrot_gpu(max_iter=10000, width=1024, height=1024)
#     plot_zoomable_mandelbrot(1000)
resolution = "800x800"  # Initialize the variable in the global scope
max_iter_value = 100  # Initialize the variable in the global scope
def on_resolution_selected(selected_resolution):
    global resolution
    resolution = selected_resolution
    messagebox.showwarning("Selected Resolution", f"Selected resolution: {resolution}")

def set_max_iter():
    global max_iter_value
    max_iter_value = max_iter_entry.get()
    try:
        max_iter_value = int(max_iter_value)
        messagebox.showinfo("Max Iteration Set", f"Max Iteration set to: {max_iter_value}")
        # Call your function to set the max_iter value here
    except ValueError:
        messagebox.showwarning("Invalid Input", "Error: Please enter a valid integer for Max Iteration")

def show_image(width, height, max_iter, regions, processors, compute_once):
    # Create a new Tkinter window for displaying the Mandelbrot image
    image_window = tk.Tk()
    image_window.title("Mandelbrot Set")

    # Compute Mandelbrot set
    mandelbrot = ZoomableMandelbrot(image_window, max_iter=max_iter_value, regions=regions,
                                    processors=processors, width=width, height=height, ComputeOnce=compute_once)

def ComputeAll():
    width, height = map(int, resolution.split('x'))
    messagebox.showinfo("Compute All", f"Testing for {minToMaxProcessors} processors")
    for processors in minToMaxProcessors:
        ZoomableMandelbrot(root, max_iter=max_iter_value, regions="auto", processors=processors, width=width, height=height, ComputeOnce=False)
    
    messagebox.showinfo("Efficiency", f"Efficiency: {Efficiency}\nAverage Efficiency: {np.mean(Efficiency)}")
    messagebox.showinfo("Speedup", f"Speedup: {Speedup}\nAverage Speedup: {np.mean(Speedup)}")
    Efficiency.clear()
    Speedup.clear()

def ComputeOnce():
    width, height = map(int, resolution.split('x'))
    show_image(width, height, max_iter=100, regions="auto", processors=maximumPhysicalCores, compute_once=True)

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Mandelbrot Set Viewer")
    
    resolutions = ["800x800", "1024x1024", "1280x720", "1920x1080"]
    selected_resolution = tk.StringVar(root)
    selected_resolution.set(resolutions[0])  # Default resolution

    button_compute_once = tk.Button(root, text="Tüm Çekirdeklerle Hesapla", command=ComputeOnce, width=20)
    button_compute_once.pack(side="left", padx=30, pady=20)  # Add horizontal padding between buttons

    button_compute_all = tk.Button(root, text="Sırayla Tüm Çekirdekler", command=ComputeAll, width=20)
    button_compute_all.pack(side="right", padx=20, pady=20)  # Add horizontal padding between buttons

    # Dropdown menu for selecting resolution
    resolution_dropdown = tk.OptionMenu(root, selected_resolution, *resolutions, command=on_resolution_selected)
    resolution_dropdown.pack(pady=20)

    # Entry field for setting max_iter
    max_iter_entry_label = tk.Label(root, text="Max Iteration:")
    max_iter_entry_label.pack()
    max_iter_entry = tk.Entry(root, width=10)
    max_iter_entry.pack()
    set_max_iter_button = tk.Button(root, text="Set Max Iteration", command=set_max_iter)
    set_max_iter_button.pack(pady=10)

    root.mainloop()
    #plot_mandelbrot_gpu(max_iter=10000, width=1024, height=1024)