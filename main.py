from tkinter import messagebox
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from numba import cuda
import time
from concurrent.futures import ProcessPoolExecutor
import os
import tkinter as tk

from mandelbrot_plotting import  ZoomableMandelbrot, Efficiency, Speedup, runTime
from mandelbrot_gpu import plot_mandelbrot_gpu
from mandelbrot_core import processCount

maximumPhysicalCores = os.cpu_count() // 2

minToMaxProcessors = list(range(1,maximumPhysicalCores+1))

singleCoreTimeForSizes = []
multiCoreTimeForSizes = []
speedupForSizes = []
efficiencyForSizes = []


# if __name__ == '__main__':
    
#     plot_mandelbrot_gpu(max_iter=10000, width=1024, height=1024)
#     plot_zoomable_mandelbrot(1000)
resolution = "800x800"  # Initialize the variable in the global scope
max_iter_value = 100  # Initialize the variable in the global scope

def butona_tiklandi():
    plot_mandelbrot_gpu(10000, 1024,1024,1)

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
    runTime.clear()
    width, height = map(int, resolution.split('x'))
    messagebox.showinfo("Compute All", f"Testing for {minToMaxProcessors} processors")
    for processors in minToMaxProcessors:
        ZoomableMandelbrot(root, max_iter=max_iter_value, regions="auto", processors=processors, width=width, height=height, ComputeOnce=False)

    print(runTime)

    Speedup = [runTime[0] / time for time in runTime]
    print(Speedup)

    Efficiency = [100 * speedup / processor for speedup, processor in zip(Speedup, minToMaxProcessors)]
    print(Efficiency)

    plt.close()

    plt.plot(minToMaxProcessors, Efficiency, marker='o', linestyle='-', color='tab:blue', label="Efficiency")
    plt.xlabel("Number of Processors")
    plt.ylabel("Efficiency (%)", color='tab:blue')
    plt.tick_params(axis='y', labelcolor='tab:blue')

    # Create a secondary y-axis for Speedup on the right side
    ax2 = plt.gca().twinx()
    ax2.plot(minToMaxProcessors, Speedup, marker='s', linestyle='--', color='tab:orange', label="Speedup")
    ax2.set_ylabel("Speedup", color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title('Efficiency and Speedup vs. Number of Processors')
    plt.grid(True)
    plt.show()
    
    Efficiency.clear()
    Speedup.clear()
    runTime.clear()

def ComputeOnce():
    fig, ax = plt.subplots(figsize=(10, 10))
    width, height = map(int, resolution.split('x'))
    ZoomableMandelbrot(ax, max_iter=max_iter_value, regions="auto",processors=maximumPhysicalCores, width=width, height=height, ComputeOnce=True)
    
    plt.show()

def SizeBenchmark():
    singleCoreTimeForSizes = []
    multiCoreTimeForSizes = []

    # Calculate runtime for single core and multi-core for each resolution
    for resolution in resolutions:
        width, height = map(int, resolution.split('x'))
        
        # Single core computation
        runTime.clear()
        ZoomableMandelbrot(root, max_iter=max_iter_value, regions="auto", processors=1, width=width, height=height, ComputeOnce=False)
        singleCoreTimeForSizes.append(runTime[0])  # Append actual runtime to singleCoreTimeForSizes

        # Multi-core computation
        runTime.clear()
        ZoomableMandelbrot(root, max_iter=max_iter_value, regions="auto", processors=maximumPhysicalCores, width=width, height=height, ComputeOnce=False)
        multiCoreTimeForSizes.append(runTime[0])  # Append actual runtime to multiCoreTimeForSizes

    # Calculate Speedup and Efficiency for each resolution
    speedupForSizes = [singleCoreTimeForSizes[i] / multiCoreTimeForSizes[i] for i in range(len(resolutions))]
    efficiencyForSizes = [100 * speedup / maximumPhysicalCores for speedup in speedupForSizes]

    print("Single-Core Time:", singleCoreTimeForSizes)
    print("Multi-Core Time:", multiCoreTimeForSizes)
    print("Speedup:", speedupForSizes)
    print("Efficiency:", efficiencyForSizes)

    # Plotting Speedup and Efficiency
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Speedup on the left y-axis
    ax1.plot(resolutions, speedupForSizes, marker='s', linestyle='--', color='tab:blue', label='Speedup')
    ax1.set_ylabel('Speedup', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create a secondary y-axis for Efficiency on the right side
    ax2 = ax1.twinx()
    ax2.set_ylim(0, 50)
    ax2.plot(resolutions, efficiencyForSizes, marker='o', linestyle='-', color='tab:orange', label='Efficiency')
    ax2.set_ylabel('Efficiency (%)', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    ax1.set_xlabel('Resolution')
    ax1.set_xticks(resolutions)
    ax1.set_xticklabels(resolutions, rotation=45, ha='right')
    ax1.set_title(f'Speedup and Efficiency vs. Resolution at {maximumPhysicalCores} Processors and {max_iter_value} Iterations')

    # Set legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plotting Single-Core and Multi-Core Runtime
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot Single-Core and Multi-Core Runtime
    ax.plot(resolutions, singleCoreTimeForSizes, marker='v', linestyle='-', color='tab:red', label='Single-Core Runtime')
    ax.plot(resolutions, multiCoreTimeForSizes, marker='^', linestyle='-', color='tab:green', label='Multi-Core Runtime')

    ax.set_xlabel('Resolution')
    ax.set_ylabel('Time (s)')
    ax.set_xticks(resolutions)
    ax.set_xticklabels(resolutions, rotation=45, ha='right')
    ax.set_title(f'Single Processor and {maximumPhysicalCores} Processors Runtime vs. Resolution at {max_iter_value} Iterations')

    ax.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Mandelbrot Set Viewer")
    
    resolutions = [ "400x400", "600x600" ,"800x800", "1024x1024", "1200x1200"]
    selected_resolution = tk.StringVar(root)
    selected_resolution.set(resolutions[0])  # Default resolution

    SizeBenchmark()

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



    buton = tk.Button(root, text="GPU ILE CALISTIR", command=butona_tiklandi)
    buton.pack()

    root.mainloop()
    #plot_mandelbrot_gpu(max_iter=10000, width=1024, height=1024)