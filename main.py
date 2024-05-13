from tkinter import messagebox
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from numba import cuda
import time
from concurrent.futures import ProcessPoolExecutor
import os
import tkinter as tk
import threading

from mandelbrot_plotting import  ZoomableMandelbrot, Efficiency, Speedup, runTime
from mandelbrot_gpu import plot_mandelbrot_gpu, Compute_Md_Gpu, block_size, compute_mandelbrot_gpuFORONE,all_time
from mandelbrot_core import processCount

maximumPhysicalCores = os.cpu_count() // 2

minToMaxProcessors = list(range(1,maximumPhysicalCores+1))

singleCoreTimeForSizes = []
multiCoreTimeForSizes = []
speedupForSizes = []
efficiencyForSizes = []

resolutions = [ "400x400", "600x600" ,"800x800", "1024x1024", "1200x1200"]


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

def ComputeOnceGPU():
    width, height = map(int, resolution.split('x'))
    plot_mandelbrot_gpu(max_iter_value,width,height,1)

def ComputeAllGPU():
    width, height = map(int, resolution.split('x'))
    runTimeGPU, speedUp_GPU, EfficiencyGPU = Compute_Md_Gpu(max_iter_value,width,height,1)

    plt.close()

    plt.plot(block_size, EfficiencyGPU, marker='o', linestyle='-', color='tab:blue', label="Efficiency")
    plt.xlabel("Number of GPU Blocks")
    plt.ylabel("Efficiency (%)", color='tab:blue')
    plt.tick_params(axis='y', labelcolor='tab:blue')

    # Create a secondary y-axis for Speedup on the right side
    ax2 = plt.gca().twinx()
    ax2.plot(block_size, speedUp_GPU, marker='s', linestyle='--', color='tab:orange', label="Speedup")
    ax2.set_ylabel("Speedup", color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title('Efficiency and Speedup vs. Number of GPU Blocks')
    plt.grid(True)
    plt.show()

def onSelectedComputeUnit(selected_compute_unit):
    print(selected_compute_unit)
    if selected_compute_unit == "GPU":
        button_compute_once.config(command=ComputeOnceGPU)
        button_compute_all.config(command=ComputeAllGPU)
        button_size_benchmark.config(command=SizeBenchmarkGPU)
    elif selected_compute_unit == "CPU":
        button_compute_once.config(command=ComputeOnce)
        button_compute_all.config(command=ComputeAll)
        button_size_benchmark.config(command=SizeBenchmark)

def SizeBenchmarkGPU():
    singleCoreTimeForSizes = []
    multiCoreTimeForSizes = []

    # Calculate runtime for single core and multi-core for each resolution
    for resolution in resolutions:
        width, height = map(int, resolution.split('x'))
        print(width, height)
        # Single core computation
        all_time.clear()
        compute_mandelbrot_gpuFORONE(-2.0, 1.0, -1.5, 1.5, width, height, max_iter_value, 1,1)
        singleCoreTimeForSizes.append(all_time[0])  # Append actual runtime to singleCoreTimeForSizes

        # Multi-core computation
        all_time.clear()
        compute_mandelbrot_gpuFORONE(-2.0, 1.0, -1.5, 1.5, width, height, max_iter_value, 1,32)
        multiCoreTimeForSizes.append(all_time[0])  # Append actual runtime to multiCoreTimeForSizes

    # Calculate Speedup and Efficiency for each resolution
    speedupForSizes = [singleCoreTimeForSizes[i] / multiCoreTimeForSizes[i] for i in range(len(resolutions))]
    efficiencyForSizes = [100 * speedup / 32**2 for speedup in speedupForSizes]

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
    ax1.set_title(f'Speedup and Efficiency vs. Resolution at {32} Block and {max_iter_value} Iterations')

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
    ax.set_title(f'Single Block and {32} Block Runtime vs. Resolution at {max_iter_value} Iterations')

    ax.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
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

    # add thread for each function

    def start_size_benchmarkGPU():
        thread_size_benchmarkGPU = threading.Thread(target=SizeBenchmarkGPU)
        thread_size_benchmarkGPU.start()

    def start_compute_all_gpu():
        thread_compute_all_gpu = threading.Thread(target=ComputeAllGPU)
        thread_compute_all_gpu.start()

    def start_benchmark():
        thread_Benchmark = threading.Thread(target=SizeBenchmark)
        thread_Benchmark.start() 

    def start_compute_once():
        thread_compute_once = threading.Thread(target=ComputeOnce)
        thread_compute_once.start()

    def start_compute_all():
        thread_compute_all = threading.Thread(target=ComputeAll)
        thread_compute_all.start()

    def start_compute_once_gpu():
        thread_compute_once_gpu = threading.Thread(target=ComputeOnceGPU)
        thread_compute_once_gpu.start()
    
    compute_units = ["GPU", "CPU"]
    selected_compute_unit = tk.StringVar(root)
    selected_compute_unit.set(compute_units[1])  # Default compute unit
    selected_resolution = tk.StringVar(root)
    selected_resolution.set(resolutions[0])  # Default resolution

    global button_compute_once, button_compute_all
    button_compute_once = tk.Button(root, text="Tüm Çekirdeklerle Hesapla", command=ComputeOnce, width=20)
    button_compute_once.pack(side="left", padx=30, pady=20)  # Add horizontal padding between buttons

    button_compute_all = tk.Button(root, text="Sırayla Tüm Çekirdekler", command=ComputeAll, width=20)
    button_compute_all.pack(side="right", padx=20, pady=20)  # Add horizontal padding between buttons

    compute_units_dropdown = tk.OptionMenu(root, selected_compute_unit, *compute_units, command=onSelectedComputeUnit)
    compute_units_dropdown.pack(pady=10)

    button_size_benchmark = tk.Button(root, text="Size Benchmark", command=SizeBenchmark, width=20)
    button_size_benchmark.pack(pady=10)
        
    # Dropdown menu for selecting resolution
    resolution_dropdown = tk.OptionMenu(root, selected_resolution, *resolutions, command=on_resolution_selected)
    resolution_dropdown.pack(pady=10)

    # Entry field for setting max_iter
    max_iter_entry_label = tk.Label(root, text="Max Iteration:")
    max_iter_entry_label.pack()
    max_iter_entry = tk.Entry(root, width=10)
    max_iter_entry.pack()
    set_max_iter_button = tk.Button(root, text="Set Max Iteration", command=set_max_iter)
    set_max_iter_button.pack(pady=10)


    
    # buton = tk.Button(root, text="GPU ILE CALISTIR", command=butona_tiklandi)
    # buton.pack()

    root.mainloop()
    #plot_mandelbrot_gpu(max_iter=10000, width=1024, height=1024)