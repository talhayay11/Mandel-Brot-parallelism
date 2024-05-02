import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from numba import cuda
import time
from concurrent.futures import ProcessPoolExecutor
import os
import tkinter as tk
from tkinter import ttk
from mandelbrot_core import compute_mandelbrot_region
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

Efficiency = []
Speedup = []
runTime = []

class ZoomableMandelbrot:
    def __init__(self, master,max_iter, regions, processors, width=800, height=800, ComputeOnce=True):
        self.master = master
        self.max_iter = max_iter
        self.processors = processors
        self.width = width
        self.height = height
        self.ComputeOnce = ComputeOnce



        self.figure = Figure(figsize=(5, 5))
        
        if self.ComputeOnce:
            self.ax = self.figure.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
            self.canvas.draw()
            self.canvas_widget = self.canvas.get_tk_widget()
            self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        if processors == 1:
            self.regions = 1
        elif regions == "auto":
            self.regions = processors * 2
        else:
            self.regions = 24
        
        self.press = None
        self.xmin, self.xmax = -2.0, 2.0
        self.ymin, self.ymax = -2.0, 2.0

        
        if self.ComputeOnce:

            self.rect = None
            self.start_x = None
            self.start_y = None


            self.canvas_widget.bind("<ButtonPress-1>", self.on_press)
            self.canvas_widget.bind("<B1-Motion>", self.on_motion)
            self.canvas_widget.bind("<ButtonRelease-1>", self.on_release)


        # if self.ComputeOnce:
        #     self.ax.add_patch(self.rect)
        #self.ax.add_patch(self.rect)
        self.plot_mandelbrot(self.regions, self.processors)

    def on_press(self, event):
        print("Pressed", event.x, event.y)  # Koordinatları kontrol et
        self.press = (event.x, event.y)
        self.start_x = event.x
        self.start_y = event.y
        if self.rect:
            self.canvas_widget.delete(self.rect)
        # Yeni bir dikdörtgen oluştur
        self.rect = self.canvas_widget.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline='red')


    def on_motion(self, event):
        if self.press is None:
            return
        xpress, ypress = self.press
        dx = event.x - xpress
        dy = event.y - ypress
        if self.rect:
            self.canvas_widget.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

        self.canvas.draw()

    def on_release(self, event):
        if self.press is None:
            return
        
        xpress, ypress = self.press
        xrelease, yrelease = event.x, event.y
        self.press = None

        # Piksel koordinatlarını data koordinatlarına dönüştürme
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()

        dx = xmax - xmin
        dy = ymax - ymin

        # Piksel'den data koordinatlarına çevirme
        data_xpress = xmin + (xpress / self.width) * dx
        data_ypress = ymin + ((self.height - ypress) / self.height) * dy
        data_xrelease = xmin + (xrelease / self.width) * dx
        data_yrelease = ymin + ((self.height - yrelease) / self.height) * dy

        # Hesaplanan sınırlar
        xmin = min(data_xpress, data_xrelease)
        xmax = max(data_xpress, data_xrelease)
        ymin = min(data_ypress, data_yrelease)
        ymax = max(data_ypress, data_yrelease)

        # Seçilen bölgenin merkezi
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0

        # Yeni sınırların hesaplanması (%50 zoom)
        new_width = (xmax - xmin) / 4.0  # %50 genişliğinin yarısı
        new_height = (ymax - ymin) / 4.0  # %50 yüksekliğinin yarısı

        # Yeni sınırlar
        new_xmin = center_x - new_width
        new_xmax = center_x + new_width
        new_ymin = center_y - new_height
        new_ymax = center_y + new_height

        if self.rect:
            self.canvas_widget.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

        self.canvas_widget.delete(self.rect)


        # Mandelbrot setinin yeni sınırlarla çizilmesi
        self.plot_mandelbrot(self.regions, self.processors, new_xmin, new_xmax, new_ymin, new_ymax)




    def plot_mandelbrot(self,  regions, processors,  xmin=None, xmax=None, ymin=None, ymax=None):
        if xmin is None:
                xmin, xmax = self.xmin, self.xmax
        if ymin is None:
            ymin, ymax = self.ymin, self.ymax    

        print(f"Running on {self.regions} regions and {self.max_iter} iterations.")

            # Calculate width and height of each region
        region_width = int(self.width // self.regions)
        region_height = int(self.height // self.regions)

            # Adjust the region boundaries to ensure integer number of pixels per region
        x_step = (xmax - xmin) / self.width
        y_step = (ymax - ymin) / self.height
        x_steps = np.linspace(xmin, xmax, self.regions + 1)
        y_steps = np.linspace(ymin, ymax, self.regions + 1)

        regions = []
        for i in range(self.regions):
                for j in range(self.regions):
                    x_min = x_steps[i]
                    x_max = x_steps[i + 1]
                    y_min = y_steps[j]
                    y_max = y_steps[j + 1]
                    regions.append((f"Region-{i}-{j}", x_min, x_max, y_min, y_max, region_width, region_height, self.max_iter))

        start_time = time.time()
        with ProcessPoolExecutor(max_workers=processors) as executor:
            results = list(executor.map(compute_mandelbrot_region, regions))

        final_result = np.zeros((self.height, self.width))
        for j in range(self.regions):
            for i in range(self.regions):
                idx = j * self.regions + i
                result = results[idx]
                final_result[j * region_height:(j + 1) * region_height, i * region_width:(i + 1) * region_width] = result

        elapsed_time = time.time() - start_time
        runTime.append(elapsed_time)
        print(f"Number of processors: {processors} Total computation time: {elapsed_time:.2f} seconds")


        # if self.ComputeOnce==False:
        #     start_time_sequential = time.time()
        #     compute_mandelbrot_region(("Sequential", self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter))
        #     elapsed_time_sequential = time.time() - start_time_sequential

        #     # Compute the speedup
        #     speedup = elapsed_time_sequential / elapsed_time
        #     Speedup.append(speedup)

        #     # Compute the efficiency
        
        #     efficiency = speedup / processors
        #     Efficiency.append(efficiency)

        #     #print(f"Speedup: {speedup:.2f}")
        #     #print(f"Efficiency: {efficiency:.2f}")

        if self.ComputeOnce:
            self.ax.clear()
            self.ax.imshow(final_result.T, extent=[self.xmin, self.xmax, self.ymin, self.ymax], origin='lower', cmap='hot')
            self.ax.set_title(f"Process count: {processors} Computation time: {elapsed_time:.2f} seconds")
            self.ax.axis("off")
            self.canvas.draw_idle()
        


    