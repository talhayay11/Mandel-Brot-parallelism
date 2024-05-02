from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

@jit(nopython=True)
def mandelbrot(c, z0, max_iter):
    z = z0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return max_iter

@jit(nopython=True)
def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, c, max_iter):
    real = np.linspace(xmin, xmax, width)
    imag = np.linspace(ymin, ymax, height)
    result = np.empty((width, height), dtype=np.int32)
    for i in range(width):
        for j in range(height):
            result[i, j] = mandelbrot(c, real[i] + 1j * imag[j], max_iter)
    return result

def plot_mandelbrot(ax, c_value, max_iter, resolution=(800, 800)):
    ax.clear()
    width, height = resolution
    result = mandelbrot_set(-2.0, 2.0, -2.0, 2.0, width, height, c_value, max_iter)
    ax.imshow(result.T, extent=[-2.0, 2.0, -2.0, 2.0], origin='lower', cmap='hot')
    ax.set_title(f"Mandelbrot Set for c = {c_value}")
    plt.draw()

def on_move(event):
    if event.button == 1 and event.inaxes == ax:
        x, y = event.xdata, event.ydata
        real_range = np.linspace(-2.0, 2.0, 200)  # Reduced resolution
        imag_range = np.linspace(-2.0, 2.0, 200)  # Reduced resolution
        real_part = real_range[int((x + 2.0) / 4.0 * 200)]
        imag_part = imag_range[int((y + 2.0) / 4.0 * 200)]
        c_value = complex(real_part, imag_part)
        plot_mandelbrot(ax, c_value, 100, resolution=(200, 200))

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
txt_box = widgets.TextBox(plt.axes([0.1, 0.05, 0.7, 0.075]), 'Enter c:', initial="0.285+0.01j")
plot_mandelbrot(ax, complex(txt_box.text), 100)

fig.canvas.mpl_connect('button_press_event', on_move)
fig.canvas.mpl_connect('motion_notify_event', on_move)

plt.show()
