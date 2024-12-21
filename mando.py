import numpy as np
import matplotlib.pyplot as plt

def mandelbrot_set(width, height, bound_value, max_iterations, n=2):
    real = np.linspace(-2, 2, width)
    imag = np.linspace(-2, 2, height)
    complex_array = real[np.newaxis, :] + imag[:, np.newaxis] * 1j
    iteration_array = np.zeros((height, width), dtype=int)
    for i in range(height):
        for j in range(width):
            c = complex_array[i, j]
            z = 0
            iteration = 0
            while abs(z) < bound_value and iteration <= max_iterations:
                z = z**n + c
                iteration += 1
            if iteration > max_iterations:
                iteration_array[i, j] = 0
            else:
                iteration_array[i, j] = iteration
    return iteration_array

width, height = 800, 800
bound_value = 2
max_iterations = 100
n = 50
mandelbrot_image = mandelbrot_set(width, height, bound_value, max_iterations, n)
plt.figure(figsize=(10, 10))
plt.imshow(mandelbrot_image, extent=(-20, 20, -20, 20), cmap='hot')
plt.colorbar(label='Iteration Count')
plt.title(f'Mandelbrot Set (n={n}, Bound={bound_value}, Max Iterations={max_iterations})')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()
