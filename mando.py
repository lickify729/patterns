"""
Given graphical information, color maps, and a custom imaginary equation
this program generates fractals. 

Source: https://www.geeksforgeeks.org/mandelbrot-fractal-set-visualization-in-python/
12/15/24

"""
import numpy as np
import matplotlib.pyplot as plt

# fractal function
def mandelbrot_set(width, height, bound_value, max_iterations, equation):
    # sets up graph
    real = np.linspace(-2, 2, width)
    imag = np.linspace(-2, 2, height)
    complex_array = real[np.newaxis, :] + imag[:, np.newaxis] * 1j
    iteration_array = np.zeros((height, width), dtype=int)

    # implements geeksforgeeks' algorithm but extended to not just basic mandelbrot equation
    for i in range(height):
        for j in range(width):
            c = complex_array[i, j]
            z = 0
            iteration = 0
            while abs(z) < bound_value and iteration < max_iterations:
                try:
                    z = eval(equation)
                except Exception as e:
                    print(f"Error in equation: {e}")
                    return None
                iteration += 1
            iteration_array[i, j] = iteration if iteration < max_iterations else 0

    return iteration_array

# get user inputs
width = int(input("Enter the width of the image: "))
height = int(input("Enter the height of the image: "))
bound_value = float(input("Enter the boundary value (Ex: 2): "))
max_iterations = int(input("Enter the maximum number of iterations: "))

print("Enter a custom Mandelbrot equation using 'z' and 'c' (Ex: 'z**2 + c'): ")
equation = input("Equation: ")

# color maps
print("\nAvailable color maps options (choose one):")
colormap_options = plt.colormaps()
print(", ".join(colormap_options))  # Print all colormap options
colormap = input("\nColor Maps: ")

# make sure inputs are correct
if 'z' not in equation or 'c' not in equation:
    print("Equation must include both 'z' and 'c'.")
elif colormap not in colormap_options:
    print("Invalid colormap. Please choose from the list provided.")
# calls function
else:
    mandelbrot_image = mandelbrot_set(width, height, bound_value, max_iterations, equation)

    if mandelbrot_image is not None:
        # display
        plt.figure(figsize=(10, 10))
        plt.imshow(mandelbrot_image, extent=(-2, 2, -2, 2), cmap=colormap)
        plt.colorbar(label='Iteration Count')
        plt.title(f'Custom Mandelbrot Set (Equation: {equation}, Bound={bound_value}, Max Iterations={max_iterations})')
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.show()
