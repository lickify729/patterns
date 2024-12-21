"""
Given a set of points in the format (x, y)
This program outputs relevant statistical information:
Standard deviation, R value, and RMSE
This program also plots 2 graphs:
-One with the set of points, regression line, and square
-The other is the residual plot 

Micah Mok
11/15/2024
"""

import numpy as np
import matplotlib.pyplot as plt

# Function to collect points from the user input
# 3 ways to get input: manually, paste multiple points, or by file
def getPoints():
    print("Choose an input method:")
    print("1. Enter points manually")
    print("2. Paste multiple points (ex: (1,2) (3,4) (5,6))")
    print("3. Import points from a file (each point in '(x,y)' format on a new line)")
    choice = input("Enter your choice (1, 2, or 3): ").strip()

    points = []

    if choice == "1":
        print("Enter points using the (x,y) format (ex: (4,6)). Type 'done' when done.")
        while True:
            user_input = input("Enter a point: ").strip()
            if user_input.lower() == "done":
                break
            if user_input.startswith("(") and user_input.endswith(")"):
                try:
                    x, y = map(float, user_input[1:-1].split(","))
                    points.append((x, y))
                except ValueError:
                    print("Invalid format. Please use the (x,y) format.")

    elif choice == "2":
        user_input = input("Paste your points (ex:(1,2) (3,4) (5,6))").strip()
        try:
            points = [
                tuple(map(float, point[1:-1].split(",")))
                for point in user_input.split()
                if point.startswith("(") and point.endswith(")")
            ]
        except ValueError:
            print("Invalid format. Please use the (x,y) (x2,y2) (x3,y3) format.")
            return getPoints()

    elif choice == "3":
        print("Ensure your file has one point per line: (x,y)")
        print("To find the file path: Right-click the file, select Properties/Get Info, and copy the full path.")
        file_path = input("Enter the path to the file: ").strip()
        try:
            with open(file_path, "r") as file:
                for line in file:
                    line = line.strip()
                    if line.startswith("(") and line.endswith(")"):
                        try:
                            x, y = map(float, line[1:-1].split(","))
                            points.append((x, y))
                        except ValueError:
                            print(f"Invalid format in the file in line: {line}")
        except FileNotFoundError:
            print(f"The file for all the points is not found: {file_path}")
            return getPoints()

    return points

# Function to calculate regression line 
def regressionLine(points):
    x, y = zip(*points)
    x, y = np.array(x), np.array(y)
    m = np.cov(x, y, bias=True)[0, 1] / np.var(x)
    b = y.mean() - m * x.mean()
    return m, b

# Function to calculate R-value
def calculateRvalue(points):
    x, y = zip(*points)
    return np.corrcoef(x, y)[0, 1]

# Function to calculate RMSE (Root Mean Square Error)
def calculateRMSE(points, m, b):
    x, y = zip(*points)
    y_pred = m * np.array(x) + b
    return np.sqrt(np.mean((np.array(y) - y_pred) ** 2))

# Function to plot the regression line, the data points, and squares
def plotRegression(points, m, b, title, xlabel, ylabel, r_value, rmse):
    x, y = zip(*points)
    x, y = np.array(x), np.array(y)
    y_pred = m * x + b
    square_areas = []

    # plots points
    plt.scatter(x, y, color='blue', label='Data points')
    sign = '-' if b < 0 else '+'
    plt.plot(x, y_pred, color='red', label=f'Regression line: y = {m:.2f}x {sign} {abs(b):.2f}')

    # calculate residuals and plots squares
    for xi, yi, y_hat in zip(x, y, y_pred):
        residual = yi - y_hat
        square_side = abs(residual)
        square_areas.append(square_side ** 2)

        # plots squares so that only one corner of the square intersects with the regression line
        # using the vertical line from (xi,yi) to (xi,y_hat) as a side of the square
        if residual > 0:
            corner_x, corner_y = xi, y_hat
            square_x = [corner_x, corner_x, corner_x - square_side, corner_x - square_side, corner_x]
            square_y = [corner_y, corner_y + square_side, corner_y + square_side, corner_y, corner_y]
        else:
            corner_x, corner_y = xi, y_hat
            square_x = [corner_x, corner_x, corner_x + square_side, corner_x + square_side, corner_x]
            square_y = [corner_y, corner_y - square_side, corner_y - square_side, corner_y, corner_y]

        plt.plot(square_x, square_y, color='green')

    # sum of square areas
    sum_square_areas = sum(square_areas)
    plt.text(0.05, 0.85, f"Sum of Square Areas = {sum_square_areas:.2f}",
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))
    # labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.text(0.05, 0.95, f"R-value: {r_value:.2f}\nRMSE: {rmse:.2f}",
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))
    plt.legend()
    plt.grid(True)
    plt.show()

# Function plots residuals
def plotResiduals(points, m, b):
    x, y = zip(*points)
    x, y = np.array(x), np.array(y)
    y_pred = m * x + b
    residuals = y - y_pred

    plt.scatter(x, residuals, color='purple', label='Residuals')
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Residual Plot (Linear Regression)')
    plt.xlabel('X values')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.legend()
    plt.show()

# runs program - getting data from user
def main():
    points = getPoints()

    if len(points) < 2:
        print("At least two points are required to calculate a regression line.")
        return

    m, b = regressionLine(points)
    r = calculateRvalue(points)
    rmse = calculateRMSE(points, m, b)

    print(f"Regression line: y = {m:.2f}x {('+' if b >= 0 else '-')}{abs(b):.2f}")
    print(f"R-value: {r:.2f}")
    print(f"RMSE: {rmse:.2f}")

    title = input("Enter the title of the graph: ").strip()
    xlabel = input("Enter the label for the x-axis: ").strip()
    ylabel = input("Enter the label for the y-axis: ").strip()

    plotRegression(points, m, b, title, xlabel, ylabel, r, rmse)
    plotResiduals(points, m, b)

# Run the program
if __name__ == "__main__":
    main()

