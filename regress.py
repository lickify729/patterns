import numpy as np
import matplotlib.pyplot as plt

# Function to collect points from the user input
def getPoints():
    print("Enter points using the (x,y) format (e.g., (4, 6) or (3,5)). Type 'done' when done entering points.")
    points = []
    while True:
        user_input = input("Enter a point: ").strip()
        if user_input.lower() == "done":
            break
        if user_input.startswith("(") and user_input.endswith(")"):
            try:
                x, y = map(float, user_input[1:-1].split(","))
                points.append((x, y))
            except ValueError:
                print("Invalid format. Please enter the point in the format (x,y).")
        else:
            print("Invalid format. Please enter the point in the format (x,y).")
    return points

# Function to calculate regression line 
def regressionLine(points):
    x_values, y_values = zip(*points)
    x = np.array(x_values)
    y = np.array(y_values)
    m = np.cov(x, y, bias=True)[0, 1] / np.var(x)
    b = y.mean() - m * x.mean()
    return m, b

# Function to calculate R-value
def calculateRvalue(points):
    x_values, y_values = zip(*points)
    x = np.array(x_values)
    y = np.array(y_values)
    r = np.corrcoef(x, y)[0, 1]
    return r

# Function to calculate RMSE (Root Mean Square Error)
def calculateRMSE(points, m, b):
    x_values, y_values = zip(*points)
    x = np.array(x_values)
    y = np.array(y_values)
    y_pred = m * x + b
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    return rmse

# Function to plot the regression line, the data points, and squares
def plotRegression(points, m, b, title, xlabel, ylabel, r_value, rmse):
    x_values, y_values = zip(*points)
    x = np.array(x_values)
    y_pred = m * x + b
    square_areas = []  
    
    # plots points
    plt.scatter(x, y_values, color='blue', label='Data points')  
    # graph regression line
    plt.plot(x, y_pred, color='red', label=f'Regression line: y = {m:.2f}x + {b:.2f}')  
    
    for xi, yi, y_hat in zip(x, y_values, y_pred):
        plt.plot([xi, xi], [yi, y_hat], color='black', linestyle='--')
      
        # Calculate residual
        residual = yi - y_hat
        square_area = residual ** 2
        square_areas.append(square_area)
        square_side = abs(residual)
        
        if residual > 0:  
            square_x = [xi - square_side, xi - square_side, xi, xi, xi - square_side]
            square_y = [y_hat, y_hat + square_side, y_hat + square_side, y_hat, y_hat]
        else:  
            square_x = [xi, xi, xi + square_side, xi + square_side, xi]
            square_y = [y_hat, y_hat - square_side, y_hat - square_side, y_hat, y_hat]
        
        plt.plot(square_x, square_y, color='green') 
    
    # sum of squares
    sum_square_areas = sum(square_areas)
    plt.text(0.05, 0.85, f"Sum of Square Areas = {sum_square_areas:.2f}",
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    annotation_text = f"R-value: {r_value:.2f}\nRMSE: {rmse:.2f}"
    plt.text(0.05, 0.95, annotation_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.grid(True)
    plt.show()


# Function plots residuals
def plotResiduals(points, m, b):
    x_values, y_values = zip(*points)
    x = np.array(x_values)
    y_pred = m * x + b
    residuals = y_values - y_pred
    
    plt.scatter(x, residuals, color='purple', label='Residuals')
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Residual Plot (Linear Regression)')
    plt.xlabel('X values')
    plt.ylabel('Residuals')
    plt.grid(True)
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

    print(f"Regression line: y = {m:.2f}x + {b:.2f}")
    print("R-value: " + str(r))
    print("RMSE: " + str(rmse))

    title = input("Enter the title of the graph: ").strip()
    xlabel = input("Enter the label for the x-axis: ").strip()
    ylabel = input("Enter the label for the y-axis: ").strip()
    
    plotRegression(points, m, b, title, xlabel, ylabel, r, rmse)
    plotResiduals(points, m, b)

if __name__ == "__main__":
    main()
