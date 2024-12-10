import numpy as np
import math
import os
import matplotlib.pyplot as plt
# Stat 401
# 7.9 Case Study

def Sxx(x_bar, x):
    S_xx = 0
    for i in range(0, len(x)):
        S_xx = S_xx + (x[i] - x_bar) ** 2
    return S_xx

def Sxy(x_bar, y_bar, x, y):
    S_xy = 0
    for i in range(0, len(x)):
        S_xy = S_xy + ((x[i] - x_bar)*(y[i] - y_bar))
    return S_xy

def Beta_1(sxy, sxx):
    return sxy / sxx

def Beta_0(x,y, beta_1, n):
    return (np.sum(y) - (beta_1*np.sum(x))) / n

def Y_hat_e(x, beta_0, beta_1, y):
    e = np.array([])
    y_hat = np.array([])
    for i in range (0, len(x), 1):
        y_hat_i = beta_0 + (beta_1*x[i])
        y_hat = np.append(y_hat, y_hat_i)
        e_i = y[i] - y_hat[i]
        e = np.append(e, e_i)
    return y_hat, e




def main():
    x = np.array([9.5, 9.8, 8.3, 8.6, 7.0, 17.4, 15.2, 16.7, 15.0, 14.8, 25.6, 24.4, 19.5, 22.8, 19.8, 
                  8.4, 11.0, 9.9, 6.4, 8.2, 15.0, 16.4, 15.4, 14.5, 13.6, 23.4, 23.3, 21.2, 21.7, 21.3])
    y = np.array([14814.00, 14007.00, 7573.00, 9714.00, 5304.00, 43243.00, 28028.00, 49499.00, 26222.00, 26751.00, 96305.00, 72594.00, 32207.00, 70453.00, 38138.00, 
                  17502.00, 19443.00, 14191.00, 8076.00, 10728.00, 25319.00, 41792.00, 25312.00, 22148.00, 18036.00, 104170.00, 49512.00, 48218.00, 47661.00, 53045.00])
    y_hat = np.array([])
    e = np.array([])
    n = len(x)
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    print("x_bar: " + str(x_bar))
    print("y_bar: " + str(y_bar))


    sxx = Sxx(x_bar, x)
    syy = Sxx(y_bar, y)
    sxy = Sxy(x_bar, y_bar, x, y)

    print("Sxx: " + str(sxx))
    print("Syy: " + str(syy))
    print("Sxy: " + str(sxy))

    r = sxy / math.sqrt((sxx*syy))
    R_2 = r ** 2
    print ("R^2 = "+str(R_2))
    print("r = "+str(r))

    beta_1 = Beta_1(sxy, sxx)
    beta_0 = Beta_0(x,y, beta_1, n)
    y_hat, e = Y_hat_e(x, beta_0, beta_1, y, ) 

    print("Beta_0: "+str(beta_0))
    print("Beta_1: "+ str(beta_1))
    print("Y_hat: ")
    print(y_hat)
    print("Residuals: ")
    print(e)

    output_dir = "regression_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Scatter Plot with Regression Line
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color="blue", label="Data Points")
    plt.plot(x, y_hat, color="red", label="Regression Line")
    plt.title("Scatter Plot with Regression Line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    output_file_scatter_plot= f"{output_dir}/predicted_vs_actual.png"
    plt.savefig(output_file_scatter_plot)


    # Residual Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, e, color="green", label="Residuals")
    plt.axhline(0, color="red", linestyle="--", label="Zero Residual Line")
    plt.title("Residual Plot")
    plt.xlabel("x")
    plt.ylabel("Residuals")
    plt.legend()
    plt.grid()
    output_file_residuals = f"{output_dir}/residuals_vs_density.png"
    plt.savefig(output_file_residuals)


main()
