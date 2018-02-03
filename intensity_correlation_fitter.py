# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:38:00 2017

@author: Richard Thurston

The goal of this program is to load, fit, and plot intensity auto correlation
data to a gausian pulse
"""
from sys import argv
from os import path

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize 
from scipy import constants
import pandas as pd    

def read_input_dat_file(file_name, daq_channel):
    """
    uses pandas to give the time delays, mean, and std dev of cross correlation
    data collected given an input time series like file and the channel used
    to make the measurement
    """
    data = pd.read_csv(file_name, sep="\t", header=None).T
    grouped = data.groupby(0)
    scale = 2*1000/(constants.c*constants.femto/constants.micro)
    print("scale = {0} fs/mm".format(scale))
    x = np.array(grouped.mean().index) # conversion from mm to fs delay
    x = (x - np.mean(x))*scale 
    y = np.array(grouped.mean().values[:,daq_channel])
    std_dev = np.array(grouped.std().values[:, daq_channel])
    return x, y, std_dev

def auto_correlation():
    return lambda x, a0, b0, c0, d: gausian_cross_correlation(x, a0, b0, c0, a0, b0, c0, d)

def gausian_cross_correlation(x, a0, b0, c0, a1, b1, c1, d):
    return (a0**2)*(a1**2)*np.exp(-((-b0 + b1 + x)**2)/(2*(c0**2 + c1**2)))*np.sqrt(2*np.pi/(c0**2 + c1**2)) + d

def gausian(x, a, b, c, d):
    return a*np.exp(-((b - x)**2)/(2*c**2)) + d

def cross_correlation(a, b, c):
    return lambda x, a0, b0, c0, d: gausian_cross_correlation(x, a0, b0, c0, a, b, c, d)

def main():
    if len(argv) == 4: #auto correlation case
        input_file = argv[1]
        output_dir = argv[2]
        channel = int(argv[3])
        fitting_fn = auto_correlation()
        fitting_type = "_auto_correlation"
    elif len(argv) == 7: #cross correlation case
        input_file = argv[1]
        output_dir = argv[2]
        channel = int(argv[3])
        fitting_fn = cross_correlation(float(argv[4]), float(argv[5]), float(argv[6]))
        fitting_type = "_cross_correlation"
    else:
        msg = ("Incorrect number of arguments. Please give the input file name, "
               "output file dirrectory, and daq channel number to fit the data "
               "to an auto correlation or additionally give a, b, and c fit "
               "parameters for a guassian pulse to fit a cross correlation\n"
               "Format: input_file_name output_file_dir channel_number a b c")
        raise AssertionError(msg)

    #fit the data
    guess = lambda x, y:[np.max(y) - np.min(y), np.mean(x), (np.max(x) - np.min(x))/2, np.min(y)]
    x_vals, y_vals, yerror = read_input_dat_file(input_file, channel)
    guess0 = guess(x_vals, y_vals)
    print(guess0)
    popt_cc, pcov_cc = optimize.curve_fit(fitting_fn, x_vals, y_vals, p0=guess0, sigma=yerror,
                                       bounds=((0, -np.inf, 0, -np.inf), (np.inf, np.inf, np.inf, np.inf)))
    popt_g, pcov_g = optimize.curve_fit(gausian, x_vals, y_vals, p0=guess0, sigma=yerror,
                                       bounds=((0, -np.inf, 0, -np.inf), (np.inf, np.inf, np.inf, np.inf)))
    # Make the output message
    fit_params = "a = {0} \nb = {1} \nc = {2} \noffset = {3}"
    fit_params_msg = ("Intensity Correlation Parameters\n" + fit_params.format(*popt_cc) +
                      "\nFWHM = {0}".format(2*np.sqrt(2*np.log(2))*popt_cc[2]) +
                      "\nGausian Parameters\n" + fit_params.format(*popt_g) +
                      "\nFWHM = {0}".format(2*np.sqrt(2*np.log(2))*popt_g[2]))
    # Save the dataset to a tab deliminated csv type file
    df = pd.DataFrame({"Time Delay(fs)":x_vals,
                       "Intensity Correlation Fit": fitting_fn(x_vals, *popt_cc),
                       "Gausian Fit": gausian(x_vals, *popt_g),
                       "Signal": y_vals,
                       "Std Dev":yerror})
    name = path.splitext(path.basename(input_file))[0] + fitting_type + "_data.tsv"
    df.to_csv(path.join(output_dir, name), sep='\t')
    # Save the fit parameters to a text file and print them to the terminal
    print(fit_params_msg)
    name = path.splitext(path.basename(input_file))[0] + fitting_type + "_optimized_parameters.txt"
    with open(path.join(output_dir, name), "w") as f:
        f.write(fit_params_msg)
    # Plot the Intensity Correlation data and the fit of the data
    plt.errorbar(x_vals, y_vals, yerror)
    plt.plot(x_vals, fitting_fn(x_vals, *popt_cc))
    name = path.splitext(path.basename(input_file))[0] + fitting_type + "_plot.jpg"
    plt.savefig(path.join(output_dir, name))
    plt.close()
    
if __name__ == "__main__":
    main()
