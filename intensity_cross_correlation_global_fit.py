# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 11:09:24 2018

@author: Richard Thurston

The goal of this script is to take a set of 3 autocorrelation datasets and do
a global parameter fit to reconstruct the gaussian intensity profiles of the
three pulses and to nicely format the results.
"""

import os
import sys
import yaml
import argparse

import matplotlib.pyplot as plt
import symfit
import numpy as np
from scipy import constants
import pandas as pd

def parse_yaml_config(config_path):
    """
    takes a yaml config file and parses it to give a list of dicts containing
    the channel and associated file.
    
    Yaml should have the following format:
    drive_probe:
      file: <drive_probe_cross_correlation_dataset_path>
      channel: <daq channel used>
    pump_drive:
      file: <pump_drive_cross_correlation_dataset_path>
      channel: <daq channel used>
    pump_probe:
      file: <pump_probe_cross_correlation_dataset_path>
      channel: <daq channel used>
    """
    header_keys = ["drive_probe", "pump_drive", "pump_probe"]
    
    with open(config_path, 'r') as stream:
        config_raw = yaml.load(stream)

    try:
        config_list = [config_raw[hkey] for hkey in header_keys]
    except KeyError as error:
        print("required header {0} is missing from {1}".format(error, config_path))
        sys.exit(-1)
    try:
        file_list = [entry["file"] for entry in config_list]
        channel_list = [entry["channel"] for entry in config_list]
    except KeyError as error:
        print("required {0} entry is missing from {1}".format(error, config_path))
        sys.exit(-1)

    return file_list, channel_list

def read_input_dat_file(file_name, daq_channel):
    """
    uses pandas to give the time delays, mean, and std dev of cross correlation
    data collected given an input time series like file and the channel used
    to make the measurement
    """
    print("reading {}".format(file_name))
    data = pd.read_csv(file_name, sep="\t", header=None).T
    grouped = data.groupby(0) # group entries by the motor possition in col 0
    x = np.array(grouped.mean().index) 
    y = np.array(grouped.mean().values[:,daq_channel])
    std_dev = np.array(grouped.std().values[:, daq_channel])
    return x, y, std_dev

def plot_data_and_fits(x_vals, y_vals, errors, y_fits, file_paths, output_dir):
    """
    This function plots a data set with its errors, overlays the fit of that
    dataset, and saves the plot to a specified output directory
    """
    print("plotting")
    for x, y, error, fit, file_path in zip(x_vals, y_vals, errors, y_fits, file_paths):
        plt.errorbar(x, y, error)
        plt.plot(x, fit)
        name, ext = os.path.splitext(os.path.basename(file_path))
        plt.savefig(os.path.join(output_dir, "global_fit_plot_{}.jpg".format(name)))
        plt.close()

def save_data_and_fits_to_tsv(x_vals, y_vals, errors, y_fits, file_paths, output_dir):
    """
    This function saves the dataset to a tab deliminated csv type file
    """
    print("plotting")
    for x, y, error, fit, file_path in zip(x_vals, y_vals, errors, y_fits, file_paths):
        df = pd.DataFrame({"Time Delay(fs)": x,
                       "Gaussian Cross Correlation Fit": fit,
                       "Signal": y,
                       "Std Dev": error})
        name = os.path.splitext(os.path.basename(file_path))[0] 
        df.to_csv(os.path.join(output_dir,"global_fit_data_{}.tsv".format(name)), sep='\t')
        
def global_fitting(x_vals, y_vals, errors, file_paths, output_dir, names):
    """
    This function makes a symfit model for an intensity cross correlation, fits
    the given xy datasets to that model, and returns the following:
        the raw fit_result sympy object
        the fit datasets over the same range as the original datasets
        a fit message to help interperate the results
    """
    #make the symfit model and fit the gaussian cross correlation
    #note in this model a, b, c, d equate to the scale, mean, stddev, and offset
    guess = lambda x, y:[np.max(y) - np.min(y), np.mean(x), (np.max(x) - np.min(x))/10, np.min(y)]
    x0, x1, x2, y0, y1, y2 = symfit.variables("x0, x1, x2, y0, y1, y2")
    a0, b0, c0, d0 = symfit.parameters("a0, b0, c0, d0")
    a1, b1, c1, d1 = symfit.parameters("a1, b1, c1, d1") 
    a2, b2, c2, d2 = symfit.parameters("a2, b2, c2, d2")

    guess_vals = [] 
    for x, y in zip(x_vals, y_vals):
        guess_vals += guess(x, y)

    for param, guess in zip([a0, b0, c0, d0, a1, b1, c1, d1, a2, b2, c2, d2], guess_vals):
        param.value = guess
        print("param_{}_guess = {}".format(param, guess))
    
    model = {y0: (a0**2)*(a1**2)*symfit.exp(-((-b0 + b1 + x0)**2)/(2*(c0**2 + c1**2)))*symfit.sqrt(2*symfit.pi/(c0**2 + c1**2)) + d0,
             y1: (a0**2)*(a2**2)*symfit.exp(-((-b0 + b2 + x1)**2)/(2*(c0**2 + c2**2)))*symfit.sqrt(2*symfit.pi/(c0**2 + c2**2)) + d1,
             y2: (a2**2)*(a1**2)*symfit.exp(-((-b2 + b1 + x2)**2)/(2*(c2**2 + c1**2)))*symfit.sqrt(2*symfit.pi/(c2**2 + c1**2)) + d2}
    fit = symfit.Fit(model, x0=x_vals[0], x1=x_vals[1], x2=x_vals[2], y0=y_vals[0], y1=y_vals[1], y2=y_vals[2],
                     constraints=[symfit.Ge(c0, 0), symfit.Ge(c1, 0), symfit.Ge(c2, 0), symfit.Ge(a0, 0), symfit.Ge(a1, 0), symfit.Ge(a2, 0)])
    fit_result = fit.execute()
    y_results = fit.model(x0=x_vals[0], x1=x_vals[1], x2=x_vals[2], **fit_result.params)  
    
    # Make the output message
    fit_params_msg = ("\n\n  FWHM {1} and {2} = {0} fs".format(2*np.sqrt(2*np.log(2))*fit_result.value(c0), names[0], names[1]) +
                      "\nstddev {1} and {2} = {0} fs".format(2*np.sqrt(2*np.log(2))*fit_result.stdev(c0), names[0], names[1]) +
                      "\n  FWHM {1} and {2} = {0} fs".format(2*np.sqrt(2*np.log(2))*fit_result.value(c1), names[0], names[2]) +
                      "\nstddev {1} and {2} = {0} fs".format(2*np.sqrt(2*np.log(2))*fit_result.stdev(c1), names[0], names[2]) +
                      "\n  FWHM {1} and {2} = {0} fs".format(2*np.sqrt(2*np.log(2))*fit_result.value(c2), names[2], names[1]) +
                      "\nstddev {1} and {2} = {0} fs".format(2*np.sqrt(2*np.log(2))*fit_result.stdev(c2), names[2], names[1]))
    
    return fit_result, y_results, fit_params_msg

def auto_correlation():
    return lambda x, a0, b0, c0, d: gausian_cross_correlation(x, a0, b0, c0, a0, b0, c0, d)

def gausian_cross_correlation(x, a0, b0, c0, a1, b1, c1, d):
    return (a0**2)*(a1**2)*np.exp(-((-b0 + b1 + x)**2)/(2*(c0**2 + c1**2)))*np.sqrt(2*np.pi/(c0**2 + c1**2)) + d

def gausian(x, a, b, c, d):
    return a*np.exp(-((b - x)**2)/(2*c**2)) + d

def cross_correlation(a, b, c):
    return lambda x, a0, b0, c0, d: gausian_cross_correlation(x, a0, b0, c0, a, b, c, d)

def main():
    program_text = ("""
    This program takes three input cross correlation data files
    specified in an input yaml style configuration file that contains
    the file paths and the nxdaq channels that were used to capture 
    the data and does a global parameter fit using gaussian intensity
    profiles. Note that the input config file should have the entries 
    for \"drive_probe\", \"pump_drive\", \"pump_probe\" where each 
    entry has values for \"file\" and \"channel\" which specify the
    file path and nxdaq channel number used to collect the data.
    """)
    #parse commandline arguments
    parser = argparse.ArgumentParser(description=program_text)
    parser.add_argument("config", type=str, help="The yaml file pointing to the datasets to be fit")
    parser.add_argument("output_dir", type=str, help="The output folder used to store output files")
    parser.add_argument("-g", "--global_fit", action="store_false", help="Only output results from the global fitting routine")
    parser.add_argument("-c", "--cross_fit", action="store_false", help="Only output results from the cross correlation fitting routine")
    args = parser.parse_args()
    
    #parse the yaml config fiile
    file_paths, file_channels = parse_yaml_config(args.config)
    
    #read input data files
    scale = 2*1000/(constants.c*constants.femto/constants.micro) # conversion factor from mm of delay to fs of delay
    x_vals, y_vals, errors, xmean = [], [], [], []
    for file, channel in zip(file_paths, file_channels):
        x, y, error = read_input_dat_file(file, channel)
        xmean.append(np.mean(x))
        x = (x - np.mean(x))*scale # conversion from mm to fs delay scale = 2/c
        x_vals.append(x)
        y_vals.append(y)
        errors.append(error)
    
    #global fit
    names = ["drive_probe", "pump_drive", "pump_probe"] #these are in this order from how config.yaml gets parsed
    fit_result, y_results, global_fit_params_msg =global_fitting(x_vals, y_vals, errors, file_paths, args.output_dir, names)
    
    # Save the dataset to a tab deliminated csv type file
    save_data_and_fits_to_tsv(x_vals, y_vals, errors, y_results, file_paths, args.output_dir)
    
    #make global fit output message
    msg_0 = "".join(["\n{} used as input for {}".format(path, name) for name, path in zip(names, file_paths)])
    msg_1 = ("\n\nxmean_{1} = {0} mm".format(xmean[0], names[0]) +
             "\nxmean_{1} = {0} mm".format(xmean[1], names[1]) +
             "\nxmean_{1} = {0} mm".format(xmean[2], names[2]) +
             "\n\nt0_{1} = {0} mm".format(x_vals[0][np.argmax(y_results[0])]/scale + xmean[0], names[0]) +
             "\nt0_{1} = {0} mm".format(x_vals[1][np.argmax(y_results[1])]/scale + xmean[1], names[1])+
             "\nt0_{1} = {0} mm".format(x_vals[2][np.argmax(y_results[2])]/scale + xmean[2], names[2]))
    global_fit_params_msg = msg_0 + global_fit_params_msg + msg_1
    
    # Save the fit parameters to a text file and print them to the terminal
    print(global_fit_params_msg)
    print(fit_result)
    with open(os.path.join(args.output_dir, "global_fit_optimized_parameters.txt"), "w") as f:
        f.write("note: in this model a, b, c, d equate to the\npulse height, mean, stddev, and vertical offset\n")
        f.write(global_fit_params_msg)
        f.write("\n\nconstants below are in units of voltage and fs of delay")
        f.write(fit_result.__str__())
        
    # Plot the Intensity Cross Correlation data and the fit of the data
    plot_data_and_fits(x_vals, y_vals, errors, y_results, file_paths, args.output_dir)
    
if __name__ == "__main__":
    main()
