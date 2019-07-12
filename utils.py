import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, Matern
from sklearn.gaussian_process import GaussianProcessRegressor

# Modules for utility functions to handle simulation data

'''
Function to get list of dat files in the simulation directory

Input
    - sim_dir:  Directory to simulated .dat files

Output
    - file_list:    List of .dat files
'''
def get_file_list(sim_dir):

    # Get list of all files in simulation directory
    all_files = os.listdir(sim_dir)

    # Get list of all .dat files
    sim_files = [fname for fname in all_files if '.dat' in fname]

    return sim_files

'''
Function to read simulated .dat files and extract information

Input
    - sim_file: .dat file containing information about the supernova

Output
    - phot_df: pandas dataframe containing lightcurve information
    - sn_data: a dictionary containing information about the supernova
                * id:         simulation id number
                * redshift:   simulated redshift of the supernova
                * type:       simulated type of the supernova
'''

def read_files(sim_file):

    # Open file and read text
    with open(sim_file, 'r') as fp:

        # Get contents and split into information and photometry
        contents = fp.read().split('Observations')

        # Get unique id, type, and redshift
        sn_data = [item.split(':\t') for item in contents[0].split('\n')]
        
        sn_id = sn_data[0][1]
        sn_type = sn_data[1][1]
        sn_redshift = sn_data[6][1]

        data_dict = {'id':sn_id,
                     'type':sn_type,
                     'redshift':sn_redshift}

        # Extract photometry information
        photometry = contents[1].split('\n')
        
        # Extract columns and data
        cols = [str(hdr) for hdr in photometry[1].split(',')]  
        data = [val.split(',') for val in photometry[2:-1]]
        
        # Create pandas dataframe
        phot_df = pd.DataFrame(columns=cols)
        
        # Add data to dataframe
        for i, row in enumerate(data):
            
            row = [float(x) for x in row]
            phot_df.loc[i] = row
        
        return phot_df, data_dict

'''
Function to plot light curves as magnitude over time

Input
    - x: time
    - y: magnitude
    - y_err: magnitude error
    - _label: label for axis _

Output 
    - Plot a light curve
'''
def plotlc_mag(x, y, y_err, xlabel, ylabel):
    
    plt.errorbar(x, y, yerr=y_err, linestyle='None', marker='o', color='#143C87')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gca().invert_yaxis()
    plt.show()

        
'''
Function to extract features from light curves as detailed in Richards et al 2011

Input
    - phot_df:  dataframe containing photometry information

Output
    - lc_features:  a dictionary of extracted features
        * Details of features are in Table 5 of R11
'''

def get_r11_features(mag, magerr, time):

    # Determine flux using 10 ** (0.4 * mag)
    flux = 10 ** (-0.4 * mag)

    # Max magnitude
    peak_mag = min(mag)

    # Amplitude
    amplitude = (max(mag) - min(mag))

    # Fraction of points beyond 1 std from mean magnitude
    mean_mag = np.mean(mag)
    std_mag = np.std(mag)

    mean_diff = np.abs(mag - mean_mag)  # Calculate difference from mean
    m_beyond = mean_diff[mean_diff > std_mag]

    beyond_1std = len(m_beyond) / len(mag)

    # Mean variance
    mean_var = std_mag / mean_mag

    # Range of cumulative sum
    c_sum = (1/(len(mag)*std_mag)) * np.cumsum(mag - mean_mag)
    rcs = max(c_sum) - min(c_sum)

    # Max slope
    dmag = mag[1:] - mag[:-1]
    dt = time[1:] - time[:-1]

    slope = np.abs(dmag / dt)
    #print(slope)
    slope[np.abs(slope) == np.inf] = 0  # Convert infinite values to zero
    max_slope = max(slope)

    # Calculate Q3 - Q1
    q1 = np.percentile(mag, 25)
    q3 = np.percentile(mag, 75)
    quartile_range = q3 - q1

    # Median absolute deviation
    abs_dev = np.abs(mag - np.median(mag))
    med_abs_dev = np.median(abs_dev)

    # Fraction of points within amplitude/5 of median magnitude
    amp_fifth = amplitude / 5
    m_in_amp = mean_diff[mean_diff < amp_fifth]

    med_buffer_range = len(m_in_amp) / len(mag)

    # Largest absolure departure from median flux, divided by median flux
    median_flux = np.median(flux)
    flux_diff = np.abs(flux - median_flux)

    percent_amplitude = max(flux_diff) / median_flux

    # Kurtosis of flxues
    kurtosis = stats.kurtosis(flux)

    # Skew of fluxes
    skew = stats.skew(flux)

    flux_95 = np.percentile(flux, 95.0) # Calculate 95th percentile for fluxes
    flux_5 = np.percentile(flux, 5.0)   # Calculate 5th percentile for fluxes

    flux_mid90 = flux_95 - flux_5  # Calculate mid 90% of fluxes

    # Flux percentile mid 20% ratio
    flux_60 = np.percentile(flux, 60.0)
    flux_40 = np.percentile(flux, 40.0)
    flux_mid20 = flux_60 - flux_40
    flux_percentile_ratio_mid20 = flux_mid20 / flux_mid90

    # Flux percentile mid 35% ratio
    flux_67 = np.percentile(flux, 67.5)
    flux_32 = np.percentile(flux, 32.5)
    flux_mid35 = flux_67 - flux_32
    flux_percentile_ratio_mid35 = flux_mid35 / flux_mid90

    # Flux percentile mid 50% ratio
    flux_75 = np.percentile(flux, 75)
    flux_25 = np.percentile(flux, 25)
    flux_mid50 = flux_75 - flux_25
    flux_percentile_ratio_mid50 = flux_mid50 / flux_mid90

    # Flux percentile mid 65% ratio
    flux_82 = np.percentile(flux, 82.5)
    flux_17 = np.percentile(flux, 17.5)
    flux_mid65 = flux_82 - flux_17
    flux_percentile_ratio_mid65 = flux_mid65 / flux_mid90

    # Flux percentile mid 80% ratio
    flux_90 = np.percentile(flux, 90)
    flux_10 = np.percentile(flux, 10)
    flux_mid80 = flux_90 - flux_10
    flux_percentile_ratio_mid80 = flux_mid80 / flux_mid90

    # Ratio of flux mid 90 over median flux
    percent_diff_flux_percentile = flux_mid90 / median_flux

    features = {'mean':mean_mag,
                'mean_variance':mean_var,
                'skew':skew,
                'kurtosis':kurtosis,
                'std':std_mag,
                'beyond1std':beyond_1std,
                'interquartile_range':quartile_range,
                'amplitude':amplitude,
                'max_slope':max_slope,
                'median_absolute_deviation':med_abs_dev,
                'median_buffer_range':med_buffer_range,
                'peak_mag':peak_mag,
                'percent_amplitude':percent_amplitude,
                'range_of_cumulative_sum':rcs,
                'flux_percentile_ratio_mid20':flux_percentile_ratio_mid20,
                'flux_percentile_ratio_mid35':flux_percentile_ratio_mid35,
                'flux_percentile_ratio_mid50':flux_percentile_ratio_mid50,
                'flux_percentile_ratio_mid65':flux_percentile_ratio_mid65,
                'flux_percentile_ratio_mid80':flux_percentile_ratio_mid80,
                'percent_flux_difference':percent_diff_flux_percentile}
    
    return features

'''
Function to fit a Gaussian process to some set of points

Input
    - x_in: x-axis of data to fit
    - y_in: y-axis of data to fit
    - y_error: error in y
    - x_pad: [min, max], how much to pad the x values when fitting
    - l_scale: length scale hyperparameter, default = 10

Output
    - x_fit
    - y_fit
    - y_fit_sigma
'''

def gaussian_regression(x_in, y_in, y_error, x_pad, l_scale=5):

    # Transform y values, subtract by median
    median_y = np.median(y_in)
    y_in = y_in - median_y
    
    # Pad the range of x values to fit over
    min_pad = x_pad[0]
    max_pad = x_pad[1]
    
    x_min = min(x_in) - min_pad
    x_max = min(x_in) + max_pad
    
    len_range = min_pad + max_pad
    
    # Set minimum length scale 
    l_min = l_scale#max(x_in[1:] - x_in[:-1])
    
    # Mesh the input space for evaluations of the function, prediction,
    # and mean-square error. Use approx. 1 day spacing
    x_space = np.atleast_2d(np.arange(int(x_min), int(x_max), 1.0)).T
    x_fit = np.atleast_2d(x_in).T
    
    # Define kernels
    radial_basis_func = RBF(length_scale=l_min, length_scale_bounds=(l_min, 1000))
    exp_sine_sq = ExpSineSquared(length_scale=100, length_scale_bounds=(100, 5e3), periodicity=5e2
                                , periodicity_bounds=(5e2, 1e3))
    
    kernel = 1.0*radial_basis_func + 1.0*radial_basis_func + 1.0*radial_basis_func
    
    gpr = GaussianProcessRegressor(kernel = kernel, alpha=y_error, n_restarts_optimizer=5)
    
    # Fit data using Maximum Likelihood Estimation of the parameters
    gpr.fit(x_fit, y_in)
    
    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, y_pred_sigma = gpr.predict(x_space, return_std=True)
    
    # Get log marginal likelihood
    log_m_likelihood = gpr.log_marginal_likelihood(gpr.kernel_.theta)
    
    # Get R^2 of the prediction
    r_score = gpr.score(gpr.X_train_, gpr.y_train_)
    
    # Get fit parameters
    fit_params = gpr.kernel_.get_params()
    
    fit_params['log_marginal_likelihood'] = log_m_likelihood
    fit_params['r^2_score'] = r_score
    
    # Transform fit back
    y_pred = y_pred + median_y
    
    return x_space.flatten(), y_pred, y_pred_sigma

'''
Function to split data into training/test set with a balanced training set

Input
    - features: array-like structure containing features
    - labels: array-like structure containing features
    - test_size: if int, then number of samples in test size. if float, then the fraction of all data used for testing
    - random_state

Output
    - X_train - training set features
    - X_test - test set features
    - y_train - training set labels
    - y_test - test set labels
'''

def balanced_train_test_split(features, labels, test_size, random_state):

    random.seed(random_state)
    
    if type(test_size) == int:
        train_size = len(features) - test_size
    if type(test_size) == float:
        train_size = int(len(features) * (1 - test_size))
        test_size = len(features) - train_size
    
    # Get number of classes
    class_labels = labels.unique()
    n_classes = len(class_labels)
    
    sub_train_size = train_size // n_classes
    remainder = train_size % n_classes
    
    # Determine number of samples per class
    class_train_size = []
    for i in range(n_classes):
        if i == (n_classes - 1):
            class_train_size.append(sub_train_size + remainder)
        else:
            class_train_size.append(sub_train_size)
    
    # Get indices of dataframe by different types
    index_split = []
    for size, clabel in zip(class_train_size, class_labels):
        df_type = labels[labels == clabel]
        
        # Randomly select sample
        indices = list(df_type.index)
        sample_idx = random.sample(indices, size)
        
        index_split.extend(sample_idx)
    
    X_train = features.iloc[index_split]
    y_train = labels.iloc[index_split]
    
    X_test = features.drop(index_split)
    y_test = labels.drop(index_split)
    
    return X_train, X_test, y_train, y_test