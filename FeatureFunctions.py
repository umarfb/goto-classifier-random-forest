# Library to calculate FATS features (Feature Analysis for Time Series), adapted
# for Python 3.x

import numpy as np 
import pandas as pd
import math
import statsmodels
from statsmodels.tsa.stattools import acf
from scipy import stats 
import scipy.optimize as opt
import scipy.interpolate as interp 

import lomb

'''
Input data needs to be in the format [magnitude, time, magnitude_error]

Show all available features
'''
def list_all_features():

    print('Input data needs to be in the format [magnitude, time, magnitude_error]')
    features_list = ['PeakMag', 'Amplitude', 'Rcs', 'StetsonK', 'Meanvariance', 'Autocor_length',
                     'Con', 'Beyond1std', 'SmallKurtosis', 'Std', 'Skew', 'MaxSlope',
                      'MedianAbsDev', 'MedianBRP', 'FluxPercentileRatioMin20', 'FluxPercentileRatioMin35',
                      'FluxPercentileRatioMin50', 'FluxPercentileRatioMin65', 'FluxPercentileRatioMin80',
                      'PercentDifferenceFluxPercentile', 'PercentAmplitude', 'LinearTrend', 'Mean',
                      'Q31', 'AndersonDarling', 'PeriodLS', 'PeriodFit', 'Psi_CS', 'Psi_eta',
                      'Freq1_harmonics_amplitude_0', 'Freq1_harmonics_amplitude_1', 'Freq1_harmonics_amplitude_2',
                      'Freq1_harmonics_amplitude_3', 'Freq2_harmonics_amplitude_0', 'Freq2_harmonics_amplitude_1',
                      'Freq2_harmonics_amplitude_2', 'Freq2_harmonics_amplitude_3', 'Freq3_harmonics_amplitude_0',
                      'Freq3_harmonics_amplitude_1', 'Freq3_harmonics_amplitude_2', 'Freq3_harmonics_amplitude_3',
                      'Freq1_harmonics_rel_phase_0', 'Freq1_harmonics_rel_phase_1', 'Freq1_harmonics_rel_phase_2',
                      'Freq1_harmonics_rel_phase_3', 'Freq2_harmonics_rel_phase_0', 'Freq2_harmonics_rel_phase_1',
                      'Freq2_harmonics_rel_phase_2', 'Freq2_harmonics_rel_phase_3', 'Freq3_harmonics_rel_phase_0',
                      'Freq3_harmonics_rel_phase_1', 'Freq3_harmonics_rel_phase_2', 'Freq3_harmonics_rel_phase_3',
                      'Gskew', 'StructureFunction_index_21', 'StructureFunction_index_31', 'StructureFunction_index_32',
                      'MeanSpacing', 'AboveThreshold', 'NumEpochs']
    n_features = len(features_list)

    print('Number of features: {0}'.format(n_features))
    for f in features_list:
        print(f)


'''
PeakMag

Peak magnitude
'''
def PeakMag(data):

    return min(data[0])

'''
Amplitude

Half the difference between median of the maximum 5% and median of the minimum 5% of magnitudes

'''
def Amplitude(data):

    magnitude = data[0]
    N = len(magnitude)
    sorted_mag = sorted(magnitude)

    amplitude = np.median(sorted_mag[-int(math.ceil(0.05 * N)):] -
                np.median(sorted_mag[0:int(math.ceil(0.05 * N))])) / 2.0
    
    return amplitude

'''
RCS

The range of the cumulative sum of magnitudes

'''
def Rcs(data):

    magnitude = data[0]
    sigma = np.std(magnitude)
    N = len(magnitude)
    m = np.mean(magnitude)
    s = np.cumsum(magnitude - m) / (N * sigma)
    R = np.max(s) - np.min(s)

    return R

'''
StetsonK

A feature based on the Welch/Stetson variability index. A robust kurtosis measure

'''
def StetsonK(data):

    magnitude = data[0]
    error = np.array(data[2])

    mean_mag = (np.sum(magnitude/(error*error)) /
                np.sum(1.0 / (error*error)))
    
    N = len(magnitude)
    sigmap = (np.sqrt(N / (N - 1)) * 
                (magnitude - mean_mag) / error)
    
    K = (1 / np.sqrt(N * 1.0) *
        np.sum(np.abs(sigmap)) / np.sqrt(np.sum(sigmap ** 2)))
    
    return K
'''
Meanvariance

A simple variability index. Light curves with strong variability will have a high value

'''
def Meanvariance(data):

    magnitude = data[0]
    meanvar = np.std(magnitude) / np.mean(magnitude)

    return meanvar


'''
Autocor_length

Autocorrelation function length - the linear dependence of a signal with itself at two
two points in time.

'''
def Autocor_length(data):

    magnitude = data[0]
    nlags = 100

    AC = acf(magnitude, nlags=nlags)
    k = next((index for index, value in
                enumerate(AC) if value < np.exp(-1)), None)
    
    while k is None:
        nlags += 100
        AC = acf(magnitude, nlags=nlags)
        k = next((index for index, value in
                enumerate(AC) if value < np.exp(-1)), None)
    
    return k

'''
Con

Count the number of three consecutive measurements that are out of 2 sigma range,
normalised by N - 2

'''
def Con(data):

    magnitude = data[0]
    N = len(magnitude)
    N_consecutive = 3

    if N < N_consecutive:
        return 0
    
    sigma = np.std(magnitude)
    m = np.mean(magnitude)
    count = 0

    for i in range(N - N_consecutive + 1):
        flag = 0
        for j in range(N_consecutive):
            if(magnitude[i + j] > m + 2*sigma or magnitude[i + j] < m - 2*sigma):
                flag = 1
            else:
                flag = 0
                break
        if flag:
            count += 1
    con = count * 1.0 / (N - N_consecutive + 1)
    return con

'''
Beyond1std

Fraction of points beyond 1 st. dev from the weighted (by errors) mean of magnitudes

'''
def Beyond1std(data):

    magnitude = data[0]
    error = np.array(data[2])
    N = len(magnitude)

    weighted_mean = np.average(magnitude, weights=(1 / error **2))

    # Std. dev with respect to weighted mean

    var = sum((magnitude - weighted_mean) ** 2)
    std = np.sqrt((1.0 / (N - 1)) * var)

    count = np.sum(np.logical_or(magnitude > weighted_mean + std,
                                    magnitude < weighted_mean - std))
    
    beyond1std = float(count) / N
    return beyond1std

'''
SmallKurtosis

Small sample kurtosis of magnitudes

'''
def SmallKurtosis(data):

    magnitude = data[0]
    N = len(magnitude)
    m = np.mean(magnitude)
    std = np.std(magnitude)

    S = sum(((magnitude - m) / std) ** 4)

    c1 = float(N * (N + 1)) / ((N - 1) * (N - 2) * (N - 3))
    c2 = float(3 * (N - 1) ** 2) / ((N - 2) * (N - 3))

    smallkurtosis = (c1 * S) - c2
    return smallkurtosis

'''
Std

Standard deviation of the magnitudes

'''
def Std(data):

    magnitude = data[0]

    return np.std(magnitude)

'''
Skew

Skewness of the magnitudes

'''
def Skew(data):

    magnitude = data[0]

    return stats.skew(magnitude)

'''
MaxSlope

Examine successive (time-sorted) magnitudes, find the maximal first difference
(delta magnitude over delta time)

'''
def MaxSlope(data):

    magnitude = data[0]
    time = data[1]
    slope = np.abs(magnitude[1:] - magnitude[:-1]) / (time[1:] - time[:-1])

    # Replace infinite values with 0
    slope[slope == np.inf] = 0

    return max(slope)
    
'''
MedianAbsDev

The median discrepancy of the data from the median data

'''
def MedianAbsDev(data):

    magnitude = data[0]
    median = np.median(magnitude)

    devs = np.abs(magnitude - median)
    medianabsdev = np.median(devs)

    return medianabsdev

'''
MedianBRP

The fraction of photometric points within amplitude/10

'''
def MedianBRP(data):

    magnitude = data[0]
    median = np.median(magnitude)
    amplitude = (np.max(magnitude) - np.min(magnitude)) / 10
    N = len(magnitude)

    count = np.sum(np.logical_and(magnitude < median + amplitude,
                                    magnitude > median - amplitude))
    
    medianbrp = float(count) / N

    return medianbrp

'''
PairSlopeTrend

Consider the last 30 (time-sorted) measurements of source magnitude, the fraction of
increasing first differences minus fraction of decreasing first differences

'''
def PairSlopeTrend(data):

    magnitude = data[0]
    n_last = 30
    data_last = magnitude[-n_last:]

    pstrend = float(len(np.where(np.diff(data_last) > 0)[0]) -
                    len(np.where(np.diff(data_last) <=0)[0])) / 30
    
    return pstrend

'''
FluxPercentileRatioMid20

'''
def FluxPercentileRatioMid20(data):

    magnitude = data[0]
    sorted_mag = np.sort(magnitude)
    #lc_len = len(sorted_mag)

    F_60= np.percentile(sorted_mag, 60.0)#int(math.ceil(0.60 * lc_len))
    F_40= np.percentile(sorted_mag, 40.0)#int(math.ceil(0.60 * lc_len))
    F_5 = np.percentile(sorted_mag, 5.0)#int(math.ceil(0.05 * lc_len))
    F_95 = np.percentile(sorted_mag, 95.0)#int(math.ceil(0.95 * lc_len))

    F_40_60 = F_60 - F_40
    F_5_95 = F_95 - F_5
    F_mid20 = F_40_60 / F_5_95

    return F_mid20

'''
FluxPercentileRatioMid35

'''
def FluxPercentileRatioMid35(data):

    magnitude = data[0]
    sorted_mag = np.sort(magnitude)
    #lc_len = len(sorted_mag)

    F_675 = np.percentile(sorted_mag, 67.5)#int(math.ceil(0.60 * lc_len))
    F_325 = np.percentile(sorted_mag, 32.5)#int(math.ceil(0.60 * lc_len))
    F_5 = np.percentile(sorted_mag, 5.0)#int(math.ceil(0.05 * lc_len))
    F_95 = np.percentile(sorted_mag, 95.0)#int(math.ceil(0.95 * lc_len))

    F_325_675 = F_675 - F_325
    F_5_95 = F_95 - F_5
    F_mid35 = F_325_675 / F_5_95

    return F_mid35

'''
FluxPercentileRatioMid50

'''
def FluxPercentileRatioMid50(data):

    magnitude = data[0]
    sorted_mag = np.sort(magnitude)
    #lc_len = len(sorted_mag)

    F_25 = np.percentile(sorted_mag, 25.0)#int(math.ceil(0.60 * lc_len))
    F_75 = np.percentile(sorted_mag, 75.0)#int(math.ceil(0.60 * lc_len))
    F_5 = np.percentile(sorted_mag, 5.0)#int(math.ceil(0.05 * lc_len))
    F_95 = np.percentile(sorted_mag, 95.0)#int(math.ceil(0.95 * lc_len))

    F_25_75 = F_75 - F_25
    F_5_95 = F_95 - F_5
    F_mid50 = F_25_75 / F_5_95

    return F_mid50

'''
FluxPercentileRatioMid65

'''
def FluxPercentileRatioMid65(data):

    magnitude = data[0]
    sorted_mag = np.sort(magnitude)
    #lc_len = len(sorted_mag)

    F_175 = np.percentile(sorted_mag, 17.5)#int(math.ceil(0.60 * lc_len))
    F_825 = np.percentile(sorted_mag, 82.5)#int(math.ceil(0.60 * lc_len))
    F_5 = np.percentile(sorted_mag, 5.0)#int(math.ceil(0.05 * lc_len))
    F_95 = np.percentile(sorted_mag, 95.0)#int(math.ceil(0.95 * lc_len))

    F_175_825 = F_825 - F_175
    F_5_95 = F_95 - F_5
    F_mid65 = F_175_825 / F_5_95

    return F_mid65

'''
FluxPercentileRatioMid80

'''
def FluxPercentileRatioMid80(data):

    magnitude = data[0]
    sorted_mag = np.sort(magnitude)
    #lc_len = len(sorted_mag)

    F_10 = np.percentile(sorted_mag, 10.0)#int(math.ceil(0.60 * lc_len))
    F_90 = np.percentile(sorted_mag, 90.0)#int(math.ceil(0.60 * lc_len))
    F_5 = np.percentile(sorted_mag, 5.0)#int(math.ceil(0.05 * lc_len))
    F_95 = np.percentile(sorted_mag, 95.0)#int(math.ceil(0.95 * lc_len))

    F_10_90 = F_90 - F_10
    F_5_95 = F_95 - F_5
    F_mid80 = F_10_90 / F_5_95

    return F_mid80

'''
PercentDifferenceFluxPercentile

Ratio of F_5_95 over the median magnitude

'''
def PercentDifferenceFluxPercentile(data):

    magnitude = data[0]
    sorted_mag = np.sort(magnitude)
    median = np.median(magnitude)

    F_5 = np.percentile(sorted_mag, 5.0)#int(math.ceil(0.05 * lc_len))
    F_95 = np.percentile(sorted_mag, 95.0)#int(math.ceil(0.95 * lc_len))
    F_5_95 = F_95 - F_5

    percentdiff = F_5_95 / median

    return percentdiff

'''
PercentAmplitude

Largest percentage difference between either the max or min magnitude and the median

'''
def PercentAmplitude(data):

    magnitude = data[0]
    median_data = np.median(magnitude)
    distance_median = np.abs(magnitude - median_data)
    max_distance = np.max(distance_median)

    percent_amplitude = max_distance / median_data

    return percent_amplitude

'''
LinearTrend

Slope of a linear fit to the light curve

'''
def LinearTrend(data):

    magnitude = data[0]
    time = data[1]
    regression_slope = stats.linregress(time, magnitude)[0]

    return regression_slope

'''
Mean

Mean magnitude

'''
def Mean(data):

    magnitude = data[0]
    mean_mag = np.mean(magnitude)

    return mean_mag

'''
Q31

Difference between third anf first quartiles

'''
def Q31(data):

    magnitude = data[0]
    q31 = np.percentile(magnitude, 75) - np.percentile(magnitude, 25)

    return q31

'''
Anderson-Darling test

A statistical test of whether a given sample of data is drawn from a given probability
distribution.

'''
def AndersonDarling(data):

    magnitude = data[0]
    andersond = stats.anderson(magnitude)[0]

    andersond_test = 1 / (1.0 + np.exp(-10 * (andersond - 0.3)))

    return andersond_test

'''
PeriodLS

Lomb-Scargle algorithm (Scargle 1982) used for period finding and frequency analysis. It
is capable of handling unevenly spaced data points

'''
def PeriodLS(data):

    magnitude = data[0]
    time = data[1]
    ofac = 6

    global new_time
    global prob
    global period

    fx, fy, nout, jmax, prob = lomb.fasper(time, magnitude, ofac, 100.)
    period = fx[jmax]
    T = 1.0 / period
    new_time = np.mod(time, 2 * T) / (2 * T)

    return T

'''
PeriodFit

Return values of LS Peridiogram fit

'''
def PeriodFit(data):

    try:
        return prob
    except:
        print('Error: please run PeriodLS first to generate values for Period_fit')


'''
Psi_CS

Range of a cumulative sum applied to the phase-folded light curve (generated using
the period estimated from the Lomb-Scargle method).

'''
def Psi_CS(data):

    try:
        magnitude = data[0]
        time = data[1]
        folded_data = magnitude[np.argsort(new_time)]
        sigma = np.std(folded_data)
        N = len(folded_data)
        m = np.mean(folded_data)
        s = np.cumsum(folded_data - m) * 1.0 / (N * sigma)
        R = np.max(s) - np.min(s)

        return R
    except:
        print('Error: please run PeriodLS first to generate values for Psi_CS')

'''
Psi_eta

Variability index eta (Von Neumann et al. 1941) is the ratio of mean of the square of successive
differences to the variance of data points. It checks if any trends exist in the data. Psi_eta is
the variability index calculated from the folded light curve

'''
def Psi_eta(data):

    try:
        magnitude = data[0]
        folded_data = magnitude[np.argsort(new_time)]

        N = len(folded_data)
        sigma2 = np.var(folded_data)

        psi_eta = (1.0 / ((N - 1) * sigma2) *
                    np.sum(np.power(folded_data[1:] - folded_data[:-1], 2)))
        
        return psi_eta
    except:
        print('Error: please run PeriodLS first to generate values for Psi_eta')

'''
CAR_sigma

CAR (Brockwell 2002) is a continuous time auto regressive model. It provides a natural and consistent
way of estimating a characteristic time scale and variance of light curves.

'''
def CAR_sigma(data):

    def CAR_Lik(parameters, t, x, error_vars):

        sigma = parameters[0]
        tau = parameters[1]

        b = np.mean(x) / tau
        epsilon = 1e-300
        cte_neg = -np.infty
        num_datos = len(x)

        Omega = []
        x_hat = []
        a = []
        x_ast = []

        Omega.append((tau * (sigma ** 2)) / 2.)
        x_hat.append(0.)
        a.append(0.)
        x_ast.append(x[0] - b * tau)

        loglik = 0.

        for i in range(1, num_datos):

            a_new = np.exp(-(t[i] - t[i - 1]) / tau)
            x_ast.append(x[i] - b * tau)
            x_hat.append(
                a_new * x_hat[i - 1] +
                (a_new * Omega[i - 1] / (Omega[i - 1] + error_vars[i - 1])) *
                (x_ast[i - 1] - x_hat[i - 1]))

            Omega.append(
                Omega[0] * (1 - (a_new ** 2)) + ((a_new ** 2) * Omega[i - 1]) *
                (1 - (Omega[i - 1] / (Omega[i - 1] + error_vars[i - 1]))))

            print('a_{0}: {1}'.format(i, a_new))
            print('Omega_{0}: {1}'.format(i, Omega[i]))

            lik_inter_A = (2 * np.pi * (Omega[i] + error_vars[i])) ** -0.5
            lik_inter_B = np.exp(-0.5 * ((x_hat[i] - x_ast[i]) **2 ) / (Omega[i] + error_vars[i]))
            lik_inter = lik_inter_A * lik_inter_B

            print('Likelihood_{0}: {1}'.format(i, lik_inter))
            loglik_inter = np.log(lik_inter)

            #print('log_likelihood_{0}: {1}'.format(i, loglik_inter))
            loglik += loglik_inter

            if(loglik <= cte_neg):
                print('CAR likelihood tends to infinity')
                return None
        
        #print(-loglik)
        return -loglik
    
    def calculateCAR(time, data, error):

        x0 = [10., 0.5]
        bnds = ((0, 100), (0, 100))

        res = opt.minimize(CAR_Lik, x0, args=(time, data, error),
                        method='L-BFGS-B', bounds=bnds)
        
        sigma = res.x[0]
        global tau
        tau = res.x[1]

        return sigma
    
    N = len(data[0])
    magnitude = data[0]
    time = data[1]
    error = data[2] ** 2

    a = calculateCAR(time, magnitude, error)

    return a

'''
CAR_tau

Relaxation time for the CAR process

'''
def CAR_tau(data):

    try:
        return tau
    except:
        print('Error: please run CAR_sigma first to generate values for CAR_tau')

'''
CAR_mean

Variability of the time series on timescales shorter than tau

'''
def CAR_mean(data):

    magnitude = data[0]
    try:
        return np.mean(magnitude) / tau
    except:
        print('Error: please run CAR_sigma first to generate values for CAR_mean')

'''
Freq1_harmonics_amplitude

'''
def Freq1_harmonics_amplitude_0(data):

    magnitude = data[0]
    time = data[1]

    time = time - np.min(time)

    global A
    global PH
    global scaledPH

    A = []
    PH = []
    scaledPH = []

    def model(x, a, b, c, Freq):
        return a*np.sin(2*np.pi*Freq*x) + b*np.cos(2*np.pi*Freq*x) + c
    
    for i in range(3):

        wk1, wk2, nout, jmax, prob = lomb.fasper(time, magnitude, 6., 100.)

        fundamental_Freq = wk1[jmax]

        # fit to a_i sin(2pi f_i t) + b_i cos(2 pi f_i t) + b_i,o

        # a, b are the parameters we care about
        # c is a constant offset
        # f is the fundamental Frequency
        def yfunc(Freq):
            def func(x, a, b, c):
                return a*np.sin(2*np.pi*Freq*x)+b*np.cos(2*np.pi*Freq*x)+c
            return func

        Atemp = []
        PHtemp = []
        popts = []

        for j in range(4):
            popt, pcov = opt.curve_fit(yfunc((j+1)*fundamental_Freq), time, magnitude)
            Atemp.append(np.sqrt(popt[0]**2+popt[1]**2))
            PHtemp.append(np.arctan(popt[1] / popt[0]))
            popts.append(popt)

        A.append(Atemp)
        PH.append(PHtemp)

        for j in range(4):
            magnitude = np.array(magnitude) - model(time, popts[j][0], popts[j][1], popts[j][2], (j+1)*fundamental_Freq)

    for ph in PH:
        scaledPH.append(np.array(ph) - ph[0])

    return A[0][0]

'''
Freq1_harmomincs_amplitude_1

'''
def Freq1_harmonics_amplitude_1(data):

    try:
        return A[0][1]
    except:
        print('Error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics')

'''
Freq1_harmomincs_amplitude_2

'''
def Freq1_harmonics_amplitude_2(data):

    try:
        return A[0][2]
    except:
        print('Error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics')

'''
Freq1_harmomincs_amplitude_3

'''
def Freq1_harmonics_amplitude_3(data):

    try:
        return A[0][3]
    except:
        print('Error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics')

'''
Freq2_harmonics_amplitude_0

'''
def Freq2_harmonics_amplitude_0(data):

    try:
        return A[1][0]
    except:
        print('Error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics')

'''
Freq2_harmonics_amplitude_1

'''
def Freq2_harmonics_amplitude_1(data):

    try:
        return A[1][1]
    except:
        print('Error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics')

'''
Freq2_harmonics_amplitude_2

'''
def Freq2_harmonics_amplitude_2(data):

    try:
        return A[1][2]
    except:
        print('Error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics')

'''
Freq2_harmonics_amplitude_3

'''
def Freq2_harmonics_amplitude_3(data):

    try:
        return A[1][3]
    except:
        print('Error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics')

'''
Freq3_harmonics_amplitude_0

'''
def Freq3_harmonics_amplitude_0(data):

    try:
        return A[2][0]
    except:
        print('Error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics')

'''
Freq3_harmonics_amplitude_1

'''
def Freq3_harmonics_amplitude_1(data):

    try:
        return A[2][1]
    except:
        print('Error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics')

'''
Freq3_harmonics_amplitude_2

'''
def Freq3_harmonics_amplitude_2(data):

    try:
        return A[2][2]
    except:
        print('Error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics')

'''
Freq3_harmonics_amplitude_3

'''
def Freq3_harmonics_amplitude_3(data):

    try:
        return A[2][3]
    except:
        print('Error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics')

'''
Freq1_harmonics_rel_phase_0

'''
def Freq1_harmonics_rel_phase_0(data):

    try:
        return scaledPH[0][0]
    except:
        print('Error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics')

'''
Freq1_harmonics_rel_phase_1

'''
def Freq1_harmonics_rel_phase_1(data):

    try:
        return scaledPH[0][1]
    except:
        print('Error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics')

'''
Freq1_harmonics_rel_phase_2

'''
def Freq1_harmonics_rel_phase_2(data):

    try:
        return scaledPH[0][2]
    except:
        print('Error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics')

'''
Freq1_harmonics_rel_phase_3

'''
def Freq1_harmonics_rel_phase_3(data):

    try:
        return scaledPH[0][3]
    except:
        print('Error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics')

'''
Freq2_harmonics_rel_phase_0

'''
def Freq2_harmonics_rel_phase_0(data):

    try:
        return scaledPH[1][0]
    except:
        print('Error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics')

'''
Freq2_harmonics_rel_phase_1

'''
def Freq2_harmonics_rel_phase_1(data):

    try:
        return scaledPH[1][1]
    except:
        print('Error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics')

'''
Freq2_harmonics_rel_phase_2

'''
def Freq2_harmonics_rel_phase_2(data):

    try:
        return scaledPH[1][2]
    except:
        print('Error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics')

'''
Freq2_harmonics_rel_phase_3

'''
def Freq2_harmonics_rel_phase_3(data):

    try:
        return scaledPH[1][3]
    except:
        print('Error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics')

'''
Freq3_harmonics_rel_phase_0

'''
def Freq3_harmonics_rel_phase_0(data):

    try:
        return scaledPH[2][0]
    except:
        print('Error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics')

'''
Freq3_harmonics_rel_phase_1

'''
def Freq3_harmonics_rel_phase_1(data):

    try:
        return scaledPH[2][1]
    except:
        print('Error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics')

'''
Freq3_harmonics_rel_phase_0

'''
def Freq3_harmonics_rel_phase_2(data):

    try:
        return scaledPH[2][2]
    except:
        print('Error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics')

'''
Freq3_harmonics_rel_phase_0

'''
def Freq3_harmonics_rel_phase_3(data):

    try:
        return scaledPH[2][3]
    except:
        print('Error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics')

'''
Gskew

Median-based measure of skew

'''
def Gskew(data):

    magnitude = np.array(data[0])
    median_mag = np.median(magnitude)
    F_3_value = np.percentile(magnitude, 3)
    F_97_value = np.percentile(magnitude, 97)

    return (np.median(magnitude[magnitude <= F_3_value]) +
                np.median(magnitude[magnitude >= F_97_value])
                - 2*median_mag)

'''
StructureFunction_index_21

'''
def StructureFunction_index_21(data):

    magnitude = data[0]
    time = data[1]

    global m_21
    global m_31
    global m_32

    Nsf = 100
    Np = 100
    sf1 = np.zeros(Nsf)
    sf2 = np.zeros(Nsf)
    sf3 = np.zeros(Nsf)
    f = interp.interp1d(time, magnitude)

    time_int = np.linspace(np.min(time), np.max(time), Np)
    mag_int = f(time_int)

    for tau in np.arange(1, Nsf):
        sf1[tau-1] = np.mean(np.power(np.abs(mag_int[0:Np-tau] - mag_int[tau:Np]) , 1.0))
        sf2[tau-1] = np.mean(np.abs(np.power(np.abs(mag_int[0:Np-tau] - mag_int[tau:Np]) , 2.0)))
        sf3[tau-1] = np.mean(np.abs(np.power(np.abs(mag_int[0:Np-tau] - mag_int[tau:Np]) , 3.0)))
    sf1_log = np.log10(np.trim_zeros(sf1))
    sf2_log = np.log10(np.trim_zeros(sf2))
    sf3_log = np.log10(np.trim_zeros(sf3))

    m_21, b_21 = np.polyfit(sf1_log, sf2_log, 1)
    m_31, b_31 = np.polyfit(sf1_log, sf3_log, 1)
    m_32, b_32 = np.polyfit(sf2_log, sf3_log, 1)

    return m_21

'''
StructureFunction_index_31

'''
def StructureFunction_index_31(data):

    try:
        return m_31
    except:
        print('error: please run StructureFunction_index_21 first to generate values for all Structure Function')

'''
StructureFunction_index_32

'''
def StructureFunction_index_32(data):

    try:
        return m_32
    except:
        print('error: please run StructureFunction_index_21 first to generate values for all Structure Function')

'''
MeanSpacing

Mean time spacing between two consecutive observations
'''
def MeanSpacing(data):

    time = data[1]

    d_time = time[1:] - time[:-1]
    meanspacing = np.mean(d_time)

    return meanspacing

'''
AboveThreshold

Difference between median magnitude and detection threshold
'''
def AboveThreshold(data, mag_lim):

    magnitude = data[0]
    median = np.median(magnitude)

    abv_thresh = mag_lim - median

    return abv_thresh

'''
NumEpochs

Number of epochs observed

'''
def NumEpochs(data):

    time = list(map(int, data[1]))
    epochs = np.unique(time)

    return len(epochs)
