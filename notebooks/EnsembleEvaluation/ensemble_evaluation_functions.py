import copy
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import scipy

from pandas import DataFrame
from properscoring import crps_ensemble
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def crps_score(A, B, plotCRPS=False):

    nA = A.size
    asorted = np.sort(A.reshape(-1))
    aproportional = 1. * np.arange(nA) / (nA - 1)

    nB = B.size
    bsorted = np.sort(B.reshape(-1))
    bproportional = 1. * np.arange(nB) / (nB - 1)

    if nB > nA:
        iii = np.linspace(0, nB - 1, num=nA).astype(int)
        bsorted2 = bsorted[iii]
    else:
        bsorted2 = bsorted

    if plotCRPS:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        ax.plot(asorted, aproportional, color='black')
        ax.plot(bsorted, bproportional, color='mediumseagreen')
        fig.show()
        
    dp = aproportional[2] - aproportional[1]
    score = np.sum(np.abs(np.subtract(asorted, bsorted2))) * dp
    return score

def ignorance_score(ytrue, ymean, ystd, 
    plotPDF=False, nBins=10, probMin=0.0001):
    
    nPts = ytrue.shape[0]
    if len(ytrue.shape) > 1:
        ytrue = ytrue.reshape(-1)
    if len(ymean.shape) > 1:
        ymean = ymean.reshape(-1)
    if len(ystd.shape) > 1:
        ystd = ystd.reshape(-1)
    ymin = ymean - 3. * ystd
    ymax = ymean + 3. * ystd

    ign_score = 0.
    for i in range(nPts):
        minH = ymin[i]
        maxH = ymax[i]
        trueH = ytrue[i]
        if trueH >= minH and trueH <= maxH:
            bin_edges = np.array([value for value in np.linspace(
                minH, maxH, num=nBins + 1)])
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            indx = find_nearest(bin_centers, trueH)

            dist = scipy.stats.norm(ymean[i], ystd[i])
            probability = dist.pdf(bin_centers[indx])
            ign_score += np.log2(probability)

            if plotPDF and i == 0:
                probabilities = [dist.pdf(value) for value in bin_centers]
                plt.plot(bin_centers, probabilities)
                plt.xlabel('Target Value')
                plt.ylabel('Density')
                plt.show()
        else:
            ign_score += np.log2(probMin)

    ign_score = -1. * ign_score / nPts
    return ign_score

def ignorance_score_ens(ytrue, ypred, 
    nBins=10, plotPDF=False, probMin=0.0001):
    nPts = ytrue.size

    ytrue = ytrue.reshape(-1)
    ymean = np.mean(ypred, axis=-1).reshape(-1)
    ystd = np.std(ypred, axis=-1).reshape(-1)
    ymin = ymean - 3. * ystd
    ymax = ymean + 3. * ystd

    ign_score = 0.
    for i in range(nPts):
        minH = ymin[i]
        maxH = ymax[i]
        trueH = ytrue[i]

        if trueH >= minH and trueH <= maxH:
            bin_edges = [value for value in np.linspace(\
                minH, maxH, num=nBins + 1)]
            density, bin_centers = get_histogram(\
                ypred[i, :], bins=bin_edges, density=True)
            indx = find_nearest(bin_centers, trueH)
            probability = np.max([density[indx], probMin])
            ign_score += np.log2(probability)

            if plotPDF and i == 0:
                plt.plot(bin_centers, density)
                plt.xlabel('Target Value')
                plt.ylabel('Density')
        else:
            ign_score += np.log2(probMin)

    ign_score = -ign_score / nPts
    return ign_score

def mse(A, B):
    return np.mean((A - B)**2)

def rmse(A, B):
    return np.sqrt(np.mean((A - B)**2))

def get_attributes_points(y_true, y_pred,
                           bin_edges=None,
                           min_val=None,
                           max_val=None,
                           mean_val=None,
                           nBins=10,
                           showR2=True,
                           nTicks=5,
                           verbose=True):

    if min_val is None or max_val is None:
        min_val, max_val = get_min_max(y_pred, y_true)

    if mean_val is None:
        mean_val = y_true.mean()

    if bin_edges is None:
        bin_edges = create_contours(min_val, max_val, nBins + 1)

    pred_vals = np.zeros((nBins)) - 999.
    obs_vals = np.zeros((nBins)) - 999.
    y_on_pred = np.zeros((y_pred.shape)) - 999.
    for i in range(nBins):
        gref = np.logical_and(\
            y_pred >= bin_edges[i],
            y_pred < bin_edges[i + 1])
        if np.any(gref):
            pred_vals[i] = np.mean(y_pred[gref])
            obs_vals[i] = np.mean(y_true[gref])
            y_on_pred[gref] = y_true[gref]

    obs_counts, obs_bins = get_histogram(
        y_true, bins=bin_edges)
    pred_counts, pred_bins = get_histogram(
        y_pred, bins=bin_edges)

    yclimo = np.ones_like(y_true) * mean_val
    mseclimo = mse(y_true, yclimo)
    msepred = mse(y_true, y_pred)
    msess = (mseclimo - msepred) / mseclimo

    r2score = r2_score(y_on_pred.reshape(-1), y_pred.reshape(-1))
    corcoef = np.corrcoef(\
        y_pred.reshape(-1), y_on_pred.reshape(-1))[0, 1]
    slope, intercept, rvalue, _, _ = scipy.stats.linregress(\
        y_pred.reshape(-1), y_on_pred.reshape(-1))

    tick_vals = list_to_int(
        np.linspace(min_val, max_val, nTicks))

    if verbose:
        print("Climatology MSE: {:.4f}".format(mseclimo))
        print("Predicted MSE: {:.4f}".format(msepred))
        print("MSE Skill Score (MSESS): {:.4f}".format(msess))
        print("Cond YObs vs. Pred R2 Score: {:.4f}  r2 Value: {:.4f}".\
            format(r2score, rvalue * rvalue))
        print("Cond YObs vs. Pred Pearson's Cor Coef: {:.4f}  r Value: {:.4f}".\
            format(corcoef, rvalue))
        print("Cond YObs Min: {:.3f}, Mean: {:.3f}, Max: {:.3f}".\
            format(y_on_pred.min(), y_on_pred.mean(), y_on_pred.max()))
        print("Pred Min: {:.3f}, Mean: {:.3f}, Max: {:.3f}".\
            format(y_pred.min(), y_pred.mean(), y_pred.max()))

    if showR2:
        mycolor = 'tab:blue'
        linecolor = 'black'
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(y_pred, y_on_pred, 
        alpha=0.6, color=mycolor, s=1)
        ax.set_xlabel("Pred Spread")
        ax.set_ylabel("Error")

        yline1 = min_val * slope + intercept
        yline2 = max_val * slope + intercept
        ax.plot([min_val, max_val], [yline1, yline2], \
            color=mycolor)
        ax.plot([min_val, max_val], [min_val, max_val], 
            color=linecolor, linestyle='--')
        plt.show()

    aDict = {
        'attr_cor_coef': rvalue,
        'attr_min_val': min_val,
        'attr_max_val': max_val,
        'attr_mean_val': mean_val,
        'attr_msess': msess,
        'attr_obs_bins': obs_bins,
        'attr_obs_counts': obs_counts,
        'attr_obs_vals': obs_vals,
        'attr_pred_bins': pred_bins,
        'attr_pred_counts': pred_counts,
        'attr_pred_vals': pred_vals,
        'attr_r2_score': r2score,
        'attr_tick_vals': tick_vals}

    return aDict

def get_discard_points(ytrue, ymean, ystd, 
    bins=None):

    if bins is None:
        nbins = 10
        bins = np.linspace(0., 0.9, nbins)
    else:
        nbins = len(bins)

    ytrue1d = ytrue.reshape(-1)
    ymean1d = ymean.reshape(-1)
    ystd1d = ystd.reshape(-1)
    nsamples = ystd1d.shape[0]

    yrefs = np.argsort(ystd1d)
    ytrueSorted = ytrue1d[yrefs]
    ymeanSorted = ymean1d[yrefs]

    rmseOut = np.empty((nbins))
    for i in range(nbins):
        iCutoff = nsamples - int(nsamples * bins[i])
        ytrueH = ytrueSorted[:iCutoff]
        ymeanH = ymeanSorted[:iCutoff]
        rmseOut[i] = rmse(ytrueH, ymeanH)

    dtiA = np.empty((nbins - 1))
    for i in range(nbins - 1):
        dtiA[i] = (rmseOut[i] - rmseOut[i + 1])

    dtmf = 0.
    for i in range(nbins - 1):
        if rmseOut[i] >= rmseOut[i + 1]:
            indicator = 1.
        else:
            indicator = 0.
        dtmf += indicator
    dtmf *= 1 / (nbins - 1)

    discardDict = {
        'discard_bins': bins,
        'discard_dtis': dtiA,
        'discard_mf': dtmf,
        'discard_imprv': np.mean(dtiA[:-1]),
        'discard_vals': rmseOut}

    return discardDict

def get_pit_dvalue(pit_counts):
    dvalue = 0.
    nbins = pit_counts.shape[0]
    nbinsI = 1./nbins

    pitTot = np.sum(pit_counts)
    pit_freq = np.divide(pit_counts, pitTot)
    for i in range(nbins):
        dvalue += (pit_freq[i] - nbinsI) * (pit_freq[i] - nbinsI)
    dvalue = np.sqrt(dvalue/nbins)
    return dvalue

def get_pit_evalue(nbins, tsamples):
    evalue = (1. - 1./nbins)/(tsamples * nbins)
    return np.sqrt(evalue)

PIT_BIN_EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
def get_pit_points(y_true, y_pred, y_std,
                   pit_bins=PIT_BIN_EDGES):

    nBins = len(pit_bins) - 1
    nSamples = y_true.size
    pit_values = scipy.stats.norm.cdf(\
      x=y_true, loc=y_pred, scale=y_std).reshape(-1)
    weights = np.ones_like(pit_values) / nSamples
    pit_counts, bin_centers = get_histogram(\
      pit_values, bins=pit_bins, weights=weights)

    pDict = {
      'pit_centers': bin_centers,
      'pit_counts': pit_counts,
      'pit_dvalue': get_pit_dvalue(pit_counts),
      'pit_evalue': get_pit_evalue(nBins, nSamples),
      'pit_values': pit_values,
      'pit_weights': weights}
      
    return pDict

def get_pit_points_ens(ytrue, ypred,
    pit_bins=PIT_BIN_EDGES):
    """
    Program to calculate Probability Integral Transform (PIT)
    points from ensemble predictions.

    Assumes a single target.

    Input:
    ytrue: An array containing the true samples with a single target.
    ypred: An array containing the predictions for a single target,
      with the ensemble members in the last dimension 
      i.e. shape = (..., nEns), nEns = # of ensemble members

    Output:
    Dictionary containing the PIT information.
    """

    nTr = ytrue.size
    nPr = ypred.size
    if nTr == nPr:
        print("Using ensemble version of get_pit_points, ")
        print("   but the predictions are not ensembles.")
        return {}

    nBins = len(pit_bins) - 1
    nEns = ypred.shape[-1]
    nSamples = ytrue.size
    pit_evalue = get_pit_evalue(nBins, nSamples)

    ytrueT = ytrue.reshape(-1)
    ypredT = ypred.reshape((nSamples, nEns))
    ypredTS = np.sort(ypredT, axis=1)

    ytrueTE = np.repeat(
      ytrueT[..., np.newaxis], nEns, axis=-1)
    pred_diff = np.abs(np.subtract(ytrueTE, ypredTS))
    pit_values = np.divide(np.argmin(pred_diff, axis=-1), nEns)
    weights = np.ones_like(pit_values) / nSamples
    pit_counts, bin_centers = get_histogram(\
        pit_values, bins=pit_bins, weights=weights)

    pDict = {
      'pit_centers': bin_centers,
      'pit_counts': pit_counts,
      'pit_dvalue': get_pit_dvalue(pit_counts),
      'pit_evalue': pit_evalue,
      'pit_values': pit_values,
      'pit_weights': weights}
      
    return pDict

def get_spread_skill_points(y_true, y_pred, y_std,
                            nBins=10,
                            bins=None,
                            showR2=True,
                            spread_last=None,
                            verbose=True):

    if y_true.shape != y_pred.shape:
        print("Mismatching shapes:")
        print("   y_true: {}".format(y_true.shape))
        print("   y_pred: {}".format(y_pred.shape))
        return {}
    nPts = y_true.size

    if not bins:
        minBin = np.min([0., y_std.min()])
        maxBin = np.ceil(np.max([rmse(y_true, y_pred), y_std.max()]))
        bins = create_contours(minBin, maxBin, nBins + 1)
    else:
        nBins = len(bins) - 1

    ssRel = 0.
    error = np.zeros((nBins)) - 999.
    spread = np.zeros((nBins)) - 999.
    y_on_error = np.zeros((y_pred.shape)) - 999.
    for i in range(nBins):
        refs = np.logical_and(\
            y_std >= bins[i], y_std < bins[i + 1])
        nPtsBin = np.count_nonzero(refs)
        if nPtsBin > 0:
            ytrueBin = y_true[refs]
            ymeanBin = y_pred[refs]
            error[i] = rmse(ytrueBin, ymeanBin)
            spread[i] = np.mean(y_std[refs])
            y_on_error[refs] = np.abs(y_true[refs] - y_pred[refs])
            ssRel += (nPtsBin/nPts) * np.abs(error[i] - spread[i])

    if spread_last is not None:
        spread[-1] = spread_last
    spread_counts, bin_centers  = get_histogram(\
        y_std, bins=bins)

    ssRatio = np.mean(y_std)/rmse(y_true, y_pred)
    ssr2 = r2_score(y_on_error.reshape(-1), y_std.reshape(-1))
    sscor = np.corrcoef(y_std.reshape(-1), y_on_error.reshape(-1))[0, 1]
    slope, intercept, rvalue, _, _ = scipy.stats.linregress(\
        y_std.reshape(-1), y_on_error.reshape(-1))

    if verbose:
        print("Spread Skill Ratio: {:.4f}".format(ssRatio))
        print("       Reliability: {:.4f}".format(ssRel))
        print("          R2 Score: {:.4f}  r2 Value: {:.4f}".\
            format(ssr2, rvalue * rvalue))
        print("Pearson's Cor Coef: {:.4f}  r Value: {:.4f}".\
            format(sscor, rvalue))
        print("   Cond YError Min: {:.3f}, Mean: {:.3f}, Max: {:.3f}".\
            format(y_on_error.min(), y_on_error.mean(), y_on_error.max()))
        print("      Pred Std Min: {:.3f}, Mean: {:.3f}, Max: {:.3f}".\
            format(y_std.min(), y_std.mean(), y_std.max()))

    if showR2:
        mycolor = 'tab:blue'
        linecolor = 'black'
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(y_std, y_on_error, 
        alpha=0.6, color=mycolor, s=1)
        ax.set_xlabel("Pred Spread")
        ax.set_ylabel("Error")

        yline1 = minBin * slope + intercept
        yline2 = maxBin * slope + intercept
        ax.plot([minBin, maxBin], [yline1, yline2], \
            color=mycolor)
        ax.plot([minBin, maxBin], [minBin, maxBin], 
            color=linecolor, linestyle='--')
        plt.show()

    sDict = {
        'ss_bin_edges': bins,
        'ss_bin_centers': bin_centers,
        'ss_cor_coef': rvalue,
        'ss_error_vals': error,
        'ss_ratio': ssRatio,
        'ss_reliability': ssRel,
        'ss_r2_score': ssr2,
        'ss_spread_counts': spread_counts,
        'ss_spread_vals': spread}
    return sDict

def create_contours(minVal, maxVal, nContours, match=False):
    if match:
        xVal = np.max([np.abs(minVal), np.abs(maxVal)])
        interval = 2 * xVal / (nContours - 1)
    else:
        interval = (maxVal - minVal) / (nContours - 1)
    contours = np.empty((nContours))
    for i in range(nContours):
        contours[i] = minVal + i * interval
    return contours


#### HELPER FUNCTIONS


def get_histogram(var, bins=10, density=False, weights=None):
    counts, bin_edges = np.histogram(
        var, bins=bins, density=density, weights=weights)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return counts, bin_centers

def find_nearest(array, value, smaller=False, larger=False):
    if larger:
        # The implementation below is used so that when a value is halfway
        # between two points, the larger index is returned (like infind)
        return len(array) - np.abs((array - value))[::-1].argmin() - 1
        
    if smaller:
        return np.where(array <= value)[0][-1]
    
    return (np.abs(array - value)).argmin()
### 2D Functions

def get_spread_skill_2d(y_true,ensemble):
    ssrel = np.zeros_like(y_true[0])
    ssrat = np.zeros_like(y_true[0])
    for ilat in range(y_true.shape[1]):
        for ilon in range(ensemble.shape[2]):
            spread_skill_dict = get_spread_skill_points(y_true[:,ilat,ilon],
                                                ensemble[:,ilat,ilon].mean(axis=1),
                                                ensemble[:,ilat,ilon].std(axis=1),
                                                showR2=False,
                                                verbose=False)
            ssrel[ilat,ilon] = spread_skill_dict['ss_reliability']
            ssrat[ilat,ilon] = spread_skill_dict['ss_ratio']
    return ssrel, ssrat


def get_discard_skill_2d(y_true,ensemble):
    mf = np.zeros_like(y_true[0])
    di = np.zeros_like(y_true[0])
    for ilat in range(y_true.shape[1]):
        for ilon in range(ensemble.shape[2]):
            discardDict = get_discard_points(y_true[:,ilat,ilon],
                                            ensemble[:,ilat,ilon].mean(axis=1),
                                            ensemble[:,ilat,ilon].std(axis=1),None)
            mf[ilat,ilon] = discardDict['discard_mf']
            di[ilat,ilon] = discardDict['discard_imprv']
    return mf,di

def get_pitd_2d(y_true,ensemble):
    pitd = np.zeros_like(y_true[0])
    for ilat in range(y_true.shape[1]):
        for ilon in range(ensemble.shape[2]):
            pitDict = get_pit_points_ens(y_true[:,ilat,ilon], ensemble[:,ilat,ilon])
            pitd[ilat,ilon] = pitDict['pit_dvalue']
    return pitd

def get_crps_2d(y_true,ensemble):
    crps = np.zeros_like(y_true[0])
    for ilat in range(y_true.shape[1]):
        for ilon in range(ensemble.shape[2]):
            crps[ilat,ilon] = crps_ensemble(y_true[:,ilat,ilon],ensemble[:,ilat,ilon]).mean()
    return crps

def get_ign_2d(y_true,ensemble):
    ign = np.zeros_like(y_true[0])
    for ilat in range(y_true.shape[1]):
        for ilon in range(ensemble.shape[2]):
            ign[ilat,ilon] = ignorance_score_ens(y_true[:,ilat,ilon],ensemble[:,ilat,ilon])
    return ign

def get_ign_2d_v2(y_true,ensemble):
    #Computes the score with the standard deviation instead of the full ensemble
    ign = np.zeros_like(y_true[0])
    for ilat in range(y_true.shape[1]):
        for ilon in range(ensemble.shape[2]):
            ign[ilat,ilon] = ignorance_score(y_true[:,ilat,ilon], ensemble[:,ilat,ilon].mean(axis=1),
                                             ensemble[:,ilat,ilon].std(axis=1),
                                             plotPDF=False, nBins=10, 
                                             probMin=0.0001)
    return ign