# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep, LSQUnivariateSpline, PchipInterpolator
from scipy.interpolate import CubicSpline, Akima1DInterpolator
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression
from functools import partial
from random import shuffle
from sklearn import linear_model
from sklearn.metrics import r2_score
import scipy as sp
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback


# %% [markdown]
# # Age Model

# %%
def linear_inverse_SR(x, a=4):
    """

    Args:
        x (depth): 
        a (slope) from [-2,2]
    
    Returns:
        linear inverse SR so that depth = 0 and 1 when time = 0 and 1
    """
    b = 1 - a/2
    return a*x + b  


def constant_inverse_SR(x):
    """

    Args:
        x (array): depth

    Returns:
        cosntant SR
    """
    return np.ones_like(x)

def sine_inverse_SR(x, A=0.2, k=2):
        
    inv_SR = A*np.cos(np.pi*k*x) + 1
    
    return inv_SR

def logistic_inverse_SR(x, A=0.4, k=20):
    """
    Args:
        x (_type_): depth
        A (float, optional): Range of the inverse SR. Defaults to 0.4.
        k (int, optional): steepness of the SR at depth=0.5. Defaults to 20.

    Returns:
        array of size x: logistic inverse SR (with average=1) so that  time = 0 and 1 when depth = 0 and 1 
    """
    return 1/(1+np.exp(-k*(x-0.5)))*A + (1-A/2)

def heart_wave_inverse_SR(x, A=0.2, k=5):
    """

    Args:
        x (_type_): depth
        A (float, optional): Amplitude. Defaults to 0.2.
        k (int, optional): number of peaks of the sine. Defaults to 5.

    Returns:
        array of size x: heart_wave_inverse_SR so that  time = 0 and 1 when depth = 0 and 1 
    """
    B = 1 - A/(2*np.pi*k)*(1-np.cos(k*np.pi))
    y =  A*np.sin(k*2*np.pi*(x-0.25)) + B
    a, b = np.searchsorted(x, [0.25, 0.75])
    y[:a] = np.ones_like(x[:a])*B
    y[b:] = np.ones_like(x[b:])*B
    return y

# def zigzag_inverse_SR(x, k=2):
#     knots = np.linspace(0,1,k+2)
#     np.piecewise(x, [ x < knots[i] for i in range(k+1)], [-1, 1])



# %% [markdown]
# # Transfer model

# %%
def generate_X_linReg(Amp, freq, times):
    ts_cos = Amp[None]*np.sin(freq[None]*times[:,None])
    ts_sin = Amp[None]*np.cos(freq[None]*times[:,None])
    return np.concatenate([ts_cos, ts_sin], axis=1)


# %% [markdown]
# ## Metric

# %%
def interpolate_Pchip(xy, xs):
    spl_xy = PchipInterpolator(*xy)
    return spl_xy(xs)
def interpolate_BSpline(SR, depth):
    spl_SR = splrep(*SR)
    SR_interpolate = splev(depth, spl_SR)
    return SR_interpolate

def interpolate_CubicSpline(SR, depth):
    if len(SR[0]) == 1:
        SR_interpolate = np.ones_like(depth)*SR[1]
    else:
        spl_SR = CubicSpline(*SR)
        SR_interpolate = spl_SR(depth)
    return SR_interpolate
    
def interpolate_Akima(SR, depth):
    if len(SR[0]) == 2:
        slope = (SR[1][1]-SR[1][0])/(SR[0][1]-SR[0][0])
        SR_interpolate = slope*depth + SR[1][0]
    else:
        SR_interpolate = Akima1DInterpolator(SR[0],SR[1])(depth)
    return SR_interpolate

def metric_v0(invSR, data, fs, interpolator=interpolate_CubicSpline, metric_type="BIC", *args, **kwargs):
    """_summary_: 
    
        metric  (according to the metric type) for the linear model with predictors variable are fourier harmonics of frequencies fs
        of fitting the data, when used the age model derived from the invSR.
        age model: given points of inverse SR, interpolate with an interpolator and integrate along depth
        to obtain time.
    Args:
        invSR (array[2,n]): inverse of sedimetation rates and their corresponding depth 
        interpolator: interpolator
        data (array[2,m]): the data and corresponding depth
        fs (_type_): list of frequencies of the model
        metric_type (str, optional):"BIC", "AIC" or "RSS". Defaults to "BIC".
            Returns:
        int: metric
           
    """

    depth, y_data = data

    invSR_interpolate = interpolator(invSR, depth)
    # invSR_interpolate[invSR_interpolate<0] = 0
    
    time = sp.integrate.cumulative_trapezoid(invSR_interpolate, depth, initial=0)
    X = generate_X_linReg(np.ones_like(fs), fs, time)

    reg_model = LinearRegression().fit(X, y_data)
    # Residual Sum Square
    RSS = np.sum((y_data - reg_model.predict(X))**2)
    N = len(y_data)
    n_params_SR = len(invSR[0])
    n_params = n_params_SR + 1 + len(fs)*2 + 2 
    if metric_type == "BIC":
        coef = np.log(N)
    elif metric_type == "AIC":
        coef = 2
    elif metric_type == "RSS":
        coef = 0
    metric = N*np.log(RSS/N) + coef*n_params
    return metric
    
def metric_v1(invSR, data, fs, interpolator=interpolate_CubicSpline, metric_type="BIC", *args, **kwargs):
    """_summary_: 
    
        metric  (according to the metric type) for the linear model with predictors variable are fourier harmonics of frequencies fs
        of fitting the data, when used the age model derived from the invSR.
        age model: given points of inverse SR, interpolate with an interpolator and integrate along depth
        to obtain time.
    Args:
        invSR (array[2,n]): inverse of sedimetation rates and their corresponding depth 
        interpolator: interpolator
        data (array[2,m]): the data and corresponding depth
        fs (_type_): list of frequencies of the model
        metric_type (str, optional):"BIC", "AIC" or "RSS". Defaults to "BIC".
            Returns:
        int: metric
           
    """

    depth, y_data = data

    invSR_interpolate = interpolator(invSR, depth)
    # invSR_interpolate[invSR_interpolate<0] = 0
    
    time = sp.integrate.cumulative_trapezoid(invSR_interpolate, depth, initial=0)
    X = generate_X_linReg(np.ones_like(fs), fs, time)

    reg_model = LinearRegression().fit(X, y_data)
    # Residual Sum Square
    RSS = np.sum((y_data - reg_model.predict(X))**2)
    N = len(y_data)
    n_params_SR = len(invSR[0])
    # n_params = n_params_SR 
    if metric_type == "BIC":
        coef = np.log(N)
    elif metric_type == "AIC":
        coef = 2
    elif metric_type == "RSS":
        coef = 0
    metric = N*np.log(RSS/N) + coef*n_params_SR
    return metric
    
def metric_piecewise(depth_invSR, data, fs, interpolator=interpolate_CubicSpline, n_pieces=1, invSR_lims=None, metric_type="r2", *args, **kwargs):
    """_summary_: 
    
        metric  (according to the metric type) for the linear model with predictors variable are fourier harmonics of frequencies fs
        of fitting the data, when used the age model derived from the invSR.
        age model: given points of inverse SR, interpolate with an interpolator and integrate along depth
        to obtain time.
    Args:
        depth_invSR (array[2,n]): inverse of sedimetation rates and their corresponding depth 
        interpolator: interpolator
        data (array[2,m]): the data and corresponding depth
        fs (_type_): list of frequencies of the model
        metric_type (str, optional):"BIC", "AIC" or "RSS". Defaults to "BIC".
            Returns:
        int: metric
           
    """

    depth, y_data = data

    invSR_interpolate = interpolator(depth_invSR, depth)
    if invSR_lims is not None:
        invSR_interpolate[invSR_interpolate<invSR_lims[0]] = invSR_lims[0]
        invSR_interpolate[invSR_interpolate>invSR_lims[1]] = invSR_lims[1]
    
    time = sp.integrate.cumulative_trapezoid(invSR_interpolate, depth, initial=0)
    X = generate_X_linReg(np.ones_like(fs), fs, time)

    reg_model = LinearRegression().fit(X, y_data)
    y_pred = reg_model.predict(X)
    
    depth_pieces = np.linspace(depth[0], depth[-1], n_pieces+1)
    # r2 = np.zeros(n_pieces)
    # RSS = np.zeros(n_pieces)
    metrix = np.zeros(n_pieces)

    for i in range(n_pieces):
        j1, j2 = np.searchsorted(depth, depth_pieces[i],"left"),  np.searchsorted(depth, depth_pieces[i+1], "right")
        if metric_type == "r2":
            metrix[i] = r2_score(y_data[j1:j2], y_pred[j1:j2])
        elif metric_type == "RSS":
            metrix[i] = np.mean((y_data[j1:j2] - y_pred[j1:j2])**2)
        else:
            print("metric type not defined")
    return metrix

def metric_piecewise_reg(depth_invSR, data, fs, spline=CubicSpline, n_pieces=1, invSR_lims=None, metric_type="r2", lambda_reg=0, *args, **kwargs):
    """_summary_: 
    
        metric  (according to the metric type) for the linear model with predictors variable are fourier harmonics of frequencies fs
        of fitting the data, when used the age model derived from the invSR.
        age model: given points of inverse SR, interpolate with an interpolator and integrate along depth
        to obtain time.
        with penalty terms for roughness with coefficient lambda : integral of squared of second derivative of the invSR function    
    Args:
        depth_invSR (array[2,n]): inverse of sedimetation rates and their corresponding depth 
        interpolator: interpolator
        data (array[2,m]): the data and corresponding depth
        fs (_type_): list of frequencies of the model
        metric_type (str, optional):"BIC", "AIC" or "RSS". Defaults to "BIC".
            Returns:
        int: metric
           
    """

    depth, y_data = data

    invSR_spline = spline(*depth_invSR)
    invSR_interpolate = invSR_spline(depth)
    if lambda_reg != 0:
        second_der = invSR_spline.derivative(2)
        roughness = sp.integrate.quad(lambda x: second_der(x)**2, depth[0], depth[-1])
    else:
        roughness = 0
    if invSR_lims is not None:
        invSR_interpolate[invSR_interpolate<invSR_lims[0]] = invSR_lims[0]
        invSR_interpolate[invSR_interpolate>invSR_lims[1]] = invSR_lims[1]
    
    time = sp.integrate.cumulative_trapezoid(invSR_interpolate, depth, initial=0)
    X = generate_X_linReg(np.ones_like(fs), fs, time)

    reg_model = LinearRegression().fit(X, y_data)
    y_pred = reg_model.predict(X)
    
    depth_pieces = np.linspace(depth[0], depth[-1], n_pieces+1)
    # r2 = np.zeros(n_pieces)
    # RSS = np.zeros(n_pieces)
    metrix = np.zeros(n_pieces)

    for i in range(n_pieces):
        j1, j2 = np.searchsorted(depth, depth_pieces[i],"left"),  np.searchsorted(depth, depth_pieces[i+1], "right")
        if metric_type == "r2":
            metrix[i] = r2_score(y_data[j1:j2], y_pred[j1:j2]) - lambda_reg*roughness
        elif metric_type == "RSS":
            metrix[i] = np.mean((y_data[j1:j2] - y_pred[j1:j2])**2)  - lambda_reg*roughness
        else:
            print("metric type not defined")
    return metrix

def invSR_to_pred(depth_invSR, data, invSR_lims, fs, interpolator=interpolate_CubicSpline):
    depth, y = data
    invSR_interpolate = interpolator(depth_invSR, depth)
    invSR_interpolate[invSR_interpolate<invSR_lims[0]] = invSR_lims[0]
    invSR_interpolate[invSR_interpolate>invSR_lims[1]] = invSR_lims[1]
    time = sp.integrate.cumulative_trapezoid(invSR_interpolate, depth, initial=0)
    X = generate_X_linReg(np.ones_like(fs), fs, time)
    reg_model = LinearRegression().fit(X, y)
    y_pred = reg_model.predict(X)
    return time, y_pred

def invSR_to_pred_v2(depth_invSR, data, invSR_lims, fs, interpolator=interpolate_CubicSpline):
    depth, y = data
    invSR_interpolate = interpolator(depth_invSR, depth)
    invSR_interpolate[invSR_interpolate<invSR_lims[0]] = invSR_lims[0]
    invSR_interpolate[invSR_interpolate>invSR_lims[1]] = invSR_lims[1]
    time = sp.integrate.cumulative_trapezoid(invSR_interpolate, depth, initial=0)
    X = generate_X_linReg(np.ones_like(fs), fs, time)
    reg_model = LinearRegression().fit(X, y)
    y_pred = reg_model.predict(X)
    return time, y_pred, reg_model.coef_

def metric_piecewise_short(y_data, y_pred, n_pieces, metric_type="r2"):
    metrix = np.zeros(n_pieces)
    n = len(y_data)
    j2 = 0
    for i in range(n_pieces):
        j1, j2 = j2, n//(n_pieces)*(i+1)
        if metric_type == "r2":
            metrix[i] = r2_score(y_data[j1:j2], y_pred[j1:j2])
        elif metric_type == "RSS":
            metrix[i] = np.mean((y_data[j1:j2] - y_pred[j1:j2])**2)
        else:
            print("metric type not defined")
    return metrix 

class invSRinference(ElementwiseProblem):

    def __init__(self, depth_genes, genes_lims, interpolator, data, fs, n_pieces=1, metric=metric_piecewise):

        self.invSR_lims = genes_lims
        self.depth_genes = depth_genes 
        self.interpolator = interpolator
        self.data = data
        self.fs = fs

        N_genes = len(depth_genes)
        xl = np.zeros(N_genes) + genes_lims[0]
        xu = np.zeros(N_genes) + genes_lims[1]
        self.func_metric = partial(metric, data=data, fs=fs, interpolator=interpolator, n_pieces=n_pieces, invSR_lims=genes_lims)
        super().__init__(n_var=N_genes, n_obj=n_pieces, xl=xl, xu=xu)

    def _evaluate(self, genes, out, *args, **kwargs):
        
        out["F"] = -self.func_metric([self.depth_genes, genes])
class invSR_p0_inference(ElementwiseProblem):
    """ same as invSRinference, include p0 as genes, input fsp is fs defined without p0 (first fives values are g1,..,5)"""
    def __init__(self, depth_genes, invSR_lims, p0_lims, interpolator, data, fsp, n_pieces=1, metric=metric_piecewise):

        self.invSR_lims = invSR_lims
        self.p0_lims = p0_lims
        self.depth_genes = depth_genes 
        self.interpolator = interpolator
        self.data = data
        self.fsp = fsp

        N_genes_invSR = len(depth_genes)
        xl = np.zeros(N_genes_invSR+1) + invSR_lims[0]
        xu = np.zeros(N_genes_invSR+1) + invSR_lims[1]
        xl[0] = p0_lims[0]; xu[0] = p0_lims[1]
        self.func_metric = partial(metric, data=data, interpolator=interpolator, n_pieces=n_pieces, invSR_lims=invSR_lims)
        super().__init__(n_var=N_genes_invSR+1, n_obj=n_pieces, xl=xl, xu=xu)

    def _evaluate(self, genes, out, *args, **kwargs):
        genes_p0 = genes[0]
        genes_invSR = genes[1:]
        fs = self.fsp.copy()
        fs[:5] += genes_p0
        # print(len(self.depth_genes), len(genes_invSR))
        out["F"] = -self.func_metric([self.depth_genes, genes_invSR], fs=fs)
        
class invSR_p0_inference_constrained(ElementwiseProblem):
    """ same as invSRinference, include p0 as genes, input fsp is fs defined without p0 (first fives values are g1,..,5)
        constrained on invSR and p0
        suchthat fs/2 > (p0+g4)/2/np.pi where fs = sampling rate = 1/dt 
        this is <=> T_duration * (p0+g4)/2/pi < (n-1)/2
    """
    def __init__(self, depth_genes, invSR_lims, p0_lims, interpolator, data, fsp, n_pieces=1, metric=metric_piecewise_short):

        self.invSR_lims = invSR_lims
        self.p0_lims = p0_lims
        self.depth_genes = depth_genes 
        self.interpolator = interpolator
        depth, y = data
        self.depth = depth
        self.y = y
        self.fsp = fsp
        self.metric = metric
        self.n_pieces = n_pieces
        N_genes_invSR = len(depth_genes)
        xl = np.zeros(N_genes_invSR+1) + invSR_lims[0]
        xu = np.zeros(N_genes_invSR+1) + invSR_lims[1]
        xl[0] = p0_lims[0]; xu[0] = p0_lims[1]
        self.invSR_to_predx = partial(invSR_to_pred, data=data, interpolator=interpolator, invSR_lims=invSR_lims)
        super().__init__(n_var=N_genes_invSR+1, n_obj=n_pieces, n_ieq_constr=1, xl=xl, xu=xu)
        
    def _evaluate(self, genes, out, *args, **kwargs):
        genes_p0 = genes[0]
        genes_invSR = genes[1:]
        fs = self.fsp.copy()
        fs[:5] += genes_p0
        time, y_pred = self.invSR_to_predx([self.depth_genes, genes_invSR], fs=fs)
        out["F"] = -self.metric(self.y, y_pred, self.n_pieces)
        out["G"] = [ (time[-1]-time[0])*fs[3]/2/np.pi - (len(self.y)-1)/2]
        # print(time[-1])
class Callback_getF(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []

    def notify(self, algorithm):
        b = algorithm.pop.get("F").mean(axis=1).min()
        self.data["best"].append(b)
# %%
def log_likelihood_whitenoise(invSR, data, fs, sigma, interpolator=interpolate_CubicSpline, *args, **kwargs):
    """_summary_: 
    
        loglikelihood for the linear model with predictors variable are fourier harmonics of frequencies fs
        of fitting the data, when used the age model derived from the invSR.
        age model: given points of inverse SR, interpolate with an interpolator and integrate along depth
        to obtain time.
    Args:
        invSR (array[2,n]): inverse of sedimetation rates and their corresponding depth 
        interpolator: interpolator
        data (array[2,m]): the data and corresponding depth
        fs (_type_): list of frequencies of the model
        sigma2: variance of the residual misfit 
           
    """

    depth, y_data = data

    invSR_interpolate = interpolator(invSR, depth)
    # invSR_interpolate[invSR_interpolate<0] = 0
    
    time = sp.integrate.cumulative_trapezoid(invSR_interpolate, depth, initial=0)
    X = generate_X_linReg(np.ones_like(fs), fs, time)

    reg_model = LinearRegression().fit(X, y_data)
    y_pred = reg_model.predict(X)
    # Residual Sum Square
    RSS = np.sum((y_data - y_pred)**2)
    N = len(y_data)

    return -0.5 * RSS / sigma**2 - N*np.log(sigma)

def log_likelihood_rednoise(invSR, data, fs, noise_params, interpolator=interpolate_CubicSpline, *args, **kwargs):
    """_summary_: 
    
        loglikelihood for the
        linear model with predictors variable are fourier harmonics of frequencies fs, with the age model derived from the invSR.
        age model: given points of inverse SR, interpolate with an interpolator and integrate along depth to obtain time.
        with red noise defined by AR1 with noise params = [sigma, rho] (see  Appendix Meyers and Malinverno 2018)
    Args:
        invSR (array[2,n]): inverse of sedimetation rates and their corresponding depth 
        interpolator: interpolator
        data (array[2,m]): the data and corresponding depth
        fs (_type_): list of frequencies of the model
        noise_params = [sigma, rho]: paramters of AR1 noise
    """

    depth, y_data = data
    N_data = len(y_data)
    invSR_interpolate = interpolator(invSR, depth)
    # invSR_interpolate[invSR_interpolate<0] = 0
    
    time = sp.integrate.cumulative_trapezoid(invSR_interpolate, depth, initial=0)
    X = generate_X_linReg(np.ones_like(fs), fs, time)

    reg_model = LinearRegression().fit(X, y_data)
    y_pred = reg_model.predict(X)
    
    
    # Residual Sum Square
    residual = y_data - y_pred
    sigma, rho = noise_params
    coef = 1/(sigma**2)/(1-rho**2)
    coef = 1/(sigma**2)/(1-rho**2)
    diag_vec = np.ones(N_data)*(1+rho**2)*coef
    diag_vec[0] = coef; diag_vec[-1] = coef
    offdiag_vec = np.ones(N_data-1)*-rho*coef
    offset = [-1,0,1]
    invCov = sp.sparse.diags([offdiag_vec, diag_vec, offdiag_vec],offset) 
    main_part = residual.T@invCov@residual
    
    return -0.5*main_part - 0.5*(2*N_data*np.log(sigma) + (N_data-1)*np.log(1-rho**2)) 


def log_uniform(x, x_lims):    
    if np.all((x > x_lims[0])  &  (x < x_lims[1])):
        return 0.0
    return -np.inf

def log_loguniform(x, x_lims):    
    if np.all((x > x_lims[0])  &  (x < x_lims[1])):
        return -np.log(x)
    return -np.inf

def log_gaussian(x, params):
    muy, sigma = params
    return -0.5*(x-muy)**2/sigma**2

def log_posterior_whitenoise_v0(invSR, depth_invSR, data, fs, sigma, interpolator=interpolate_CubicSpline, invSR_lims=[0,2]):
    "same as log_posterior_whitenoise but sigma is known"
    lp = log_uniform(invSR, invSR_lims)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_whitenoise([depth_invSR, invSR], data, fs, sigma, interpolator=interpolator)
    # return lp + log_likelihood(*arg, **kwargs)

def log_posterior_whitenoise_v1(params, depth_invSR, data, fs, interpolator=interpolate_CubicSpline, invSR_lims=[0,2], sigma_lims=[1e-4, 2]):
    """log of posterior for the
        linear model with predictors variable are fourier harmonics of constant frequencies fs
        with the age model derived from the invSR.
        age model: given points of inverse SR, interpolate with an interpolator and integrate along depth to obtain time.
        with white noise
        log posterior = log likelihood + log prior
        prior for invSR is uniform, sigma is loguniform
    Args:
        params: array[1 + N_genes] = [sigma (whitenoise), array of invSR]
        depth_invSR (_type_): depth of invSR
        data (array[2,m]):  depth and corresponding data
        fs (array): frequencies 
        interpolator (_type_, optional). Defaults to interpolate_CubicSpline.
        invSR_lims (list, optional): limit of inSR. Defaults to [0,2].
        invSR_lims (list, optional): limit of inSR. Defaults to [0,2].
    Returns:
        log posterior = log likelihood + log prior
    """
    sigma =  params[0]
    invSR = params[1:]
    lp_invSR = log_uniform(invSR, invSR_lims)
    lp_sigma = log_loguniform(sigma, sigma_lims)    
    lp = lp_invSR + lp_sigma
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_whitenoise([depth_invSR, invSR], data, fs, sigma, interpolator=interpolator)

def log_posterior_rednoise_v1(params, depth_invSR, data, fs, interpolator=interpolate_CubicSpline, invSR_lims=[0,2], sigma_lims=[1e-4, 2], rho_lims=[0,1]):
    """ same as log_posterior_whitenoise_v1 but red noise instead of whitenoise
    params: array[1 + N_genes] = [sigma, rho (from AR1), array of invSR]
    """
    sigma, rho = params[:2]
    invSR = params[2:]
    lp_invSR = log_uniform(invSR, invSR_lims)
    lp_rho = log_uniform(rho, rho_lims)
    lp_sigma = log_loguniform(sigma, sigma_lims)
    lp = lp_invSR + lp_rho + lp_sigma
    # if not np.isfinite(lp_invSR) or not np.isfinite(lp_rho) or not np.isfinite(lp_sigma):
    #     return -np.inf
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_rednoise([depth_invSR, invSR], data, fs, [sigma, rho], interpolator=interpolator)

def log_posterior_whitenoise(params, depth_invSR, data, prior_freq, interpolator=interpolate_CubicSpline, invSR_lims=[0,2],
                             sigma_lims=[1e-4, 2]):
    """ same as log_posterior_whitenoise_v1 but frequencies are no longer constant but treated as parameters
    params: array[1 + N_genes] = [sigma, rho (from AR1), array of invSR]
    prior_fs: array[N_freq, 2] :  parameters of prior for frequencies fs, (Gaussian with muy and sigma as params) 
    """
    sigma =  params[0]
    N_freq = prior_freq.shape[0]
    freqs = params[1:1+N_freq]
    fs = freqs[0] + freqs[1:]
    invSR = params[1+N_freq:]
    
    
    lp_invSR = log_uniform(invSR, invSR_lims)
    lp_sigma = log_loguniform(sigma, sigma_lims)    
    lp_fs = sum([log_gaussian(freqs[i], prior_freq[i]) for i in range(N_freq) ])
    lp = lp_invSR + lp_sigma + lp_fs
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_whitenoise([depth_invSR, invSR], data, fs, sigma, interpolator=interpolator)


def log_posterior_whitenoise_EP(params, depth_invSR, data, prior_freq, interpolator=interpolate_CubicSpline, invSR_lims=[0,2],
                             sigma_lims=[1e-4, 2]):
    """ same as log_posterior_whitenoise_v1 but frequencies are no longer constant but treated as parameters
    params: array[1 + N_genes] = [sigma, rho (from AR1), array of invSR]
    prior_fs: array[N_freq, 2] :  parameters of prior for frequencies fs, (Gaussian with muy and sigma as params) 
    """
    sigma =  params[0]
    N_freq = prior_freq.shape[0]
    freqs = params[1:1+N_freq]
    fs = np.hstack([freqs[0] + freqs[1:], [freqs[2]-freqs[5], freqs[4]-freqs[3] , freqs[4]-freqs[2], freqs[3]-freqs[5], freqs[3]-freqs[2]]])
    invSR = params[1+N_freq:]
    
    
    lp_invSR = log_uniform(invSR, invSR_lims)
    lp_sigma = log_loguniform(sigma, sigma_lims)    
    lp_fs = sum([log_gaussian(freqs[i], prior_freq[i]) for i in range(N_freq) ])
    lp = lp_invSR + lp_sigma + lp_fs
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_whitenoise([depth_invSR, invSR], data, fs, sigma, interpolator=interpolator)


def log_posterior_rednoise_EP(params, depth_invSR, data, prior_freq, interpolator=interpolate_CubicSpline, invSR_lims=[0,2],
                             sigma_lims=[1e-4, 2]):
    """ same as log_posterior_whitenoise_EP but with rednoise
    """
    sigma, rho = params[:2]
    N_freq = prior_freq.shape[0]
    freqs = params[2:2+N_freq]
    fs = np.hstack([freqs[0] + freqs[1:], [freqs[2]-freqs[5], freqs[4]-freqs[3] , freqs[4]-freqs[2], freqs[3]-freqs[5], freqs[3]-freqs[2]]])
    invSR = params[2+N_freq:]
    
    
    lp_invSR = log_uniform(invSR, invSR_lims)
    lp_sigma = log_loguniform(sigma, sigma_lims)    
    lp_fs = sum([log_gaussian(freqs[i], prior_freq[i]) for i in range(N_freq) ])
    lp = lp_invSR + lp_sigma + lp_fs
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + log_likelihood_rednoise([depth_invSR, invSR], data, fs, [sigma, rho], interpolator=interpolator)
