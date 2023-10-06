'''
This script defines a Gaussian process (GP) classification model and associated utility functions.
It first imports required libraries, including PyTorch, GPyTorch, NumPy, and Matplotlib.
It then defines a custom Bernoulli likelihood class for use in the GP, along with two other classes to perform feature scaling.

Next, the script defines a function for preparing input data for the GP classification model, including scaling and converting
the labels to the range [0,1]. It then defines the GP classification model itself, along with functions for training and
evaluating it. Additionally, a function is provided for creating an evaluation grid for the model and another for plotting
the model's classification results.

Finally, the script defines a function for finding the input values that maximize the differential entropy of the GP's output.
This is a measure of the "uncertainty" of the model's predictions, with higher entropy indicating greater uncertainty.

It includes the following classes and functions:

CustomBernoulliLikelihood: a custom likelihood class used for the GP, which is based on the Bernoulli distribution and includes two hyperparameters: psi_gamma (g) and psi_lambda (l)

logFreq and logContrast: two classes used to transform the input data (frequency and contrast) logarithmically

prepare_data: a function that preprocesses the input data, including separating the features and labels, applying logarithmic transformations, scaling the features, and converting the labels to values between 0 and 1 if they are -1 and 1

GPClassificationModel: a class that represents the GP classification model with a constant mean and a kernel consisting of a linear kernel and an RBF kernel

train_gp_classification_model: a function that trains the GP model using the provided training inputs, labels, and likelihood using the Adam optimizer and the marginal log likelihood as the loss function

plot_classification_results: a function that plots the classification results of the GP model, including the evaluation grid, the predicted probabilities, and the original data points

find_best_entropy: a function that finds the input values that maximize the differential entropy of the GP's output, which is a measure of the "uncertainty" of the model's predictions, with higher entropy indicating greater uncertainty.

random_samples_from_data: Takes in data, and some other params to label the data, and chooses n points from this dataset

simulate_labeling: Labels points using a cubic spline, and psi_sigma, psi_gamma, psi_lambda parameters. See function for detailed comments

create_cubic_spline: Pass in a curve, get a cubic spline

get_data_bounds: returns the min and max for x and y axis

create_evaluation_grid: takes in min and max values for both axis, as well as the size of the grid, and returns a grid

evaluate_posterior_mean: evaluates the GP model (the mean) on some data points

transform_dataset: transforms the passed in dataset (as I'm writing this, it performs the identity transformation (so nothing))
'''
import sys

import torch
import gpytorch
from IPython.core.display_functions import clear_output
from gpytorch.distributions import base_distributions
import warnings
import numpy as np
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, LinearKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.models import ApproximateGP
from gpytorch.means import ZeroMean
from gpytorch.means import ConstantMean
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from gpytorch.likelihoods import _OneDimensionalLikelihood
from gpytorch.functions import inv_matmul, log_normal_cdf
from torch.distributions import Normal, Bernoulli
import torch.distributions as dist	
from torch.nn import functional as F
from scipy.interpolate import CubicSpline
from scipy.special import erf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import qmc
from pathlib import Path
from PIL import Image
import os
import time
import json


class CustomBernoulliLikelihood(_OneDimensionalLikelihood):
    """
    Bernoulli likelihood object with two hyperparameters used to fit the GP model to the training data
    """

    def __init__(self, g, l):
        """
        :param g: psi_gamma - the percentage of guesses you get right when random guessing
        :param l: psi_lambda - the percetnage of guesses you get wrong when you know what you're doing
        """
        super().__init__()
        if not (0 <= g <= 1):
            raise ValueError("g must be in [0, 1]")
        if not (0 <= l <= 1):
            raise ValueError("l must be in [0, 1]")
        self.g = torch.tensor(g)
        self.l = torch.tensor(l)
        self.quadrature = GaussHermiteQuadrature1D()

    def forward(self, function_samples, **kwargs):
        output_probs = Normal(0, 1).cdf(function_samples)
        output_probs = output_probs.mul(1 - self.l - self.g).add(self.g)
        return Bernoulli(probs=output_probs)

    def marginal(self, function_dist, **kwargs):
        mean = function_dist.mean
        var = function_dist.variance
        link = mean.div(torch.sqrt(1 + var))
        output_probs = Normal(0, 1).cdf(link)
        output_probs = output_probs.mul(1 - self.l - self.g).add(self.g)
        return Bernoulli(probs=output_probs)
    
    # default _OneDimensionalLikelihood expected_log_prob used


class CustomMeanModule(gpytorch.means.Mean):
    def __init__(self, models_and_likelihoods, scale_factor=1.0, gaussian_lengthscale=None):
        '''
        models_and_likelihoods: a list of tuples of this form - [(model1, likelihood1), (model2,likelihood2),...]
        '''
        super().__init__()
        self.models_and_likelihoods = models_and_likelihoods
        self.scale_factor = scale_factor
        self.gaussian_lengthscale = gaussian_lengthscale

    def forward(self, x):

        # Compute the latent distribution
        total_mean = 0
        n = 0
        with torch.no_grad():
            for gp_model, likelihood in self.models_and_likelihoods:
                n += 1
                gp_model.eval()
                likelihood.eval()
                latent_pred = gp_model(x)
                # posterior_pred = self.likelihood(latent_pred)
                total_mean += latent_pred.mean
        # Return the scaled posterior mean
        avg = (total_mean / n)
        if self.gaussian_lengthscale is None:
            return self.scale_factor * avg
        else:
            return self.scale_factor * (avg - (
                    avg * np.exp(-(1 / (self.gaussian_lengthscale * self.gaussian_lengthscale)) * avg * avg)))


class logFreq:
    """
    A class that defines a logarithmic frequency transformation.
    """

    def __init__(self):
        self.n = 0.125  # the smallest raw freq we expect

    def forward(self, data):
        return np.log2(data / self.n)

    def inverse(self, transformed_data):
        return self.n * np.power(2.0, transformed_data)


class logContrast:
    """
    A class that defines a invert-logarithmic contrast transformation.
    """

    def __init__(self):
        self.n = 1  # the smallest contrast sensitivity we expect (1/contrast is in the range [1, inf)

    def forward(self, data):
        return -1 * np.log10(data * self.n)  # the '-' inverts the data!!!

    def inverse(self, transformed_data):
        return self.n * np.power(10.0, -1 * transformed_data)


class MixtureNormalPrior(gpytorch.priors.Prior):	
    '''
    Prior enforced using two normal distributions with a specified weight for each distribution
    '''
    def __init__(self, mean1, std1, mean2, std2, weight1):	
        super().__init__(validate_args=False)	
        self.mean1 = mean1	
        self.std1 = std1	
        self.mean2 = mean2	
        self.std2 = std2	
        self.weight1 = weight1	
    def log_prob(self, x):
        prob1 = self.weight1 * torch.distributions.Normal(self.mean1, self.std1).log_prob(x)	
        prob2 = (1 - self.weight1) * torch.distributions.Normal(self.mean2, self.std2).log_prob(x)	
        return torch.logsumexp(torch.stack([prob1, prob2]), dim=0)

		
class GPClassificationModel(gpytorch.models.ApproximateGP):	

    """	
    A class that represents a Gaussian process classification model with a constant mean and a kernel consisting of a linear kernel and a RBF kernel.	
    """	

    def __init__(self, train_x, mean_module, 	
                linear_variance_prior=gpytorch.priors.SmoothedBoxPrior(.1, 200, sigma=1),	
                rbf_outputscale_prior=gpytorch.priors.SmoothedBoxPrior(.1, 150, sigma=1),	
                rbf_lengthscale_prior=MixtureNormalPrior(mean1=0.3, std1=0.05, mean2=0.215, std2=0.015, weight1=0.10),	
                min_lengthscale=None):	
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(train_x.size(0))	
        variational_strategy = gpytorch.variational.VariationalStrategy(	
            self, train_x, variational_distribution, learn_inducing_locations=False	
        )	
        super(GPClassificationModel, self).__init__(variational_strategy)	
        self.mean_module = mean_module	
        	
        # Linear kernel with constraints and priors	
        self.linear_kernel = gpytorch.kernels.LinearKernel(active_dims=[1])
        linear_variance_constraint = gpytorch.constraints.GreaterThan(.1)	
        self.linear_kernel.variance_constraint = linear_variance_constraint	
        self.linear_kernel.register_prior("variance_prior", linear_variance_prior, "variance")	

        # RBF kernel with constraints and priors	
        if min_lengthscale is not None:	
            RBF_lengthscale_constraint = gpytorch.constraints.GreaterThan(min_lengthscale)	
        self.rbf_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=[0], lengthscale_constraint=RBF_lengthscale_constraint))	
        self.rbf_kernel.base_kernel.register_prior("lengthscale_prior", rbf_lengthscale_prior, "lengthscale")	

        # Set constraints and priors for the outputscale of rbf_kernel	
        rbf_outputscale_constraint = gpytorch.constraints.GreaterThan(.1)	
        self.rbf_kernel.outputscale_constraint = rbf_outputscale_constraint	
        self.rbf_kernel.register_prior("outputscale_prior", rbf_outputscale_prior, "outputscale")

        # Combining the kernels	
        self.covar_module = self.linear_kernel + self.rbf_kernel	

    def forward(self, x):	
        mean_x = self.mean_module(x)	
        covar_x = self.covar_module(x)	
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)	
        return latent_pred


def train_gp_classification_model(model, likelihood, Xt, yt, beta=1, training_iterations=2000, lr=0.1,
                                  visualize_loss=False, progress_bar=True):
    """
    A function that trains the GP model using the provided training inputs, labels, and likelihood. It uses the Adam
    optimizer and the marginal log likelihood as the loss function.
    :param model: GP model
    :param likelihood: Likelihood (i.e. bernoulli, or custom bernoulli)
    :param Xt: Training data (shape?)
    :param yt: Labels (shape?)
    :param training_iterations: how many iterations you want to optimize the hyperparams
    :param lr: learning rate
    :param beta: Regularizer. Smaller beta is more regularized.
    :param visualize_loss: Debugging mostly. Plots loss inline. Default False.
    """

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # "Loss" for GPs - the marginal log likelihood
    # num_data refers to the number of training datapoints
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, yt.numel(), beta=beta)

    # Initialize loss history
    loss_history = []

    iterator = range(training_iterations)
    if progress_bar:
        iterator = tqdm(range(training_iterations), desc="Training progress", ncols=100)

    for i in iterator:
        # Zero backpropped gradients from previous iteration
        optimizer.zero_grad()
        # Get predictive output
        output = model(Xt)
        # Calc loss and backprop gradients
        loss = -mll(output, yt)
        loss.backward()
        loss_history.append(loss.item())
        optimizer.step()

        # Update progress bar and plot the loss if visualize_loss is True
        if visualize_loss and (i + 1) % 10 == 0:
            clear_output(wait=True)
            plt.plot(loss_history, label="Loss")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()


def getQcsfRMSE(xx, cs, peakSensitivity, peakFrequency, logBandwidth, delta, qcsf):
    """
    Args:
        xx (numpy.ndarray): Mesh grid X coordinate.
        cs (scipy.interpolate.CubicSpline object): ground truth CSF.
        peakSensitivity (float): sensitivity greatest value on truncated log-parabola
        over all frequencies
        peakFrequency (float): log2(CPD) frequency value of peakSensitivity
        logBandwidth (float): log2(CPD) full width at half max
        delta (float): sensitivity difference between truncated line and peakSensitivity
        qcsf (function): qcsf predicting function

    Returns:
        Root Mean Square Error (RMSE) calculated on each for each well-defined frequency value
        on our mesh grid. Returns in units of log_10(contast sensitivity).

    """
    xmedium = xx[0, :]

    # Get Frequency defined on same values as xmedium
    frequency = logFreq().inverse(xmedium)

    # Get csf Predictions
    ymedium = qcsf(peakSensitivity, peakFrequency, logBandwidth, delta, frequency).reshape(-1)

    # Calculate Ground truth posterior using spline.
    discrete_x2 = cs(xmedium)

    # Calculate RMSE between qCSF prediction, ground truth where ground truth well defined (>0).
    rmse = np.sqrt(np.mean((ymedium[discrete_x2 > 0] - discrete_x2[discrete_x2 > 0]) ** 2))
    return rmse


def getRMSE(xx, yy, zz, level, cs):
    """
    Args:
        xx (numpy.ndarray): Mesh grid X coordinate.
        yy (numpy.ndarray): Mesh grid Y coordinate.
        zz (numpy.ndarray): evaluate_posterior_mean(model, likelihood, grid_transformed) reshaped to xx.shap
        level: inflection point on psychometric curve (point of interest) (1-psi_lambda+psi_gamma)/2.
        cs: scipy.interpolate.CubicSpline object of ground truth CSF.
    """
    # zzmin is an array of square of distance to level, which we can use to find our level prediction from the PP.
    zzmin = (zz - level) ** 2
    # Get y indices of predictive posterior closest to level at each xindex.
    yindex = np.int64(np.argmin(zzmin, axis=0))
    yplot = yy[:, 0]
    # Get associated x values
    xmedium = xx[0, :]
    # Create array containing raw contrast values of level in predictive posterior.
    ymedium = yplot[yindex[:]]
    # Calculate Ground truth posterior using spline.
    discrete_x2 = cs(xmedium)
    # Calculate RMSE between predictive posterior, ground truth where ground truth well defined (>0).
    rmse = np.sqrt(np.mean((ymedium[discrete_x2 > 0] - discrete_x2[discrete_x2 > 0]) ** 2))
    return rmse


def find_best_entropy(model, likelihood, X_eval, xx):
    """
    A function that computes the differential entropy on the evaluation grid, finds the indices of the maximum
    differential entropy, and computes the corresponding coordinates.
    """
    H = lambda x: -x * torch.log2(x) - (1 - x) * torch.log2(1 - x)  # the likelihood used
    C = torch.sqrt(3.1415 * torch.log(torch.tensor(2.)) * 0.5)

    # Compute differential entropy on evaluation grid
    y_eval = likelihood(model(X_eval)).mean.detach()
    sd = likelihood(model(X_eval)).variance.detach() ** 0.5
    Denom1 = (sd ** 2 + 1) ** 0.5
    first_pre = torch.distributions.Normal(0., 1.).cdf(torch.div(y_eval, Denom1))
    first = H(first_pre)
    Nom = torch.ones_like(y_eval) * C
    Denom2 = sd ** 2 + Nom ** 2
    second = torch.div(Nom, Denom2 ** 0.5) * torch.exp(torch.div(y_eval ** 2, -2 * Denom2))
    de = first - second
    # de has the entropies for each point. It is the same shape as X_eval

    # Find indices of maximum differential entropy
    sorted_indices = [x for x in range(de.shape[0])]
    sorted_indices.sort(key=lambda x: de[x], reverse=True)

    # Return the entropies grid (so it can be plotted if you want)
    entropies = de.reshape(xx.shape)

    return entropies, sorted_indices


def random_samples_from_data(data, cs, psi_gamma, psi_lambda, n, replacement=False, inds=None, strict=False, sigmoid_type='logistic', psi_sigma=.08):
    """
      Draws n random data points
      Labels produced via cs, psi_sigma, psi_gamma, psi_lambda
      :param data: mx2 matrix - the data to draw points from and label them
      :param cs: a CubicSpline
      :param psi_sigma: spread parameter
      :param sigmoid_type: which shape sigmoid to use
      :param psi_gamma: success percentage when guessing at random
      :param psi_lambda: (lambda) lapse rate, i.e. the error percentage when you should get it right
      :param n: number of data points to generate
      :param replacement: set to True if you want to sample with replacement
      :param inds: Optional, the specific indices of the grid you want to sample
      :return: X, a nx2 matrix, and y, a length n vector
    """
    m = data.shape[0]

    if replacement is False and m < n:
        n = m
        warnings.warn("You are attempting to sample more points than are in the grid." +
                      "This is likely a mistake", DeprecationWarning)

    valid_indices = np.arange(0, m)
    if inds is None:
        chosen_indices = np.random.choice(valid_indices, size=n, replace=replacement)
    else:
        chosen_indices = inds

    x1 = data[chosen_indices, 0]
    x2 = data[chosen_indices, 1]

    y = simulate_labeling(x1, x2, cs, psi_gamma, psi_lambda, strict=strict, sigmoid_type=sigmoid_type, psi_sigma=psi_sigma)	

    # return the generated values and labels
    X = np.vstack((x1, x2)).T
    return X, y


def halton_samples_from_data(xx, yy, cs, psi_gamma, psi_lambda, n, strict=False, sigmoid_type='logistic', psi_sigma=.08):	
    l_bounds = [0, 0]
    u_bounds = [xx.shape[1], xx.shape[0]]

    sampler = qmc.Halton(d=2, scramble=False)

    samples = sampler.integers(l_bounds, u_bounds=u_bounds, n=n)
    samples = np.array(samples)

    x1_indices = samples[:, 0]
    x2_indices = samples[:, 1]

    x1 = xx[0, x1_indices]
    x2 = yy[x2_indices, 0]

    X = np.vstack((x1, x2)).T
    y = simulate_labeling(x1, x2, cs, psi_gamma, psi_lambda, strict=strict, sigmoid_type=sigmoid_type, psi_sigma=psi_sigma)

    return X, y


def simulate_labeling(x1, x2, cs, psi_gamma, psi_lambda, strict=False, sigmoid_type='logistic', psi_sigma=.08):
    """
    Labels the points according to the specified sigmoid and spline
	:param x1: vector - First features	
    :param x2: vector - Second features	
    :param cs: a CubicSpline	
    :param psi_gamma: success percentage when guessing at random	
    :param psi_lambda: (lambda) lapse rate, i.e. the error percentage when you should get it right	
    :param strict: if True, labels are deterministically set based on the threshold level	
    :param mode: 'logistic' or 'probit' to choose the function type	
    :return: the labels	
    """

    probabilities = simulate_sigmoid(x1, x2, cs, psi_gamma, psi_lambda, sigmoid_type=sigmoid_type, psi_sigma=psi_sigma)
    random_values = np.random.uniform(size=probabilities.shape[0])
    level = (1 - psi_lambda + psi_gamma) / 2
    if strict:
        y = (level < probabilities).astype(int)
    else:
        y = (random_values < probabilities).astype(int)

    return y


def simulate_sigmoid(x1, x2, cs, psi_gamma, psi_lambda, sigmoid_type='logistic', psi_sigma=.08):	
    """	
    See comment below for details	
    :param x1: vector - First features	
    :param x2: vector - Second features	
    :param cs: a CubicSpline	
    :param psi_gamma: success percentage when guessing at random	
    :param psi_lambda: (lambda) lapse rate, i.e. the error percentage when you should get it right	
    :param sigmoid_type: 'logistic' or 'normal_cdf' to choose the function type	
    :return: the labels	
    """

    '''
      A classic sigmoid produces outputs between 0 and 1. We want values between psi_gamma and 1 - psi_lambda.
      To do this, we shrink everything by a factor of 1 - psi_lambda - psi_gamma, then add psi_gamma at the end. You
      can check this gives us the correct output.
      We use psi_sigma to affect the rate at which moving to the right/left increases/decreases the output of the sigmoid.
      A smaller psi_sigma means the sigmoid is "steeper" or "squished horizontally" --- larger rate of increase per horizontal step.
      If a y value is above the curve, we want to move it to the left on the sigmoid (-1 labels).
      If a y value is below the curve, we want to move it to the right on the sigmoid (+1 labels).
      We do this by inputting true_y - y as the input to the sigmoid. You can check for yourself that
      this gives us the desired properties.
      We use these values as probabilites, and generate random values between 0 and 1, and the odds that a value
      is < a probability is that probability itself. So we just see if it's less than the probability
      '''
    c = 1 - psi_lambda - psi_gamma	
    true_y = cs(x1)	
    warnings.filterwarnings('ignore')	
    	
    if sigmoid_type == 'logistic':	
        probabilities = c / (1 + np.exp(-1.0/psi_sigma * (true_y - x2))) + psi_gamma	
    elif sigmoid_type == 'normal_cdf':	
        probabilities = c * 0.5 * (1 + erf((true_y - x2) / (psi_sigma * np.sqrt(2)))) + psi_gamma	
    else:	
        raise ValueError("Invalid sigmoid_type. Choose either 'logistic' or 'normal_cdf'.")	
    if len(probabilities.shape) == 0:	
        return np.array([probabilities])	
    return np.array(probabilities)


def create_cubic_spline(curve):
    """
  This method creates a cubic spline to approximate the given curve.
  :param curve: A nx2 numpy matrix. First column is x values. Second column is y values.
  :return: The cubic spline.
  """
    x = curve[:, 0]
    y = curve[:, 1]

    cs = CubicSpline(x, y)
    return cs


def get_data_bounds(data):
    """
  This method returns the left, right, upper, lower bounds for a data set
  :param data: A nx2 numpy matrix. First column is x values. Second column is y values.
  :return: The 4 bounds (left, right, bottom, top)
  """

    x_min = np.min(data[:, 0])
    x_max = np.max(data[:, 0])
    y_min = np.min(data[:, 1])
    y_max = np.max(data[:, 1])

    return x_min, x_max, y_min, y_max

    # return the generated values and labels
    return raw_x1, raw_x2, y


def create_evaluation_grid_resolution(x_min, x_max, y_min, y_max, x_resolution=15, y_resolution=30):
    """
    Creates an evaluation grid covering the specified bounds
    :param x_min: min of x axis
    :param x_max: max of x axis
    :param y_min: min of y axis
    :param y_max: max of y axis
    :param x_resolution: Number of grid columns per spatial frequency octave (default 15)
    :param y_resolution: Number of grid rows per contrast decade (default 30)
    :return: The evaluation data, the x grid points (mesh), the y grid points (mesh)
    """
    x_side_length = int((x_max - x_min) * x_resolution + 1)
    y_side_length = int((y_max - y_min) * y_resolution + 1)

    X_eval, xx, yy = create_evaluation_grid(x_min, x_max, y_min, y_max, x_side_length, y_side_length)

    return X_eval, xx, yy, x_side_length, y_side_length


def create_evaluation_grid(x_min, x_max, y_min, y_max, x_side_length, y_side_length):
    """
    Creates an evaluation grid covering the specified bounds
    :param x_min: min of x axis
    :param x_max: max of x axis
    :param y_min: min of y axis
    :param y_max: max of y axis
    :param side_length: If side length is 10, there will be 100 data points in the grid
    :return: The evaluation data, the x grid points (mesh), the y grid points (mesh)
    """

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, x_side_length), np.linspace(y_min, y_max, y_side_length))

    # creates a (side_length^2)x2 matrix -- the test data
    X_eval = np.vstack((xx.reshape(-1), yy.reshape(-1))).T

    return X_eval, xx, yy


def evaluate_posterior_mean(model, likelihood, X):
    """
      Evaluates the GP model on X
      :param model: GP model
      :param likelihood: likelihood (bernoulli)
      :param X: nx2 matrix, the data
      :return: the posterior mean evaluated at X
      """

    # Go into eval mode
    model.eval()
    likelihood.eval()

    # Get likelihood
    with torch.no_grad():
        # Get classification predictions
        observed_pred = likelihood(model(X))

        predicted_values = observed_pred.mean.numpy()

    return predicted_values


def transform_dataset(data, phi=None):
    """
   Transforms the data (logged data) to be used by the GP.
   :param data: nx2 matrix - your data
   :param phi: the function to transform your data
   :return: the transformed data
   """

    # don't modify the original data
    X = data.copy()

    if phi is not None:
        X = phi(X)

    # Convert the data to PyTorch tensors
    Xt = torch.from_numpy(X).float()

    return Xt


def scale_data_within_range(data, ran, x_min, x_max, y_min, y_max):
    """
    Scales your data set within the given range.

    IMPORTANT NOTE: make sure the bounds (x_min, ... y_max) are relatively tight. Else, you're
    effectively shrinking your range. For example, if you generate data between 0 and 4, but you pass
    in 0 and 8 for the bounds, you will really be scaling your data into the first half
    of whatever range you pass in

    :param data: nx2 numpy matrix
    :param ran: the range you want to scale into (a, b)
    :param x_min: min x value
    :param x_max: max x value
    :param y_min: min y value
    :param y_max: max y value
    :return: transformed data
    """

    a, b = ran

    X = data.copy()

    # get the percent of range that the data takes up
    X[:, 0] = (X[:, 0] - x_min) / (x_max - x_min)
    X[:, 1] = (X[:, 1] - y_min) / (y_max - y_min)

    # get corresponding value in range (using the percentages)
    X = (X * (b - a)) + a

    return X


def get_mean_module(name, params):
    if name == 'constant_mean':
        return ConstantMean()
    elif name == 'prior_gp_mean':
        models_and_likelihoods = []
        for param_dict in params:
            state_dict = torch.load(param_dict['state_dict_path'])
            Xt = torch.load(param_dict['Xt_path'])
            scale_factor = param_dict['scale_factor']
            gaussian_lengthscale = param_dict['gaussian_lengthscale']
            min_lengthscale = param_dict['min_lengthscale']
            psi_gamma = param_dict['psi_gamma']
            psi_lambda = param_dict['psi_lambda']
            prior_model = GPClassificationModel(Xt, mean_module=ConstantMean(), min_lengthscale=min_lengthscale)
            prior_model.load_state_dict(state_dict)
            prior_likelihood = CustomBernoulliLikelihood(psi_gamma, psi_lambda)

            models_and_likelihoods.append((prior_model, prior_likelihood))

        prior_mean = CustomMeanModule(models_and_likelihoods, scale_factor=scale_factor,
                                      gaussian_lengthscale=gaussian_lengthscale)

        return prior_mean


def sample_and_train_gp(
        cs,
        grid,
        xx,
        yy,
        mean_module_name='constant_mean',
        mean_module_params=None,
	    sigmoid_type = 'logistic',	
        psi_sigma = .08,	
        psi_gamma=.01,	
        psi_lambda=.01,
        lr=.125,
        num_initial_training_iters=500,
        num_new_points_training_iters=50,
        num_new_points=100,
        beta_for_regularization=1,
        train_on_all_points_after_sampling=False,
        train_on_all_points_iters=1500,
        phi=None,
        print_training_hyperparameters=False,
        print_training_iters=True,
        progress_bar=True,
        min_lengthscale=None,
        random_seed=None,
        calculate_rmse=True,
        calculate_entropy=True,
        calculate_posterior=True,
        initial_Xs=None,
        initial_ys=None,
        sampling_strategy=None,
        strict_labeling=False,
        num_ghost_points=0,
        timepoints=None
):
    '''
    initial_Xs is a nx2 numpy array
    initial_ys is a n numpy array
    '''

    # setup timing
    times = []
    if timepoints:
        timepoints_set = set(timepoints)

    # set random seed to get reproducible results
    if random_seed is not None:
        np.random.seed(random_seed)

    # check sampling strategy
    if sampling_strategy == 'active':
        AL_FLAG = True
    elif sampling_strategy == 'random':
        AL_FLAG = False
    else:
        raise Exception('Invalid sampling strategy given. Valid options are "active" or "random".')

    # define the transformation function (usually it does scaling)
    def f(d):
        if phi is not None:
            return phi(d)
        else:
            return d

    # transform the grid
    grid_transformed = transform_dataset(grid, phi=f)

    # We need to loop through and train on each slice of the initial points to calculate rmses or posteriors
    num_initial_points = len(initial_Xs) - num_ghost_points
    num_total_points = num_initial_points + num_new_points

    # make sure you didn't mess up
    if len(initial_Xs) != len(initial_ys):
        raise Exception('Initial Xs and ys do not have the same length.')

    # create copies of the data (so it doesn't affect things outside the function)
    X = np.copy(initial_Xs)
    y = np.copy(initial_ys)

    # store the data during training
    rmse_list = []
    posterior_list = []
    entropy_list = []

    # to calculate data on initial points, need to go through each slice separately
    if calculate_rmse or calculate_posterior:

        # i's value will be the index of the non-ghost points
        # for example, if 2 ghost and 10 halton, i=2,3,4,...,11
        for i in range(num_ghost_points, len(initial_Xs)):
            if timepoints:
                startTime = time.perf_counter()

            if print_training_iters:
                print(f'iteration {i - num_ghost_points + 1}/{num_total_points}')

            Xt = transform_dataset(X[:i + 1, :], phi=f)  # notice the X[:i+1, :]
            yt = torch.from_numpy(y[:i + 1]).float()

            model = GPClassificationModel(Xt,
                                          mean_module=get_mean_module(mean_module_name, mean_module_params),
                                          min_lengthscale=min_lengthscale)
            likelihood = CustomBernoulliLikelihood(psi_gamma, psi_lambda)
            train_gp_classification_model(model,
                                          likelihood,
                                          Xt,
                                          yt,
                                          beta=beta_for_regularization,
                                          training_iterations=num_initial_training_iters,
                                          lr=lr,
                                          progress_bar=progress_bar)
            
            if print_training_hyperparameters:	
                print("kernel 0 variance:" , model.covar_module.kernels[0].variance.item())	
                print("kernel 1 outputscale:" , model.covar_module.kernels[1].outputscale.item())	
                print("kernel 1 lengthscale:" , model.covar_module.kernels[1].base_kernel.lengthscale.item())

            if calculate_rmse:
                Z = evaluate_posterior_mean(model, likelihood, grid_transformed)
                zz = Z.reshape(xx.shape)
                level = (1 - psi_lambda + psi_gamma) / 2
                rmse_list.append(getRMSE(xx, yy, zz, level, cs))

            if calculate_posterior:
                posterior_list.append((model, likelihood))

            curr_num_pts = i - num_ghost_points + 1
            if timepoints and curr_num_pts in timepoints_set:
                times.append((curr_num_pts, time.perf_counter() - startTime))

    startTime = time.perf_counter()

    # next we proceed as usual, training on all points for initial_training_iters
    Xt = transform_dataset(X, phi=f)
    yt = torch.from_numpy(y).float()
    model = GPClassificationModel(Xt,
                                  mean_module=get_mean_module(mean_module_name, mean_module_params),
                                  min_lengthscale=min_lengthscale)
    likelihood = CustomBernoulliLikelihood(psi_gamma, psi_lambda)
    train_gp_classification_model(model,
                                  likelihood,
                                  Xt,
                                  yt,
                                  beta=beta_for_regularization,
                                  training_iterations=num_initial_training_iters,
                                  lr=lr,
                                  progress_bar=progress_bar)
    
    if print_training_hyperparameters:	
        print("kernel 0 variance:" , model.covar_module.kernels[0].variance.item())	
        print("kernel 1 outputscale:" , model.covar_module.kernels[1].outputscale.item())	
        print("kernel 1 lengthscale:" , model.covar_module.kernels[1].base_kernel.lengthscale.item())

    # main loop, repeatedly grabs the next point according to sampling_strategy
    for i in range(num_new_points):

        if print_training_iters:
            print(f'iteration {i + num_initial_points + 1}/{num_total_points}')

        # grab the next point
        if AL_FLAG:
            entropy_grid, best_indices = find_best_entropy(model, likelihood, grid_transformed, xx)
            if calculate_entropy:
                entropy_list.append(entropy_grid)
            new_x = grid[best_indices[0], :]
            new_y = simulate_labeling(new_x[0], new_x[1], cs, psi_gamma, psi_lambda, strict=strict_labeling, sigmoid_type=sigmoid_type, psi_sigma = psi_sigma)	
        else:	
            new_x, new_y = random_samples_from_data(grid, cs, psi_gamma, psi_lambda, 1, strict=strict_labeling, sigmoid_type=sigmoid_type, psi_sigma = psi_sigma)

        # append it to the dataset - MUST BE APPENDED! DON'T PUT ANYWHERE ELSE (or copying over params will fail)
        X_new = np.vstack((X, new_x))
        y_new = np.hstack((y, new_y))

        # transform dataset
        Xt = transform_dataset(X_new, phi=f)
        yt = torch.from_numpy(y_new).float()

        # store the old model state
        old_inducing_points = model.variational_strategy.inducing_points
        old_mean = model.variational_strategy._variational_distribution.variational_mean
        old_covar = model.variational_strategy._variational_distribution.chol_variational_covar
        old_covar_state_dict = model.covar_module.state_dict()
        old_mean_state_dict = model.mean_module.state_dict()
        n = old_inducing_points.shape[0]

        # create new model
        model_new = GPClassificationModel(Xt,
                                          mean_module=get_mean_module(mean_module_name, mean_module_params),
                                          min_lengthscale=min_lengthscale)

        # Copy over hyper parameters
        model_new.covar_module.load_state_dict(old_covar_state_dict)
        model_new.mean_module.load_state_dict(old_mean_state_dict)

        # Copy over variational parameters
        with torch.no_grad():
            # IMPORTANT: we need to make sure the transformation is not local, or else copying over the
            # old points will not be consistent. For example, we can't scale wrt min and max of our data,
            # it has to be wrt a global min/max value (such as the bounds of the grid)
            model_new.variational_strategy.inducing_points[:n, :] = old_inducing_points
            model_new.variational_strategy._variational_distribution.variational_mean[:n] = old_mean
            model_new.variational_strategy._variational_distribution.chol_variational_covar[:n, :n] = old_covar

        # train the model for a few iterations
        try:
            train_gp_classification_model(model_new,
                                          likelihood,
                                          Xt,
                                          yt,
                                          beta=beta_for_regularization,
                                          training_iterations=num_new_points_training_iters,
                                          lr=lr,
                                          progress_bar=progress_bar)
            model = model_new
            X = X_new
            y = y_new

            if print_training_hyperparameters:	
                print("kernel 0 variance:" , model.covar_module.kernels[0].variance.item())	
                print("kernel 1 outputscale:" , model.covar_module.kernels[1].outputscale.item())	
                print("kernel 1 lengthscale:" , model.covar_module.kernels[1].base_kernel.lengthscale.item())

        except:
            print("resetting model hypers")
            model_from_scratch = GPClassificationModel(Xt,
                                                       mean_module=get_mean_module(mean_module_name,
                                                                                   mean_module_params),
                                                       min_lengthscale=min_lengthscale)
            train_gp_classification_model(model_from_scratch,
                                          likelihood,
                                          Xt,
                                          yt,
                                          beta=beta_for_regularization,
                                          training_iterations=1500,
                                          lr=lr,
                                          progress_bar=True)
            model = model_from_scratch
            X = X_new
            y = y_new
            if print_training_hyperparameters:	
                print("kernel 0 variance:" , model.covar_module.kernels[0].variance)	
                print("kernel 1 outputscale:" , model.covar_module.kernels[1].outputscale)	
                print("kernel 1 lengthscale:" , model.covar_module.kernels[1].base_kernel.lengthscale)
        if calculate_rmse:
            Z = evaluate_posterior_mean(model, likelihood, grid_transformed)
            zz = Z.reshape(xx.shape)
            level = (1 - psi_lambda + psi_gamma) / 2
            rmse_list.append(getRMSE(xx, yy, zz, level, cs))

        if calculate_posterior:
            posterior_list.append((model, likelihood))

        curr_num_pts = num_initial_points + i + 1
        if timepoints and curr_num_pts in timepoints_set:
            times.append((curr_num_pts, time.perf_counter() - startTime))

    # trains the model from scratch if set to true
    if train_on_all_points_after_sampling:
        Xt = transform_dataset(X, phi=f)
        yt = torch.from_numpy(y).float()
        model = GPClassificationModel(Xt,
                                      mean_module=get_mean_module(mean_module_name, mean_module_params),
                                      min_lengthscale=min_lengthscale)
        train_gp_classification_model(model,
                                      likelihood,
                                      Xt,
                                      yt,
                                      beta=beta_for_regularization,
                                      training_iterations=train_on_all_points_iters,
                                      lr=lr,
                                      progress_bar=progress_bar)

    # this prints out the learned length scale of the final model
    if print_training_hyperparameters:	
        print("kernel 0 variance:" , model.covar_module.kernels[0].variance.item())	
        print("kernel 1 outputscale:" , model.covar_module.kernels[1].outputscale.item())	
        print("kernel 1 lengthscale:" , model.covar_module.kernels[1].base_kernel.lengthscale.item())

    if AL_FLAG:
        return model, likelihood, X, y, rmse_list, entropy_list, posterior_list, times
    else:
        return model, likelihood, X, y, rmse_list, posterior_list, times


def create_and_save_plots(
        my_dict,
        path,
        title,
        start_index=0,
        zero_index=True,
        latent_color='purple',
        mean_color='turquoise',
        level=0.5,
        xticks_labels=np.array([1, 4, 16, 64]),
        yticks_labels=np.array([1, 10, 100, 1000])
):
    xx = my_dict['xx']
    yy = my_dict['yy']
    X = my_dict['X']
    y = my_dict['y']
    left = my_dict['left']
    right = my_dict['right']
    cs = my_dict['cs']
    psi_gamma = my_dict['psi_gamma']
    psi_lambda = my_dict['psi_lambda']
    x_min = my_dict['x_min']
    x_max = my_dict['x_max']
    y_min = my_dict['y_min']
    y_max = my_dict['y_max']
    xs = my_dict['xs']
    ys = my_dict['ys']
    grid = my_dict['grid']
    f = my_dict['f']
    posterior_list = my_dict['posterior_list']

    for i, (model, likelihood) in enumerate(posterior_list):

        plt.clf()

        beforeX = X[:i + start_index + 1, :]
        beforey = y[:i + start_index + 1]

        # transform evaluation grid so it can be used by GP
        grid_transformed = transform_dataset(grid, phi=f)

        # get the predictions on the eval grid
        Z = evaluate_posterior_mean(model, likelihood, grid_transformed)
        zz = Z.reshape(xx.shape)

        # plot the contour field
        resolution = 111
        plt.pcolormesh(xx, yy, zz, cmap='gist_gray', vmin=0, vmax=1)
        cbar = plt.colorbar()  # this needs to be called right after contourf
        cbar.set_ticks([0, .25, .5, .75, 1])
        cbar.set_ticklabels(['0', '.25', '.5', '.75', '1'])

        # plot the training data
        plt.scatter(beforeX[beforey == 1, 0].reshape(-1),
                    beforeX[beforey == 1, 1].reshape(-1), label='Success', marker='.', c='blue')
        plt.scatter(beforeX[beforey == 0, 0].reshape(-1),
                    beforeX[beforey == 0, 1].reshape(-1), label='Failure', marker='.', c='red')

        # plot the spline
        latent_x1 = np.linspace(left, right, 750)
        latent_x2 = cs(latent_x1)
        plt.plot(latent_x1, latent_x2, color=latent_color)

        # plot the level curve
        plt.contour(xx, yy, zz, levels=[level], colors=mean_color)

        # specify the tick marks here
        # _labels are the numbers you want to display
        # _values are the underlying values corresponding to these labels
        # in this case, the underlying values are log10 of the labels
        # _labels and _values must be the same length
        xticks_values = logFreq().forward(xticks_labels)

        yticks_values = logContrast().forward(1 / yticks_labels)

        plt.xticks(xticks_values, xticks_labels)
        plt.yticks(yticks_values, yticks_labels)

        # fit to the grid
        x_padding = (x_max - x_min) / (2 * (xs - 1))
        y_padding = (y_max - y_min) / (2 * (ys - 1))
        plt.xlim(x_min - x_padding, x_max + x_padding)
        plt.ylim(y_min - y_padding, y_max + y_padding)

        # title and axis labels
        plt.title(title, fontdict={'fontsize': 12})
        plt.xlabel('Spacial Frequency cycles/degree (log scale)', fontdict={'fontsize': 12})
        plt.ylabel('Contrast Sensitivity (log scale)', fontdict={'fontsize': 12})

        # save
        diff_index = 0
        if zero_index:
            diff_index = 1
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(path + f'{i + 1 - diff_index}-{title}')
        plt.close()


def create_gif(path):
    image_filenames = [file for file in os.listdir(path) if file.endswith('.png')]
    image_filenames = sorted(image_filenames, key=lambda fn: int(fn.split('-')[0]))

    images = []
    for fn in image_filenames:
        images.append(Image.open(path + fn))

    images[0].save(
        path + 'summary_gif.gif',
        save_all=True,
        append_images=images[1:],
        duration=500,
        loop=0,
    )


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_directory_exists(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json_from_file(path):
    with open(path, 'r') as file:
        jsonobj = json.load(file)
    return jsonobj
