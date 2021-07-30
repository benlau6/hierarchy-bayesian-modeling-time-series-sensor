import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import pymc3 as pm
import theano.tensor as tt
import arviz as az
from scipy.stats import norm
import warnings
import abc
from ..utils import add_subplot
from .data import gaussian
warnings.simplefilter(action="ignore", category=FutureWarning)


class BayesModel(metaclass=abc.ABCMeta):
    
    def __init__(self):
        self.model = pm.Model()
        
    @abc.abstractmethod
    def fit(self, X, y, num_samples=1000, num_burnin=2000):
        return NotImplemented
    
    def plot_trace(self):
        with self.model:
            pm.plot_trace(self.trace)
            plt.show()

    @abc.abstractmethod
    def plot_posterior_predictive(self):   
        return NotImplemented
    
    def get_params(self, params=None):
        stats = ['mean']#, 'sd']
        summary = az.summary(self.trace)
        if params:
            summ = summary[stats].T[params]
        else:
            summ = summary[stats].T[params]
        return summ
            
        
class DirchletProcessModel(BayesModel):
    
    def __init__(self, K=30):
        super().__init__()
        self.K = K
        
    def fit(self, y, num_samples=1000, num_burnin=2000):
        N = y.shape[0]
        
        def stick_breaking(beta):
            portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
            return beta * portion_remaining
    
        with self.model:
            alpha = pm.Gamma("alpha", 1.0, 1.0)
            beta = pm.Beta("beta", 1.0, alpha, shape=self.K)
            w = pm.Deterministic("w", stick_breaking(beta))

            tau = pm.Gamma("tau", 1.0, 1.0, shape=self.K)
            lambda_ = pm.Gamma("lambda_", 10.0, 1.0, shape=self.K)
            mu = pm.Normal("mu", 0, tau=lambda_ * tau, shape=self.K)
            y_obs = pm.NormalMixture("obs", w, mu, tau=lambda_ * tau, observed=y)
            
            self.trace = pm.sample(num_samples, tune=num_burnin, return_inferencedata=True)
    
    def plot_posterior_predictive(self):
        with self.model:
            self.ppc = pm.sample_posterior_predictive(self.trace, var_names=["mu", "tau", "obs"])
            
    def plot_num_cluster(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_w = np.arange(self.K) + 1

        ax.bar(plot_w - 0.5, self.trace.posterior.w.values.mean(0).mean(0), width=1.0, lw=0)

        ax.set_xlim(0.5, self.K)
        ax.set_xlabel("Component")

        ax.set_ylabel("Posterior expected mixture weight");
        plt.show()
        
        
class GaussianMixtureModel(BayesModel):
    
    def __init__(self, K=2):
        super().__init__()
        self.K = K
        
    def fit(self, y, num_samples=1000, num_burnin=2000):
        N = y.shape[0]
    
        with self.model:
            w = pm.Dirichlet("w", np.ones(self.K))

            mu = pm.Normal(
                "mu",
                np.zeros(self.K),
                1.0,
                shape=self.K,
                transform=pm.transforms.ordered,
                testval=[1, 2],
            )
            tau = pm.Gamma("tau", 1.0, 1.0, shape=self.K)

            y_obs = pm.NormalMixture("obs", w, mu, tau=tau, observed=y)
            
            self.trace = pm.sample(num_samples, tune=num_burnin, return_inferencedata=True)
            
    def _get_ppc(self):
        with self.model:
            self.ppc = pm.sample_posterior_predictive(self.trace, var_names=["mu", "tau", "obs"], keep_size=True)
    
    def plot_posterior_predictive(self):
        if not hasattr(self, 'ppc'):
            self._get_ppc()
        
        with self.model:
            self.trace.add_groups(posterior_predictive=self.ppc)
            az.plot_ppc(self.trace)
            
    def _get_dist(self):
        if not hasattr(self, 'ppc'):
            self._get_ppc()
        
        with self.model:
            mu_1, mu_2 = self.ppc['mu'].mean(0).mean(0)
            tau_1, tau_2 = self.ppc['tau'].mean(0).mean(0)
            self.dist_1 = gaussian(mu_1, np.sqrt(1/tau_1))   
            self.dist_2 = gaussian(mu_2, np.sqrt(1/tau_2))  
            
    def get_clustered_y(self, y):    
        if not hasattr(self, 'dist_1'):
            self._get_dist()
            
        idxes = []
        for yi in y:
            idx = abs(yi-self.dist_1.mu) > abs(yi-self.dist_2.mu)
            idxes.append(int(idx)+1)
        idxes = np.array(idxes)

        return idxes
        
    def plot_all(self, t, y, fig=None):
        if fig is None:
            fig = plt.figure(figsize=(18, 1))
        self.plot_cluster(t, y, fig)
        fig.tight_layout()
        
    def plot_cluster(self, t, y, fig=None):
        if fig is None:
            fig = plt.figure(figsize=(18, 1))
        ax = add_subplot()
        clusters = self.get_clustered_y(y)
        cluster1 = np.where(clusters==1, True, False)
        ax.scatter(x=t[cluster1], y=y[cluster1], color='b', label=str(self.dist_1))
        ax.scatter(x=t[~cluster1], y=y[~cluster1], color='r', label=str(self.dist_2))
        plt.legend()
        plt.ylabel('y')
        plt.xlabel('time')
        fig.tight_layout()
        
    def plot_cluster_dist(self, t, y, fig=None):
        if fig is None:
            fig = plt.figure(figsize=(18, 1))
        ax = add_subplot()
        clusters = self.get_clustered_y(y)
        cluster1 = np.where(clusters==1, True, False)
        ax.scatter(x=t[cluster1], y=y[cluster1], color='b', label=str(self.dist_1))
        ax.scatter(x=t[~cluster1], y=y[~cluster1], color='r', label=str(self.dist_2))
        ax.axhline(self.dist_1.mu, color='b')
        ax.axhline(self.dist_2.mu, color='r')
        ax.fill_between(x=t, y1=self.dist_1.mu-self.dist_1.sigma, y2=self.dist_1.mu+self.dist_1.sigma, color='b', alpha=0.3)
        ax.fill_between(x=t, y1=self.dist_2.mu-self.dist_2.sigma, y2=self.dist_2.mu+self.dist_2.sigma, color='r', alpha=0.3)
        plt.legend()
        plt.ylabel('y')
        plt.xlabel('time')
        fig.tight_layout()
        
        
class TMixtureModel(GaussianMixtureModel):
    
    def fit(self, y, num_samples=1000, num_burnin=2000):
        N = y.shape[0]
    
        with self.model:
            w = pm.Dirichlet("w", np.ones(self.K))

            mu = pm.Normal(
                "mu",
                np.zeros(self.K),
                1.0,
                shape=self.K,
                transform=pm.transforms.ordered,
                testval=[1, 2],
            )
            tau = pm.Gamma("tau", 1.0, 1.0, shape=self.K)
            
            nu = pm.Gamma('nu', alpha=2, beta=0.1, shape=self.K)

            components  = pm.StudentT.dist(nu=nu, mu=mu, lam=tau, shape=self.K)

            y_obs = pm.Mixture("obs", w, components, observed=y)
            
            self.trace = pm.sample(num_samples, tune=num_burnin, return_inferencedata=True)
            
    def get_clustered_y(self, y, threshold = 0.05):    
        if not hasattr(self, 'dist_1'):
            self._get_dist()
            
        prob1s = norm().sf(abs((y-self.dist_1.mu)/self.dist_1.sigma))
        prob2s = norm().sf(abs((y-self.dist_2.mu)/self.dist_2.sigma))

        clusters = []
        for p1, p2 in zip(prob1s, prob2s):
            max_ = max(p1, p2)
            if max_ < threshold:
                clusters.append(-1)
            elif max_ == p1:
                clusters.append(1)
            elif max_ == p2:
                clusters.append(2)
        clusters = np.array(clusters)

        return clusters
        
    def plot_all(self, t, y, fig=None):
        if fig is None:
            fig = plt.figure(figsize=(18, 1))
        self.plot_cluster(t, y, fig)
        fig.tight_layout()
        
    def plot_cluster(self, t, y, fig=None):
        if fig is None:
            fig = plt.figure(figsize=(18, 1))
        ax = add_subplot()
        clusters = self.get_clustered_y(y)
        cluster1 = np.where(clusters==1, True, False)
        cluster2 = np.where(clusters==2, True, False)
        outliers = np.where(clusters==-1, True, False)
        ax.scatter(x=t[cluster1], y=y[cluster1], color='b', label=str(self.dist_1))
        ax.scatter(x=t[cluster2], y=y[cluster2], color='g', label=str(self.dist_2))
        ax.scatter(x=t[outliers], y=y[outliers], color='r', label='outliers')
        plt.legend()
        plt.ylabel('y')
        plt.xlabel('time')
        fig.tight_layout()
        
    def plot_cluster_dist(self, t, y, fig=None):
        if fig is None:
            fig = plt.figure(figsize=(18, 1))
        ax = add_subplot()
        clusters = self.get_clustered_y(y)
        cluster1 = np.where(clusters==1, True, False)
        cluster2 = np.where(clusters==2, True, False)
        outliers = np.where(clusters==-1, True, False)
        ax.scatter(x=t[cluster1], y=y[cluster1], color='b', label=str(self.dist_1))
        ax.scatter(x=t[cluster2], y=y[cluster2], color='g', label=str(self.dist_2))
        ax.scatter(x=t[outliers], y=y[outliers], color='r', label='outliers')
        ax.axhline(self.dist_1.mu, color='b')
        ax.axhline(self.dist_2.mu, color='g')
        ax.fill_between(x=t, y1=self.dist_1.mu-self.dist_1.sigma, y2=self.dist_1.mu+self.dist_1.sigma, color='b', alpha=0.3)
        ax.fill_between(x=t, y1=self.dist_2.mu-self.dist_2.sigma, y2=self.dist_2.mu+self.dist_2.sigma, color='g', alpha=0.3)
        plt.legend()
        plt.ylabel('y')
        plt.xlabel('time')
        fig.tight_layout()