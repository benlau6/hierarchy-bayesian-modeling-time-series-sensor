import matplotlib.pyplot as plt
import numpy as np
from ..utils import add_subplot
from .data import gaussian

class KalmanFilterChangePointModel():
    
    def __init__(self, changepoint_tolerance=5, residual_tolerance=1):
        self.changepoint_tolerance = changepoint_tolerance
        self.residual_tolerance = residual_tolerance  

    def predict(self, posterior, movement):
        x, P = posterior # mean and variance of posterior
        dx, Q = movement # mean and variance of movement
        x = x + dx
        P = P + Q
        return gaussian(x, P)
    
    def update(self, prior, measurement, num_seq_anomalies, prev_residual_sign):
        x, P = prior        # mean and variance of prior
        z, R = measurement  # mean and variance of measurement
        
        y = z - x        # residual
        
        residual_sign = 1 if y>0 else -1
        have_changepoint = False
        not_seq_anomaly = True
        if abs(y) > self.residual_tolerance:
            if residual_sign == prev_residual_sign:
                num_seq_anomalies += 1
                not_seq_anomaly = False

        if num_seq_anomalies>self.changepoint_tolerance:
            have_changepoint = True
        
        if have_changepoint or not_seq_anomaly:
            num_seq_anomalies = 0
                        
        K = P / (P + R)  # Kalman gain

        if have_changepoint:
            x = z
        else:
            x = x + K*y      # posterior
            
        P = (1 - K) * P  # posterior variance
        return gaussian(x, P), num_seq_anomalies, residual_sign
        
    def fit(self, zs, movement, sensor_var=1):
        N = zs.size
        x = gaussian(zs[0], 1.)
        xs = np.zeros((N, 2))
        num_seq_anomalies = 0
        residual_sign = 1
        for i, z in enumerate(zs):
            measurement = gaussian(z, sensor_var)
            prior = self.predict(x, movement)  
            x, num_seq_anomalies, residual_sign = self.update(prior, measurement, num_seq_anomalies=num_seq_anomalies, prev_residual_sign=residual_sign)
            xs[i] = prior
        self.xs = xs
    
    def plot_all(self, ts, zs, fig=None):
        if fig is None:
            fig = plt.figure(figsize=(18, 1))
        self.plot_y(ts, zs)
        self.plot_var(ts, zs)
        fig.tight_layout()
        
    def plot_y(self, ts, zs):
        ax = add_subplot()
        ax.plot(ts, self.xs[:,0], color='orange')
        ax.fill_between(ts, self.xs[:,0]-2*self.xs[:,1], self.xs[:,0]+2*self.xs[:,1], color='moccasin')
        ax.scatter(x=ts, y=zs)
        plt.ylabel('y')
        plt.xlabel('time step')

    def plot_var(self, ts, zs):
        ax = add_subplot()
        ax.plot(ts, self.xs[:,1])
        plt.ylabel('Variance')
        plt.xlabel('time step')