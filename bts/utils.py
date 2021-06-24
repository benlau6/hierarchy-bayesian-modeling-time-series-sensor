import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import namedtuple

gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: '𝒩(μ={:.3f}, 𝜎²={:.3f})'.format(s[0], s[1])

def t2dt(t, start_time='2021-06-01', unit='hour'):
    '''
    t: numpy arr of time step
    unit: unit time step
    start_time: datetime at time 0
    '''
    start = pd.to_datetime(start_time)
    dt = start + pd.TimedeltaIndex(t, unit=unit) 
    dt = pd.to_datetime(dt)
    return dt

def add_subplot(height=5):
    fig = plt.gcf()
    n = len(fig.axes)
    for i in range(n):
        fig.axes[i].change_geometry(n + 1, 1, i + 1)
    w, h = fig.get_size_inches()
    fig.set_size_inches(w, h + height)
    return fig.add_subplot(len(fig.axes) + 1, 1, len(fig.axes) + 1)