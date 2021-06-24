import matplotlib.pyplot as plt
from collections import namedtuple

gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: '𝒩(μ={:.3f}, 𝜎²={:.3f})'.format(s[0], s[1])

def add_subplot(height=5):
    fig = plt.gcf()
    n = len(fig.axes)
    for i in range(n):
        fig.axes[i].change_geometry(n + 1, 1, i + 1)
    w, h = fig.get_size_inches()
    fig.set_size_inches(w, h + height)
    return fig.add_subplot(len(fig.axes) + 1, 1, len(fig.axes) + 1)