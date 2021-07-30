import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def add_subplot(height=5):
    fig = plt.gcf()
    n = len(fig.axes)
    for i in range(n):
        fig.axes[i].change_geometry(n + 1, 1, i + 1)
    w, h = fig.get_size_inches()
    fig.set_size_inches(w, h + height)
    return fig.add_subplot(len(fig.axes) + 1, 1, len(fig.axes) + 1)

def select_model(p=0.6):
    comparison = az.compare({"switchpoint": sp_model.trace, "baseline": bl_model.trace})
    stats = ['rank', 'weight', 'loo', 'd_loo']
    comp = comparison[stats]

    chosen_model_idx = 0 if comp.iloc[0]['weight']>p else 1
    chosen_model = comp.index[chosen_model_idx]
    return chosen_model, comp

def dt2t(dt):
    return ((dt - dt[0]).total_seconds().astype(int)//3600).values