import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta as td

def sharpe(rets, hzn = 'h', hr = 0):
    if rets.std() == 0:
        sha = 0
    else:
        if hzn == 'h':
            sha = np.sqrt(365 * 24) * (rets.mean() - hr) / (rets - hr).std()
        elif hzn == 'd':
            rets_cum = pd.concat([pd.Series(index = [rets.index.min() - td(hours = 1)], data = 1), (1 + rets).cumprod()])
            rets_d_cum = rets_cum[(rets_cum.index.hour == 23) | (rets_cum.index == rets_cum.index[0]) | (rets_cum.index == rets_cum.index[-1])]
            rets_d = np.sqrt(td(days = 1) / rets_d_cum.index.diff()[1:]) * rets_d_cum.pct_change()[1:]
            sha = np.sqrt(365) * (rets_d.mean() - hr) / (rets_d - hr).std()
    return sha

def sortino(rets, hzn = 'h', hr = 0):
    if hzn == 'h':
        n = len(rets)
        dr = np.sqrt(sum(np.array([min(rets.iloc[i] - hr, 0) for i in range(n)]) ** 2) / (n - 1))
        sor = np.sqrt(365 * 24) * (rets.mean() - hr) / dr
    elif hzn == 'd':
        rets_cum = pd.concat([pd.Series(index = [rets.index.min() - td(hours = 1)], data = 1), (1 + rets).cumprod()])
        rets_d_cum = rets_cum[(rets_cum.index.hour == 23) | (rets_cum.index == rets_cum.index[0]) | (rets_cum.index == rets_cum.index[-1])]
        rets_d = np.sqrt(td(days = 1) / rets_d_cum.index.diff()[1:]) * rets_d_cum.pct_change()[1:]
        n = len(rets_d)
        dr = np.sqrt(sum(np.array([min(rets_d.iloc[i] - hr, 0) for i in range(n)]) ** 2) / (n - 1))
        sor = np.sqrt(365) * (rets_d.mean() - hr) / dr
    return sor

def cagr(port_val):
    gr = port_val.iloc[-1] / port_val.iloc[0]
    ny = ((port_val.index.max() - port_val.index.min()).days + (port_val.index.max() - port_val.index.min()).seconds / (24 * 60 * 60)) / 365
    return (gr ** (1 / ny)) - 1

def cdgr(port_val):
    gr = port_val.iloc[-1] / port_val.iloc[0]
    nd = (port_val.index.max() - port_val.index.min()).days + (port_val.index.max() - port_val.index.min()).seconds / (24 * 60 * 60)
    return (gr ** (1 / nd)) - 1

def win_rate(rets_g):
    n_tr_tot = (rets_g != 0).sum().sum()
    n_w_tot = (rets_g > 0).sum().sum()
    n_tr = (rets_g.sum(axis = 1) != 0).sum()
    n_w = (rets_g.sum(axis = 1) > 0).sum()
    return n_tr_tot, n_w_tot / n_tr_tot, n_tr, n_w / n_tr

def mdd(port_val):
    return 1. - np.min(np.flip(np.minimum.accumulate(np.flip(port_val))) / port_val)

def hvar(port_val, p = 0.99):
    port_val_d = port_val[(port_val.index.hour == 23) | (port_val.index == port_val.index[0]) | (port_val.index == port_val.index[-1])]
    rets_d = np.sqrt(td(days = 1) / port_val_d.index.diff()[1:]) * port_val_d.pct_change()[1:]
    return (-rets_d).quantile(p), (-port_val.pct_change()).quantile(p)