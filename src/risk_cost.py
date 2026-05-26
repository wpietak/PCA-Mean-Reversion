import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta as td


def covar(cal_int, recal, lam, df_r, h):

    v_lam = np.array([lam ** (cal_int - 1 - i) for i in range(cal_int)])
    varis = df_r[cal_int:].copy()
    varis.iloc[0, :] = (1 - lam) * v_lam.dot(df_r[:cal_int] ** 2)
    for i in varis.index[1:]:
        varis.loc[i] = (1 - lam) * df_r.shift(1).loc[i] ** 2 + lam * varis.shift(1).loc[i]

    cor_mats = dict()
    t = df_r.index[0] + td(hours = cal_int)
    while t < df_r.index.max():
        df_r_cal = df_r[(t - td(hours = cal_int)):(t - td(hours = 1))]
        df_r_cal_cum = (1 + df_r_cal).cumprod()
        df_r_cal_cum = pd.concat([pd.DataFrame(index = [df_r_cal_cum.index.min() - td(hours = 1)], data = 1, columns = df_r_cal_cum.columns), df_r_cal_cum])
        df_r_cal_h = df_r_cal_cum.loc[df_r_cal_cum.index.hour % h == h - 1].pct_change()[1:]
        cor_mats[t] = df_r_cal_h.corr()
        t += td(hours = recal)

    return varis, cor_mats


def fix_cost(df, cal_int, fees, df_bas, fun_rat_pred, fun_rat):
    idx = df[1:][cal_int:].index
    fe_bas_pr = (2 * fees + df_bas / df.shift(1)).loc[idx]
    fe_bas_o = (fees + (df_bas / df.shift(1)) / 2).loc[idx]
    fe_bas_c = (fees + (df_bas / df) / 2).loc[idx]
    fr_pr = fun_rat_pred.loc[idx]
    fr_o = fun_rat.loc[idx]
    fr_c = fun_rat.shift(-1).loc[idx]
    return fe_bas_pr, fe_bas_o, fe_bas_c, fr_pr, fr_o, fr_c


def var_cost(memes_bin, memes_byb, klines_dict, imp_mod_params, memes, df_r, cal_int):

    im_pr = pd.DataFrame()
    im_re = pd.DataFrame()
    for m in memes_bin.keys():
        m_h_vol_df = klines_dict['bin']['h'][m][['time', 'quote_volume']].rename(columns = {'time': 'hour', 'quote_volume': 'hourly_volume'})
        m_h_vol_df['hourly_volume_ewma'] = m_h_vol_df['hourly_volume'].ewm(alpha = 0.1).mean()
        m_h_vol_df['hourly_volume_ewma_lag'] = m_h_vol_df['hourly_volume_ewma'].shift(1)
        m_h_vol_df = m_h_vol_df.set_index('hour')
        m_h_rv_df = klines_dict['bin']['min'][m][['hour', 'real_var']].rename(columns = {'real_var': 'real_var_h'}).groupby('hour', as_index = True).sum()
        m_h_rv_df['real_var_h_ewma'] = m_h_rv_df['real_var_h'].ewm(alpha = 0.1).mean()
        m_h_rv_df['real_var_h_ewma_lag'] = m_h_rv_df['real_var_h_ewma'].shift(1)
        im_pr[m] = 2 * imp_mod_params['wo_coef'][m] * np.sqrt(m_h_rv_df['real_var_h_ewma_lag']) / m_h_vol_df['hourly_volume_ewma_lag']
        im_re[m] = imp_mod_params['cc_coef'][m] * np.sqrt(m_h_rv_df['real_var_h_ewma']) / m_h_vol_df['hourly_volume_ewma'] ** imp_mod_params['cc_exp'][m]
    for m in memes_byb.keys():
        m_h_vol_df = klines_dict['byb']['h'][m].reset_index()[['time', 'foreignNotional']].rename(columns = {'time': 'hour', 'foreignNotional': 'hourly_volume'})
        m_h_vol_df['hourly_volume_ewma'] = m_h_vol_df['hourly_volume'].ewm(alpha = 0.1).mean()
        m_h_vol_df['hourly_volume_ewma_lag'] = m_h_vol_df['hourly_volume_ewma'].shift(1)
        m_h_vol_df = m_h_vol_df.set_index('hour')
        m_h_rv_df = klines_dict['byb']['min'][m][['hour', 'real_var']].rename(columns = {'real_var': 'real_var_h'}).groupby('hour', as_index = True).sum()
        m_h_rv_df['real_var_h_ewma'] = m_h_rv_df['real_var_h'].ewm(alpha = 0.1).mean()
        m_h_rv_df['real_var_h_ewma_lag'] = m_h_rv_df['real_var_h_ewma'].shift(1)
        im_pr[m] = 2 * imp_mod_params['wo_coef'][m] * np.sqrt(m_h_rv_df['real_var_h_ewma_lag']) / m_h_vol_df['hourly_volume_ewma_lag']
        im_re[m] = imp_mod_params['cc_coef'][m] * np.sqrt(m_h_rv_df['real_var_h_ewma']) / m_h_vol_df['hourly_volume_ewma'] ** imp_mod_params['cc_exp'][m]
    im_pr = im_pr[memes].loc[df_r[cal_int:].index]
    im_re = im_re[memes].loc[df_r[cal_int:].index]

    return im_pr, im_re