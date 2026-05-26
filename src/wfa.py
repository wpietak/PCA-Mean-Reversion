import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from datetime import datetime as dt
from datetime import timedelta as td
from statsmodels.tsa.arima.model import ARIMA
import warnings
import random
import copy
from src.hyper_optimization import *
from src.state import *



def ret_pred_son_rf(df_full_r, cal_int, recal, memes_pca, nPC_ord, npcs_max, h):

    resids_cal_full_dnpc = dict()
    resids_full_dnpc = dict()

    for i in range(1, npcs_max + 1):
        
        t = df_full_r.index[0] + td(hours = cal_int)
        resids_cal_full = dict()
        resids_full = dict()
                
        while t < df_full_r.index.max():
            
            df_r_cal = df_full_r[memes_pca][(t - td(hours = cal_int)):(t - td(hours = 1))]
            df_r_dep = df_full_r[memes_pca][t:(t + td(hours = recal - 1))]
                
            pca = PCA(n_components = i)
            pca.fit(df_r_cal)
                    
            comps_r_cal = pca.transform(df_r_cal)
            sys_r_cal = pca.inverse_transform(comps_r_cal)
            resids_cal = (df_r_cal - sys_r_cal)
                    
            sys_r = df_r_dep.dot(pca.components_.T).dot(pca.components_)
            sys_r.columns = df_r_dep.columns       
            resids = (df_r_dep - sys_r)
        
            resids_cal_full[t] = resids_cal
            resids_full[t] = resids
                    
            t += td(hours = recal)
    
        resids_cal_full_dnpc[i] = resids_cal_full
        resids_full_dnpc[i] = resids_full
        
    res_pred_full = pd.DataFrame()
    t = df_full_r.index[0] + td(hours = cal_int)
            
    while t < df_full_r.index.max():
        
        res_pred = pd.DataFrame()
        
        for m in nPC_ord.keys():
            p, q = nPC_ord[m][1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                arma = ARIMA(resids_cal_full_dnpc[nPC_ord[m][0]][t][m], order = [p, 0, q], freq = td(hours = h)).fit()
            arma_up = arma.append(resids_full_dnpc[nPC_ord[m][0]][t][m], refit = False)
            res_pred[m] = arma_up.predict(start = t)
            if p > 0:
                if arma.arparams[0] > 0.85:
                    res_pred[m] = 0.        
        
        res_pred_full = pd.concat([res_pred_full, res_pred])
                
        t += td(hours = recal)

    return res_pred_full



def WFA(is_per, oos_per, memes_all, memes, memes_fix, df_full_r, df_l_r, df_h_r, cal_int, recal, h, cor_mats, init_capital, lev_cap, risk_av, varis, buf, sl, fe_bas_pr, fe_bas_o, fe_bas_c, fr_pr, fr_o, fr_c, im_pr, im_re, volu_exp_fin, n_pop, n_ite, b, npcd, npcs_max, npcs_w, r_th_an, n_ret_del, conc_pen_ga, bkt_w, p_tour, n_tour, p_cross, p_mut, imi_rat, n_top, n_nbr, swa_rat, main_bkt_w, orders, ord_sim, main_comb_w, clo_comb_w, sem_comb_w, r_th_an_l, r_th_an_h, n_top_l, n_top_h, conc_pen_comb2, n_nbrs, comb_drop_th, sh_disp_pen_met, sh_nbr_d_max_ga = None, sh_nbr_d_max_comb2 = None, tp_len_fr_th = None, n_top_top = None, sh_nbr_d_inc = None, sh_dif_pen = None):

    
    dates_per = pd.DataFrame({'i': df_full_r.index[int(cal_int / h):], 't': pd.to_datetime("nan")})
    t = df_full_r.index[0] + td(hours = cal_int)
    while t < df_full_r.index.max():
        for i in np.arange(t, (t + td(hours = recal)), td(hours = 1)):
            dates_per.loc[dates_per['i'] == i, 't'] = t
        t += td(hours = recal)

    r_th_l, r_th_h = (1 + np.array([r_th_an_l, r_th_an_h])) ** (is_per / (365 * 24)) - 1

    nbr_combs = dict()
    for npcs in range(1, npcs_max + 1):
        if npcs == 1:
            npcs_ns = [npcs + 1, npcs + 2]
        elif npcs == npcs_max:
            npcs_ns = [npcs - 1, npcs - 2]
        else:
            npcs_ns = [npcs - 1, npcs + 1]
        for o in orders:
            nbr_combs[(npcs, o)] = dict()
            nbr_combs[(npcs, o)]['clo'] = list(pd.concat([pd.Series([(npcs, o_n) for o_n in ord_sim[o]['clo']]), pd.Series([(npcs_n, o) for npcs_n in npcs_ns])]))
            nbr_combs[(npcs, o)]['sem'] = list(pd.concat([pd.Series([(npcs, o_n) for o_n in ord_sim[o]['sem']]), pd.Series([(npcs_n, o_n) for o_n in ord_sim[o]['clo'] for npcs_n in npcs_ns])]))
            nbr_combs[(npcs, o)]['far'] = list(pd.concat([pd.Series([(npcs, o_n) for o_n in ord_sim[o]['far']]), pd.Series([(npcs_n, o_n) for o_n in ord_sim[o]['sem'] for npcs_n in npcs_ns])]))

    wf_t = df_full_r.index[0] + td(hours = cal_int + is_per)

    #gens_perf = dict()
    #gens_perf_cache = dict()
    #top_perf = dict()
    #armas_rf_pred = dict()
    #cum_rets_ind = dict()
    #top_combs = dict()
    #joi_tr_perf = dict()

    if oos_df.port_val_oos.empty:
        capital_oos = init_capital
    else:
        capital_oos = oos_df.port_val_oos.iloc[-1]

    #rets_g_oos = pd.DataFrame()
    #w_oos = pd.DataFrame()
    #port_val_oos = pd.DataFrame()
    
    
    while wf_t < df_full_r.index.max():

        
        print(wf_t, dt.now())
        
        
        df_full_r_is = df_full_r.loc[(wf_t - td(hours = cal_int + is_per)):(wf_t - td(hours = h))]
        df_l_r_is = df_l_r.loc[(wf_t - td(hours = cal_int + is_per)):(wf_t - td(hours = h))]
        df_h_r_is = df_h_r.loc[(wf_t - td(hours = cal_int + is_per)):(wf_t - td(hours = h))]
        varis_is = varis.loc[(wf_t - td(hours = cal_int + is_per)):(wf_t - td(hours = h))]
        fe_bas_pr_is = fe_bas_pr.loc[(wf_t - td(hours = is_per)):(wf_t - td(hours = h))]
        fe_bas_o_is = fe_bas_o.loc[(wf_t - td(hours = is_per)):(wf_t - td(hours = h))]
        fe_bas_c_is = fe_bas_c.loc[(wf_t - td(hours = is_per)):(wf_t - td(hours = h))]
        fr_pr_is = fr_pr.loc[(wf_t - td(hours = is_per)):(wf_t - td(hours = h))]
        fr_o_is = fr_o.loc[(wf_t - td(hours = is_per)):(wf_t - td(hours = h))]
        fr_c_is = fr_c.loc[(wf_t - td(hours = is_per)):(wf_t - td(hours = h))]
        im_pr_is = im_pr.loc[(wf_t - td(hours = is_per)):(wf_t - td(hours = h))]
        im_re_is = im_re.loc[(wf_t - td(hours = is_per)):(wf_t - td(hours = h))]

        df_full_r_oos = df_full_r.loc[(wf_t - td(hours = cal_int)):(wf_t + td(hours = oos_per - h))]
        df_l_r_oos = df_l_r.loc[(wf_t - td(hours = cal_int)):(wf_t + td(hours = oos_per - h))]
        df_h_r_oos = df_h_r.loc[(wf_t - td(hours = cal_int)):(wf_t + td(hours = oos_per - h))]
        varis_oos = varis.loc[(wf_t - td(hours = cal_int)):(wf_t + td(hours = oos_per - h))]
        fe_bas_pr_oos = fe_bas_pr.loc[wf_t:(wf_t + td(hours = oos_per - h))]
        fe_bas_o_oos = fe_bas_o.loc[wf_t:(wf_t + td(hours = oos_per - h))]
        fe_bas_c_oos = fe_bas_c.loc[wf_t:(wf_t + td(hours = oos_per - h))]
        fr_pr_oos = fr_pr.loc[wf_t:(wf_t + td(hours = oos_per - h))]
        fr_o_oos = fr_o.loc[wf_t:(wf_t + td(hours = oos_per - h))]
        fr_c_oos = fr_c.loc[wf_t:(wf_t + td(hours = oos_per - h))]
        im_pr_oos = im_pr.loc[wf_t:(wf_t + td(hours = oos_per - h))]
        im_re_oos = im_re.loc[wf_t:(wf_t + td(hours = oos_per - h))]

        
        args_fix = (df_full_r_is, cal_int, recal, init_capital, risk_av, varis_is, fe_bas_pr_is, fr_pr_is, im_pr_is)
        
        print("goa", dt.now())
        gens_perf_t, gens_perf_cache_t = goa_ar2_rf(n_pop, n_ite, b, memes_all, memes, memes_fix, npcd, npcs_max, npcs_w, r_th_an, n_ret_del, conc_pen_ga, bkt_w, p_tour, n_tour, p_cross, p_mut, imi_rat, args_fix)

        
        print("top bkts nbh", dt.now())
        top_perf_t = top_ind_nbh(gens_perf_t[n_ite - 1], gens_perf_cache_t, n_top, n_nbr, swa_rat, memes, memes_fix, npcs_max, npcs_w, r_th_an, n_ret_del, conc_pen_ga, main_bkt_w, args_fix)

        
        if sh_disp_pen_met == "removal w/thr":
            
            top_perf_t_sel = top_perf_t.loc[(top_perf_t['Sharpe_nbrs'] - top_perf_t['Sharpe']) / top_perf_t['Sharpe'] > -sh_nbr_d_max_ga].sort_values('Sharpe_nbh', ascending = False).copy()
            if len(top_perf_t_sel) > tp_len_fr_th * n_top:
                memes_pca = list(set(top_perf_t_sel['ind'].iloc[0].loc[top_perf_t_sel['ind'].iloc[0]].index) | set(memes_fix))
            else:
                memes_pca = list(set(top_perf_t['ind'].iloc[0].loc[top_perf_t['ind'].iloc[0]].index) | set(memes_fix))
        
        elif sh_disp_pen_met == "3-step removal w/thr":
            
            top_perf_t_sel = top_perf_t.loc[(top_perf_t['Sharpe_nbrs'] - top_perf_t['Sharpe']) / top_perf_t['Sharpe'] > -sh_nbr_d_max_ga].sort_values('Sharpe_nbh', ascending = False).copy()
            if len(top_perf_t_sel) > tp_len_fr_th * n_top:
                memes_pca = list(set(top_perf_t_sel['ind'].iloc[0].loc[top_perf_t_sel['ind'].iloc[0]].index) | set(memes_fix))
            else:
                top_perf_t_sel = top_perf_t.loc[(top_perf_t['Sharpe_nbrs'] - top_perf_t['Sharpe']) / top_perf_t['Sharpe'] > -(sh_nbr_d_max_ga + sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                if len(top_perf_t_sel) > tp_len_fr_th * n_top:
                    memes_pca = list(set(top_perf_t_sel['ind'].iloc[0].loc[top_perf_t_sel['ind'].iloc[0]].index) | set(memes_fix))
                else:
                    top_perf_t_sel = top_perf_t.loc[(top_perf_t['Sharpe_nbrs'] - top_perf_t['Sharpe']) / top_perf_t['Sharpe'] > -(sh_nbr_d_max_ga + 2 * sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                    if len(top_perf_t_sel) > tp_len_fr_th * n_top:
                        memes_pca = list(set(top_perf_t_sel['ind'].iloc[0].loc[top_perf_t_sel['ind'].iloc[0]].index) | set(memes_fix))
                    else:
                        memes_pca = list(set(top_perf_t['ind'].iloc[0].loc[top_perf_t['ind'].iloc[0]].index) | set(memes_fix))
        
        elif sh_disp_pen_met == "top top":
            
            top_perf_t_sel = top_perf_t[:n_top_top].loc[(top_perf_t[:n_top_top]['Sharpe_nbrs'] - top_perf_t[:n_top_top]['Sharpe']) / top_perf_t[:n_top_top]['Sharpe'] > -sh_nbr_d_max_ga].sort_values('Sharpe_nbh', ascending = False).copy()
            if len(top_perf_t_sel) > 0:
                memes_pca = list(set(top_perf_t_sel['ind'].iloc[0].loc[top_perf_t_sel['ind'].iloc[0]].index) | set(memes_fix))
            else:
                top_perf_t_sel = top_perf_t[:n_top_top].loc[(top_perf_t[:n_top_top]['Sharpe_nbrs'] - top_perf_t[:n_top_top]['Sharpe']) / top_perf_t[:n_top_top]['Sharpe'] > -(sh_nbr_d_max_ga + sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                if len(top_perf_t_sel) > 0:
                    memes_pca = list(set(top_perf_t_sel['ind'].iloc[0].loc[top_perf_t_sel['ind'].iloc[0]].index) | set(memes_fix))
                else:
                    top_perf_t_sel = top_perf_t[:n_top_top].loc[(top_perf_t[:n_top_top]['Sharpe_nbrs'] - top_perf_t[:n_top_top]['Sharpe']) / top_perf_t[:n_top_top]['Sharpe'] > -(sh_nbr_d_max_ga + 2 * sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                    if len(top_perf_t_sel) > 0:
                        memes_pca = list(set(top_perf_t_sel['ind'].iloc[0].loc[top_perf_t_sel['ind'].iloc[0]].index) | set(memes_fix))
                    else:
                        memes_pca = list(set(top_perf_t['ind'].iloc[0].loc[top_perf_t['ind'].iloc[0]].index) | set(memes_fix))
        
        elif sh_disp_pen_met == "double top top":
            
            n_top_top = int(1 / 4 * n_top)
            n_top_top_2 = int(1 / 2 * n_top)
            top_perf_t_sel = top_perf_t[:n_top_top].loc[(top_perf_t[:n_top_top]['Sharpe_nbrs'] - top_perf_t[:n_top_top]['Sharpe']) / top_perf_t[:n_top_top]['Sharpe'] > -sh_nbr_d_max_ga].sort_values('Sharpe_nbh', ascending = False).copy()
            if len(top_perf_t_sel) > 0:
                memes_pca = list(set(top_perf_t_sel['ind'].iloc[0].loc[top_perf_t_sel['ind'].iloc[0]].index) | set(memes_fix))
            else:
                top_perf_t_sel_1 = top_perf_t[:n_top_top].loc[(top_perf_t[:n_top_top]['Sharpe_nbrs'] - top_perf_t[:n_top_top]['Sharpe']) / top_perf_t[:n_top_top]['Sharpe'] > -(sh_nbr_d_max_ga + sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                top_perf_t_sel_2 = top_perf_t[n_top_top:n_top_top_2].loc[(top_perf_t[n_top_top:n_top_top_2]['Sharpe_nbrs'] - top_perf_t[n_top_top:n_top_top_2]['Sharpe']) / top_perf_t[n_top_top:n_top_top_2]['Sharpe'] > -sh_nbr_d_max_ga].sort_values('Sharpe_nbh', ascending = False).copy()
                top_perf_t_sel = pd.concat([top_perf_t_sel_1, top_perf_t_sel_2])
                if len(top_perf_t_sel) > 0:
                    memes_pca = list(set(top_perf_t_sel['ind'].iloc[0].loc[top_perf_t_sel['ind'].iloc[0]].index) | set(memes_fix))
                else:
                    top_perf_t_sel_1 = top_perf_t[:n_top_top].loc[(top_perf_t[:n_top_top]['Sharpe_nbrs'] - top_perf_t[:n_top_top]['Sharpe']) / top_perf_t[:n_top_top]['Sharpe'] > -(sh_nbr_d_max_ga + 2 * sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                    top_perf_t_sel_2 = top_perf_t[n_top_top:n_top_top_2].loc[(top_perf_t[n_top_top:n_top_top_2]['Sharpe_nbrs'] - top_perf_t[n_top_top:n_top_top_2]['Sharpe']) / top_perf_t[n_top_top:n_top_top_2]['Sharpe'] > -(sh_nbr_d_max_ga + sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                    top_perf_t_sel = pd.concat([top_perf_t_sel_1, top_perf_t_sel_2])
                    if len(top_perf_t_sel) > 0:
                        memes_pca = list(set(top_perf_t_sel['ind'].iloc[0].loc[top_perf_t_sel['ind'].iloc[0]].index) | set(memes_fix))
                    else:
                        memes_pca = list(set(top_perf_t['ind'].iloc[0].loc[top_perf_t['ind'].iloc[0]].index) | set(memes_fix))
        
        elif sh_disp_pen_met == "dif pen":
            
            top_perf_t['Sharpe_nbh_pen'] = top_perf_t['Sharpe_nbh'] + sh_dif_pen * (top_perf_t['Sharpe_nbrs'] - top_perf_t['Sharpe'])
            top_perf_t = top_perf_t.sort_values('Sharpe_nbh_pen', ascending = False)
            memes_pca = list(set(top_perf_t['ind'].iloc[0].loc[top_perf_t['ind'].iloc[0]].index) | set(memes_fix))

        
        print("ret pred", dt.now())
        armas_rf_pred_t = ret_pred_don_rf(df_full_r_is, cal_int, recal, memes_pca, memes, npcs_max, orders, h)

        
        memes_tr = list(set(memes_pca) & set(memes))
        args_fix_ext = (df_full_r_is, df_l_r_is, df_h_r_is, cal_int, recal, h, init_capital, lev_cap, risk_av, varis_is, buf, sl, fe_bas_pr_is, fe_bas_o_is, fe_bas_c_is, fr_pr_is, fr_o_is, fr_c_is, im_pr_is, im_re_is, volu_exp_fin)

        print("ind tra rets", dt.now())
        cum_rets_ind_t = ind_rets_don_rf(armas_rf_pred_t, memes_tr, npcs_max, orders, n_ret_del, ord_sim, main_comb_w, clo_comb_w, sem_comb_w, args_fix_ext)

        top_combs_t = dict()
        for m in memes_tr:
            cum_rets_mel = pd.melt(cum_rets_ind_t[m].reset_index(), id_vars = "index").sort_values('value', ascending = False)
            n_th_l, n_th_h = len(cum_rets_mel.loc[cum_rets_mel['value'] > r_th_l]), len(cum_rets_mel.loc[cum_rets_mel['value'] > r_th_h])
            if n_th_l == 0:
                continue
            cum_rets_mel_sel = cum_rets_mel[:(min(n_th_l, n_top_l) + min(n_th_h, n_top_h - n_top_l))]
            top_combs_t[m] = [(npcs, o) for npcs, o in zip(cum_rets_mel_sel['variable'], cum_rets_mel_sel['index'])]
            if len(memes_tr) > 1:
                top_combs_t[m].append((0, 0))

        
        args_fix_ext_2 = (df_full_r_is, df_l_r_is, df_h_r_is, cal_int, recal, h, dates_per, cor_mats, init_capital, lev_cap, risk_av, varis_is, buf, sl, fe_bas_pr_is, fe_bas_o_is, fe_bas_c_is, fr_pr_is, fr_o_is, fr_c_is, im_pr_is, im_re_is, volu_exp_fin)

        print("joi tra perf", dt.now())
        joi_tr_perf_t = joi_rets_don_rf(armas_rf_pred_t, top_combs_t, n_ret_del, conc_pen_comb2, nbr_combs, n_nbrs, main_comb_w, clo_comb_w, sem_comb_w, comb_drop_th, args_fix_ext_2)

        
        gens_perf[wf_t] = gens_perf_t
        gens_perf_cache[wf_t] = gens_perf_cache_t
        top_perf[wf_t] = top_perf_t
        armas_rf_pred[wf_t] = armas_rf_pred_t
        cum_rets_ind[wf_t] = cum_rets_ind_t
        top_combs[wf_t] = top_combs_t
        joi_tr_perf[wf_t] = joi_tr_perf_t

        
        if sh_disp_pen_met == "removal w/thr":

            joi_tr_perf_t_sel = joi_tr_perf_t.loc[(joi_tr_perf_t['Sharpe_nbrs'] - joi_tr_perf_t['Sharpe']) / joi_tr_perf_t['Sharpe'] > -sh_nbr_d_max_comb2].sort_values('Sharpe_nbh', ascending = False).copy()
            if len(joi_tr_perf_t_sel) > tp_len_fr_th * len(joi_tr_perf_t):
                top_comb2 = joi_tr_perf_t_sel['comb2'].iloc[0]
            else:
                top_comb2 = joi_tr_perf_t['comb2'].iloc[0]

        elif sh_disp_pen_met == "3-step removal w/thr":

            joi_tr_perf_t_sel = joi_tr_perf_t.loc[(joi_tr_perf_t['Sharpe_nbrs'] - joi_tr_perf_t['Sharpe']) / joi_tr_perf_t['Sharpe'] > -sh_nbr_d_max_comb2].sort_values('Sharpe_nbh', ascending = False).copy()
            if len(joi_tr_perf_t_sel) > tp_len_fr_th * len(joi_tr_perf_t):
                top_comb2 = joi_tr_perf_t_sel['comb2'].iloc[0]
            else:
                joi_tr_perf_t_sel = joi_tr_perf_t.loc[(joi_tr_perf_t['Sharpe_nbrs'] - joi_tr_perf_t['Sharpe']) / joi_tr_perf_t['Sharpe'] > -(sh_nbr_d_max_comb2 + sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                if len(joi_tr_perf_t_sel) > tp_len_fr_th * len(joi_tr_perf_t):
                    top_comb2 = joi_tr_perf_t_sel['comb2'].iloc[0]
                else:
                    joi_tr_perf_t_sel = joi_tr_perf_t.loc[(joi_tr_perf_t['Sharpe_nbrs'] - joi_tr_perf_t['Sharpe']) / joi_tr_perf_t['Sharpe'] > -(sh_nbr_d_max_comb2 + 2 * sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                    if len(joi_tr_perf_t_sel) > tp_len_fr_th * len(joi_tr_perf_t):
                        top_comb2 = joi_tr_perf_t_sel['comb2'].iloc[0]
                    else:
                        top_comb2 = joi_tr_perf_t['comb2'].iloc[0]

        elif sh_disp_pen_met == "top top":

            joi_tr_perf_t_sel = joi_tr_perf_t[:n_top_top].loc[(joi_tr_perf_t[:n_top_top]['Sharpe_nbrs'] - joi_tr_perf_t[:n_top_top]['Sharpe']) / joi_tr_perf_t[:n_top_top]['Sharpe'] > -sh_nbr_d_max_comb2].sort_values('Sharpe_nbh', ascending = False).copy()
            if len(joi_tr_perf_t_sel) > 0:
                top_comb2 = joi_tr_perf_t_sel['comb2'].iloc[0]
            else:
                joi_tr_perf_t_sel = joi_tr_perf_t[:n_top_top].loc[(joi_tr_perf_t[:n_top_top]['Sharpe_nbrs'] - joi_tr_perf_t[:n_top_top]['Sharpe']) / joi_tr_perf_t[:n_top_top]['Sharpe'] > -(sh_nbr_d_max_comb2 + sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                if len(joi_tr_perf_t_sel) > 0:
                    top_comb2 = joi_tr_perf_t_sel['comb2'].iloc[0]
                else:
                    joi_tr_perf_t_sel = joi_tr_perf_t[:n_top_top].loc[(joi_tr_perf_t[:n_top_top]['Sharpe_nbrs'] - joi_tr_perf_t[:n_top_top]['Sharpe']) / joi_tr_perf_t[:n_top_top]['Sharpe'] > -(sh_nbr_d_max_comb2 + 2 * sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                    if len(joi_tr_perf_t_sel) > 0:
                        top_comb2 = joi_tr_perf_t_sel['comb2'].iloc[0]
                    else:
                        top_comb2 = joi_tr_perf_t['comb2'].iloc[0]

        elif sh_disp_pen_met == "double top top":

            n_top_top = int(1 / 4 * len(joi_tr_perf_t))
            n_top_top_2 = int(1 / 2 * len(joi_tr_perf_t))
            joi_tr_perf_t_sel = joi_tr_perf_t[:n_top_top].loc[(joi_tr_perf_t[:n_top_top]['Sharpe_nbrs'] - joi_tr_perf_t[:n_top_top]['Sharpe']) / joi_tr_perf_t[:n_top_top]['Sharpe'] > -sh_nbr_d_max_comb2].sort_values('Sharpe_nbh', ascending = False).copy()
            if len(joi_tr_perf_t_sel) > 0:
                top_comb2 = joi_tr_perf_t_sel['comb2'].iloc[0]
            else:
                joi_tr_perf_t_sel_1 = joi_tr_perf_t[:n_top_top].loc[(joi_tr_perf_t[:n_top_top]['Sharpe_nbrs'] - joi_tr_perf_t[:n_top_top]['Sharpe']) / joi_tr_perf_t[:n_top_top]['Sharpe'] > -(sh_nbr_d_max_comb2 + sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                joi_tr_perf_t_sel_2 = joi_tr_perf_t[n_top_top:n_top_top_2].loc[(joi_tr_perf_t[n_top_top:n_top_top_2]['Sharpe_nbrs'] - joi_tr_perf_t[n_top_top:n_top_top_2]['Sharpe']) / joi_tr_perf_t[n_top_top:n_top_top_2]['Sharpe'] > -sh_nbr_d_max_comb2].sort_values('Sharpe_nbh', ascending = False).copy()
                joi_tr_perf_t_sel = pd.concat([joi_tr_perf_t_sel_1, joi_tr_perf_t_sel_2])
                if len(joi_tr_perf_t_sel) > 0:
                    top_comb2 = joi_tr_perf_t_sel['comb2'].iloc[0]
                else:
                    joi_tr_perf_t_sel_1 = joi_tr_perf_t[:n_top_top].loc[(joi_tr_perf_t[:n_top_top]['Sharpe_nbrs'] - joi_tr_perf_t[:n_top_top]['Sharpe']) / joi_tr_perf_t[:n_top_top]['Sharpe'] > -(sh_nbr_d_max_comb2 + 2 * sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                    joi_tr_perf_t_sel_2 = joi_tr_perf_t[n_top_top:n_top_top_2].loc[(joi_tr_perf_t[n_top_top:n_top_top_2]['Sharpe_nbrs'] - joi_tr_perf_t[n_top_top:n_top_top_2]['Sharpe']) / joi_tr_perf_t[n_top_top:n_top_top_2]['Sharpe'] > -(sh_nbr_d_max_comb2 + sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                    joi_tr_perf_t_sel = pd.concat([joi_tr_perf_t_sel_1, joi_tr_perf_t_sel_2])
                    if len(joi_tr_perf_t_sel) > 0:
                        top_comb2 = joi_tr_perf_t_sel['comb2'].iloc[0]
                    else:
                        top_comb2 = joi_tr_perf_t['comb2'].iloc[0]

        elif sh_disp_pen_met == "dif pen":

            joi_tr_perf_t['Sharpe_nbh_pen'] = joi_tr_perf_t['Sharpe_nbh'] + sh_dif_pen * (joi_tr_perf_t['Sharpe_nbrs'] - joi_tr_perf_t['Sharpe'])
            joi_tr_perf_t = joi_tr_perf_t.sort_values('Sharpe_nbh_pen', ascending = False)
            top_comb2 = joi_tr_perf_t['comb2'].iloc[0]
        
        nPC_ord = dict()
        for i, m in enumerate(top_combs_t.keys()):
            if top_comb2[i] != (0, 0):
                nPC_ord[m] = top_comb2[i]
        memes_tr_fin = list(nPC_ord.keys())
        
        
        print("oos", dt.now())
        res_pred_oos = ret_pred_son_rf(df_full_r_oos, cal_int, recal, memes_pca, nPC_ord, npcs_max, h)
        rets_g_t, w_t, port_val_t = joi_rets_on_rf(res_pred_oos, memes_tr_fin, df_full_r_oos, df_l_r_oos, df_h_r_oos, cal_int, recal, h, dates_per, cor_mats, capital_oos, lev_cap, risk_av, varis_oos, buf, sl, fe_bas_pr_oos, fe_bas_o_oos, fe_bas_c_oos, fr_pr_oos, fr_o_oos, fr_c_oos, im_pr_oos, im_re_oos, volu_exp_fin)
        capital_oos = port_val_t.iloc[-1]

        oos_df.rets_g_oos = pd.concat([oos_df.rets_g_oos, rets_g_t])
        oos_df.w_oos = pd.concat([oos_df.w_oos, w_t])
        if oos_df.port_val_oos.empty:
            oos_df.port_val_oos = port_val_t
        else:
            oos_df.port_val_oos = pd.concat([oos_df.port_val_oos, port_val_t[1:]])

        
        wf_t += td(hours = oos_per)
        
        print()

    
    #return gens_perf, gens_perf_cache, top_perf, armas_rf_pred, cum_rets_ind, top_combs, joi_tr_perf, rets_g_oos, w_oos, port_val_oos



def gen_oos_rets(lab, oos_per, memes_fix, df_full_r, df_l_r, df_h_r, cal_int, recal, h, cor_mats, init_capital, lev_cap, risk_av, varis, buf, sl, fe_bas_pr, fe_bas_o, fe_bas_c, fr_pr, fr_o, fr_c, im_pr, im_re, volu_exp_fin, npcs_max, n_top, sh_disp_pen_met, sh_nbr_d_max_ga = None, sh_nbr_d_max_comb2 = None, tp_len_fr_th = None, n_top_top = None, sh_nbr_d_inc = None, sh_dif_pen = None):
    
    capital_oos = init_capital
    
    dates_per = pd.DataFrame({'i': df_full_r.index[int(cal_int / h):], 't': pd.to_datetime("nan")})
    t = df_full_r.index[0] + td(hours = cal_int)
    while t < df_full_r.index.max():
        for i in np.arange(t, (t + td(hours = recal)), td(hours = 1)):
            dates_per.loc[dates_per['i'] == i, 't'] = t
        t += td(hours = recal)
    
    for j, wf_t_str in enumerate(pd.read_csv('Results/WFA_' + lab + '/WFA_OoS_start_date.csv')['0']):
        
        wf_t = pd.to_datetime(wf_t_str)
        
        df_full_r_oos = df_full_r.loc[(wf_t - td(hours = cal_int)):(wf_t + td(hours = oos_per - h))]
        df_l_r_oos = df_l_r.loc[(wf_t - td(hours = cal_int)):(wf_t + td(hours = oos_per - h))]
        df_h_r_oos = df_h_r.loc[(wf_t - td(hours = cal_int)):(wf_t + td(hours = oos_per - h))]
        varis_oos = varis.loc[(wf_t - td(hours = cal_int)):(wf_t + td(hours = oos_per - h))]
        fe_bas_pr_oos = fe_bas_pr.loc[wf_t:(wf_t + td(hours = oos_per - h))]
        fe_bas_o_oos = fe_bas_o.loc[wf_t:(wf_t + td(hours = oos_per - h))]
        fe_bas_c_oos = fe_bas_c.loc[wf_t:(wf_t + td(hours = oos_per - h))]
        fr_pr_oos = fr_pr.loc[wf_t:(wf_t + td(hours = oos_per - h))]
        fr_o_oos = fr_o.loc[wf_t:(wf_t + td(hours = oos_per - h))]
        fr_c_oos = fr_c.loc[wf_t:(wf_t + td(hours = oos_per - h))]
        im_pr_oos = im_pr.loc[wf_t:(wf_t + td(hours = oos_per - h))]
        im_re_oos = im_re.loc[wf_t:(wf_t + td(hours = oos_per - h))]
    
        top_perf_t = pd.read_excel("Results/WFA_" + lab + "/top_perf.xlsx", sheet_name = 'wf_' + str(j))
        top_combs_t = pd.read_excel("Results/WFA_" + lab + "/top_combs.xlsx", sheet_name = 'wf_' + str(j))
        joi_tr_perf_t = pd.read_excel("Results/WFA_" + lab + "/joi_tr_perf.xlsx", sheet_name = 'wf_' + str(j))
    
        top_perf_t['ind'] = [pd.Series([True if tf == 'True' else False for tf in top_perf_t['enc_tf'][i][1:-1].replace(' ', '').split(',')], index = top_perf_t['enc_idx'][i][1:-1].replace("'", "").replace(" ", "").split(',')) for i in range(n_top)]
        
        if sh_disp_pen_met == "removal w/thr":
            
            top_perf_t_sel = top_perf_t.loc[(top_perf_t['Sharpe_nbrs'] - top_perf_t['Sharpe']) / top_perf_t['Sharpe'] > -sh_nbr_d_max_ga].sort_values('Sharpe_nbh', ascending = False).copy()
            if len(top_perf_t_sel) > tp_len_fr_th * n_top:
                memes_pca = list(set(top_perf_t_sel['ind'].iloc[0].loc[top_perf_t_sel['ind'].iloc[0]].index) | set(memes_fix))
            else:
                memes_pca = list(set(top_perf_t['ind'].iloc[0].loc[top_perf_t['ind'].iloc[0]].index) | set(memes_fix))
        
        elif sh_disp_pen_met == "3-step removal w/thr":
            
            top_perf_t_sel = top_perf_t.loc[(top_perf_t['Sharpe_nbrs'] - top_perf_t['Sharpe']) / top_perf_t['Sharpe'] > -sh_nbr_d_max_ga].sort_values('Sharpe_nbh', ascending = False).copy()
            if len(top_perf_t_sel) > tp_len_fr_th * n_top:
                memes_pca = list(set(top_perf_t_sel['ind'].iloc[0].loc[top_perf_t_sel['ind'].iloc[0]].index) | set(memes_fix))
            else:
                top_perf_t_sel = top_perf_t.loc[(top_perf_t['Sharpe_nbrs'] - top_perf_t['Sharpe']) / top_perf_t['Sharpe'] > -(sh_nbr_d_max_ga + sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                if len(top_perf_t_sel) > tp_len_fr_th * n_top:
                    memes_pca = list(set(top_perf_t_sel['ind'].iloc[0].loc[top_perf_t_sel['ind'].iloc[0]].index) | set(memes_fix))
                else:
                    top_perf_t_sel = top_perf_t.loc[(top_perf_t['Sharpe_nbrs'] - top_perf_t['Sharpe']) / top_perf_t['Sharpe'] > -(sh_nbr_d_max_ga + 2 * sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                    if len(top_perf_t_sel) > tp_len_fr_th * n_top:
                        memes_pca = list(set(top_perf_t_sel['ind'].iloc[0].loc[top_perf_t_sel['ind'].iloc[0]].index) | set(memes_fix))
                    else:
                        memes_pca = list(set(top_perf_t['ind'].iloc[0].loc[top_perf_t['ind'].iloc[0]].index) | set(memes_fix))
        
        elif sh_disp_pen_met == "top top":
            
            top_perf_t_sel = top_perf_t[:n_top_top].loc[(top_perf_t[:n_top_top]['Sharpe_nbrs'] - top_perf_t[:n_top_top]['Sharpe']) / top_perf_t[:n_top_top]['Sharpe'] > -sh_nbr_d_max_ga].sort_values('Sharpe_nbh', ascending = False).copy()
            if len(top_perf_t_sel) > 0:
                memes_pca = list(set(top_perf_t_sel['ind'].iloc[0].loc[top_perf_t_sel['ind'].iloc[0]].index) | set(memes_fix))
            else:
                top_perf_t_sel = top_perf_t[:n_top_top].loc[(top_perf_t[:n_top_top]['Sharpe_nbrs'] - top_perf_t[:n_top_top]['Sharpe']) / top_perf_t[:n_top_top]['Sharpe'] > -(sh_nbr_d_max_ga + sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                if len(top_perf_t_sel) > 0:
                    memes_pca = list(set(top_perf_t_sel['ind'].iloc[0].loc[top_perf_t_sel['ind'].iloc[0]].index) | set(memes_fix))
                else:
                    top_perf_t_sel = top_perf_t[:n_top_top].loc[(top_perf_t[:n_top_top]['Sharpe_nbrs'] - top_perf_t[:n_top_top]['Sharpe']) / top_perf_t[:n_top_top]['Sharpe'] > -(sh_nbr_d_max_ga + 2 * sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                    if len(top_perf_t_sel) > 0:
                        memes_pca = list(set(top_perf_t_sel['ind'].iloc[0].loc[top_perf_t_sel['ind'].iloc[0]].index) | set(memes_fix))
                    else:
                        memes_pca = list(set(top_perf_t['ind'].iloc[0].loc[top_perf_t['ind'].iloc[0]].index) | set(memes_fix))
        
        elif sh_disp_pen_met == "double top top":
            
            n_top_top = int(1 / 4 * n_top)
            n_top_top_2 = int(1 / 2 * n_top)
            top_perf_t_sel = top_perf_t[:n_top_top].loc[(top_perf_t[:n_top_top]['Sharpe_nbrs'] - top_perf_t[:n_top_top]['Sharpe']) / top_perf_t[:n_top_top]['Sharpe'] > -sh_nbr_d_max_ga].sort_values('Sharpe_nbh', ascending = False).copy()
            if len(top_perf_t_sel) > 0:
                memes_pca = list(set(top_perf_t_sel['ind'].iloc[0].loc[top_perf_t_sel['ind'].iloc[0]].index) | set(memes_fix))
            else:
                top_perf_t_sel_1 = top_perf_t[:n_top_top].loc[(top_perf_t[:n_top_top]['Sharpe_nbrs'] - top_perf_t[:n_top_top]['Sharpe']) / top_perf_t[:n_top_top]['Sharpe'] > -(sh_nbr_d_max_ga + sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                top_perf_t_sel_2 = top_perf_t[n_top_top:n_top_top_2].loc[(top_perf_t[n_top_top:n_top_top_2]['Sharpe_nbrs'] - top_perf_t[n_top_top:n_top_top_2]['Sharpe']) / top_perf_t[n_top_top:n_top_top_2]['Sharpe'] > -sh_nbr_d_max_ga].sort_values('Sharpe_nbh', ascending = False).copy()
                top_perf_t_sel = pd.concat([top_perf_t_sel_1, top_perf_t_sel_2])
                if len(top_perf_t_sel) > 0:
                    memes_pca = list(set(top_perf_t_sel['ind'].iloc[0].loc[top_perf_t_sel['ind'].iloc[0]].index) | set(memes_fix))
                else:
                    top_perf_t_sel_1 = top_perf_t[:n_top_top].loc[(top_perf_t[:n_top_top]['Sharpe_nbrs'] - top_perf_t[:n_top_top]['Sharpe']) / top_perf_t[:n_top_top]['Sharpe'] > -(sh_nbr_d_max_ga + 2 * sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                    top_perf_t_sel_2 = top_perf_t[n_top_top:n_top_top_2].loc[(top_perf_t[n_top_top:n_top_top_2]['Sharpe_nbrs'] - top_perf_t[n_top_top:n_top_top_2]['Sharpe']) / top_perf_t[n_top_top:n_top_top_2]['Sharpe'] > -(sh_nbr_d_max_ga + sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                    top_perf_t_sel = pd.concat([top_perf_t_sel_1, top_perf_t_sel_2])
                    if len(top_perf_t_sel) > 0:
                        memes_pca = list(set(top_perf_t_sel['ind'].iloc[0].loc[top_perf_t_sel['ind'].iloc[0]].index) | set(memes_fix))
                    else:
                        memes_pca = list(set(top_perf_t['ind'].iloc[0].loc[top_perf_t['ind'].iloc[0]].index) | set(memes_fix))
        
        elif sh_disp_pen_met == "dif pen":
            
            top_perf_t['Sharpe_nbh_pen'] = top_perf_t['Sharpe_nbh'] + sh_dif_pen * (top_perf_t['Sharpe_nbrs'] - top_perf_t['Sharpe'])
            top_perf_t = top_perf_t.sort_values('Sharpe_nbh_pen', ascending = False)
            memes_pca = list(set(top_perf_t['ind'].iloc[0].loc[top_perf_t['ind'].iloc[0]].index) | set(memes_fix))
        
        if sh_disp_pen_met == "removal w/thr":

            joi_tr_perf_t_sel = joi_tr_perf_t.loc[(joi_tr_perf_t['Sharpe_nbrs'] - joi_tr_perf_t['Sharpe']) / joi_tr_perf_t['Sharpe'] > -sh_nbr_d_max_comb2].sort_values('Sharpe_nbh', ascending = False).copy()
            if len(joi_tr_perf_t_sel) > tp_len_fr_th * len(joi_tr_perf_t):
                top_comb2_str = joi_tr_perf_t_sel['comb2'].iloc[0]
            else:
                top_comb2_str = joi_tr_perf_t['comb2'].iloc[0]

        elif sh_disp_pen_met == "3-step removal w/thr":

            joi_tr_perf_t_sel = joi_tr_perf_t.loc[(joi_tr_perf_t['Sharpe_nbrs'] - joi_tr_perf_t['Sharpe']) / joi_tr_perf_t['Sharpe'] > -sh_nbr_d_max_comb2].sort_values('Sharpe_nbh', ascending = False).copy()
            if len(joi_tr_perf_t_sel) > tp_len_fr_th * len(joi_tr_perf_t):
                top_comb2_str = joi_tr_perf_t_sel['comb2'].iloc[0]
            else:
                joi_tr_perf_t_sel = joi_tr_perf_t.loc[(joi_tr_perf_t['Sharpe_nbrs'] - joi_tr_perf_t['Sharpe']) / joi_tr_perf_t['Sharpe'] > -(sh_nbr_d_max_comb2 + sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                if len(joi_tr_perf_t_sel) > tp_len_fr_th * len(joi_tr_perf_t):
                    top_comb2_str = joi_tr_perf_t_sel['comb2'].iloc[0]
                else:
                    joi_tr_perf_t_sel = joi_tr_perf_t.loc[(joi_tr_perf_t['Sharpe_nbrs'] - joi_tr_perf_t['Sharpe']) / joi_tr_perf_t['Sharpe'] > -(sh_nbr_d_max_comb2 + 2 * sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                    if len(joi_tr_perf_t_sel) > tp_len_fr_th * len(joi_tr_perf_t):
                        top_comb2_str = joi_tr_perf_t_sel['comb2'].iloc[0]
                    else:
                        top_comb2_str = joi_tr_perf_t['comb2'].iloc[0]

        elif sh_disp_pen_met == "top top":

            joi_tr_perf_t_sel = joi_tr_perf_t[:n_top_top].loc[(joi_tr_perf_t[:n_top_top]['Sharpe_nbrs'] - joi_tr_perf_t[:n_top_top]['Sharpe']) / joi_tr_perf_t[:n_top_top]['Sharpe'] > -sh_nbr_d_max_comb2].sort_values('Sharpe_nbh', ascending = False).copy()
            if len(joi_tr_perf_t_sel) > 0:
                top_comb2_str = joi_tr_perf_t_sel['comb2'].iloc[0]
            else:
                joi_tr_perf_t_sel = joi_tr_perf_t[:n_top_top].loc[(joi_tr_perf_t[:n_top_top]['Sharpe_nbrs'] - joi_tr_perf_t[:n_top_top]['Sharpe']) / joi_tr_perf_t[:n_top_top]['Sharpe'] > -(sh_nbr_d_max_comb2 + sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                if len(joi_tr_perf_t_sel) > 0:
                    top_comb2_str = joi_tr_perf_t_sel['comb2'].iloc[0]
                else:
                    joi_tr_perf_t_sel = joi_tr_perf_t[:n_top_top].loc[(joi_tr_perf_t[:n_top_top]['Sharpe_nbrs'] - joi_tr_perf_t[:n_top_top]['Sharpe']) / joi_tr_perf_t[:n_top_top]['Sharpe'] > -(sh_nbr_d_max_comb2 + 2 * sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                    if len(joi_tr_perf_t_sel) > 0:
                        top_comb2_str = joi_tr_perf_t_sel['comb2'].iloc[0]
                    else:
                        top_comb2_str = joi_tr_perf_t['comb2'].iloc[0]

        elif sh_disp_pen_met == "double top top":

            n_top_top = int(1 / 4 * len(joi_tr_perf_t))
            n_top_top_2 = int(1 / 2 * len(joi_tr_perf_t))
            joi_tr_perf_t_sel = joi_tr_perf_t[:n_top_top].loc[(joi_tr_perf_t[:n_top_top]['Sharpe_nbrs'] - joi_tr_perf_t[:n_top_top]['Sharpe']) / joi_tr_perf_t[:n_top_top]['Sharpe'] > -sh_nbr_d_max_comb2].sort_values('Sharpe_nbh', ascending = False).copy()
            if len(joi_tr_perf_t_sel) > 0:
                top_comb2_str = joi_tr_perf_t_sel['comb2'].iloc[0]
            else:
                joi_tr_perf_t_sel_1 = joi_tr_perf_t[:n_top_top].loc[(joi_tr_perf_t[:n_top_top]['Sharpe_nbrs'] - joi_tr_perf_t[:n_top_top]['Sharpe']) / joi_tr_perf_t[:n_top_top]['Sharpe'] > -(sh_nbr_d_max_comb2 + sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                joi_tr_perf_t_sel_2 = joi_tr_perf_t[n_top_top:n_top_top_2].loc[(joi_tr_perf_t[n_top_top:n_top_top_2]['Sharpe_nbrs'] - joi_tr_perf_t[n_top_top:n_top_top_2]['Sharpe']) / joi_tr_perf_t[n_top_top:n_top_top_2]['Sharpe'] > -sh_nbr_d_max_comb2].sort_values('Sharpe_nbh', ascending = False).copy()
                joi_tr_perf_t_sel = pd.concat([joi_tr_perf_t_sel_1, joi_tr_perf_t_sel_2])
                if len(joi_tr_perf_t_sel) > 0:
                    top_comb2_str = joi_tr_perf_t_sel['comb2'].iloc[0]
                else:
                    joi_tr_perf_t_sel_1 = joi_tr_perf_t[:n_top_top].loc[(joi_tr_perf_t[:n_top_top]['Sharpe_nbrs'] - joi_tr_perf_t[:n_top_top]['Sharpe']) / joi_tr_perf_t[:n_top_top]['Sharpe'] > -(sh_nbr_d_max_comb2 + 2 * sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                    joi_tr_perf_t_sel_2 = joi_tr_perf_t[n_top_top:n_top_top_2].loc[(joi_tr_perf_t[n_top_top:n_top_top_2]['Sharpe_nbrs'] - joi_tr_perf_t[n_top_top:n_top_top_2]['Sharpe']) / joi_tr_perf_t[n_top_top:n_top_top_2]['Sharpe'] > -(sh_nbr_d_max_comb2 + sh_nbr_d_inc)].sort_values('Sharpe_nbh', ascending = False).copy()
                    joi_tr_perf_t_sel = pd.concat([joi_tr_perf_t_sel_1, joi_tr_perf_t_sel_2])
                    if len(joi_tr_perf_t_sel) > 0:
                        top_comb2_str = joi_tr_perf_t_sel['comb2'].iloc[0]
                    else:
                        top_comb2_str = joi_tr_perf_t['comb2'].iloc[0]

        elif sh_disp_pen_met == "dif pen":

            joi_tr_perf_t['Sharpe_nbh_pen'] = joi_tr_perf_t['Sharpe_nbh'] + sh_dif_pen * (joi_tr_perf_t['Sharpe_nbrs'] - joi_tr_perf_t['Sharpe'])
            joi_tr_perf_t = joi_tr_perf_t.sort_values('Sharpe_nbh_pen', ascending = False)
            top_comb2_str = joi_tr_perf_t['comb2'].iloc[0]
        
        top_comb2_str_sep0 = top_comb2_str.replace(' ', '')[1:-1].split("),(")
        top_comb2_str_sep = [c[1:] if i == 0 else c[:-1] if i == len(top_comb2_str_sep0) - 1 else c for i, c in enumerate(top_comb2_str_sep0)]
        top_comb2_list = []
        for c in top_comb2_str_sep:
            if c[0] == '0':
                top_comb2_list.append((0,0))
            else:
                top_comb2_list.append((int(c[0]), (int(c[3]), int(c[5]))))
        top_comb2 = tuple(top_comb2_list)
        nPC_ord = dict()
        for i, m in enumerate(top_combs_t.columns):
            if top_comb2[i] != (0, 0):
                nPC_ord[m] = top_comb2[i]
        memes_tr_fin = list(nPC_ord.keys())
    
        res_pred_oos = ret_pred_son_rf(df_full_r_oos, cal_int, recal, memes_pca, nPC_ord, npcs_max, h)
        rets_g_t, w_t, port_val_t = joi_rets_on_rf(res_pred_oos, memes_tr_fin, df_full_r_oos, df_l_r_oos, df_h_r_oos, cal_int, recal, h, dates_per, cor_mats, capital_oos, lev_cap, risk_av, varis_oos, buf, sl, fe_bas_pr_oos, fe_bas_o_oos, fe_bas_c_oos, fr_pr_oos, fr_o_oos, fr_c_oos, im_pr_oos, im_re_oos, imp_mod_params['cc_exp'])
        capital_oos = port_val_t.iloc[-1]
    
        oos_df.rets_g_oos = pd.concat([oos_df.rets_g_oos, rets_g_t])
        oos_df.w_oos = pd.concat([oos_df.w_oos, w_t])
        if oos_df.port_val_oos.empty:
            oos_df.port_val_oos = port_val_t
        else:
            oos_df.port_val_oos = pd.concat([oos_df.port_val_oos, port_val_t[1:]])