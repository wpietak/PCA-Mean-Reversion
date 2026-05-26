import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from datetime import datetime as dt
from datetime import timedelta as td
from statsmodels.tsa.arima.model import ARIMA
import warnings
import random
import copy
from src.performance import *


def ar2_rf_test(i, memes_pca, memes_sub, df_full_r, cal_int, recal, init_capital, risk_av, varis, fe_bas_pr, fr_pr, im_pr):

    t = df_full_r.index[0] + td(hours = cal_int)
    res_pred_full = pd.DataFrame()
    
    while t < df_full_r.index.max():
        
        df_r_cal = df_full_r[memes_pca][(t - td(hours = cal_int)):(t - td(hours = 1))]
        df_r_dep = df_full_r[memes_pca][(t - td(hours = 2)):(t + td(hours = recal - 1))]
            
        pca = PCA(n_components = i)
        pca.fit(df_r_cal)
        
        comps_r_cal = pca.transform(df_r_cal)
        sys_r_cal = pca.inverse_transform(comps_r_cal)
        resids_cal = df_r_cal - sys_r_cal
            
        sys_r = df_r_dep.dot(pca.components_.T).dot(pca.components_)
        sys_r.columns = df_r_dep.columns
        resids = df_r_dep - sys_r

        res_pred = pd.DataFrame()
            
        for m in memes_sub:
            rho_1, rho_2 = resids_cal[m].autocorr(1), resids_cal[m].autocorr(2)
            beta_1, beta_2 = (rho_1 - rho_1 * rho_2) / (1 - rho_1 ** 2), (rho_2 - rho_1 ** 2) / (1 - rho_1 ** 2)
            res_pred[m] = (beta_1 * resids[m].shift(1) + beta_2 * resids[m].shift(2))[2:]
            
        res_pred_full = pd.concat([res_pred_full, res_pred])

        t += td(hours = recal)

    mu = res_pred_full.copy()
    fc = fe_bas_pr[memes_sub] + np.sign(mu) * fr_pr[memes_sub]
    mu = (mu - np.sign(mu) * fc) * (abs(mu) > fc)
    w = mu / (2 * im_pr[memes_sub] * init_capital + risk_av * varis[memes_sub])
    ar2_rf_rets_i = w * df_full_r[memes_sub][cal_int:] - abs(w) * (fc + im_pr[memes_sub] * init_capital * abs(w))

    return ar2_rf_rets_i


def goa_ar2_rf(n_pop, n_ite, b, memes_all, memes, memes_fix, npcd, npcs_max, npcs_w, r_th_an, n_ret_del, conc_pen_ga, bkt_w, p_tour, n_tour, p_cross, p_mut, imi_rat, args_fix, memes_to_pick_prev = None, cache = None, init_pop = None):

    if memes_to_pick_prev is None:
        memes_to_pick = list(set(memes_all) - set(memes_fix))
    else:
        memes_to_pick = memes_to_pick_prev
    min_te = min(npcd) - len(memes_fix)
    max_te = max(npcd) - len(memes_fix)

    r_th = (1 + r_th_an) ** (len(args_fix[0][args_fix[1]:]) / (365 * 24)) - 1
    
    pis = np.array([np.exp(- b * i) for i in range(1, n_pop + 1)])
    pis = pis / sum(pis)

    n_imi = round(imi_rat * n_pop)

    if init_pop is None:
        popu = [pd.Series(list(pd.Series(memes_to_pick).isin(random.sample(memes_to_pick, random.choice(npcd) - len(memes_fix)))), index = memes_to_pick) for i in range(n_pop)]
    else:
        popu = init_pop
        n_pop = len(init_pop)
    
    gens_perf = dict()
    if cache is None:
        popu_perf_cache = dict()
    else:
        popu_perf_cache = cache
    
    for ni in range(n_ite):

        popu_perf_nbh = dict()

        for ind_no, ind_main in enumerate(popu):

            bkt_nbr_1 = ind_main.copy()
            bkt_nbr_2 = ind_main.copy()
            el_swa = [random.choice(ind_main[ind_main].index), random.choice(ind_main[~ind_main].index)]
            el_swi = random.choice(memes_to_pick)
            bkt_nbr_1.loc[el_swa] = ~bkt_nbr_1.loc[el_swa]
            bkt_nbr_2.loc[el_swi] = ~bkt_nbr_2.loc[el_swi]
            
            for ind in [ind_main, bkt_nbr_1, bkt_nbr_2]:
            
                ind_tup = tuple(ind)
                
                if {ind_tup}.issubset(set(popu_perf_cache.keys())):
                    continue
                #elif ni > 0:
                #    if {ind_tup}.issubset(set(popu_perf_old.keys())):
                #        popu_perf_dict[ind_tup] = popu_perf_old[ind_tup]
                #        continue
                
                memes_picked = list(ind.loc[ind].index)
                memes_pca = list(set(memes_fix) | set(memes_picked))
                memes_tr = list(set(memes_pca) & set(memes))
                rets_ind = dict()
                rets_cum = pd.DataFrame()
    
                for i in range(1, npcs_max + 1):
    
                    rets_i = ar2_rf_test(i, memes_pca, memes_tr, *args_fix)
                    for j in range(n_ret_del):
                        for m in memes_tr:
                            rets_i.loc[rets_i.idxmax()[m], m] = 0.
                    rets_cum = pd.concat([rets_cum, ((1 + rets_i).prod() - 1).rename(i)], axis = 1)
                    rets_ind[i] = rets_i
    
                rets_cum_avg = pd.DataFrame()
                for i in range(1, npcs_max + 1):
                    if i == 1:
                        rets_cum_avg[i] = npcs_w * rets_cum[i] + (1 - npcs_w) / 2 * (rets_cum[i + 1] + rets_cum[i + 2])
                    elif i == npcs_max:
                        rets_cum_avg[i] = npcs_w * rets_cum[i] + (1 - npcs_w) / 2 * (rets_cum[i - 1] + rets_cum[i - 2])
                    else:
                        rets_cum_avg[i] = npcs_w * rets_cum[i] + (1 - npcs_w) / 2 * (rets_cum[i - 1] + rets_cum[i + 1])
                
                top_is = pd.concat([rets_cum.idxmax(axis = 1).rename('i'), rets_cum.max(axis = 1).rename('r')], axis = 1)
                top_is_sel = top_is.loc[top_is.index.isin(memes_fix) | (top_is['r'] > r_th)]
                rets_comb, rets_comb_1, rets_comb_2 = [pd.DataFrame() for j in range(3)]
                for m in top_is_sel.iterrows():
                    top_i = int(m[1]['i'])
                    if top_i == 1:
                        i_neig = [top_i + 1, top_i + 2]
                    elif top_i == npcs_max:
                        i_neig = [top_i - 1, top_i - 2]
                    else:
                        i_neig = [top_i - 1, top_i + 1]
                    random.shuffle(i_neig)
                    rets_comb = pd.concat([rets_comb, rets_ind[top_i][m[0]]], axis = 1)
                    rets_comb_1 = pd.concat([rets_comb_1, rets_ind[i_neig[0]][m[0]]], axis = 1)
                    rets_comb_2 = pd.concat([rets_comb_2, rets_ind[i_neig[1]][m[0]]], axis = 1)
                rets, rets_1, rets_2 = rets_comb.sum(axis = 1), rets_comb_1.sum(axis = 1), rets_comb_2.sum(axis = 1)
                #conc = (((1 + rets_comb).prod() - 1) ** 2).sum() / (((1 + rets).prod() - 1) ** 2)
                #conc_1 = (((1 + rets_comb_1).prod() - 1) ** 2).sum() / (((1 + rets_1).prod() - 1) ** 2)
                #conc_2 = (((1 + rets_comb_2).prod() - 1) ** 2).sum() / (((1 + rets_2).prod() - 1) ** 2)
                conc = np.max([sharpe(rets) - sharpe(rets_comb.drop(columns = m).sum(axis = 1)) for m in rets_comb.columns])
                conc_1 = np.max([sharpe(rets_1) - sharpe(rets_comb_1.drop(columns = m).sum(axis = 1)) for m in rets_comb_1.columns])
                conc_2 = np.max([sharpe(rets_2) - sharpe(rets_comb_2.drop(columns = m).sum(axis = 1)) for m in rets_comb_2.columns])
                #conc_t = np.max([sharpe(rets) - sharpe(rets.loc[~rets.index.isin(rets[ti:(ti + td(hours = conc_int_td))].index)]) for ti in rets.index[:(-conc_int_td)]])
                #conc_t_1 = np.max([sharpe(rets_1) - sharpe(rets_1.loc[~rets_1.index.isin(rets_1[ti:(ti + td(hours = conc_int_td))].index)]) for ti in rets_1.index[:(-conc_int_td)]])
                #conc_t_2 = np.max([sharpe(rets_2) - sharpe(rets_2.loc[~rets_2.index.isin(rets_2[ti:(ti + td(hours = conc_int_td))].index)]) for ti in rets_2.index[:(-conc_int_td)]])
                #conc, conc_1, conc_2 = (conc_m + conc_t) / 2, (conc_m_1 + conc_t_1) / 2, (conc_m_2 + conc_t_2) / 2
                #popu_perf_cache[ind_tup] = npcs_w * sharpe(rets) * (1 - conc_pen_ga * conc) + (1 - npcs_w) / 2 * (sharpe(rets_1) * (1 - conc_pen_ga * conc_1) + sharpe(rets_2) * (1 - conc_pen_ga * conc_2))
                popu_perf_cache[ind_tup] = npcs_w * (sharpe(rets) - conc_pen_ga * conc) + (1 - npcs_w) / 2 * (sharpe(rets_1) - conc_pen_ga * conc_1 + sharpe(rets_2) - conc_pen_ga * conc_2)
            
            popu_perf_nbh[ind_no] = bkt_w * popu_perf_cache[tuple(ind_main)] + (1 - bkt_w) / 2 * (popu_perf_cache[tuple(bkt_nbr_1)] + popu_perf_cache[tuple(bkt_nbr_2)])

        popu_perf = pd.DataFrame()
        popu_perf['ind'] = popu
        popu_perf['Sharpe'] = [popu_perf_cache[tuple(ind_main)] for ind_main in popu]
        popu_perf['Sharpe_nbh'] = [popu_perf_nbh[ind_no] for ind_no in range(n_pop)]
        popu_perf = popu_perf.sort_values('Sharpe_nbh', ascending = False)
        gens_perf[ni] = popu_perf
        
        new_popu = list()
        new_popu.append(popu_perf['ind'].iloc[0])
        if (n_pop - n_imi) % 2 == 0:
            new_popu.append(popu_perf['ind'].iloc[1])
        
        for pair in range((n_pop - n_imi - 1) // 2):
            
            if np.random.uniform(low = 0., high = 1.) < 1 - p_tour:
                parent_1 = random.choices(popu_perf['ind'], weights = pis, k = 1)[0]
            else:
                parent_1 = popu_perf.sample(frac = n_tour / n_pop).sort_values('Sharpe_nbh', ascending = False).iloc[0, 0]
            if np.random.uniform(low = 0., high = 1.) < 1 - p_tour:
                parent_2 = random.choices(popu_perf['ind'], weights = pis, k = 1)[0]
            else:
                parent_2 = popu_perf.sample(frac = n_tour / n_pop).sort_values('Sharpe_nbh', ascending = False).iloc[0, 0]
            
            child_1, child_2 = parent_1.copy(), parent_2.copy()
            parents = pd.concat([parent_1, parent_2], axis = 1)
            parents_dif = parents.loc[parents[0] != parents[1]]
            
            if np.random.uniform(low = 0., high = 1.) < 0.5:
                cross_tf = np.random.uniform(low = 0., high = 1., size = len(parents_dif)) < p_cross
                el_cr = parents_dif.loc[pd.Series(cross_tf, index = parents_dif.index)]
                child_1.loc[el_cr.index] = parent_2.loc[el_cr.index]
                child_2.loc[el_cr.index] = parent_1.loc[el_cr.index]
                if sum(child_1) < min_te:
                    child_1[random.sample(list(parent_2.loc[el_cr.index][~parent_2.loc[el_cr.index]].index), k = min_te - sum(child_1))] = True
                elif sum(child_1) > max_te:
                    child_1[random.sample(list(parent_2.loc[el_cr.index][parent_2.loc[el_cr.index]].index), k = sum(child_1) - max_te)] = False
                if sum(child_2) < min_te:
                    child_2[random.sample(list(parent_1.loc[el_cr.index][~parent_1.loc[el_cr.index]].index), k = min_te - sum(child_2))] = True
                elif sum(child_2) > max_te:
                    child_2[random.sample(list(parent_1.loc[el_cr.index][parent_1.loc[el_cr.index]].index), k = sum(child_2) - max_te)] = False
            else:
                nec_max = min(len(parents_dif.loc[parents_dif[0]]), len(parents_dif.loc[parents_dif[1]]))
                cross_tf = np.random.uniform(low = 0., high = 1., size = nec_max) < p_cross
                if nec_max == 0:
                    frac_1, frac_2 = 0, 0
                else:
                    frac_1, frac_2 = nec_max / len(parents_dif.loc[parents_dif[0]]), nec_max / len(parents_dif.loc[~parents_dif[0]])
                parents_dif_sub1 = parents_dif.loc[parents_dif[0]].sample(frac = frac_1)
                parents_dif_sub2 = parents_dif.loc[~parents_dif[0]].sample(frac = frac_2)
                el_1_cr = parents_dif_sub1.loc[pd.Series(cross_tf, index = parents_dif_sub1.index)]
                el_2_cr = parents_dif_sub2.loc[pd.Series(cross_tf, index = parents_dif_sub2.index)]
                child_1.loc[el_1_cr.index] = parent_2.loc[el_1_cr.index]
                child_1.loc[el_2_cr.index] = parent_2.loc[el_2_cr.index]
                child_2.loc[el_1_cr.index] = parent_1.loc[el_1_cr.index]
                child_2.loc[el_2_cr.index] = parent_1.loc[el_2_cr.index]
            
            if np.random.uniform(low = 0., high = 1.) < 0.5:
                if sum(child_1) == min_te:
                    pot_mut_1 = random.choice(child_1[~child_1].index)
                elif sum(child_1) == max_te:
                    pot_mut_1 = random.choice(child_1[child_1].index)
                else:
                    pot_mut_1 = random.choice(memes_to_pick)
                if np.random.uniform(low = 0., high = 1.) < p_mut:
                    child_1.loc[pot_mut_1] = ~child_1.loc[pot_mut_1]
                if sum(child_2) == min_te:
                    pot_mut_2 = random.choice(child_2[~child_2].index)
                elif sum(child_2) == max_te:
                    pot_mut_2 = random.choice(child_2[child_2].index)
                else:
                    pot_mut_2 = random.choice(memes_to_pick)
                if np.random.uniform(low = 0., high = 1.) < p_mut:
                    child_2.loc[pot_mut_2] = ~child_2.loc[pot_mut_2]
            else:
                pot_mut_1 = random.choice(child_1[child_1].index), random.choice(child_1[~child_1].index)
                if np.random.uniform(low = 0., high = 1.) < p_mut:
                    child_1.loc[list(pot_mut_1)] = ~child_1.loc[list(pot_mut_1)]
                pot_mut_2 = random.choice(child_2[child_2].index), random.choice(child_2[~child_2].index)
                if np.random.uniform(low = 0., high = 1.) < p_mut:
                    child_2.loc[list(pot_mut_2)] = ~child_2.loc[list(pot_mut_2)]
            
            new_popu.append(child_1), new_popu.append(child_2)

        for imi in range(n_imi):
            imig = pd.Series(list(pd.Series(memes_to_pick).isin(random.sample(memes_to_pick, random.choice(npcd) - len(memes_fix)))), index = memes_to_pick)
            new_popu.append(imig)
        
        popu = new_popu
        #popu_perf_dict_old = popu_perf_dict.copy()

    return gens_perf, popu_perf_cache


def top_ind_nbh(gens_perf_fin, popu_perf_cache, n_top, n_nbr, swa_rat, memes, memes_fix, npcs_max, npcs_w, r_th_an, n_ret_del, conc_pen_ga, main_bkt_w, args_fix):

    popu_perf = gens_perf_fin.copy()
    popu_perf['ind_tup'] = [tuple(ind) for ind in popu_perf['ind']]
    top_inds_tup = list(popu_perf[['ind_tup', 'Sharpe_nbh']].groupby('ind_tup', as_index = False).max().sort_values('Sharpe_nbh', ascending = False)[:n_top]['ind_tup'])
    top_inds = [pd.Series(ind, index = popu_perf['ind'].iloc[0].index) for ind in top_inds_tup]
    top_perf_nbh = dict()
    top_perf_nbh_dev2 = dict()
    r_th = (1 + r_th_an) ** (len(args_fix[0][args_fix[1]:]) / (365 * 24)) - 1

    for ind_main in top_inds:

        ind_nbrs = []
        for i in range(n_nbr):
            bkt_nbr_i = ind_main.copy()
            if i < round(n_nbr * swa_rat):
                el_sw = [random.choice(ind_main[ind_main].index), random.choice(ind_main[~ind_main].index)]
            else:
                el_sw = random.choice(ind_main.index)
            bkt_nbr_i.loc[el_sw] = ~bkt_nbr_i.loc[el_sw]
            ind_nbrs.append(bkt_nbr_i)
            
        for ind in ind_nbrs:
            
            ind_tup = tuple(ind)
                
            if {ind_tup}.issubset(set(popu_perf_cache.keys())):
                continue
                
            memes_picked = list(ind.loc[ind].index)
            memes_pca = list(set(memes_fix) | set(memes_picked))
            memes_tr = list(set(memes_pca) & set(memes))
            rets_ind = dict()
            rets_cum = pd.DataFrame()
    
            for i in range(1, npcs_max + 1):
    
                rets_i = ar2_rf_test(i, memes_pca, memes_tr, *args_fix)
                for j in range(n_ret_del):
                    for m in memes_tr:
                        rets_i.loc[rets_i.idxmax()[m], m] = 0.
                rets_cum = pd.concat([rets_cum, ((1 + rets_i).prod() - 1).rename(i)], axis = 1)
                rets_ind[i] = rets_i
    
            rets_cum_avg = pd.DataFrame()
            for i in range(1, npcs_max + 1):
                if i == 1:
                    rets_cum_avg[i] = npcs_w * rets_cum[i] + (1 - npcs_w) / 2 * (rets_cum[i + 1] + rets_cum[i + 2])
                elif i == npcs_max:
                    rets_cum_avg[i] = npcs_w * rets_cum[i] + (1 - npcs_w) / 2 * (rets_cum[i - 1] + rets_cum[i - 2])
                else:
                    rets_cum_avg[i] = npcs_w * rets_cum[i] + (1 - npcs_w) / 2 * (rets_cum[i - 1] + rets_cum[i + 1])
                
            top_is = pd.concat([rets_cum.idxmax(axis = 1).rename('i'), rets_cum.max(axis = 1).rename('r')], axis = 1)
            top_is_sel = top_is.loc[top_is.index.isin(memes_fix) | (top_is['r'] > r_th)]
            rets_comb, rets_comb_1, rets_comb_2 = [pd.DataFrame() for j in range(3)]
            for m in top_is_sel.iterrows():
                top_i = int(m[1]['i'])
                if top_i == 1:
                    i_neig = [top_i + 1, top_i + 2]
                elif top_i == npcs_max:
                    i_neig = [top_i - 1, top_i - 2]
                else:
                    i_neig = [top_i - 1, top_i + 1]
                random.shuffle(i_neig)
                rets_comb = pd.concat([rets_comb, rets_ind[top_i][m[0]]], axis = 1)
                rets_comb_1 = pd.concat([rets_comb_1, rets_ind[i_neig[0]][m[0]]], axis = 1)
                rets_comb_2 = pd.concat([rets_comb_2, rets_ind[i_neig[1]][m[0]]], axis = 1)
            rets, rets_1, rets_2 = rets_comb.sum(axis = 1), rets_comb_1.sum(axis = 1), rets_comb_2.sum(axis = 1)
            #conc = (((1 + rets_comb).prod() - 1) ** 2).sum() / (((1 + rets).prod() - 1) ** 2)
            #conc_1 = (((1 + rets_comb_1).prod() - 1) ** 2).sum() / (((1 + rets_1).prod() - 1) ** 2)
            #conc_2 = (((1 + rets_comb_2).prod() - 1) ** 2).sum() / (((1 + rets_2).prod() - 1) ** 2)
            conc = np.max([sharpe(rets) - sharpe(rets_comb.drop(columns = m).sum(axis = 1)) for m in rets_comb.columns])
            conc_1 = np.max([sharpe(rets_1) - sharpe(rets_comb_1.drop(columns = m).sum(axis = 1)) for m in rets_comb_1.columns])
            conc_2 = np.max([sharpe(rets_2) - sharpe(rets_comb_2.drop(columns = m).sum(axis = 1)) for m in rets_comb_2.columns])
            #conc_t = np.max([sharpe(rets) - sharpe(rets.loc[~rets.index.isin(rets[ti:(ti + td(hours = conc_int_td))].index)]) for ti in rets.index[:(-conc_int_td)]])
            #conc_t_1 = np.max([sharpe(rets_1) - sharpe(rets_1.loc[~rets_1.index.isin(rets_1[ti:(ti + td(hours = conc_int_td))].index)]) for ti in rets_1.index[:(-conc_int_td)]])
            #conc_t_2 = np.max([sharpe(rets_2) - sharpe(rets_2.loc[~rets_2.index.isin(rets_2[ti:(ti + td(hours = conc_int_td))].index)]) for ti in rets_2.index[:(-conc_int_td)]])
            #conc, conc_1, conc_2 = (conc_m + conc_t) / 2, (conc_m_1 + conc_t_1) / 2, (conc_m_2 + conc_t_2) / 2
            #popu_perf_cache[ind_tup] = npcs_w * sharpe(rets) * (1 - conc_pen_ga * conc) + (1 - npcs_w) / 2 * (sharpe(rets_1) * (1 - conc_pen_ga * conc_1) + sharpe(rets_2) * (1 - conc_pen_ga * conc_2))
            popu_perf_cache[ind_tup] = npcs_w * (sharpe(rets) - conc_pen_ga * conc) + (1 - npcs_w) / 2 * (sharpe(rets_1) - conc_pen_ga * conc_1 + sharpe(rets_2) - conc_pen_ga * conc_2)
            
        top_perf_nbh[tuple(ind_main)] = main_bkt_w * popu_perf_cache[tuple(ind_main)] + (1 - main_bkt_w) * np.mean([popu_perf_cache[tuple(nbr)] for nbr in ind_nbrs])
        top_perf_nbh_dev2[tuple(ind_main)] = np.mean([(popu_perf_cache[tuple(nbr)] - popu_perf_cache[tuple(ind_main)]) ** 2 for nbr in ind_nbrs])

    top_perf = pd.DataFrame()
    top_perf['ind'] = top_inds
    top_perf['Sharpe'] = [popu_perf_cache[tuple(ind_main)] for ind_main in top_inds]
    top_perf['Sharpe_nbh'] = [top_perf_nbh[tuple(ind_main)] for ind_main in top_inds]
    top_perf['Sharpe_nbrs'] = (top_perf['Sharpe_nbh'] - main_bkt_w * top_perf['Sharpe']) / (1 - main_bkt_w)
    top_perf['Sharpe_nbh_msd'] = [top_perf_nbh_dev2[tuple(ind_main)] for ind_main in top_inds]
    top_perf = top_perf.sort_values('Sharpe_nbh', ascending = False)

    return top_perf


def ret_pred_don_rf(df_full_r, cal_int, recal, memes_pca, memes, npcs_max, orders, h):

    memes_tr = list(set(memes_pca) & set(memes))
    armas_rf_pred = dict()

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
    
        armas_rf_pred[i] = dict()
        
        for o in orders:

            armas_rf_pred[i][o] = pd.DataFrame()    
            p, q = o
        
            t = df_full_r.index[0] + td(hours = cal_int)
                    
            while t < df_full_r.index.max():
            
                resids_cal = resids_cal_full[t]
                resids = resids_full[t]
            
                res_pred = pd.DataFrame()
                
                for m in memes_tr:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        arma = ARIMA(resids_cal[m], order = [p, 0, q], freq = td(hours = h)).fit()
                    arma_up = arma.append(resids[m], refit = False)
                    res_pred[m] = arma_up.predict(start = t)
                    if p > 0:
                        if arma.arparams[0] > 0.85:
                            res_pred[m] = 0.             
                
                armas_rf_pred[i][o] = pd.concat([armas_rf_pred[i][o], res_pred])
                        
                t += td(hours = recal)

    return armas_rf_pred


def ind_rets_on_rf(res_pred, memes_sub, df_full_r, df_l_r, df_h_r, cal_int, recal, h, init_capital, lev_cap, risk_av, varis, buf, sl, fe_bas_pr, fe_bas_o, fe_bas_c, fr_pr, fr_o, fr_c, im_pr, im_re, volu_exp_fin):

    t_0 = df_full_r.index[0] + td(hours = cal_int)
    rets_g = df_full_r[(cal_int - 1):].copy()[memes_sub]
    rets_g.iloc[:, :] = 0.
    w = rets_g.copy()
    #capital = init_capital
    capital = pd.Series(init_capital, index = memes_sub)
    #port_val = pd.Series(capital, index = df_r[(cal_int - 1):].index)
    imp_pred = im_pr[memes_sub]
    imp_real = im_re[memes_sub]
    pos_val_bc, pos_val_ac, pnl_o, sl_trig = [pd.Series(0., index = memes_sub) for i in range(4)]

    df_r_dep = df_full_r[memes_sub]
    df_l_r_dep = df_l_r[memes_sub]
    df_h_r_dep = df_h_r[memes_sub]
            
    df_r_dep_cum = (1 + df_r_dep).cumprod()
    df_r_dep_cum = pd.concat([pd.DataFrame(index = [df_r_dep_cum.index.min() - td(hours = 1)], data = 1, columns = df_r_dep_cum.columns), df_r_dep_cum])
    df_r_dep_h = df_r_dep_cum.loc[df_r_dep_cum.index.hour % h == h - 1].pct_change()[1:]
    
    df_l_r_dep_cum = df_r_dep_cum.copy()
    df_l_r_dep_cum.iloc[1:, :] = df_r_dep_cum.shift(1)[1:] * (1 + df_l_r_dep[memes_sub])
    df_l_r_dep_cum['day'] = pd.Series(df_l_r_dep_cum.index, index = df_l_r_dep_cum.index).dt.floor('d')
    df_l_r_dep_cum['floordiv_h'] = df_l_r_dep_cum.index.hour // h
    df_l_r_dep_cum_h = df_l_r_dep_cum.groupby(['day', 'floordiv_h'], as_index = False).min()
    df_l_r_dep_cum_h['time'] = df_l_r_dep_cum_h['day'] + td(hours = h) * df_l_r_dep_cum_h['floordiv_h'] + td(hours = h - 1)
    df_l_r_dep_cum_h = df_l_r_dep_cum_h.set_index('time')[memes_sub]
    df_l_r_dep_h = df_l_r_dep_cum_h / df_r_dep_cum.loc[df_r_dep_cum.index.hour % h == h - 1].shift(1) - 1
    df_l_r_dep_h = df_l_r_dep_h.loc[df_r_dep_h.index]
    
    df_h_r_dep_cum = df_r_dep_cum.copy()
    df_h_r_dep_cum.iloc[1:, :] = df_r_dep_cum.shift(1)[1:] * (1 + df_h_r_dep[memes_sub])
    df_h_r_dep_cum['day'] = pd.Series(df_h_r_dep_cum.index, index = df_h_r_dep_cum.index).dt.floor('d')
    df_h_r_dep_cum['floordiv_h'] = df_h_r_dep_cum.index.hour // h
    df_h_r_dep_cum_h = df_h_r_dep_cum.groupby(['day', 'floordiv_h'], as_index = False).max()
    df_h_r_dep_cum_h['time'] = df_h_r_dep_cum_h['day'] + td(hours = h) * df_h_r_dep_cum_h['floordiv_h'] + td(hours = h - 1)
    df_h_r_dep_cum_h = df_h_r_dep_cum_h.set_index('time')[memes_sub]
    df_h_r_dep_h = df_h_r_dep_cum_h / df_r_dep_cum.loc[df_r_dep_cum.index.hour % h == h - 1].shift(1) - 1
    df_h_r_dep_h = df_h_r_dep_h.loc[df_r_dep_h.index]
    
    fixed_cost_pred = fe_bas_pr[memes_sub] + fr_pr[memes_sub] * np.sign(res_pred[memes_sub])
    fixed_cost_o = fe_bas_o[memes_sub] + fr_o[memes_sub] * np.sign(res_pred[memes_sub])
    fixed_cost_c = fe_bas_c[memes_sub] + fr_c[memes_sub] * np.sign(res_pred[memes_sub])
        
    mu = abs(res_pred) - fixed_cost_pred
    mu = mu * (mu > buf) * np.sign(res_pred)
    mu = pd.concat([pd.DataFrame(0., columns = [t_0 - td(hours = h)], index = memes_sub).T, mu])
    #cor_mat = df_r_cal_h.corr()
        
    for i in res_pred.index:
            
        keep_pos = (mu.loc[i] * mu.loc[i - td(hours = h)] > 0) & (sl_trig == 0)
        outst_pos = pos_val_bc * keep_pos
        pos_val_fin = outst_pos + pos_val_ac * (1 - keep_pos)
        pnl = pnl_o + (pos_val_fin - capital * w.loc[i - td(hours = h)]) * np.sign(w.loc[i - td(hours = h)])
        rets_g.loc[i - td(hours = h)] = pnl / capital
        capital += pnl #.sum()
        #port_val.loc[i - td(hours = h)] = capital
        pnl_o = pd.Series(0., index = memes_sub)
            
        m_sig = (mu.loc[i] != 0)
        d = m_sig.sum()
            
        if d > 0:
                
            #cov_mat = np.diag(vols.loc[i, m_sig]).dot(cor_mat.loc[m_sig, m_sig]).dot(np.diag(vols.loc[i, m_sig]))
                
            #G = np.diag(imp_pred.loc[i, m_sig]) * capital
            G = imp_pred.loc[i, m_sig] * capital.loc[m_sig]
            #w_t_i = np.linalg.inv(2 * G + risk_av * cov_mat).dot(mu.loc[i, m_sig])
            w_t_i = mu.loc[i, m_sig] / (2 * G + risk_av * varis[memes_sub].loc[i, m_sig])
            if abs(w_t_i).sum() > lev_cap:
                w_t_i = w_t_i * lev_cap / abs(w_t_i).sum()
            w.loc[i, m_sig] = w_t_i
                
            alloc_tar = capital * w.loc[i]
            alloc_net = alloc_tar - outst_pos
            pos_inc, pos_dec = [(abs(alloc_tar) - abs(outst_pos)) * j > 0 for j in [1, -1]]
            impa_co = imp_real.loc[i] * abs(alloc_net * pos_dec) ** pd.Series([volu_exp_fin[m] for m in memes_sub], index = memes_sub)
            pnl_o = - (1 - (1 - impa_co) * (1 - fixed_cost_o.loc[i])) * abs(alloc_net * pos_dec)
            pos_val_o = outst_pos + alloc_net * (1 - fixed_cost_o.loc[i] * pos_inc)
            impa_o = imp_real.loc[i] * abs((pos_val_o - outst_pos) * pos_inc) ** pd.Series([volu_exp_fin[m] for m in memes_sub], index = memes_sub)
    
            sl_trig = (df_l_r_dep_h.loc[i] * w.loc[i] <= -sl) + (df_h_r_dep_h.loc[i] * w.loc[i] <= -sl)
            fin_ret = sl_trig * (-sl) / (abs(w.loc[i]) + (w.loc[i] == 0) * 1) + (1 - sl_trig) * df_r_dep_h.loc[i] * np.sign(w.loc[i])
            pos_val_bc = pos_val_o * (1 - impa_o) * (1 + fin_ret)
            impa_c = imp_real.loc[i] * abs(pos_val_bc) ** pd.Series([volu_exp_fin[m] for m in memes_sub], index = memes_sub)
            pos_val_ac = pos_val_bc * (1 - impa_c) * (1 - fixed_cost_c.loc[i])
                
            if i == df_full_r.index[-1]:
                pnl = pnl_o + (pos_val_ac - alloc_tar) * np.sign(w.loc[i])
                rets_g.loc[i] = pnl / capital
                capital += pnl #.sum()
    
        #if i == df_r.index[-1]:
        #    port_val.loc[i] = capital

    return rets_g[1:], w[1:]


def ind_rets_don_rf(armas_rf_pred, memes_tr, npcs_max, orders, n_ret_del, ord_sim, main_comb_w, clo_comb_w, sem_comb_w, args_fix_ext):

    cum_rets_ind_raw = dict()
    for m in memes_tr:
        cum_rets_ind_raw[m] = pd.DataFrame(index = orders, columns = list(np.arange(1, npcs_max + 1)))
    cum_rets_ind_smo = cum_rets_ind_raw.copy()
    
    for npcs in range(1, npcs_max + 1):

        for o in orders:
    
            rets_g, w = ind_rets_on_rf(armas_rf_pred[npcs][o], memes_tr, *args_fix_ext)
            for j in range(n_ret_del):
                for m in memes_tr:
                    rets_g.loc[rets_g.idxmax()[m], m] = 0.
            
            for m in memes_tr:
                cum_rets_ind_raw[m].at[o, npcs] = ((1 + rets_g).cumprod().iloc[-1, :] - 1)[m]

    for m in memes_tr:

        for npcs in range(1, npcs_max + 1):

            if npcs == 1:
                npcs_ns = [npcs + 1, npcs + 2]
            elif npcs == npcs_max:
                npcs_ns = [npcs - 1, npcs - 2]
            else:
                npcs_ns = [npcs - 1, npcs + 1]

            for o in orders:

                main_comb = cum_rets_ind_raw[m].at[o, npcs]
                clo_comb = pd.concat([pd.Series([cum_rets_ind_raw[m].at[o_n, npcs] for o_n in ord_sim[o]['clo']]), pd.Series([cum_rets_ind_raw[m].at[o, npcs_n] for npcs_n in npcs_ns])]).mean()
                sem_comb = pd.concat([pd.Series([cum_rets_ind_raw[m].at[o_n, npcs] for o_n in ord_sim[o]['sem']]), pd.Series([cum_rets_ind_raw[m].at[o_n, npcs_n] for o_n in ord_sim[o]['clo'] for npcs_n in npcs_ns])]).mean()
                far_comb = pd.concat([pd.Series([cum_rets_ind_raw[m].at[o_n, npcs] for o_n in ord_sim[o]['far']]), pd.Series([cum_rets_ind_raw[m].at[o_n, npcs_n] for o_n in ord_sim[o]['sem'] for npcs_n in npcs_ns])]).mean()
                cum_rets_ind_smo[m].at[o, npcs] = main_comb_w * main_comb + clo_comb_w * clo_comb + sem_comb_w * sem_comb + (1 - main_comb_w - clo_comb_w - sem_comb_w) * far_comb

    return cum_rets_ind_smo


def joi_rets_on_rf(res_pred, memes_sub, df_full_r, df_l_r, df_h_r, cal_int, recal, h, dates_per, cor_mats, init_capital, lev_cap, risk_av, varis, buf, sl, fe_bas_pr, fe_bas_o, fe_bas_c, fr_pr, fr_o, fr_c, im_pr, im_re, volu_exp_fin):

    t_0 = df_full_r.index[0] + td(hours = cal_int)
    rets_g = df_full_r[(cal_int - h):].copy()[memes_sub]
    rets_g.iloc[:, :] = 0.
    w = rets_g.copy()
    vols = np.sqrt(varis[memes_sub])
    capital = init_capital
    port_val = pd.Series(capital, index = df_full_r[(cal_int - 1):].index)
    imp_pred = im_pr[memes_sub]
    imp_real = im_re[memes_sub]
    pos_val_bc, pos_val_ac, pnl_o, sl_trig = [pd.Series(0., index = memes_sub) for i in range(4)]

    df_r_dep = df_full_r[memes_sub]
    df_l_r_dep = df_l_r[memes_sub]
    df_h_r_dep = df_h_r[memes_sub]
            
    df_r_dep_cum = (1 + df_r_dep).cumprod()
    df_r_dep_cum = pd.concat([pd.DataFrame(index = [df_r_dep_cum.index.min() - td(hours = 1)], data = 1, columns = df_r_dep_cum.columns), df_r_dep_cum])
    df_r_dep_h = df_r_dep_cum.loc[df_r_dep_cum.index.hour % h == h - 1].pct_change()[1:]
    
    df_l_r_dep_cum = df_r_dep_cum.copy()
    df_l_r_dep_cum.iloc[1:, :] = df_r_dep_cum.shift(1)[1:] * (1 + df_l_r_dep[memes_sub])
    df_l_r_dep_cum['day'] = pd.Series(df_l_r_dep_cum.index, index = df_l_r_dep_cum.index).dt.floor('d')
    df_l_r_dep_cum['floordiv_h'] = df_l_r_dep_cum.index.hour // h
    df_l_r_dep_cum_h = df_l_r_dep_cum.groupby(['day', 'floordiv_h'], as_index = False).min()
    df_l_r_dep_cum_h['time'] = df_l_r_dep_cum_h['day'] + td(hours = h) * df_l_r_dep_cum_h['floordiv_h'] + td(hours = h - 1)
    df_l_r_dep_cum_h = df_l_r_dep_cum_h.set_index('time')[memes_sub]
    df_l_r_dep_h = df_l_r_dep_cum_h / df_r_dep_cum.loc[df_r_dep_cum.index.hour % h == h - 1].shift(1) - 1
    df_l_r_dep_h = df_l_r_dep_h.loc[df_r_dep_h.index]
    
    df_h_r_dep_cum = df_r_dep_cum.copy()
    df_h_r_dep_cum.iloc[1:, :] = df_r_dep_cum.shift(1)[1:] * (1 + df_h_r_dep[memes_sub])
    df_h_r_dep_cum['day'] = pd.Series(df_h_r_dep_cum.index, index = df_h_r_dep_cum.index).dt.floor('d')
    df_h_r_dep_cum['floordiv_h'] = df_h_r_dep_cum.index.hour // h
    df_h_r_dep_cum_h = df_h_r_dep_cum.groupby(['day', 'floordiv_h'], as_index = False).max()
    df_h_r_dep_cum_h['time'] = df_h_r_dep_cum_h['day'] + td(hours = h) * df_h_r_dep_cum_h['floordiv_h'] + td(hours = h - 1)
    df_h_r_dep_cum_h = df_h_r_dep_cum_h.set_index('time')[memes_sub]
    df_h_r_dep_h = df_h_r_dep_cum_h / df_r_dep_cum.loc[df_r_dep_cum.index.hour % h == h - 1].shift(1) - 1
    df_h_r_dep_h = df_h_r_dep_h.loc[df_r_dep_h.index]
    
    fixed_cost_pred = fe_bas_pr[memes_sub] + fr_pr[memes_sub] * np.sign(res_pred[memes_sub])
    fixed_cost_o = fe_bas_o[memes_sub] + fr_o[memes_sub] * np.sign(res_pred[memes_sub])
    fixed_cost_c = fe_bas_c[memes_sub] + fr_c[memes_sub] * np.sign(res_pred[memes_sub])
        
    mu = abs(res_pred) - fixed_cost_pred
    mu = mu * (mu > buf) * np.sign(res_pred)
    mu = pd.concat([pd.DataFrame(0., columns = [t_0 - td(hours = h)], index = memes_sub).T, mu])
        
    for i in res_pred.index:
            
        keep_pos = (mu.loc[i] * mu.loc[i - td(hours = h)] > 0) & (sl_trig == 0)
        outst_pos = pos_val_bc * keep_pos
        pos_val_fin = outst_pos + pos_val_ac * (1 - keep_pos)
        pnl = pnl_o + (pos_val_fin - capital * w.loc[i - td(hours = h)]) * np.sign(w.loc[i - td(hours = h)])
        rets_g.loc[i - td(hours = h)] = pnl / capital
        capital += pnl.sum()
        port_val.loc[i - td(hours = h)] = capital
        pnl_o = pd.Series(0., index = memes_sub)
            
        m_sig = (mu.loc[i] != 0)
        d = m_sig.sum()
            
        if d > 0:
                
            cor_mat = cor_mats[dates_per.loc[dates_per['i'] == i, 't'].iloc[0]].loc[memes_sub, memes_sub]
            cov_mat = np.diag(vols.loc[i, m_sig]).dot(cor_mat.loc[m_sig, m_sig]).dot(np.diag(vols.loc[i, m_sig]))
                
            G = np.diag(imp_pred.loc[i, m_sig]) * capital
            w_t_i = np.linalg.inv(2 * G + risk_av * cov_mat).dot(mu.loc[i, m_sig])
            if abs(w_t_i).sum() > lev_cap:
                w_t_i = w_t_i * lev_cap / abs(w_t_i).sum()
            w.loc[i, m_sig] = w_t_i
                
            alloc_tar = capital * w.loc[i]
            alloc_net = alloc_tar - outst_pos
            pos_inc, pos_dec = [(abs(alloc_tar) - abs(outst_pos)) * j > 0 for j in [1, -1]]
            impa_co = imp_real.loc[i] * abs(alloc_net * pos_dec) ** pd.Series([volu_exp_fin[m] for m in memes_sub], index = memes_sub)
            pnl_o = - (1 - (1 - impa_co) * (1 - fixed_cost_o.loc[i])) * abs(alloc_net * pos_dec)
            pos_val_o = outst_pos + alloc_net * (1 - fixed_cost_o.loc[i] * pos_inc)
            impa_o = imp_real.loc[i] * abs((pos_val_o - outst_pos) * pos_inc) ** pd.Series([volu_exp_fin[m] for m in memes_sub], index = memes_sub)
    
            sl_trig = (df_l_r_dep_h.loc[i] * w.loc[i] <= -sl) + (df_h_r_dep_h.loc[i] * w.loc[i] <= -sl)
            fin_ret = sl_trig * (-sl) / (abs(w.loc[i]) + (w.loc[i] == 0) * 1) + (1 - sl_trig) * df_r_dep_h.loc[i] * np.sign(w.loc[i])
            pos_val_bc = pos_val_o * (1 - impa_o) * (1 + fin_ret)
            impa_c = imp_real.loc[i] * abs(pos_val_bc) ** pd.Series([volu_exp_fin[m] for m in memes_sub], index = memes_sub)
            pos_val_ac = pos_val_bc * (1 - impa_c) * (1 - fixed_cost_c.loc[i])
                
            if i == df_full_r.index[-1]:
                pnl = pnl_o + (pos_val_ac - alloc_tar) * np.sign(w.loc[i])
                rets_g.loc[i] = pnl / capital
                capital += pnl.sum()
    
        if i == df_r_dep_h.index[-1]:
            port_val.loc[i] = capital

    return rets_g[1:], w[1:], port_val


def joi_rets_don_rf(armas_rf_pred, top_combs, n_ret_del, conc_pen_comb2, nbr_combs, n_nbrs, main_comb_w, clo_comb_w, sem_comb_w, comb_drop_th, args_fix_ext_2):

    top_combs_cur = copy.deepcopy(top_combs)
    memes_tr = list(top_combs_cur.keys())
    joi_nbh_perf = dict()
    joi_nbh_perf_dev2 = dict()
    joi_perf = dict()
    comb2_cur = [top_combs_cur[m][0] for m in memes_tr]
    cur_best = tuple(comb2_cur)
    old_best = None
    ni = 0

    while 1 == 1:

        print(ni, sum([len(top_combs_cur[m]) for m in memes_tr]), dt.now())

        for i, m in enumerate(memes_tr):

            for npcs, o in top_combs_cur[m]:

                comb2_cur[i] = (npcs, o)
                
                if tuple(comb2_cur) in joi_nbh_perf.keys():
                    continue

                memes_tr_c = []
                for j, me in enumerate(memes_tr):
                    if comb2_cur[j][0] != 0:
                        memes_tr_c.append(me)

                if tuple(comb2_cur) not in joi_perf.keys():
                    
                    res_pred = pd.DataFrame()
                    for j, me in enumerate(memes_tr):
                        if comb2_cur[j][0] != 0:
                            res_pred = pd.concat([res_pred, armas_rf_pred[comb2_cur[j][0]][comb2_cur[j][1]][me]], axis = 1)
                    rets_g, w, port_val = joi_rets_on_rf(res_pred, memes_tr_c, *args_fix_ext_2)
                    for j in range(n_ret_del):
                        for me in memes_tr_c:
                            rets_g.loc[rets_g.idxmax()[me], me] = 0.
                    rets = rets_g.sum(axis = 1)
                    #conc = (((1 + rets_g).prod() - 1) ** 2).sum() / (((1 + rets).prod() - 1) ** 2)
                    if len(memes_tr_c) == 1:
                        conc = sharpe(rets)
                    else:
                        conc = np.max([sharpe(rets) - sharpe(rets_g.drop(columns = me).sum(axis = 1)) for me in memes_tr_c])
                        #conc_t = np.max([sharpe(rets) - sharpe(rets.loc[~rets.index.isin(rets[ti:(ti + td(hours = conc_int_td))].index)]) for ti in rets.index[:(-conc_int_td)]])
                        #conc = (conc_m + conc_t) / 2
                    #joi_perf[tuple(comb2_cur)] = sharpe(rets) * (1 - conc_pen_comb2 * conc)
                    joi_perf[tuple(comb2_cur)] = sharpe(rets) - conc_pen_comb2 * conc

                comb2_cur_nbr_clo = [[(0, 0) if comb == (0, 0) else random.choice(nbr_combs[comb]['clo']) for comb in comb2_cur] for j in range(round(clo_comb_w / (1 - main_comb_w) * n_nbrs))]
                comb2_cur_nbr_sem = [[(0, 0) if comb == (0, 0) else random.choice(nbr_combs[comb]['sem']) for comb in comb2_cur] for j in range(round(sem_comb_w / (1 - main_comb_w) * n_nbrs))]
                comb2_cur_nbr_far = [[(0, 0) if comb == (0, 0) else random.choice(nbr_combs[comb]['far']) for comb in comb2_cur] for j in range(round((1 - main_comb_w - clo_comb_w - sem_comb_w) / (1 - main_comb_w) * n_nbrs))]

                for nbh_typ in [comb2_cur_nbr_clo, comb2_cur_nbr_sem, comb2_cur_nbr_far]:

                    for comb2_nbr in nbh_typ:

                        if tuple(comb2_nbr) in joi_perf.keys():
                            continue

                        res_pred = pd.DataFrame()
                        for j, me in enumerate(memes_tr):
                            if comb2_nbr[j][0] != 0:
                                res_pred = pd.concat([res_pred, armas_rf_pred[comb2_nbr[j][0]][comb2_nbr[j][1]][me]], axis = 1)
                        rets_g, w, port_val = joi_rets_on_rf(res_pred, memes_tr_c, *args_fix_ext_2)
                        for j in range(n_ret_del):
                            for me in memes_tr_c:
                                rets_g.loc[rets_g.idxmax()[me], me] = 0.
                        rets = rets_g.sum(axis = 1)
                        #conc = (((1 + rets_g).prod() - 1) ** 2).sum() / (((1 + rets).prod() - 1) ** 2)
                        if len(memes_tr_c) == 1:
                            conc = sharpe(rets)
                        else:
                            conc = np.max([sharpe(rets) - sharpe(rets_g.drop(columns = me).sum(axis = 1)) for me in memes_tr_c])
                            #conc_t = np.max([sharpe(rets) - sharpe(rets.loc[~rets.index.isin(rets[ti:(ti + td(hours = conc_int_td))].index)]) for ti in rets.index[:(-conc_int_td)]])
                            #conc = (conc_m + conc_t) / 2
                        #joi_perf[tuple(comb2_nbr)] = sharpe(rets) * (1 - conc_pen_comb2 * conc)
                        joi_perf[tuple(comb2_nbr)] = sharpe(rets) - conc_pen_comb2 * conc

                comb2_nbh_clo_perf = np.mean([joi_perf[tuple(comb2_nbr)] for comb2_nbr in comb2_cur_nbr_clo])
                comb2_nbh_clo_perf_dev2 = np.mean([(joi_perf[tuple(comb2_nbr)] - joi_perf[tuple(comb2_cur)]) ** 2 for comb2_nbr in comb2_cur_nbr_clo])
                if len(comb2_cur_nbr_sem) == 0:
                    comb2_nbh_sem_perf = comb2_nbh_clo_perf
                    comb2_nbh_sem_perf_dev2 = comb2_nbh_clo_perf_dev2
                else:
                    comb2_nbh_sem_perf = np.mean([joi_perf[tuple(comb2_nbr)] for comb2_nbr in comb2_cur_nbr_sem])
                    comb2_nbh_sem_perf_dev2 = np.mean([(joi_perf[tuple(comb2_nbr)] - joi_perf[tuple(comb2_cur)]) ** 2 for comb2_nbr in comb2_cur_nbr_sem])
                if len(comb2_cur_nbr_far) == 0:
                    comb2_nbh_far_perf = comb2_nbh_sem_perf
                    comb2_nbh_far_perf_dev2 = comb2_nbh_sem_perf_dev2
                else:
                    comb2_nbh_far_perf = np.mean([joi_perf[tuple(comb2_nbr)] for comb2_nbr in comb2_cur_nbr_far])
                    comb2_nbh_far_perf_dev2 = np.mean([(joi_perf[tuple(comb2_nbr)] - joi_perf[tuple(comb2_cur)]) ** 2 for comb2_nbr in comb2_cur_nbr_far])

                joi_nbh_perf[tuple(comb2_cur)] = main_comb_w * joi_perf[tuple(comb2_cur)] + clo_comb_w * comb2_nbh_clo_perf + sem_comb_w * comb2_nbh_sem_perf + (1 - main_comb_w - clo_comb_w - sem_comb_w) * comb2_nbh_far_perf
                joi_nbh_perf_dev2[tuple(comb2_cur)] = (clo_comb_w * comb2_nbh_clo_perf_dev2 + sem_comb_w * comb2_nbh_sem_perf_dev2 + (1 - main_comb_w - clo_comb_w - sem_comb_w) * comb2_nbh_far_perf_dev2) / (1 - main_comb_w)

                cur_best = pd.Series(joi_nbh_perf.values(), index = joi_nbh_perf.keys()).idxmax()
                
                if joi_nbh_perf[tuple(comb2_cur)] < (1 - comb_drop_th) * joi_nbh_perf[cur_best]:
                    top_combs_cur[m].remove((npcs, o))

            comb2_cur[i] = cur_best[i]

        if cur_best == old_best:
            break
        old_best = cur_best

        ni += 1

    joi_tr_perf = pd.DataFrame()
    joi_tr_perf['comb2'] = list(joi_nbh_perf.keys())
    joi_tr_perf['Sharpe'] = [joi_perf[comb2] for comb2 in joi_tr_perf['comb2']]
    joi_tr_perf['Sharpe_nbh'] = [joi_nbh_perf[comb2] for comb2 in joi_tr_perf['comb2']]
    joi_tr_perf['Sharpe_nbrs'] = (joi_tr_perf['Sharpe_nbh'] - main_comb_w * joi_tr_perf['Sharpe']) / (1 - main_comb_w)
    joi_tr_perf['Sharpe_nbh_msd'] = [joi_nbh_perf_dev2[comb2] for comb2 in joi_tr_perf['comb2']]
    joi_tr_perf.sort_values('Sharpe_nbh', ascending = False, inplace = True)
    
    return joi_tr_perf