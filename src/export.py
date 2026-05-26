import pandas as pd
from src.state import *

def save_wfa_ho_res(lab, n_ite):

    pd.DataFrame([wf_t for wf_t in gens_perf.keys()]).to_csv('results/WFA_' + lab + '/WFA_OoS_start_date.csv', index = False)

    with pd.ExcelWriter("results/WFA_" + lab + "/gens_perf_last.xlsx", engine = "openpyxl") as gp_writer:
        for j, wf_t in enumerate(gens_perf.keys()):
            gp_ts = gens_perf[wf_t][n_ite - 1].copy()
            gp_ts['enc_idx'] = [list(i.index) for i in gp_ts['ind']]
            gp_ts['enc_tf'] = [tuple(i) for i in gp_ts['ind']]
            gp_ts.drop(columns = 'ind').to_excel(gp_writer, sheet_name = "wf_" + str(j), index = False, engine = "openpyxl")

    with pd.ExcelWriter("results/WFA_" + lab + "/gens_perf_cache.xlsx", engine = "openpyxl") as gpc_writer:
        for j, wf_t in enumerate(gens_perf.keys()):
            gpc_ts = pd.DataFrame({'enc_tf': gens_perf_cache[wf_t].keys(), 'Sharpe': gens_perf_cache[wf_t].values()})
            gpc_ts.to_excel(gpc_writer, sheet_name = "wf_" + str(j), index = False, engine = "openpyxl")

    with pd.ExcelWriter("results/WFA_" + lab + "/top_perf.xlsx", engine = "openpyxl") as tp_writer:
        for j, wf_t in enumerate(gens_perf.keys()):
            tp_ts = top_perf[wf_t].copy()
            tp_ts['enc_idx'] = [list(i.index) for i in tp_ts['ind']]
            tp_ts['enc_tf'] = [tuple(i) for i in tp_ts['ind']]
            tp_ts.drop(columns = 'ind').to_excel(tp_writer, sheet_name = "wf_" + str(j), index = False, engine = "openpyxl")

    with pd.ExcelWriter("results/WFA_" + lab + "/armas_pred.xlsx", engine = "openpyxl") as ap_writer:
        for j, wf_t in enumerate(gens_perf.keys()):
            for npcs in armas_rf_pred[wf_t].keys():
                for o in armas_rf_pred[wf_t][npcs].keys():
                    ap_ts = armas_rf_pred[wf_t][npcs][o].copy()
                    ap_ts.to_excel(ap_writer, sheet_name = "wf_" + str(j) + "_" + str(npcs) + "_" + str(o[0]) + str(o[1]), index = True, engine = "openpyxl")

    with pd.ExcelWriter("results/WFA_" + lab + "/cum_rets.xlsx", engine = "openpyxl") as cr_writer:
        for j, wf_t in enumerate(gens_perf.keys()):
            for m in cum_rets_ind[wf_t].keys():
                cr_ts = cum_rets_ind[wf_t][m].copy()
                cr_ts.to_excel(cr_writer, sheet_name = "wf_" + str(j) + "_" + m, index = True, engine = "openpyxl")

    with pd.ExcelWriter("results/WFA_" + lab + "/top_combs.xlsx", engine = "openpyxl") as tc_writer:
        for j, wf_t in enumerate(gens_perf.keys()):
            tc_ts = pd.DataFrame(data = top_combs[wf_t].values(), index = top_combs[wf_t].keys()).T
            tc_ts.to_excel(tc_writer, sheet_name = "wf_" + str(j), index = False, engine = "openpyxl")

    with pd.ExcelWriter("results/WFA_" + lab + "/joi_tr_perf.xlsx", engine = "openpyxl") as jtp_writer:
        for j, wf_t in enumerate(gens_perf.keys()):
            jtp_ts = joi_tr_perf[wf_t]
            jtp_ts.to_excel(jtp_writer, sheet_name = "wf_" + str(j), index = False, engine = "openpyxl")