import pandas as pd

gens_perf, gens_perf_cache, top_perf, armas_rf_pred, cum_rets_ind, top_combs, joi_tr_perf = [dict() for i in range(7)]

class Oos_df:
    def __init__(self):
        self.rets_g_oos, self.w_oos, self.port_val_oos = [pd.DataFrame() for i in range(3)]
oos_df = Oos_df()