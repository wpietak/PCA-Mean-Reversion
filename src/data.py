import pandas as pd
import numpy as np
import cryptocompare
from datetime import datetime as dt
from datetime import timedelta as td


def get_data_cryptocompare(start_date, end_date, memes_all):

    df_full = pd.DataFrame()
    df_full_l = pd.DataFrame()
    df_full_h = pd.DataFrame()
    n = int(np.ceil(((end_date - start_date).days * 24 + (end_date - start_date).seconds / 3600) / 2000))
    
    for t in range(n):
        df_t_c = pd.DataFrame()
        df_t_l = pd.DataFrame()
        df_t_h = pd.DataFrame()
        if t + 1 < n:    
            int_h = start_date + td(hours = (t + 1) * 2001 - 1)
            lim = 2000
        else:
            int_h = end_date
            int_l = start_date + td(hours = t * 2001)
            lim = int(((end_date - int_l).days * 24 + (end_date - int_l).seconds / 3600))
        for i in memes_all:
            df_i = pd.DataFrame.from_dict(cryptocompare.get_historical_price_hour(i, 'USDT', toTs = int_h, limit = lim))
            df_i['time'] = pd.to_datetime(df_i['time'], unit = 's')
            df_i = df_i.set_index('time')
            df_i_c = df_i[['close']].rename(columns = {'close': i})
            df_i_l = df_i[['low']].rename(columns = {'low': i})
            df_i_h = df_i[['high']].rename(columns = {'high': i})
            df_t_c = pd.concat([df_t_c, df_i_c], axis = 1)
            df_t_h = pd.concat([df_t_h, df_i_h], axis = 1)
            df_t_l = pd.concat([df_t_l, df_i_l], axis = 1)
        df_full = pd.concat([df_full, df_t_c], axis = 0)
        df_full_l = pd.concat([df_full_l, df_t_l], axis = 0)
        df_full_h = pd.concat([df_full_h, df_t_h], axis = 0)

    return df_full, df_full_l, df_full_h


def klines_bin(pair, start_date, end_date, freq):
    time_span = pd.to_datetime(np.arange(start_date - td(days = 10), end_date + td(days = 10), td(days = 1)))
    df_kl_m = pd.DataFrame()
    for (ye, mon) in pd.Series([(ye, mon) for ye, mon in zip(time_span.year, time_span.month)]).unique():
        if mon < 10:
            str_mon = "0" + str(mon)
        else:
            str_mon = str(mon)
        df_kl_m_i = pd.read_csv("https://data.binance.vision/data/futures/um/monthly/klines/" + pair + "/1" + freq + "/" + pair + "-1" + freq + "-" + str(ye) + "-" + str_mon + ".zip")
        df_kl_m = pd.concat([df_kl_m, df_kl_m_i])
    df_kl_m['time'] = pd.to_datetime(df_kl_m['open_time'], unit = 'ms')
    return df_kl_m


def get_fun_rat_bin(m, pair, start_date, end_date):
    time_span = pd.to_datetime(np.arange(start_date - td(days = 10), end_date + td(days = 10), td(days = 1)))
    m_fr = pd.DataFrame()
    for (ye, mon) in pd.Series([(ye, mon) for ye, mon in zip(time_span.year, time_span.month)]).unique():
        if mon < 10:
            str_mon = "0" + str(mon)
        else:
            str_mon = str(mon)
        m_fr_i = pd.read_csv("https://data.binance.vision/data/futures/um/monthly/fundingRate/" + pair + "/" + pair + "-fundingRate-" + str(ye) + "-" + str_mon + ".zip")
        m_fr_i['calc_time'] = pd.to_datetime(m_fr_i['calc_time'], unit = 'ms').dt.round('s')
        m_fr = pd.concat([m_fr, m_fr_i])
    m_fr = m_fr[['calc_time', 'last_funding_rate']].rename(columns = {'calc_time': 'time', 'last_funding_rate': m})
    return m_fr


def klines_byb(pair, start_date, end_date):
    df_kl_m = pd.DataFrame()
    df_kl_m_min = pd.DataFrame()
    time_span = pd.to_datetime(np.arange(start_date - td(days = 10), end_date + td(days = 10), td(days = 1)))
    d = 0
    df_tr_m_i = pd.DataFrame()
    for (ye, mon, day) in pd.Series([(ye, mon, day) for ye, mon, day in zip(time_span.year, time_span.month, time_span.day)]).unique():
        str_mon = f"{mon:0{2}d}"
        str_day = f"{day:0{2}d}"
        df_tr_m_i_d = pd.read_csv("https://public.bybit.com/trading/" + pair + "/" + pair + str(ye) + "-" + str_mon + "-" + str_day + ".csv.gz")
        df_tr_m_i = pd.concat([df_tr_m_i, df_tr_m_i_d])
        d += 1
        if (d == 30) | (dt(ye, mon, day) == time_span[-1].floor('d')):
            df_tr_m_i['timestamp'] = pd.to_datetime(df_tr_m_i['timestamp'], unit = 's')
            df_tr_m_i['time'] = df_tr_m_i['timestamp'].dt.floor('h')
            df_tr_m_i['min'] = df_tr_m_i['timestamp'].dt.floor('min')
            df_kl_m_i = df_tr_m_i[['time', 'price', 'foreignNotional']].groupby('time').agg({'price': 'ohlc', 'foreignNotional': 'sum'})
            df_kl_m_i.columns = df_kl_m_i.columns.get_level_values(1)
            df_kl_m_min_i = df_tr_m_i[['min', 'price']].rename(columns = {'min': 'time'}).groupby('time').ohlc()['price']
            df_kl_m = pd.concat([df_kl_m, df_kl_m_i])
            df_kl_m_min = pd.concat([df_kl_m_min, df_kl_m_min_i])
            d = 0
            df_tr_m_i = pd.DataFrame()
    return df_kl_m, df_kl_m_min


def read_fun_rat_byb(m, pair):
    m_fr = pd.read_excel("src/raw_data/funding_rate_history_" + pair + ".xlsx")
    m_fr['Time(UTC)'] = pd.to_datetime(m_fr['Time(UTC)'])
    m_fr = m_fr[['Time(UTC)', 'Funding Rate']].rename(columns = {'Time(UTC)': 'time', 'Funding Rate': m})
    return m_fr