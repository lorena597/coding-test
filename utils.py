import numpy as np
import pandas as pd
from scipy.stats import spearmanr

class Strategy(object):

    def __init__(self, price: pd.DataFrame, volume: pd.DataFrame):
        self.price = price
        self.volume = volume
        self.daily_return = self.price / self.price.shift(1) - 1.
    
    def calc_vwap(self, window):
        return (self.price * self.volume).rolling(window, min_periods = int(window/4)).sum() / self.volume.rolling(window, min_periods = int(window/4)).sum()
    
    def signal1(self, window):
        return self.price / self.price.shift(window) - 1.
    
    def signal2(self, long, short):
        return self.price.shift(short) / self.price.shift(long) - 1.
    
    def signal3(self, window):
        H = self.price.rolling(window, min_periods = int(window/4)).max()
        L = self.price.rolling(window, min_periods = int(window/4)).min()
        # H = self.price.rolling(window).max()
        # L = self.price.rolling(window).min()
        pos = (self.price - L) / (H - L)
        pos[H - L <= 0.01] = np.nan
        return pos
    
    def signal4(self, window):
        H = self.price.rolling(window, min_periods = int(window/4)).max()
        L = self.price.rolling(window, min_periods = int(window/4)).min()
        twap = self.price.rolling(window, min_periods = int(window/4)).mean()
        pos = (twap - L) / (H - L)
        pos[H - L <= 0.01] = np.nan
        return pos
    
    def signal5(self, window):
        return self.daily_return.rolling(window, min_periods = int(window/4)).std()

    def signal6(self, window):
        return self.daily_return.rolling(window, min_periods = int(window/4)).skew()

    def signal7(self, window):
        return self.daily_return.rolling(window, min_periods = int(window/4)).kurt()

    def signal8(self, window):
        daily_return_abs = abs(self.daily_return)
        path = daily_return_abs.rolling(window).sum()
        distance = self.price / self.price.shift(window) - 1.
        distance[distance <= 0.01] = np.nan
        return path / distance

    def signal9(self, window):
        def corr_by_column(x, y):
            mu_x = np.nanmean(x, axis = 0)
            mu_y = np.nanmean(y, axis = 0)
            sigma_x = np.nanstd(x, axis = 0)
            sigma_y = np.nanstd(y, axis = 0)
            return np.nanmean((x - mu_x) * (y - mu_y), axis = 0) / (sigma_x * sigma_y)
        corr = np.zeros(shape = self.price.shape); corr[:] = np.nan
        for i in range(corr.shape[0]):
            cur_price = self.price.iloc[(i-window+1):(i+1),:]
            cur_volume = self.volume.iloc[(i-window+1):(i+1),:]
            corr[i,:] = corr_by_column(cur_price, cur_volume)
        corr_df = pd.DataFrame(corr, index = self.price.index, columns = self.price.columns)
        return corr_df
    
    def signal10(self, window):
        def calc(p, v, s, w):
            vwap = np.nansum(p * v, axis = 0) / np.nansum(v, axis = 0)
            ix = np.argsort(s.values, axis = 0)
            v_sort = np.take_along_axis(v.values, ix, axis = 0)
            p_sort = np.take_along_axis(p.values, ix, axis = 0)
            vwap_1 = np.nansum(v_sort[-int(w/4):,:] * p_sort[-int(w/4):,:], axis = 0) / np.nansum(v_sort[-int(w/4):,:], axis = 0)
            return vwap_1/vwap

        S = np.abs(self.daily_return) / np.sqrt(self.volume)
        S = pd.DataFrame(S, index = self.price.index, columns = self.price.columns)
        factor = np.zeros(shape = self.price.shape); factor[:] = np.nan
        for i in range(factor.shape[0]):
            cur_price = self.price.iloc[(i-window+1):(i+1),:]
            cur_volume = self.volume.iloc[(i-window+1):(i+1),:]
            cur_smart = S.iloc[(i-window+1):(i+1),:]
            factor[i,:] = calc(cur_price,cur_volume,cur_smart,window)
        factor_df = pd.DataFrame(factor, index = self.price.index, columns = self.price.columns)
        return factor_df

class Backtest(object):

    def __init__(self, price: pd.DataFrame):
        self.price = price
        self.all_dates = self.price.index
        self.all_tickers = self.price.columns
        self.tradable_matrix = ~np.isnan(self.price)
        self.all_dates_int = np.array([int(''.join(date.split('-'))) for date in self.all_dates])
        self.all_years_int = np.array([date_int // 10000 for date_int in self.all_dates_int])

    def get_result(self, signal: pd.DataFrame):
        signal = signal.copy()
        daily_return = self.price / self.price.shift(1) - 1.
        daily_return[~self.tradable_matrix] = np.nan
        signal[~self.tradable_matrix] = np.nan
        # transform = self.normalize(self.winsorize(signal))
        transform = self.scale(self.winsorize(signal))
        transform = transform / np.nansum(transform * (transform > 0), axis = 1).reshape((-1,1))
        mask = np.nansum(~np.isnan(transform), axis = 1) == 0
        tot_rtn = np.nansum(pd.DataFrame(transform).shift(1).values * daily_return, axis = 1)
        tot_rtn[mask] = np.nan
        ix = np.array([i for i in range(len(tot_rtn)) if i % 10 == 0])
        stats_dict = self.calculate_stats(tot_rtn[ix], self.all_dates_int[ix])
        # stats_dict = self.calculate_stats(tot_rtn, self.all_dates)
        # calculate stats for every year
        dict_list = []
        unique_years = np.unique(self.all_years_int)
        for unique_year in unique_years:
            mask = self.all_years_int == unique_year
            cur_rtn = tot_rtn[mask]
            cur_dates = self.all_dates_int[mask]
            ix = np.array([i for i in range(len(cur_rtn)) if i % 10 == 0])
            dict_list.append(self.calculate_stats(cur_rtn[ix], cur_dates[ix]))
        dict_list.append(stats_dict)
        result = pd.DataFrame(dict_list, index = list(unique_years) + ['total'])
        # ic = self.calculate_ic(signal, daily_return)
        # stats_dict['IC'] = ic
        return result.round(2)
    
    @staticmethod
    def calculate_ic(signal, daily_return):
        corr_ts = []
        for i in range(len(signal)-1):
            cur_signal = signal.iloc[i,:]
            cur_return = daily_return.iloc[i+1,:]
            mask = ~np.isnan(cur_signal) & ~np.isnan(cur_return)
            cur_signal = cur_signal[mask]
            cur_return = cur_return[mask]
            corr_ts.append(spearmanr(cur_signal, cur_return))
        return np.mean(corr_ts)
    
    @staticmethod
    def calculate_stats(rtn,ds):
        ix = np.where(~np.isnan(rtn))[0][0]
        prd_rtn = np.nansum(rtn[ix:])
        ann_rtn = np.nanmean(rtn[ix:]) * 252
        ann_vol = np.nanstd(rtn[ix:]) * np.sqrt(252) 
        shrp = ann_rtn / ann_vol
        nav = np.cumsum(rtn[ix:])
        mdd, mdd_bgn, mdd_end = 0, 0, 0
        for i in range(1,len(nav)):
            dd_i = np.full(i,nav[i]) - nav[:i]
            mdd_i = np.nanmin(dd_i)
            if mdd_i <= mdd:
                mdd = mdd_i
                mdd_bgn = np.argmin(dd_i)
                mdd_end = i
        wrt = np.nansum(rtn[ix:] > 0) / len(rtn[ix:])
        return {'period return': prd_rtn * 100,
                'annual return': ann_rtn * 100,
                'annual volatility': ann_vol * 100,
                'sharpe ratio': shrp,
                'max drawdown': mdd * 100,
                'max drawdown begin date': ds[ix:][mdd_bgn],
                'max drawdown end date': ds[ix:][mdd_end],
                'winning ratio': wrt * 100}  

    @staticmethod
    def normalize(signal):
        return (signal - np.nanmean(signal, axis = 1).reshape(-1, 1)) / np.nanstd(signal, axis = 1).reshape(-1, 1)
    
    @staticmethod
    def scale(signal):
        signal = signal.rank(axis = 1)
        min_ts = np.nanmin(signal, axis = 1).reshape(-1,1)
        max_ts = np.nanmax(signal, axis = 1).reshape(-1,1)
        return (signal - min_ts) / (max_ts - min_ts) - 0.5
    
    @staticmethod
    def winsorize(signal):
        def func(row):
            median = np.nanmedian(row)
            absolute_deviation = np.abs(row - median)
            MAD = np.nanmedian(absolute_deviation)
            MAD_e = MAD * 1.4826
            row[row >= median + 3 * MAD_e] = median + 3 * MAD_e
            row[row <= median - 3 * MAD_e] = median - 3 * MAD_e
            return row
        signal.apply(func, axis = 1)
        return signal