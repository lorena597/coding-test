import numpy as np
import pandas as pd

class Strategy(object):

    def __init__(self, price: pd.DataFrame, volume: pd.DataFrame):
        self.price = price
        self.volume = volume
        self.daily_return = self.price / self.price.shift(1) - 1.
    
    def calc_vwap(self, window):
        return (self.price * self.volume).rolling(window, min_periods = int(window/4)).sum() / self.volume.rolling(window, min_periods = int(window/4)).sum()
    
    def signal1(self, window):
        return self.price / self.price.shift(window) - 1.

    def signal2(self, window):
        daily_return_abs = abs(self.daily_return)
        path = daily_return_abs.rolling(window, min_periods = int(window/4)).sum()
        distance = self.price / self.price.shift(window) - 1.
        distance[distance == 0] = np.nan
        return path / distance
    
    def signal3(self, window):
        H = self.price.rolling(window, min_periods = int(window/4)).max()
        L = self.price.rolling(window, min_periods = int(window/4)).min()
        twap = self.price.rolling(window, min_periods = int(window/4)).mean()
        return (twap - L) / (H - L)
    
    def signal4(self, window):
        return 1. / self.daily_return.rolling(window, min_periods = int(window/4)).std()
    
    def signal5(self, long, short):
        return self.price.shift(short) / self.price.shift(long) - 1.

    def signal6(self, window):
        return 1. / self.daily_return.rolling(window, min_periods = int(window/4)).skew()

    def signal7(self, window):
        return 1. / self.daily_return.rolling(window, min_periods = int(window/4)).kurt()
    
    def signal8(self, window):
        H = self.price.rolling(window, min_periods = int(window/4)).max()
        L = self.price.rolling(window, min_periods = int(window/4)).min()
        return (self.price - L) / (H - L)

    def signal9(self, window):
        # to do: corr between price and volume
        pass
    
    def signal10(self, window):
        # to do: about smart money
        pass


class Backtest(object):

    def __init__(self, price: pd.DataFrame):
        self.price = price
        self.all_dates = self.price.index
        self.all_tickers = self.price.columns
        self.tradable_matrix = ~np.isnan(self.price)

    def get_result(self, signal: pd.DataFrame):
        daily_return = self.price / self.price.shift(1) - 1.
        daily_return[~self.tradable_matrix] = np.nan
        signal[~self.tradable_matrix] = np.nan
        transform = self.normalize(self.winsorize(signal))
        transform = transform / np.nansum(transform * (transform > 0), axis = 1).reshape((-1,1))
        mask = np.nansum(~np.isnan(transform), axis = 1) == 0
        tot_rtn = np.nansum(pd.DataFrame(transform).shift(1).values * daily_return, axis = 1)
        tot_rtn[mask] = np.nan
        ix = np.array([i for i in range(len(tot_rtn)) if i % 10 == 0])
        stats_dict = self.calculate_stats(tot_rtn[ix], self.all_dates[ix])
        # stats_dict = self.calculate_stats(tot_rtn, self.all_dates)
        return stats_dict
    
    @staticmethod
    def calculate_stats(rtn,ds):
        ix = np.where(~np.isnan(rtn))[0][0]
        prd_rtn = np.nansum(rtn[ix:])
        ann_rtn = np.nanmean(rtn[ix:]) * 25
        ann_vol = np.nanstd(rtn[ix:]) * np.sqrt(25) 
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