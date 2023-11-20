import pandas as pd
import pickle
import bz2
import glob
import numpy as np


class MDProcessor:
    def __init__(self, date, base_path, start, end, export_csv=False):
        self._start = start
        self._end = end
        self._date = date
        self._base_path = base_path
        self._export_csv = export_csv

    def export_n_level_df_csv(self, n):
        df2 = self.get_n_level_book_and_trades(n)
        df2['bidprice_1'] = df2['price'].where(np.isnan(df2['bidprice_1']), df2['bidprice_1'])
        df2['bidquantity_1'] = df2['quantity'].where(np.isnan(df2['bidquantity_1']), df2['bidquantity_1'])
        df2['askquantity_1'] = df2['askquantity_1'].fillna(0)
        df2['bidquantity_1'] = df2['bidquantity_1'].apply(lambda x: int(x))
        df2['askquantity_1'] = df2['askquantity_1'].apply(lambda x: int(x))
        df2_mod = df2.copy()[['product', 'bidprice_1', 'bidquantity_1', 'askprice_1', 'askquantity_1']]
        df2_mod.index = df2_mod.index.astype('datetime64[ms]')
        df2_mod.to_csv(f'md_{self._date}.csv', na_rep='', header=False)

    def get_n_level_book_and_trades(self, n, add_midprice=True):
        book = self.get_n_levels_book(n)
        trades = self.get_trades()
        df = pd.concat([trades, book])
        df.sort_index(ascending=True, inplace=True)
        if add_midprice:
            df = self._add_midprice(df)
        return df

    def _add_midprice(self, df):
        df["midprice"] = ((df["bidprice_1"] + df["askprice_1"])/2).ffill()
        df["ismoplus"] = (~df.price.isnull()) & (df['price'] > df['midprice'])
        df["ismominus"] = (~df.price.isnull()) & (df['price'] < df['midprice'])
        df["isjumpplus"] = (df['midprice'] - df['midprice'].shift(1)) > 0
        df["isjumpminus"] = (df['midprice'] - df['midprice'].shift(1)) < 0
        return df

    def get_n_levels_book(self, n, export_csv=False):
        path = self._base_path + self._date + f'/top_{n}*'
        files = glob.glob(path)
        consolidated = pd.DataFrame()
        count = 1
        for file in files:
            df = self._decompress_pickle(file)
            print("processing file {}/{} size: {}".format(count, len(files), str(df.shape)))
            product = df['product'].values[0]
            df.drop(columns=list(
                {'feed', 'product', 'lastexchangetimestamp', 'lastsequencenumber', 'firstexchangetimestamp',
                 'firstreceipttimestamp', 'lastexchangesendtimestamp',
                 'firstexchangesendtimestamp'} & set(df.columns.tolist())), inplace=True)
            # print(df.head())
            df.dropna(inplace=True)
            # print(df.head())

            # df = df[(df[df.columns[:20]].diff(1) > 0).any(axis=1) == True]
            consolidated = pd.concat([consolidated, df])
            # print("size consolidated: {} \n".format(str(consolidated.shape)))
            count += 1

        consolidated = consolidated[(consolidated[consolidated.columns[:20]].diff(1) > 0).any(axis=1) == True]
        print("getting timestamp column")
        consolidated['timestamp'] = pd.to_datetime(consolidated['lastreceipttimestamp'], unit='ns')
        print("dropping lastreceipttimestamp")
        consolidated.drop(columns=['lastreceipttimestamp'], inplace=True)
        print("setting index")
        consolidated.set_index('timestamp', inplace=True)
        print("sorting")
        consolidated.sort_index(ascending=True, inplace=True)
        print("getting date column")
        consolidated['date'] = consolidated.index.to_series().apply(lambda x: '%d%02d%02d' % (x.year, x.month, x.day))
        # x.strftime('%Y%m%d'))
        print("getting time column")
        consolidated['time'] = consolidated.index.to_series().apply(
            lambda x: '%02d:%02d:%02d.' % (x.hour, x.minute, x.second) + '{:>03d}'.format((x.microsecond // 1000)))
        # x.strftime('%H:%M:%S.%f'))
        if self._export_csv:
            # right columns for simulation
            cols = ["date", "time"] + ["{}{}_{}".format(side, t, i) for side, t, i in
                                       zip((["ask"] * 10 + ["bid"] * 10),
                                           (["price"] * 5 + ["quantity"] * 5) * 2,
                                           [1, 2, 3, 4, 5] * 4)]
            output_file = "md_{}.csv".format(self._date)
            print("writing csv {} ...\n".format(output_file))
            consolidated.to_csv(output_file, columns=cols, index=False, header=False)
            print("done")
        consolidated['product'] = product
        return consolidated[(consolidated.index > self._start) & (consolidated.index < self._end)]

    @staticmethod
    def _decompress_pickle(file):
        data = bz2.BZ2File(file, 'rb')
        data = pickle.load(data)
        return data

    # rutina para importar todos los archivos con los trades
    # y exportar en csv
    # base_path es el directorio base
    # date es la fecha en string (donde estan los archivos de una fecha dada)
    def get_trades(self):
        path = self._base_path + self._date + '/trades*'
        files = glob.glob(path)
        print('\n'.join(files), path)
        consolidated = pd.DataFrame()
        count = 1
        for file in files:
            print("loading {}/{}".format(count, len(files)))
            df = self._decompress_pickle(file)
            df.drop(columns=list(
                {'exchangesendtimestamp', 'exchangetimestamp', 'transactiontimestamp', 'exchangereceipttimestamp',
                 'exchangeenginereceipttimestamp', 'side'} & set(df.columns.tolist())), inplace=True)
            # print(df.head())
            df.dropna(inplace=True)
            # print(df.head())
            consolidated = pd.concat([consolidated, df])
            count += 1
        print(consolidated.columns)
        consolidated['timestamp'] = pd.to_datetime(consolidated['receipttimestamp'], unit='ns')
        print("setting index")
        consolidated.set_index('timestamp', inplace=True)
        print("sorting")
        consolidated.sort_index(ascending=True, inplace=True)
        print("getting date column")
        consolidated['date'] = consolidated.index.to_series().apply(lambda x: '%d%02d%02d' % (x.year, x.month, x.day))
        # x.strftime('%Y%m%d'))
        print("getting time column")
        consolidated['time'] = consolidated.index.to_series().apply(
            lambda x: '%02d:%02d:%02d.' % (x.hour, x.minute, x.second) + '{:>03d}'.format((x.microsecond // 1000)))
        print("done")
        return consolidated[(consolidated.index > self._start) & (consolidated.index < self._end)]
