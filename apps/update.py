# Software program written by Neil Murphy in year 2019.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# By using this software, the Disclaimer and Terms distributed with the
# software are deemed accepted, without limitation, by user.
#
# You should have received a copy of the Disclaimer and Terms document
# along with this program.  If not, see... https://bit.ly/2Tlr9ii
#
#############################################################################
import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
import shutil
import sqlite3

import pandas as pd

from apps.psar import psar_manager



def file_backup():
    """Create a back up the DashData.db File"""

    """Run the update script."""
    data_folder = Path("data/")

    src = data_folder / "DashData.db"

    backup_folder = Path("data/backup/")
    backup_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".db"

    dst = backup_folder / backup_file

    shutil.copy(src, dst)


def read_file(ss):
    """
    Read data from spreadsheet into dateframe.
    :param ss: Path to spreadsheet.
    :return: dataframe
    """

    df = pd.read_csv(
        ss,
        header=0,
        usecols=[0, 1, 2, 3, 4, 5],
        names=["date", "ticker", "open", "high", "low", "close"],
        parse_dates=['date'],
        dayfirst=True
    )

    # Trading holidays show up as open, high, low at zero and close at the previous close.
    # Eliminate the holidays.
    df = df.loc[~(df[['open', 'high', 'low']].sum(axis=1) == 0)]

    # # Original excel code.
    # df = pd.read_excel(ss)
    # df["date"] = pd.to_datetime(df["date"])

    df = df.set_index(["date", "ticker"])

    return df


"""
This section will perform checks on the data.
"""

# Like check for rows with zeros and np.NaN Throw and exception and clean the data.


def return_psars(
    df, short_iaf, short_maxaf, long_iaf, long_maxaf
):
    """
    Get the psar values.
    :param df: Dataframe read from spreadsheets.
    :param short_col_name: Short psar column name.
    :param short_iaf: Short psar acceleration.
    :param short_maxaf: Short psar acceleration max.
    :param long_col_name: Long psar column name.
    :param long_iaf: Long psar acceleration.
    :param long_maxaf: Long psar acceleration max.
    :return: list with [dataframe, bad tickers].
    """

    # psar manager returns a list with df and bad stocks list.
    # psar_li is the return list.
    # bad_tickers are tickers that did not get processed likely because of empty cells.

    # Run psar_short
    psar_li = psar_manager(
        df=df, col_name="psar_short", iaf=short_iaf, maxaf=short_maxaf
    )
    bad_tickers = psar_li[1]

    # Run psar long.
    psar_li = psar_manager(
        df=psar_li[0], col_name="psar_long", iaf=long_iaf, maxaf=long_maxaf
    )

    if psar_li[1]:
        bad_tickers.append(psar_li[1])

    return [psar_li[0], bad_tickers]


# Funtion to calculate the full array of sharpe values. The first window length of each
# company will have incorrect values and should be set to np.NaN.
# Numpy and Numba used to drastically speed up the calculations.
# @jit(nopython=True)
# def sharpe(a, window):
#     """
#     Calculate sharpe on window length rolling over an array.
#     :param a: numpy one dimensional array.
#     :param window: Size of the rolling window.
#     :return: Returns an equal size numpy array.
#     """
#     result_array = np.full(len(a), np.NaN)
#
#     for n in range(len(a) - window):
#         y = a[n : n + window]
#         result_array[n + window - 1] = np.sqrt(window) * np.mean(y) / np.std(y)
#
#     return result_array


def metrics(df_psar):
    """
    Add columns here for psar return, sharpe, std etc.
    Find the information in the jupyter notebook.
    """

    # In order to calculate the following the ticker must be first in the index, then date.
    df_psar = df_psar.swaplevel().sort_index()

    # # Daily returns
    # # df_daily['return'] = qs.utils.to_returns(df_daily.loc[:, "close"])
    # df_psar["return"] = np.log(
    #     df_psar.loc[:, "close"] / df_psar.loc[:, "close"].shift(1)
    # )
    #
    # # Standard deviation of returns from -4 months to -1 month.
    # df_psar["std_dev"] = df_psar["return"].rolling(window=66).std().shift(
    #     22
    # ) * np.sqrt(66)
    #
    # # Run the sharpe function.
    # returns_array = df_psar["return"].values
    # res = sharpe(returns_array, sharpe_window)
    # df_psar["sharpe"] = pd.Series(res, index=df_psar.index)
    # df_psar.loc[
    #     df_psar["sharpe"].groupby(level=0).head(sharpe_window).index, "sharpe"
    # ] = np.NaN

    # Set 15, 30, 50, 200 day momentum.
    df_psar["ma15"] = (
        df_psar.groupby(level="ticker")["close"].rolling(window=15).mean().droplevel(0)
    )
    df_psar["ma30"] = (
        df_psar.groupby(level="ticker")["close"].rolling(window=30).mean().droplevel(0)
    )
    df_psar["ma50"] = (
        df_psar.groupby(level="ticker")["close"].rolling(window=50).mean().droplevel(0)
    )
    df_psar["ma200"] = (
        df_psar.groupby(level="ticker")["close"].rolling(window=200).mean().droplevel(0)
    )

    # Swap the index back.
    df_psar = df_psar.swaplevel().sort_index()

    return df_psar


def shorten_data(df_psar, months=18):
    # Working psar dataframe is for two years, reduce to eighteen months to drop off start up psars.
    date_last = df_psar.index.unique(0)[-1]
    date_first_from_start = df_psar.index.unique(0)[0] + relativedelta(months=4)
    date_first_from_end = date_last - relativedelta(months=months)

    # Try to go back 18 months, but forward 4 for sure.
    if date_first_from_start > date_first_from_end:
        date_first = date_first_from_start
    else:
        date_first = date_first_from_end

    df_psar = df_psar.loc[pd.IndexSlice[date_first:, :], :]

    return df_psar


def commit_to_db(df, db, table):
    """
    Commit modified to data base.
    :param df: Modified dataframe.
    :param db: Database path, string.
    :param table: Table to replace.
    :return: None
    """

    # Copy psar to the daily data table.
    conn = sqlite3.connect(db, detect_types=sqlite3.PARSE_DECLTYPES)
    df.to_sql(table, conn, if_exists="replace")
    conn.close()

def process_spreadsheet(sheet):
    """
    Process a spreadsheet into a final dataframe.
    :param sheet: Path to spreadsheet.
    :return: Two element list, [dataframe, unprocessed tickers].
    """

    df = read_file(sheet)
    psar_list = return_psars(
        df,
        short_iaf=0.02,
        short_maxaf=0.2,
        long_iaf=0.002,
        long_maxaf=0.2,
    )
    bad_tickers = psar_list[1]

    df = metrics(psar_list[0])
    df = shorten_data(df, 18)

    return [df, bad_tickers]

def run_update():
    """Run the update script."""
    data_folder = Path("data/")

    db = data_folder / "DashData.db"
    new_data = data_folder / "update.csv"

    print(db)
    print(new_data)

    # Back up the database.
    file_backup()

    # Create daily table from update sheet.
    process_daily = process_spreadsheet(new_data)
    print(process_daily[0])
    print(process_daily[1])
    commit_to_db(process_daily[0], db, "daily")
    commit_to_db(pd.DataFrame(process_daily[1], columns=["daily_bad_tickers"]), db, "daily_bad_tickers")

    return process_daily[0]


if __name__ == "__main__":
    run_update()
