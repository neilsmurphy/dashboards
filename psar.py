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
import sqlite3
import time

import pandas as pd
import plotly.graph_objects as go


def psar(barsdata, iaf=0.02, maxaf=0.2):
    """
    Calculates psar for a series.
    https://raw.githubusercontent.com/virtualizedfrog/blog_code/master/PSAR/psar.py
    :param barsdata: Dataframe for one stock only.
    :param iaf: One day is .02, one week approximately .002
    :param maxaf: Cap for the iaf as it grows.
    :return: Dictionary with date, high, low, close, psar
    """

    length = len(barsdata)
    dates = list(barsdata.index)
    high = list(barsdata["high"])
    low = list(barsdata["low"])
    close = list(barsdata["close"])
    psar = close[0 : len(close)]
    psarbull = [None] * length
    psarbear = [None] * length
    bull = True
    af = iaf
    ep = low[0]
    hp = high[0]
    lp = low[0]

    for i in range(2, length):
        if bull:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])

        reverse = False

        if bull:
            if low[i] < psar[i]:
                bull = False
                reverse = True
                psar[i] = hp
                lp = low[i]
                af = iaf
        else:
            if high[i] > psar[i]:
                bull = True
                reverse = True
                psar[i] = lp
                hp = high[i]
                af = iaf

        if not reverse:
            if bull:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + iaf, maxaf)
                if low[i - 1] < psar[i]:
                    psar[i] = low[i - 1]
                if low[i - 2] < psar[i]:
                    psar[i] = low[i - 2]
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + iaf, maxaf)
                if high[i - 1] > psar[i]:
                    psar[i] = high[i - 1]
                if high[i - 2] > psar[i]:
                    psar[i] = high[i - 2]

        if bull:
            psarbull[i] = psar[i]
        else:
            psarbear[i] = psar[i]

    return {
        "date": dates,
        "high": high,
        "low": low,
        "close": close,
        "psar": psar,
        # "psarbear": psarbear,     # Not currently using.
        # "psarbull": psarbull,     # Not currently using.
    }


def psar_manager(df, col_name, iaf=0.02, maxaf=0.2):
    """
    This version of psar_manager will create bars from successive period days.

    :param df: A dataframe from spreadsheets.
    :param col_name: PSAR column name
    :param iaf: Acceleration factor
    :param maxaf: Maximum acceleration factor.
    :return: List with 0. original dataframe with parabolic sar added and 1. list of unprocessed tickers.
    """
    tickers = df.index.unique(1)
    bad_tickers = []

    df_res = pd.DataFrame()

    for t in tickers:
        # Create ticker specific dataframe
        df_psar = df.loc[pd.IndexSlice[:, t], :].droplevel(1)

        # Drop all the np.NaN rows.
        df_psar = df_psar.dropna(axis=0)

        # Return original dataframe if number or valid rows is less than 6 months or 132 rows.
        # Parabolic SAR needs a bit of time to become accurate.
        if df_psar.shape[0] < 132:
            bad_tickers.append(t)
            continue
        else:
            pass

        df_psar[col_name] = (
            pd.DataFrame(psar(df_psar, iaf, maxaf))
            .set_index("date")
            .loc[:, "psar"]
            .round(2)
        )
        df_psar = df_psar.reset_index()
        df_psar["ticker"] = t
        df_psar = df_psar.set_index(["date", "ticker"]).sort_index()
        if df_res.shape[0] != 0:
            df_res = pd.concat([df_res, df_psar], sort=False, axis=0)
        else:
            df_res = df_psar

    df_res = df_res.sort_index()

    print("List of stocks that did not get processed: {}".format(bad_tickers))

    return [df_res, bad_tickers]


if __name__ == "__main__":
    """
    Running from main will generate printouts of df and plots.
    """

    date_start = "2018-05-01"
    date_graph_from = "2019-05-01"
    date_end = "2019-11-01"

    # Read daily into memory.
    conn = sqlite3.connect("data/DashData.db", detect_types=sqlite3.PARSE_DECLTYPES)
    df = pd.read_sql("SELECT * FROM daily", conn)
    conn.close()
    df = df.set_index(["date", "ticker"]).sort_index()

    df = df.loc[
        pd.IndexSlice[date_start:date_end, :],
        ["open", "high", "low", "close", "in_ASX200"],
    ]

    date_last = df.index.unique(0)[-1]

    s = (df.loc[pd.IndexSlice[date_last, :], "in_ASX200"] == 1).droplevel(0)
    tickers = s[s].index.values

    start_time = time.time()

    df = df.loc[pd.IndexSlice[:, tickers], :]
    iaf = 0.002
    maxaf = 0.2
    dpsar = psar_manager(df, "psar_short", 0.02, maxaf).loc[date_graph_from:, :]
    dpsar = psar_manager(dpsar, "psar_long", 0.002, maxaf).loc[date_graph_from:, :]

    end_time = time.time()

    print("Total time elapsed is: {}".format(end_time - start_time))

    print(dpsar.head())

    c1 = "#F66D44"
    c2 = "#FEAE65"
    c3 = "#E6F69D"
    c4 = "#AADEA7"
    c5 = "#64C2A6"
    c6 = "#2D87BB"

    df_loop = dpsar

    for ticker in ['APT', 'APX', 'ASB', 'CUV', 'FMG', 'MFG', 'NAN', 'NEA', 'SSM', 'XRO']:
        dpsar = df_loop.loc[pd.IndexSlice[date_graph_from:, ticker], :].droplevel(1)

        # Plot the stock.
        fig = go.Figure()

        # Add traces
        fig.add_trace(
            go.Scatter(
                x=dpsar.index,
                y=dpsar["psar_short"],
                mode="markers",
                name="psar short",
                marker_color=c5,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dpsar.index,
                y=dpsar["psar_long"],
                mode="markers",
                name="psar long",
                marker_color=c1,
                marker_opacity=0.5,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dpsar.index, y=dpsar["close"], mode="lines", name="close", marker_color=c6
            )
        )

        fig.update_layout(
            title="For {}, .02 and .002 accelerators.".format(ticker),
            xaxis_title="Date",
            yaxis_title="Price",
            font=dict(family="Courier New, monospace", size=16),
        )

        fig.show()
