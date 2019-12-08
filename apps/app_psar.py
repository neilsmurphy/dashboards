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
import math
import sqlite3
import time

import dash
import dash_core_components as dcc
import dash_daq as daq
from dash.dependencies import Input, Output
import dash_ui as dui
import dash_html_components as html
import dash_table
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from app import app

from apps import update as ud

"""
The dashboard runs in a browser and can be run locally or on a cloud server. 
The data for this database is contained in a SQLite3 database called DashData.db
The dependencies for this project can be found in requirements.txt and run using pip3.
"""

print(dcc.__version__) # 0.6.0 or above is required


# Color reference variables:
c1 = "#F66D44"
c2 = "#FEAE65"
c3 = "#E6F69D"
c4 = "#AADEA7"
c5 = "#64C2A6"
c6 = "#2D87BB"


def open_db():
    """
    Opens a connection the the database.
    :return: connection
    """
    conn = sqlite3.connect("apps/data/DashData.db", detect_types=sqlite3.PARSE_DECLTYPES)
    return conn


def set_data():
    """
    Sets the data on open or when called, or when a day is updated.
    :return: history dataframe, dates series, company series.
    """

    # Get all of the history data. Open the database and create history dataframe.
    # This dataframe will be available at all times.
    conn = open_db()

    # Get history dataframe and create date index.
    set_daily = pd.read_sql("SELECT * FROM daily", conn)
    set_daily_index = set_daily.set_index(["date", "ticker"]).sort_index()

    # Create an array of all the trading dates.
    set_dates = set_daily_index.index.unique(0)

    # Create a list of all tickers.
    set_tickers = set_daily_index.index.unique(1)

    # Close the database
    conn.close()

    return set_daily, set_dates, set_tickers


def open_status(prt=False):
    """Open data with or without showing status"""

    # prt=True will print out performance results to the terminal.
    if prt:
        start = time.time()
        open_df_daily, open_dates, open_tickers = set_data()
        end = time.time()
        print("time to open dataframe from sqlite: {:.2f}".format(end - start))

        print(
            "Memory usage in megabytes: {:.2f}".format(
                open_df_daily.memory_usage().sum() / 1000000
            )
        )

        print(open_df_daily.head())

        print("dates:\n {}".format(len(open_dates)), open_dates[0:20])
        print("tickers:\n{}".format(len(open_tickers)), open_tickers[:20])

    else:
        # Open data without printing.
        open_df_daily, open_dates, open_tickers = set_data()

    dat_out = open_df_daily.to_json(date_format="iso", orient="table")
    open_df_daily = open_df_daily.set_index(["date", "ticker"]).sort_index()

    return dat_out, open_df_daily, open_dates, open_tickers


# Loading the data here will aid in performance as module will
# load data into memory. df_daily is the daily data, dates and tickers
# are arrays with the date range and total unique list of companies.
dat, df_daily, dates, tickers = open_status(False)


def stock_selector(tickers):
    """Drop box: Returns the stock tickers for the stock selector box."""
    return [{"label": str(t), "value": t} for (i, t) in enumerate(tickers)]


def simple_stock_chart(daily, date, ticker):
    """
    Main stock chart with PSAR: Returns a plotly Dash figure object for the OHLC chart
    including PSAR.
    """

    # Create chart specific dataframe from df_daily
    df_stock = daily.loc[
        pd.IndexSlice[:date, ticker],
        ["open", "high", "low", "close", "psar_short", "psar_long"],
    ].droplevel(1)

    # Build chart
    fig_stock = {
        "data": [
            # Create OHLC.
            go.Ohlc(
                x=df_stock.index,
                open=df_stock["open"],
                high=df_stock["high"],
                low=df_stock["low"],
                close=df_stock["close"],
                name="OHLC",
                line={"width": 1.5},
                increasing_line_color=c5,
                decreasing_line_color=c1,
            ),
            # Add in PSAR one day.
            go.Scatter(
                x=df_stock.index,
                y=df_stock["psar_short"],
                name="PSAR Short",
                mode="markers",
                marker_size=4,
                marker_color=c4,
            ),
            # Add in PSAR weekly.
            go.Scatter(
                x=df_stock.index,
                y=df_stock["psar_long"],
                name="PSAR Long",
                mode="markers",
                marker_size=4,
                marker_color=c2,
            ),
        ],
        # Adjust layout properties.
        "layout": go.Layout(
            title="OHLC chart for {}".format(ticker),
            yaxis={"title": "Price in AU$"},
            xaxis={"rangeslider": dict(visible=False)},
        ),
    }
    return fig_stock


def rank_table(df, page_current, page_size):
    """

    :param df: df is a dataframe filtered by date and already ranked by return/sharpe
    :param page_current:
    :param page_size:
    :return: Dictionary object for one page of the rank table.
    """
    # Create series for return.
    ret = pd.Series(df.index, df["period_return"].values).sort_index()
    ret.name = "return"

    # Create series for sharpe.
    sharpe = pd.Series(df.index, df["period_sharpe"].values).sort_index()
    sharpe.name = "sharpe"

    # Set up variables for table DataFrame
    columns = ret.index.astype(int)
    index = ["return", "sharpe"]
    data = np.array([ret.values, sharpe.values])

    # Create table dataframe with index.
    df_top = pd.DataFrame(data=data, index=index, columns=columns).T
    df_top[" index"] = df_top.index.values

    # Convert to dictionary and return.
    return df_top.iloc[
        (page_current * page_size) : ((page_current + 1) * page_size)
    ].to_dict("records")


def reversed_psar(daily, date):
    """Create daily list of psar's reversing to below price."""

    # Reduce df_daily to relevant columns, swap the index to ticker first, sort.
    df_reversed = (
        daily.loc[:, ["close", "psar_short", "psar_long", "return"]]
        .swaplevel(axis=0)
        .sort_index()
    )

    # Bull reversal
    # Create true or false for psar reversals for one and five days.
    df_reversed["psar_short_rev"] = df_reversed["psar_short"] < df_reversed["close"]
    df_reversed["psar_long_rev"] = df_reversed["psar_long"] < df_reversed["close"]

    # Reversal will be indicated as follows;
    # True = 1, False = 0. When reversed PSAR will sum to 2 on the current day,
    # and less than two on the previous day.
    mask1 = df_reversed[["psar_short_rev", "psar_long_rev"]].sum(axis=1) == 2
    mask2 = df_reversed[["psar_short_rev", "psar_long_rev"]].shift(1).sum(axis=1) < 2
    mask3 = df_reversed[["psar_short_rev", "psar_long_rev"]].sum(axis=1) == 0
    mask4 = df_reversed[["psar_short_rev", "psar_long_rev"]].shift(1).sum(axis=1) > 0
    mask5 = df_reversed["psar_short_rev"].shift(1) == 1
    mask6 = df_reversed["psar_long_rev"].shift(1) == 0

    df_reversed_bull = df_reversed.loc[(mask1 & mask2)]
    df_reversed_bear = df_reversed.loc[(mask3 & mask4)]
    df_reversed_bull_weekly_only = df_reversed.loc[(mask1 & mask5 & mask6)]

    reversed_bull_list = (
        df_reversed_bull.loc[pd.IndexSlice[:, date], :]
        .sort_values("return", ascending=False)
        .index.get_level_values(0)
        .tolist()
    )

    reversed_bear_list = (
        df_reversed_bear.loc[pd.IndexSlice[:, date], :]
        .sort_values("return", ascending=False)
        .index.get_level_values(0)
        .tolist()
    )

    reversed_bull_weekly_only_list = (
        df_reversed_bull_weekly_only.loc[pd.IndexSlice[:, date], :]
        .sort_values("return", ascending=False)
        .index.get_level_values(0)
        .tolist()
    )

    # Return test sentence with date and a list of tickers that reversed on that date.
    return [
        "PSAR reversals Bullish: {}".format(", ".join(reversed_bull_list)),
        "PSAR reversals Bearish: {}".format(", ".join(reversed_bear_list)),
        "PSAR reversals Bullish Weekly Only: {}".format(
            ", ".join(reversed_bull_weekly_only_list)
        ),
    ]


def shares_in_asx(daily, date, return_threshold, sharpe_threshold, window_size=22):
    """ Calculate the number of PSAR shares in the ASX 200"""

    # Change index of df_daily for stocks then dates.
    daily = daily.swaplevel().sort_index()

    # Calculate returns
    daily["daily_return"] = daily.groupby(level=0)["close"].pct_change(
        periods=1, fill_method="pad"
    )
    daily["period_return"] = daily.groupby(level=0)["close"].pct_change(
        periods=window_size, fill_method="pad"
    )

    # Calculate the sharpe
    daily["period_std"] = (
        daily.groupby(level=0)["daily_return"]
        .rolling(window=window_size)
        .std()
        .droplevel(0)
    )
    daily["period_sharpe"] = daily["period_return"] / daily["period_std"]

    # Reduce the dataframe to just the current date.
    daily = daily.loc[pd.IndexSlice[:, date], :]

    mask1 = daily["psar_short"] < daily["close"]
    mask2 = daily["psar_long"] < daily["close"]
    mask3 = daily["period_return"] > return_threshold
    mask4 = daily["period_sharpe"] > sharpe_threshold
    mask5 = daily[["psar_short", "psar_long"]].min(axis=1) != 0

    # Now at one date, calculate the ranks for ['period_return', 'period_sharpe']
    df_rank = daily.loc[(mask1 & mask2 & mask5), ["period_return", "period_sharpe"]]

    df_rank = df_rank.rank(ascending=False).droplevel(1)

    psars_in_asx = daily.loc[(mask1 & mask2)].shape[0]
    psars_in_asx_over_return = daily.loc[(mask1 & mask2 & mask3)].shape[0]
    psars_in_asx_over_sharpe = daily.loc[(mask1 & mask2 & mask4)].shape[0]

    asx_list = daily.loc[(mask1 & mask2)].index.get_level_values(0).tolist()

    fig = {
        "data": [
            go.Bar(
                x=["in ASX", "Return", "Sharpe"],
                y=[psars_in_asx, psars_in_asx_over_return, psars_in_asx_over_sharpe],
                name="psars",
                marker_color=[c3, c4, c5],
            )
        ],
        # Adjust layout properties.
        "layout": go.Layout(
            title="PSAR Stocks",
            # yaxis={"title": "Price in AU$"},
            # xaxis={"rangeslider": dict(visible=False)},
        ),
    }

    return [
        fig,
        "These are stocks currently in the ASX 200 that have PSARs less than price on {}: {}".format(
            date, asx_list
        ),
        df_rank,
    ]


def asset_allocation_score(df_score, date):
    """
    Determines the assest allocation score on a particular date. Scores from 1-8.
    :param df:
    :param date:
    :return: Return integer from 1 to 8
    """

    df_score_date = df_score.loc[pd.IndexSlice[date, ["XJO", "XVI"]], :].droplevel(0)

    score = sum(
        [
            df_score_date.loc["XJO", "ma15"] > df_score_date.loc["XJO", "ma30"],
            df_score_date.loc["XJO", "ma50"] > df_score_date.loc["XJO", "ma200"],
            df_score_date.loc["XJO", "psar_short"] < df_score_date.loc["XJO", "close"],
            df_score_date.loc["XJO", "psar_long"] < df_score_date.loc["XJO", "close"],
            df_score_date.loc["XVI", "ma15"] <= df_score_date.loc["XVI", "ma30"],
            df_score_date.loc["XVI", "ma50"] > df_score_date.loc["XVI", "ma200"],
            df_score_date.loc["XVI", "psar_short"] >= df_score_date.loc["XVI", "close"],
            df_score_date.loc["XVI", "psar_long"] >= df_score_date.loc["XVI", "close"],
        ]
    )

    return int(score)


def live_quote(ticker="NEA"):

    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=3)

    if ticker == "XJO":
        ticker_quote = "^AXJO"
    elif ticker == "XVI":
        ticker_quote = "^AXVI"
    else:
        ticker_quote = ticker + ".AX"

    df_stock = yf.download(ticker_quote, start=start, end=end, interval="1m")
    df_stock = df_stock.loc[df_stock.index[-1].strftime("%Y-%m-%d") :, :]

    fig_stock = {
        "data": [
            # Create OHLC.
            go.Ohlc(
                x=df_stock.index,
                open=df_stock["Open"],
                high=df_stock["High"],
                low=df_stock["Low"],
                close=df_stock["Close"],
                name="Live",
                line={"width": 1.5},
                increasing_line_color=c5,
                decreasing_line_color=c1,
            )
        ],
        # Adjust layout properties.
        "layout": go.Layout(
            title="Live chart for {}".format(ticker),
            yaxis={"title": "Price in AU$"},
            xaxis={"rangeslider": dict(visible=True)},
        ),
    }

    return fig_stock


def update_stock_data():

    # Run the update module.
    df = ud.run_update()

    dat = df.reset_index().to_json(date_format="iso", orient="table")

    return dat


# Set up CSS Stylesheet.
external_stylesheets = [
    "https://codepen.io/rmarren1/pen/mLqGRg.css",
    "https://codepen.io/chriddyp/pen/bWLwgP.css",
]

# Launch app.
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config["suppress_callback_exceptions"] = True

# Set page size for rank table.
PAGE_SIZE = 10

grid = dui.Grid(_id="grid", num_rows=12, num_cols=12, grid_padding=50)
controlpanel = dui.ControlPanel(_id="controlpanel")

controlpanel.create_group(group="Link", group_title="")
factor_link = dcc.Link('Navigate to "/apps/factor"', href='/apps/factor'),
controlpanel.add_element(factor_link, "Link")

controlpanel.create_group(group="Date", group_title="Select a date.")

select_date = dcc.DatePickerSingle(
    id="date-picker",
    min_date_allowed=dates[0],
    max_date_allowed=dates[-1],
    initial_visible_month=dates[-1],
    date=dates[-1],
    display_format="YYYY-MM-DD",
    first_day_of_week=1,
    # day_size=50,
    with_portal=True,
    with_full_screen_portal=True,
)

controlpanel.add_element(select_date, "Date")

controlpanel.create_group(group="Ticker", group_title="Select stock.")

select_ticker = (
    dcc.Dropdown(
        id="stock-ticker-p", placeholder="Select a stock", clearable=False, value="XJO"
    ),
)

controlpanel.add_element(select_ticker, "Ticker")

controlpanel.create_group(
    group="Thresholds", group_title="Set Return, Sharpe, and Period inputs."
)

return_label = html.P("Return")

controlpanel.add_element(return_label, "Thresholds")

return_threshold = (
    dcc.Input(
        id="return-threshold",
        type="number",
        value=0.1,
        placeholder="% return threshold over period",
        # debounce=True,
        min=-1000000,
        max=1000000,
        step=0.01,
    ),
)

controlpanel.add_element(return_threshold, "Thresholds")

sharpe_label = html.P("Sharpe")

controlpanel.add_element(sharpe_label, "Thresholds")

sharpe_threshold = (
    dcc.Input(
        id="sharpe-threshold",
        type="number",
        value=1,
        placeholder="Sharpe threshold over period.",
        # debounce=True,
        min=-1000000,
        max=1000000,
        step=0.01,
    ),
)

controlpanel.add_element(sharpe_threshold, "Thresholds")

period_label = html.P("Period")

controlpanel.add_element(period_label, "Thresholds")

window_size = (
    dcc.Input(
        id="window-size",
        type="number",
        value=22,
        placeholder="In days size of return period",
        min=1,
        max=1000000,
        step=1,
    ),
)

controlpanel.add_element(window_size, "Thresholds")

controlpanel.create_group(group="Update")  # , group_title="Update data.")

update_button = dcc.ConfirmDialogProvider(
    children=html.Button("Update Data"),
    id="update-data",
    message="In the live version, this would update the data.",
)

controlpanel.add_element(update_button, "Update")

update_confirm = html.Div(id="update-confirm")

controlpanel.add_element(update_confirm, "Update")


# Add grid elements.
grid.add_element(
    col=1,
    row=1,
    width=12,
    height=12,
    element=html.Div(
        style={"background-color": "white", "height": "100%", "width": "100%"}
    ),
)

grid.add_graph(col=1, row=1, width=7, height=5, graph_id="ohlc-graph-p")

grid.add_graph(col=8, row=1, width=5, height=5, graph_id="live-quote")

grid.add_element(
    col=8,
    row=6,
    width=4,
    height=4,
    element=html.Div(
        daq.Gauge(
            id="index-gauge",
            label="Index PSAR and Momentum Score",
            color=c4,
            scale={"start": 0, "interval": 1, "labelInterval": 1},
            max=8,
            value=6,
            min=0,
        ),
        style={"background-color": "white", "width": "60%", "height": "60%", "margin-top": "0px", "margin-left": "30px"},
    ),
)

grid.add_element(
    col=1,
    row=6,
    width=4,
    height=6,
    element=html.Div(
        dash_table.DataTable(
            id="rank-table",
            columns=[
                {"name": i, "id": i} for i in sorted(["return", "sharpe", " index"])
            ],
            page_current=0,
            page_size=PAGE_SIZE,
            page_action="custom",
        )
    ),
)

grid.add_graph(col=5, row=6, width=3, height=4, graph_id="psar-return-sharpe")

grid.add_element(
    col=5,
    row=10,
    width=7,
    height=3,
    element=html.Div(
        [
            html.P(id="psar-shares-in-asx-list"),
            html.Br(),
            html.Div(id="reversed-psars-bull"),
            html.Br(),
            html.Div(id="reversed-psars-bear"),
            html.Br(),
            html.Div(id="reversed-psars-bull-weekly-only"),
            html.Div(
                id="intermediate-value", style={"display": "none"}, children=""
            ),  # Stored dataframe
        ]
    ),
)


layout = html.Div(
    dui.Layout(grid=grid, controlpanel=controlpanel),
    style={"height": "100vh", "width": "100vw", "background-color": "white"},
)


# Callback decorator and function for returning values the components above.
@app.callback(
    [
        Output("stock-ticker-p", "options"),
        Output("ohlc-graph-p", "figure"),
        Output("rank-table", "data"),
        Output("rank-table", "page_count"),
        Output("reversed-psars-bull", "children"),
        Output("reversed-psars-bear", "children"),
        Output("reversed-psars-bull-weekly-only", "children"),
        Output("psar-return-sharpe", "figure"),
        Output("psar-shares-in-asx-list", "children"),
        Output("index-gauge", "value"),
        Output("live-quote", "figure"),
    ],
    [
        Input("intermediate-value", "children"),
        Input("date-picker", "date"),
        Input("stock-ticker-p", "value"),
        Input("rank-table", "page_current"),
        Input("rank-table", "page_size"),
        Input("return-threshold", "value"),
        Input("sharpe-threshold", "value"),
        Input("window-size", "value"),
    ],
)
def update_output(
    dat_int_val,
    selected_date,
    ticker_value,
    page_current,
    page_size,
    return_threshold,
    sharpe_threshold,
    window_size,
):

    # Set the df_daily_cb (callback) dataframe, to dat if not previously stored.
    if dat_int_val:
        df_daily_cb = pd.read_json(dat_int_val, orient="table")
    else:
        df_daily_cb = pd.read_json(dat, orient="table")

    df_daily_cb["date"] = pd.to_datetime(df_daily_cb["date"])
    df_daily_cb = df_daily_cb.set_index(["date", "ticker"]).sort_index()

    # Create an array of all the trading dates.
    dates_cb = df_daily_cb.index.unique(0)

    # Create a list of all tickers.
    tickers_cb = df_daily_cb.index.unique(1)

    # Convert to datetime.
    selected_date = pd.to_datetime(selected_date)

    # Dataframe with no indices.
    df_no_index = df_daily_cb.loc[
        ~df_daily_cb.index.get_level_values(1).isin(["XJO", "XVI"])
    ]

    # Dataframe with just indices.
    df_just_index = df_daily_cb.loc[
        df_daily_cb.index.get_level_values(1).isin(["XJO", "XVI"])
    ]

    # Get the most recent item in the date's index. This will move the date if it not on a trading day. Set date.
    if selected_date:
        date = df_daily_cb.index.unique(0).asof(selected_date).strftime("%Y-%m-%d")
    else:
        date = dates_cb[-1]

    # Set ticker value.
    if ticker_value:
        ticker = ticker_value
    else:
        ticker = tickers_cb[0]

    # Run shares in asx as there are thre results to return in a list.
    # 0. Figures, 1. psar shares in asx, and 2. df_rank.
    asx = shares_in_asx(
        df_daily_cb, date, return_threshold, sharpe_threshold, window_size
    )

    # Get dataframe with rank returns and sharpe.
    df_table = asx[2]

    # Calculate maximum number of rank table pages.
    page_count = math.ceil(df_table.shape[0] / page_size)

    # Set the current page to maximum if greater than the maximum on reset.
    if page_current > page_count:
        page_current = page_count - 1

    # Reversed psar list.
    reversed_psar_li = reversed_psar(df_no_index, date)

    return (
        stock_selector(tickers_cb),
        simple_stock_chart(df_daily_cb, date, ticker),
        rank_table(df_table, page_current, page_size),
        page_count,
        reversed_psar_li[0],
        reversed_psar_li[1],
        reversed_psar_li[2],
        asx[0],
        asx[1],
        asset_allocation_score(df_daily_cb, date),
        live_quote(ticker),
    )


@app.callback(
    [Output("update-confirm", "children"), Output("intermediate-value", "children")],
    [Input("update-data", "submit_n_clicks")],
)
def update(submit_n_clicks):
    """
    Updates the dataset.
    :param submit_n_clicks: Button clicker 'n' clicks.
    :return: Update string and dataframe json file.
    """
    if not submit_n_clicks:
        return (
            "Not updated, check for errors. {}".format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ),
            "",
        )
    else:
        return (
            "Last update at {}".format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ),
            "",
        )


if __name__ == "__main__":

    app.run_server(debug=True, port=8050)
