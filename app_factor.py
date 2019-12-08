import datetime
import sys, os

import dash
import dash_daq as daq
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output
import dash_ui as dui
import numpy as np
import quantstats as qs
import pandas as pd

# import plotly.graph_objs as go
import plotly.graph_objects as go
import pymysql
import yfinance as yf
from app import app


# import factor_rank as fr

"""Retrieve all of the factor data from the SHARADAR SF1 datafile 
restricting columns to those required by the factor_rank module."""

engine = pymysql.connect("127.0.0.1", "root", "Factor1Edna!", "factor")
# engine = pymysql.connect("127.0.0.1", "root", '', "factor")

# Factor data
df_all = pd.read_sql("SELECT * FROM dash_factors", engine)
df_all = df_all.set_index("ticker")
df_all = df_all.dropna()
df_all["calendardate"] = pd.to_datetime(df_all["calendardate"])

# For demonstration, remove stocks under $5 price.
df_all = df_all[df_all.index.isin(df_all[df_all["price"] > 5].index.unique())]

quarters = df_all["calendardate"].sort_values().unique()

# Company/ticker information
df_ticker = pd.read_sql("SELECT * FROM ticker", engine)

# There are six factors being called. Set their weights evenly.
wts = {"ev": 1 / 6, "pe": 1 / 6, "re": 1 / 6, "de": 1 / 6, "ic": 1 / 6, "pc": 1 / 6}

# Values for date slider.
slider_dict = {
    0: "",
    1: "06",
    2: "",
    3: "2015",
    4: "",
    5: "06",
    6: "",
    7: "2016",
    8: "",
    9: "06",
    10: "",
    11: "2017",
    12: "",
    13: "06",
    14: "",
    15: "2018",
    16: "",
    17: "06",
    18: "",
    19: "2019",
    20: "",
    # 21: "06",
}


"""
NEW INCOMING DATA WILL NEED TO HAVE INDICES ADDED EITHER BEFORE CALL OR IN THIS MODULE. 
TWO INDICES: DATE AND TICKER.

THERE IS NO DATE. THE INCOMING DATAFRAME WILL BE ALREADY FILTERED FOR DATE FROM MYSQL
JUST ONE INDEX TICKER.


Minimum and maximum dates: 
2014-01-02 00:00:00
2019-06-12 00:00:00



FUNDAMENTAL FUNCTIONS
Fundamentals in class objects. Returning full two pandas.Series with stock as index and rank as values. 
Long series will have rank by lowest 
"""


def factor_ev(df_factor):
    """
    Rank for factor evebitda
    :param df_factor: Factor dataframe.
    :return: Returns pd.Series. long evebitda stock list
    """
    evebitda = (
        df_factor.loc[:, "evebitda"]
        .sort_values(ascending=False)
        .rank(ascending=False)
        .astype("int32")
        .rename("ev")
    )

    return evebitda


def factor_pe(df_factor):
    """
    Rank for factor pe.
    :param df_factor: Factor dataframe.
    :return: Returns pd.Series. long price/earnings stock list
    Variable df_price sent to decorator.
    """

    s_pe = df_factor.loc[:, "pe"].rename("pe")

    price_earnings = (
        s_pe[~s_pe.isin([np.inf, -np.inf])]
        .sort_values(ascending=False)
        .rank(ascending=False)
        .astype("int32")
    )

    return price_earnings


def factor_re(df_factor):
    """
    Ranks for factor roe.
    :param df_factor: Factor dataframe.
    :return: Returns pd.Series. long return on equity stock list
    """

    roe = (
        df_factor.loc[:, "roe"]
        .sort_values(ascending=False)
        .rank(ascending=False)
        .astype("int32")
        .rename("re")
    )

    return roe


def factor_de(df_factor):
    """
    Rank for factor debt to equity.
    :param df_factor: Factor dataframe.
    :return: Returns pd.Series. long debt to equity stock list
    """

    # Calculate debt to equity into a pd.Series
    s = pd.Series(df_factor["de"], name="de")

    debt_equity = s.sort_values(ascending=True).rank(ascending=True).astype("int32")

    return debt_equity


def factor_ic(df_factor):
    """
    Rank for factor interest coverage. Interest coverage is problematic if interest is zero
    as it creates infinity response. Therefore interest coverage if flipped, interest expense / ebit. Large numbers
    are bad, or short, and small numbers are good.
    :param df_factor: Factor dataframe.
    :return: Returns pd.Series. long interest coverage stock list
    """

    # Calculate interest expense to ebit into a pd.Series
    s = pd.Series(df_factor["intexp"] / df_factor["ebit"], name="ic")

    interest_coverage = (
        s.sort_values(ascending=True).rank(ascending=True).astype("int32")
    )

    return interest_coverage


def factor_pc(df_factor):
    """
    Rank for factor price to cash flow.
    :param df_factor: Factor dataframe.
    :return: Returns pd.Series. long price to cash flow stock list
    """

    # Calculate debt to equity int0 a pd.Series
    s = pd.Series(df_factor["price"] / (df_factor["fcfps"]), name="pc")

    price_cash_flow = s.sort_values(ascending=False).rank(
        ascending=False
    )  # .astype("int32")

    return price_cash_flow


def rank(df_factor, weight):
    """
    Generates final factor filtered lists of long stock and short stocks.
    :param df_factor: Factor dataframe.
    :param weight: Weight that each factor and long/short will be weighted.
                   Integer 1 means fully weighted both long and short. 0 means no weight.
    :return: Returns pd.Series. long stock recommendation list.
    """
    rank_ev = factor_ev(df_factor).sort_index()
    rank_pe = factor_pe(df_factor).sort_index()
    rank_re = factor_re(df_factor).sort_index()
    rank_de = factor_de(df_factor).sort_index()
    rank_ic = factor_ic(df_factor).sort_index()
    rank_pc = factor_pc(df_factor).sort_index()

    rank = (
        (
            (
                rank_ev * weight["ev"]
                + rank_pe * weight["pe"]
                + rank_re * weight["re"]
                + rank_de * weight["de"]
                + rank_ic * weight["ic"]
                + rank_pc * weight["pc"]
            )
            .sort_values(ascending=True)
            .rank(ascending=True)
        )
        .sort_index()
        .rename("rank")
    )

    df_result = pd.DataFrame(rank)
    df_result = df_result.join(
        [rank_ev, rank_pe, rank_re, rank_de, rank_ic, rank_pc], how="left", sort=True
    )

    return df_result


def factor(df_all, tdate):
    if not tdate:
        tdate = "2014-06-30"

    # Restrict data to tdate
    df_factor = df_all[df_all["calendardate"] == tdate]

    # There are six factors being called. It is possible to call them with weights, default is zero.
    wts = {"ev": 1 / 6, "pe": 1 / 6, "re": 1 / 6, "de": 1 / 6, "ic": 1 / 6, "pc": 1 / 6}

    result = rank(df_factor, wts)
    result = result.sort_values("rank", ascending=True).drop_duplicates()

    return result


def heatmap(stock_returns):
    # Create heatmap dataframe.
    s_heatmap = qs.stats.monthly_returns(stock_returns).unstack(level=1)
    df_heatmap = pd.DataFrame(s_heatmap).reset_index()
    df_heatmap.columns = ["Month", "Year", "Return"]

    fig_heatmap = {
        "data": [
            go.Heatmap(
                x=df_heatmap["Month"],
                y=df_heatmap["Year"],
                z=df_heatmap["Return"],
                name="Return Heatmap",
                colorscale="Portland",
            )
        ],
        "layout": go.Layout(
            title={"text": "Monthly Returns Heatmap", "xref": "paper", "x": 0.5},
            xaxis=dict(title="Months"),
            yaxis=dict(title="Years"),
        ),
    }

    return fig_heatmap


def fig_factor(df, single_stock_factors, tdate_text, company_name):
    """Factor figure"""
    fig_factor = {
        "data": [
            go.Bar(x=[x for x in range(1, df.shape[1] + 1)], y=single_stock_factors)
        ],
        "layout": go.Layout(
            xaxis={
                # "title": "Factor",
                "tickmode": "array",
                "tickvals": [x for x in range(1, df.shape[1] + 1)],
                # "tickangle": 15,
                "ticktext": [
                    "rank",
                    "evebitda",
                    "pe",
                    "roe",
                    "de",
                    "icr",
                    "p/cf",
                ],  # df.columns.tolist(),
                "showgrid": False,
            },
            yaxis={
                "title_text": "Rank (low score = top rank)",
                "range": [0, df.shape[0]],
                "showgrid": True,
            },
            title={
                "text": "Factor Rankings on {} for {}".format(tdate_text, company_name),
                "xref": "paper",
                "x": 0.5,
            },
        ),
    }
    return fig_factor


def stock_selector(tickers):
    """Drop box"""
    return [
        {"label": str(i + 1) + " " + t, "value": t} for (i, t) in enumerate(tickers)
    ]


def simple_stock_chart(df_stock, fig_stock_title):
    """Simple stock chart"""
    fig_stock = {
        "data": [
            go.Ohlc(
                x=df_stock.index,
                open=df_stock["Open"],
                high=df_stock["High"],
                low=df_stock["Low"],
                close=df_stock["Close"],
            )
        ],
        "layout": go.Layout(
            title="OHLC chart for {}".format(fig_stock_title),
            yaxis={"title": "Price in US$"},
        ),
    }
    return fig_stock


def stock_vs_bm(s_stock, s_bm, ticker, fig_stock_title):
    """Ticker vs. benchmark (SPY)"""
    stock_rebase = qs.utils.rebase(s_stock, base=100)
    bm_rebase = qs.utils.rebase(s_bm, base=100)
    fig_ticker_vs_spy = {
        "data": [
            go.Scatter(x=stock_rebase.index, y=stock_rebase.values, name=ticker),
            go.Scatter(
                x=bm_rebase.index, y=bm_rebase.values, mode="lines", name="S&P 500"
            ),
        ],
        "layout": go.Layout(
            title="{} vs. S&P 500 index".format(fig_stock_title),
            yaxis={"title": "Growth of $100"},
        ),
    }
    return fig_ticker_vs_spy


def metrics(stock_returns, bm_returns, ticker):
    """Metrics"""
    # Table of ratios
    df_metrics = qs.reports.metrics(stock_returns, bm_returns, display=False)

    df_metrics = df_metrics.reset_index()
    df_metrics.columns = ["Items", ticker, "SPY"]

    metrics_table_columns = [
        {"name": "Items", "id": "Items"},
        {"name": ticker, "id": ticker},
        {"name": "SPY", "id": "SPY"},
    ]

    metrics_table_data = df_metrics.to_dict("records")

    return [metrics_table_data, metrics_table_columns]


def my_rolling_sharpe(y):
    return np.sqrt(126) * (y.mean() / y.std())  # 21 days per month X 6 months = 126


def rolling_sharpe(stock_returns, fig_stock_title):
    """6 Month Rolling Sharpe Ratio"""
    rolling_sharpe = stock_returns.rolling(126).apply(my_rolling_sharpe, raw=True)

    rolling_sharpe = {
        "data": [
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                name="6mo Rolling Sharpe Ratio",
            )
        ],
        "layout": go.Layout(
            title="6 Month Rolling Sharpe Ratio for {}".format(fig_stock_title),
            yaxis={"title": "Sharpe Ratio (6 mo)"},
        ),
    }

    return rolling_sharpe


def drawdown_period(stock_returns, fig_stock_title):
    """6 month Drawdown Period"""
    # Number of Drawdown Periods
    dper = 5

    # Cumulative return series
    s_cum_return = (stock_returns + 1).cumprod() - 1

    # Use stock prices
    stock_prices = qs.utils.to_prices(stock_returns)

    # Convert to drawdown series
    stock_drawdown = qs.stats.to_drawdown_series(stock_prices)
    drawdown_periods = qs.stats.drawdown_details(stock_drawdown)

    # Sort by max days and filter for 5 periods to graph
    drawdown_graph = drawdown_periods.sort_values("days", ascending=False).head(dper)

    drawdown_starts = [x for x in drawdown_graph["start"]]
    drawdown_ends = [x for x in drawdown_graph["end"]]
    drawdown_days = [x for x in drawdown_graph["days"]]

    # Reset the number of drawdown periods as it may be lower.
    dper = drawdown_graph.shape[0]

    # Number reference for title.
    num_as_text = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}

    drawdown = {
        "data": [
            go.Scatter(
                x=s_cum_return.index, y=s_cum_return.values, name="Cumulative Return"
            ),
            go.Scatter(
                x=drawdown_starts,
                y=[s_cum_return[x] + 0.25 for x in drawdown_starts],
                text=drawdown_days,
                mode="text",
                showlegend=False,
            ),
        ],
        "layout": go.Layout(
            title="{} Longest Drawdown Periods for {} (days)".format(
                num_as_text[dper], fig_stock_title
            ),
            showlegend=False,
            xaxis={"showgrid": False},
            yaxis={"title": "Cumulative Return", "showgrid": False},
            shapes=[
                {
                    "type": "rect",
                    # x-reference is assigned to the x-values
                    "xref": "x",
                    # y-reference is assigned to the plot paper [0,1]
                    "yref": "paper",
                    "x0": drawdown_starts[x],
                    "y0": 0,
                    "x1": drawdown_ends[x],
                    "y1": 1,
                    "fillcolor": "#ff7f0e",
                    "opacity": 0.3,
                    "layer": "below",
                    "line_width": 1,
                }
                for x in range(dper)
            ],
        ),
    }

    return drawdown


def monthly_returns(stock_returns, ticker):
    """6 Month Rolling Sharpe Ratio"""
    monthly_returns = qs.stats.monthly_returns(stock_returns).unstack(level=0).values
    mean_returns = stock_returns.mean() * 100

    monthly_returns_fig = {
        "data": [
            {
                "x": monthly_returns * 100,
                "name": "Monthly Returns",
                "type": "histogram",
            },
            {
                "x": [mean_returns + 2.5],
                "y": [3],
                "text": ["{:.2f}%".format(mean_returns)],
                "mode": "text",
                "name": "Mean Return",
                "textfont": dict(family="sans serif", size=18, color="DarkOrange"),
            },
        ],
        "layout": go.Layout(
            title="Monthly returns for {}".format(ticker),
            yaxis={"title": "Occurances", "showgrid": True},
            xaxis={"title": "Percent Return", "showgrid": True},
            bargroupgap=0.05,
            shapes=[
                dict(
                    type="line",
                    x0=mean_returns,
                    x1=mean_returns,
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="paper",
                    line_width=4,
                    line=dict(color="DarkOrange"),
                )
            ],
        ),
    }

    return monthly_returns_fig


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config["suppress_callback_exceptions"] = True

grid = dui.Grid(_id="grid", num_rows=12, num_cols=12, grid_padding=50)


divide = 4


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

grid.add_element(
    col=1,
    row=1,
    width=3,
    height=3,
    element=html.Div(
        [
            dcc.Link('Navigate to "/apps/psar"', href='/apps/psar'),
            dcc.Markdown(
                """
            #### Factor Demo using Plot.ly Dash
    
            This is a simple demonstration of Plot.ly Dash. Selecting a quarter on the slider will return a
            ranked list of equities by factor score. Selecting an equity will update all of the analytical
            charts to see if top ranking factor stocks fair well by traditional metrics.
            """
            ),
        ],
            style={"margin-left": "10px"},
    ),
)

grid.add_element(
    col=1,
    row=3,
    width=divide - 1,
    height=2,
    element=html.Div(
        html.Div(
            daq.Slider(
                id="quarter-slider",
                min=min(slider_dict.keys()),
                max=max(slider_dict.keys()),
                value=min(slider_dict.keys()),
                marks=slider_dict,
                step=1,
                size=350,
                vertical=False,
                updatemode="mouseup",
            ),
            style={
                "width": "90%",
                "align-items": "right",
                "margin-left": "10px",
            },
        )
    ),
)


grid.add_element(
    col=1,
    row=4,
    width=divide - 1,
    height=3,
    element=html.Div(
        [
            html.Div(
                [
                    html.Div(
                        dcc.Dropdown(
                            id="stock-ticker",
                            placeholder="Select a stock",
                            clearable=False,
                        ),
                        style={
                            "width": "40%",
                            "display": "inline-block",
                            "align-items": "left",
                            "margin-bottom": "10px",
                            "margin-left": "10px",
                        },
                    ),
                    html.Div(
                        dcc.Markdown(
                            """
                            Not all stocks will return graphs due
                            to mergers, delistings, and poor data.
                            """
                        ),
                        style={
                            "width": "45%",
                            "margin-left": "20px",
                            "display": "inline-block",
                            "align-items": "right",
                            # "margin-top": "20px",
                        },
                    ),
                ]
            )
        ]
    ),
)

grid.add_element(
    col=1,
    row=5,
    width=divide - 1,
    height=8,
    element=html.Div(
        dash_table.DataTable(id="metrics"),
        style={"margin-left": "30px", "width": "70%", "align-items": "center"},
    ),
)

grid.add_graph(col=4, row=1, width=4, height=4, graph_id="factor-graph")
grid.add_graph(col=8, row=1, width=4, height=4, graph_id="ohlc-graph")
grid.add_graph(col=4, row=4, width=4, height=5, graph_id="ticker-vs-bm")
grid.add_graph(col=8, row=4, width=4, height=5, graph_id="rolling_sharpe")
grid.add_graph(col=4, row=8, width=4, height=5, graph_id="drawdown")
grid.add_graph(col=8, row=8, width=4, height=5, graph_id="heatmap")


layout = html.Div(
    dui.Layout(grid=grid),
    style={"height": "100vh", "width": "100vw", "background-color": "white"},
)



# html.Div(dcc.Graph(id="monthly_returns")),



# Calculate Factors and update factor graph only.
@app.callback(
    [Output("factor-graph", "figure"), Output("stock-ticker", "options")],
    [Input("quarter-slider", "value"), Input("stock-ticker", "value")],
)
def update_figure_main(
    selected_quarter_date,
    stock_ticker_value,  # , metrics_data_state, metrics_columns_state
):

    # Set variables that will be used for factor calculation and factor.

    # Selected quarter and text version.
    tdate = quarters[selected_quarter_date]
    tdate_text = pd.to_datetime(tdate).strftime("%B %d, %Y")

    # Get factor rank list for the selected date.
    # Returns rank of stocks and each factor rank.
    df = factor(df_all, tdate)

    # Get a unique list of tickers in rank order.
    tickers = df.index.unique().values
    if stock_ticker_value:
        ticker = stock_ticker_value
    else:
        ticker = tickers[0]

    # Try to associate ticker with the company name, if doesn't work just make the
    # company name the ticker symbol.
    try:
        company_name = df_ticker.loc[df_ticker["ticker"] == ticker, "name"].tolist()[0]
    except:
        company_name = ticker

    # Resulting graph restricted to one symbol.
    single_stock_factors = df.loc[ticker, :]

    return (
        fig_factor(df, single_stock_factors, tdate_text, company_name),
        stock_selector(tickers),
    )


# Change dropdown background color to red to warn user if no data.
@app.callback(
    [Output("stock-ticker", "style")],
    [Input("quarter-slider", "value"), Input("stock-ticker", "value")],
)
def warning_no_data(selected_quarter_date, stock_ticker_value):
    tdate = quarters[selected_quarter_date]

    # Get factor rank list for the selected date.
    # Returns rank of stocks and each factor rank.
    df = factor(df_all, tdate)

    # Get a unique list of tickers in rank order.
    tickers = df.index.unique().values
    if stock_ticker_value:
        ticker = stock_ticker_value
    else:
        ticker = tickers[0]

    # Stop terminal print output for yahoo downloads.
    oldstdout = sys.stdout
    sys.stdout = open(os.devnull, "w")  # disable output
    """Get and manage data for analysis graphs"""
    ticker_and_bm = yf.download(
        ticker + " SPY",
        start=pd.to_datetime(tdate).strftime("%Y-%m-%d"),
        end=datetime.datetime.now(),
    )
    sys.stdout = oldstdout  # enable output

    if ticker_and_bm.isnull().sum().sum():
        return ({"background-color": "indianred"},)
    else:
        return ({"background-color": "white"},)


# Get pricing data and update all other graphs (not factor) and metrics.
@app.callback(
    [
        Output("ohlc-graph", "figure"),
        Output("ticker-vs-bm", "figure"),
        Output("metrics", "data"),
        Output("metrics", "columns"),
        Output("rolling_sharpe", "figure"),
        Output("drawdown", "figure"),
        Output("heatmap", "figure"),
        # Output("monthly_returns", "figure"),
    ],
    [Input("quarter-slider", "value"), Input("stock-ticker", "value")],
)
def update_figure_others(selected_quarter_date, stock_ticker_value):

    # Selected quarter and text version.
    tdate = quarters[selected_quarter_date]

    # Get factor rank list for the selected date.
    # Returns rank of stocks and each factor rank.
    df = factor(df_all, tdate)

    # Get a unique list of tickers in rank order.
    tickers = df.index.unique().values
    if stock_ticker_value:
        ticker = stock_ticker_value
    else:
        ticker = tickers[0]

    # Stop terminal print output for yahoo downloads.
    oldstdout = sys.stdout
    sys.stdout = open(os.devnull, "w")  # disable output
    """Get and manage data for analysis graphs"""
    ticker_and_bm = yf.download(
        ticker + " SPY",
        start=pd.to_datetime(tdate).strftime("%Y-%m-%d"),
        end=datetime.datetime.now(),
    )
    sys.stdout = oldstdout  # enable output

    # Check for NaN in data and stop update if found.
    if ticker_and_bm.isnull().sum().sum():
        raise PreventUpdate()
    else:
        pass

    # Forward fill for thinly traded stocks.
    ticker_and_bm = ticker_and_bm.fillna(method="ffill")

    # Create stock and benchmark OHLCV dataframes and Adj Close Series.
    df_stock = ticker_and_bm.loc[:, pd.IndexSlice[:, ticker]].droplevel(1, axis=1)
    s_stock = df_stock["Adj Close"]

    df_bm = ticker_and_bm.loc[:, pd.IndexSlice[:, "SPY"]].droplevel(1, axis=1)
    s_bm = df_bm["Adj Close"]

    # Set returns for daily change.
    stock_returns = qs.utils.to_returns(s_stock)
    bm_returns = qs.utils.to_returns(s_bm)

    # Check to see if there is data returned for the ticker.
    if df_stock.shape[0] == 0:
        fig_stock_title = "There is no data for symbol {}".format(ticker)
    else:
        fig_stock_title = ticker

    metrics_table = metrics(stock_returns, bm_returns, ticker)

    return (
        simple_stock_chart(df_stock, fig_stock_title),
        stock_vs_bm(s_stock, s_bm, ticker, fig_stock_title),
        metrics_table[0],
        metrics_table[1],
        rolling_sharpe(stock_returns, fig_stock_title),
        drawdown_period(stock_returns, fig_stock_title),
        heatmap(stock_returns),
        # monthly_returns(stock_returns, ticker),
    )


# server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)
