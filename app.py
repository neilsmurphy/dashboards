import dash

# Set up CSS Stylesheet.
external_stylesheets = [
    "https://codepen.io/rmarren1/pen/mLqGRg.css",
    "https://codepen.io/chriddyp/pen/bWLwgP.css",
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True