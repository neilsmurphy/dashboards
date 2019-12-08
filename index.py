import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from apps import app_psar, app_factor


app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")]
)


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/apps/psar":
        return app_psar.layout
    elif pathname == "/apps/factor":
        return app_factor.layout
    else:
        return "404"

# server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)
