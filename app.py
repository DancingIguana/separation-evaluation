import dash
from dash import Dash, dcc, html, Input, Output, dash_table, page_registry

import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import os
import json


app = Dash(__name__,external_stylesheets=[dbc.themes.DARKLY,], use_pages=True)

app.layout = html.Div([
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink(page["name"], href = page["relative_path"]))
            for page in dash.page_registry.values()
        ],
        brand="Separation Evaluation",
        brand_href="#",
        color="primary",
        dark=True
    ),
	dash.page_container
])

if __name__ == '__main__':
	app.run_server(debug=True)
