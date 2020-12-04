"""
Hyperspectral tissue simulation visualization toolkit
===================================================================
The MIT License (MIT)

Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated simpa_documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

REQUIREMENTS:
===================================================================
You will need to install the following packages in order to use this script. It is recommended to create a virtualenv
to install this packages:

`pip install dash dash-table dash-daq plotly plotly-express dash-bootstrap-components pandas numpy`

USAGE:
===================================================================

"""

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import argparse


external_stylesheets = [dbc.themes.JOURNAL, '.assets/dcc.css']
app = dash.Dash(external_stylesheets=external_stylesheets, title="Manifold Analyzer")

app.layout = html.Div([
    html.H4("Simulations Analysis Multi-toolbox (SAM)"),
    html.H6("CAMI, Computer Assisted Medical Interventions"),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dbc.Input(
                id="file_path",
                type="text",
                pattern=None,
                placeholder="Path to simulation folder or file",
                persistence_type="session",
            ),
            html.Br(),
            dbc.Input(
                id="n_plots",
                type="number",
                pattern=None,
                placeholder="Number of plots",
                persistence_type="session"
            )
        ], width=3),
        dbc.Col(
            html.Div(
                id="plot_div",
                children=[
                    dcc.Graph(id="plot_1", hoverData={'points': [{'x': 0, 'y': 0, 'customdata': None}]},
                              style={"width": "50%", "display": 'inline-block'}),
                    dcc.Graph(id="plot_2", hoverData={'points': [{'x': 0, 'y': 0, 'customdata': None}]},
                              style={"width": "50%", "display": 'inline-block'})
                ]
            )
        )
    ])
], style={'padding': 40})


@app.callback(Output("plot_div", "children"),
              [Input("n_plots", "value")])
def generate_plots(n_plots):
    if n_plots is None:
        return [dcc.Graph(id="plot_1", hoverData={'points': [{'x': 0, 'y': 0, 'customdata': None}]},
                          style={"width": "50%", "display": 'inline-block'})
                ]
    n_plots = int(n_plots)
    plots = []
    for i in range(n_plots):
        p = dcc.Graph(id=f"plot_{i}", hoverData={'points': [{'x': 0, 'y': 0, 'customdata': None}]},
                      style={"width": "50%", "display": 'inline-block'})
        plots.append(p)
    return plots


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Run app in debug mode. Changes in code will update the app automatically.")
    parser.add_argument("-p", "--port", type=int,
                        help="Specify the port where the app should run (localhost:port), default is 8050")
    parser.add_argument("-host", "--host_name", type=str,
                        help="Name of the host on which the app is run, by default it is localhost")
    arguments = parser.parse_args()
    port = 8050
    host_name = "localhost"
    if arguments.port:
        port = arguments.port
    if arguments.host_name:
        host_name = arguments.host_name
    app.run_server(debug=arguments.debug, host=host_name, port=port)
