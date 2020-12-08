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
import dash_table
import dash_bootstrap_components as dbc
import dash_html_components as html
import argparse
import base64


external_stylesheets = [dbc.themes.JOURNAL, '.assets/dcc.css']
app = dash.Dash(external_stylesheets=external_stylesheets, title="SIMPA")

simpa_logo = './.assets/simpa_logo.png'
encoded_logo = (base64.b64encode(open(simpa_logo, 'rb').read())).decode()

app.layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.H4("SIMPA Visualization Tool"),
            html.H6("CAMI, Computer Assisted Medical Interventions"),
        ], width=9),
        dbc.Col([
            html.Img(src='data:image/png;base64,{}'.format(encoded_logo), width='50%')
        ], width=3)
    ]),
    html.Br(),
    dcc.Tabs([
        dcc.Tab(label="Visualization", id="tab-1", children=[
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.H6("Plotting / Handling settings"),
                    dbc.Input(
                        id="file_path",
                        type="text",
                        pattern=None,
                        placeholder="Path to simulation folder or file",
                        persistence_type="session",
                    ),
                    dcc.Dropdown(
                        id="file_selection",
                        multi=True,
                        placeholder="Simulation files",
                        persistence_type="session",
                        disabled=True
                    ),
                    html.Br(),
                    html.Hr(),
                    html.H6("Data selection"),
                    dbc.Input(
                        id="n_plots",
                        type="number",
                        pattern=None,
                        placeholder="Number of plots",
                        persistence_type="session",
                        disabled=True,
                    ),
                    dcc.Dropdown(
                        id="param",
                        multi=True,
                        placeholder="Parameter to plot",
                        persistence_type="session",
                        disabled=True,
                    ),
                    html.Br(),
                    html.Hr(),
                    html.H6("Visual settings"),
                    dcc.Dropdown(
                        id="palette_chooser",
                        multi=False,
                        placeholder="Color palette",
                        persistence_type="session",
                        disabled=True,
                    )
                ], width=2),
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Parameter visualization"),
                            html.Div(
                                id="plot_div_1",
                                children=[
                                    dcc.Graph(id="plot_11",
                                              hoverData={'points': [{'x': 0, 'y': 0, 'customdata': None}]},
                                              style={"width": "50%", "display": 'inline-block'}),
                                    dcc.Graph(id="plot_12",
                                              hoverData={'points': [{'x': 0, 'y': 0, 'customdata': None}]},
                                              style={"width": "50%", "display": 'inline-block'})
                                ]
                            )
                        ])
                    ]),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.H6("Spectra visualization"),
                            html.Div(
                                id="plot_div_2",
                                children=[
                                    dcc.Graph(id="plot_21",
                                              hoverData={'points': [{'x': 0, 'y': 0, 'customdata': None}]},
                                              style={"width": "100%", "display": 'inline-block'})
                                ]
                            )
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    html.H6("Plot information"),
                    html.Hr()
                ], width=2)
            ])
        ]),
        dcc.Tab(label="Properties", children=[
            html.Br(),
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H6("Data selection"),
                        html.Hr(),
                        html.Br(),
                        dash_table.DataTable(
                            id="data_table",
                            columns=[{"name": 'Col 1', "id": 'Col 1'}, {"name": 'Col 2', "id": 'Col 2'}],
                            data=[{'Col 1': 1, 'Col 2': 0.5}, {'Col 1': 2, 'Col 2': 1.5}],
                            style_as_list_view=True,
                            style_table={'overflowX': 'scroll', 'maxHeight': '300px',
                                         'overflowY': 'scroll', 'maxWidth': '800px'},
                            fixed_rows={'headers': True, 'data': 0},
                            style_cell={
                                'minWidth': '50px', 'maxWidth': '90px',
                                'overflow': 'hidden',
                                'textOverflow': 'ellipsis',
                                'padding': '5px',
                                'textAlign': 'center'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': 'rgb(248, 248, 248)'
                                }
                            ],
                            style_header={
                                'backgroundColor': '#e6b600ff',
                                'fontWeight': 'bold'
                            },
                            filter_action='native',
                            sort_action='native',
                            sort_mode='multi',
                            row_selectable="multi",
                            row_deletable=True,
                            persistence=False,
                        )
                    ]), width=6
                )
            ])
        ])
    ])
], style={'padding': 40})


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
