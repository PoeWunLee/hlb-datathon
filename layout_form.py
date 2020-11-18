import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State

#use template stylesheet
app = dash.Dash(__name__,external_stylesheets=[dbc.themes.DARKLY])

masterDF = pd.read_csv("cleaned_data.csv")

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
    "background-color": "#4682B4",
    "color": "white"
}
CONTENT_STYLE = {
    "margin-left":"22rem",
    "margin-right":"1rem"
}

DROPDOWN_STYLE = {
    "color" : "white",
}

# Define inputs for address
postcode_input = dbc.FormGroup(
    [
        dbc.Label("Postcode", width=4),
        dbc.Col(
            dbc.Input(id="postcode_input", placeholder="Postcode", type="number"),
            width=8,
        ),
    ],
    row=True,
)

ownership_input = dbc.FormGroup(
    [
        dbc.Label("Ownership", html_for="ownership-radios-row", width=4),
        dbc.Col(
            dbc.RadioItems(
                id="ownership-radios-row",
                options=[
                    {"label": "Freehold", "value": "freehold"},
                    {"label": "Leasehold", "value": "leasehold"},
                ],
                value="Leasehold",
                inline=True,
            ),
            width=8,
        ),
    ],
    row=True,
)

status_completion_input = dbc.FormGroup(
    [
        dbc.Label("Status Completion", html_for="status-completion-radios-row", width=4),
        dbc.Col(
            dbc.RadioItems(
                id="status-completion-radios-row",
                options=[
                    {"label": "Undercon", "value": "undercon"},
                    {"label": "Completed", "value": "completed"},
                ],
                value="completed",
                inline=True,
            ),
            width=8,
        ),
    ],
    row=True,
)

stage_completion_input = dbc.FormGroup(
    [
        dbc.Label("Completion Stage", width=4),
        dbc.Col(
            dbc.Input(id="stage_completion_input", placeholder="Stage of Completion", type="text"),
            width=8,
        ),
    ],
    row=True,
)

area_input = dbc.FormGroup(
    [
        dbc.Label("Land/Build Area", width=4),
        dbc.Col(
            dbc.Input(id="area_input", placeholder="Area", type="number"),
            width=8,
        ),
    ],
    row=True,
)

property_description_input = dbc.FormGroup(
    [
        dbc.Label("Property", width=4),
        dbc.Col(
            dbc.Input(id="property_description_input", placeholder="Property Description", type="text"),
            width=8,
        ),
    ],
    row=True,
)

# TODO: Add more types
property_types = ['Apartment', 'Semi-Detached', 'Flat', 'Condominium',
       'Terrace House', 'Shop lot', 'Bungalow', 'Office lot', 'Townhouse',
       'Commercial/Shopping Complex', 'Penthouse']

property_type_input = dbc.FormGroup(
    [
        dbc.Label("Property Type", width=4),
        dbc.Col(
            dcc.Dropdown(
                options=[{'label': property_type, 'value': property_type} for property_type in property_types],
                style={'background-color': 'white', 'color': 'black'}
            ), 
            width=8,
        ),
    ],
    row=True,
)

residential_input = dbc.FormGroup(
    [
        dbc.Label("Residential Type", html_for="residential-type-radios-row", width=4),
        dbc.Col(
            dbc.RadioItems(
                id="residential-type-radios-row",
                options=[
                    {"label": "Residential", "value": True},
                    {"label": "Non-Residential", "value": False},
                ],
                inline=True,
                value=True,
            ),
            
            width=8,
        ),
    ],
    row=True,
)

landed_type_input = dbc.FormGroup(
    [
        dbc.Label("Landed Type", html_for="landed-type-radios-row", width=4),
        dbc.Col(
            dbc.RadioItems(
                id="landed-type-radios-row",
                options=[
                    {"label": "Landed", "value": True},
                    {"label": "Non-Landed", "value": False},
                ],
                inline=True,
                value=True,
            ),
            
            width=8,
        ),
    ],
    row=True,
)


sidebar = html.Div([

        html.H3("<Our Project title>"),
        html.P("By The Universe Academy"),
        html.Hr(),
        property_type_input,
        postcode_input,
        ownership_input,
        residential_input,
        landed_type_input,
        area_input,

    ], style=SIDEBAR_STYLE)

content = html.Div([
    dcc.Dropdown(id="mode-select", options=[
        {"label": "Property Type", "value":"Property_Type"},
        {"label": "Location", "value":"Postcode"},
        {"label": "Ownership", "value":"Free_Lease_Hold_Ind"},
        {"label": "Residential Type", "value":"Residential_Type"},
        {"label": "Landed Type", "value":"Landed_Type"},
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col(dbc.Jumbotron(dbc.Container(fluid=True ,style={"padding":10}))),
        dbc.Col(dbc.Jumbotron(dbc.Container(fluid=True, style={"padding":10})))
    ]),
    dbc.Row([
        dbc.Col(dbc.Jumbotron(dbc.Container(fluid=True ,style={"padding":10}))),
        dbc.Col(dbc.Jumbotron(dbc.Container(fluid=True, style={"padding":10}))),
        dbc.Col(dbc.Jumbotron(dbc.Container(fluid=True ,style={"padding":10})))
    ]),
    dbc.Row([
        dbc.Col(dbc.Jumbotron(dbc.Container(fluid=True ,style={"padding":10})))
    ]),
    dbc.Row([
        dbc.Col(dbc.Jumbotron(dbc.Container(fluid=True ,style={"padding":10}))),
        dbc.Col(dbc.Jumbotron(dbc.Container(fluid=True ,style={"padding":10})))
    ]),

], style=CONTENT_STYLE)


#define layout
app.layout = dbc.Container([sidebar, content],

    fluid=True,
    style={"padding":10}
)




#run app on local host
if __name__ == '__main__':
    app.run_server(debug=True)
