import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

#use template stylesheet
app = dash.Dash(__name__,external_stylesheets=[dbc.themes.DARKLY])

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

sidebar = html.Div([
        html.H3("<Our Project title>"),
        html.P("By The Universe Academy"),
        html.Hr(),
        dbc.Nav([
            html.P("State"),
            dcc.Dropdown("Select state"),
            html.Br(),
            html.P("Postcode"),
            dcc.Dropdown("Select postcode"),
            html.Br(),
            html.P("Property type"),
            dcc.Dropdown("Select property type"),
            html.Br(),
            html.P("Landed option"),
            dcc.Dropdown("Select landed option"),
            dbc.NavItem(dbc.NavLink("About us", active=True, id="about-us")),
            dbc.Collapse([
                dbc.Card(dbc.CardBody(html.A("Jun Wen Kwan",href="https://au.linkedin.com/in/junwenkwan", style=DROPDOWN_STYLE))), 
                dbc.Card(dbc.CardBody(html.A("Poe Wun Lee", href="https://www.linkedin.com/in/pwunlee/", style=DROPDOWN_STYLE))),
                dbc.Card(dbc.CardBody(html.A("Kian Chong Khoo", href="https://www.linkedin.com/in/kianchongkhoo/", style=DROPDOWN_STYLE)))
            ],id="collapse")
        ],

        vertical=True,
        )
    ], style=SIDEBAR_STYLE)

content = html.Div([

    dbc.Row([
        dbc.Col(dbc.Jumbotron(dbc.Container(fluid=True ,style={"padding":10}))),
        dbc.Col(dbc.Jumbotron(dbc.Container(fluid=True, style={"padding":10}))),
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

@app.callback(
    Output("collapse","is_open"),
    [Input("about-us","n_clicks")],
    [State("collapse","is_open")]
)
def toggle_collapse(n,is_open):
    if n:
        return not is_open
    return is_open

#run app on local host
if __name__ == '__main__':
    app.run_server(debug=True)
