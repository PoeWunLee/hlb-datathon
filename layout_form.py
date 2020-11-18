import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import json

#use template stylesheet
app = dash.Dash(__name__,external_stylesheets=[dbc.themes.DARKLY])

masterDF = pd.read_csv("cleaned_data.csv")
masterDF["is_SnP_higher"] = masterDF["highest_MG_SnP_OMV"]== masterDF["MG_SnP"]

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
                    {"label": "Freehold", "value": "F"},
                    {"label": "Leasehold", "value": "L"},
                ],
                value="L",
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
                    {"label": "Undercon", "value": "UC"},
                    {"label": "Completed", "value": "CO"},
                ],
                value="CO",
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
                id="prop-type-dropdown",
                options=[{'label': property_type, 'value': property_type} for property_type in property_types],
                style={'background-color': 'white', 'color': 'black'},
                value="Apartment"
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
                options = [
                    {"label": "Residential", "value": "Residential"},
                    {"label": "Non-Residential", "value": "Non Residential"}
                ],
                inline=True,
                value="Residential"
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
                    {"label": "Landed", "value": "Landed"},
                    {"label": "Non-Landed", "value": "Non-Landed"},
                ],
                inline=True,
                value="Landed",
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
        dcc.Dropdown(id="mode-select", 
        options=[
            {"label": "Property Type", "value":"Property_Type"},
            {"label": "Location", "value":"Postcode"},
            {"label": "Ownership", "value":"Free_Lease_Hold_Ind"},
            {"label": "Residential Type", "value":"Residential_Type"},
            {"label": "Landed Type", "value":"Landed_Type"}
        ],
        style={"color":"black"}, value="Property_Type")

    ], style=SIDEBAR_STYLE)

content = html.Div([
    html.Div(id="hidden-div",style={"display":"none"}),
    dbc.Row([
        dbc.Col(dbc.Jumbotron(dbc.Container(dcc.Loading(id="jumb-1"),fluid=True ,style={"padding":0})),width="auto"),
        dbc.Col(dbc.Jumbotron(dbc.Container(dcc.Loading(id="jumb-2"),fluid=True, style={"padding":0})),width="auto")
    ]),
    dbc.Row([
        dbc.Col(dbc.Jumbotron(dbc.Container(dcc.Loading(id="jumb-3"),fluid=True ,style={"padding":0})),width="auto"),
        dbc.Col(dbc.Jumbotron(dbc.Container(dcc.Loading(id="jumb-4"),fluid=True ,style={"padding":0})),width="auto")
    ])

], style=CONTENT_STYLE)


#define layout
app.layout = dbc.Container([sidebar, content],

    fluid=True,
    style={"padding":10}
)

@app.callback(
    Output("hidden-div","children"),
    [Input("mode-select","value"),
    Input("ownership-radios-row","value"),
    Input("residential-type-radios-row","value"),
    Input("landed-type-radios-row","value"),
    Input("prop-type-dropdown","value"),
    Input("postcode_input","value")]
)
def intermediate_link(selected, owner, res, land,prop,postcode):
    refDict = {
        "Property_Type":prop,
        "Postcode":postcode,
        "Free_Lease_Hold_Ind":owner,
        "Residential_Type":res,
        "Landed_Type":land
    }
    
    if postcode==None:
        refDict["Postcode"] =47301
    return refDict[selected]

@app.callback(
    Output("jumb-4","children"),
    [Input("hidden-div","children"),
    Input("mode-select","value")]
)
def createFig4(thisInput,mode):

    myDF = masterDF.copy()
    myDF = myDF.loc[myDF[mode]==thisInput]

    #if gap more than 1 mil, log x axis
    flag=False
    if myDF["highest_MG_SnP_OMV"].max() - myDF["highest_MG_SnP_OMV"].min() > 10**6:
        flag=True

    #beautify is_SnP_higher
    myDF["is_SnP_higher"] = myDF["is_SnP_higher"].replace([True,False], ["SnP>OMV", "SnP<OMV"])
    myDF.rename(columns={"is_SnP_higher":"Legend","highest_MG_SnP_OMV":"Highest value between SnP/OMV"},inplace=True)

    fig = px.histogram(myDF,x="Highest value between SnP/OMV",color="Legend",log_x=flag)
    fig.update_layout(
        template="plotly_dark", 
        paper_bgcolor="#303030", 
        plot_bgcolor = "#303030",
        barmode="overlay",
        margin={"r":0, "t":0, "l":0, "b":0},
        width = 650,
        height = 250,
        legend={
            "yanchor":"bottom",
            "xanchor": "right"
        }
    )
    
    return dcc.Graph(figure=fig)

@app.callback(
    Output("jumb-1","children"),
    [Input("hidden-div","children"),
    Input("mode-select","value")]
)
def createFig1(thisInput,mode):

    #bar
    sumdf = masterDF.copy()
    sumdf = sumdf.groupby(["Submission_Mth",mode]).mean().reset_index()

    #moving average
    avedf = sumdf.copy()
    avedf = avedf.loc[avedf[mode]==thisInput]
    avedf["average"] = avedf["difference"].rolling(3).mean()

    fig = px.bar(avedf,x="Submission_Mth",y="difference")
    fig.add_scatter(x=avedf["Submission_Mth"],y=avedf["average"],name="3 month average")
    fig.update_layout(
        template="plotly_dark", 
        paper_bgcolor="#303030", 
        plot_bgcolor="#303030",
        margin={"r":0, "t":0, "l":0, "b":0},
        width = 500,
        height = 250,
        legend={
            "yanchor":"bottom",
            "xanchor": "right"
        }
    )

    return dcc.Graph(figure=fig)

@app.callback(
    Output("jumb-3","children"),
    [Input("hidden-div","children"),
    Input("mode-select","value")]
)
def createFig3(thisInput,mode):

    #bar
    myDF = masterDF.copy()
    myDF = myDF.loc[myDF[mode]==thisInput]

    labels=["SnP > OMV", "SnP < OMV"]
    values = [myDF.loc[myDF["is_SnP_higher"]==True]["is_SnP_higher"].count(), 
    myDF.loc[myDF["is_SnP_higher"]==False]["is_SnP_higher"].count()]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                             insidetextorientation='radial', hole=.7
                            )])
    fig.update_layout(
        template="plotly_dark", 
        paper_bgcolor="#303030",
        margin={"r":0, "t":0, "l":0, "b":0},
        width = 300,
        height = 250,
        legend={
            "yanchor":"bottom",
            "xanchor": "right"
        }
    )

    return dcc.Graph(figure=fig)

@app.callback(
    Output("jumb-2","children"),
    [Input("mode-select","value")]
)
def createFig2(mode):

    myDF = masterDF.copy()

    #use states instead of postcode
    if mode == "Postcode":
        mode = "Property_State"

    #filter exteme outliers for display
    myDF = myDF.loc[(myDF["difference"] < 10**6) & (myDF["difference"] > -10**6)]

    boxFig = go.Figure()    
    for items in myDF[mode].unique():
        boxFig.add_trace(go.Box(x=myDF["difference"][myDF[mode] == items], name=items))

    boxFig.update_layout(
        template="plotly_dark", 
        paper_bgcolor="#303030", 
        plot_bgcolor ="#303030" ,
        showlegend=False,
        margin={"r":0, "t":0, "l":0, "b":0},
        width = 450,
        height = 250
    )

    return dcc.Graph(figure=boxFig)

    
    

#run app on local host
if __name__ == '__main__':
    app.run_server(debug=True)
