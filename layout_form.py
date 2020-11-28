import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import json
import os
from run_pytorch import *


#use template stylesheet
app = dash.Dash(__name__,external_stylesheets=[dbc.themes.DARKLY])
server = app.server

masterDF = pd.read_csv("cleaned_data.csv")
masterDF["is_SnP_higher"] = masterDF["highest_MG_SnP_OMV"]== masterDF["MG_SnP"]
masterDF.loc[masterDF["difference"]==0,"is_SnP_higher"] = "SnP = OMV"
masterDF["is_SnP_higher"] = masterDF["is_SnP_higher"].replace([True,False], ["SnP > OMV", "SnP < OMV"])
encodedDF = pd.read_csv("cleaned_data_encoded_categorical_label.csv")
masterMerged = pd.merge(masterDF,encodedDF,left_index=True,right_index=True)

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
                value="Non-Landed",
            ),
            
            width=8,
        ),
    ],
    row=True,
)


sidebar = html.Div([

        html.H5("Mortgage Dashboard"),
        html.P("By The Universe Academy"),
        html.Hr(),
        property_type_input,
        postcode_input,
        ownership_input,
        residential_input,
        landed_type_input,
        #area_input,
        html.Hr(),
        html.P("Current display mode"),
        dcc.Dropdown(id="mode-select", 
        options=[
            {"label": "Property Type", "value":"Property_Type"},
            {"label": "Location", "value":"Property_State"},
            {"label": "Ownership", "value":"Free_Lease_Hold_Ind"},
            {"label": "Residential Type", "value":"Residential_Type"},
            {"label": "Landed Type", "value":"Landed_Type"}
        ],
        style={"color":"black"}, value="Property_Type"),
       html.Hr(),
       html.Div([
           html.P("About us", id="about-us"),
           #dbc.NavItem(dbc.NavLink("About us", active=True, id="about-us")),
           dbc.Collapse([
                html.A("Poe Wun Lee", href="https://www.linkedin.com/in/pwunlee/", style=DROPDOWN_STYLE),
                html.Hr(),
                html.A("Jun Wen Kwan",href="https://au.linkedin.com/in/junwenkwan", style=DROPDOWN_STYLE),
                html.Hr(),
                html.A("Kian Chong Khoo", href="https://www.linkedin.com/in/kianchongkhoo/", style=DROPDOWN_STYLE)
            ],id="collapse")])
        
    ], style=SIDEBAR_STYLE)

content = html.Div([
    html.Div(id="hidden-div",style={"display":"none"}),
    dbc.Row([
        dbc.Col(dbc.Jumbotron(dbc.Container(dcc.Loading(id="jumb-1"),fluid=True ,style={"padding":0})),width="auto"),
        dbc.Col(dbc.Jumbotron(dbc.Container(dcc.Loading(id="jumb-2"),fluid=True, style={"padding":0})),width="auto")
    ]),
    dbc.Row([
        #dbc.Col(dbc.Jumbotron(dbc.Container(dcc.Loading(id="jumb-3"),fluid=True ,style={"padding":0})),width="auto"),
        dbc.Col(dbc.Jumbotron(dbc.Container(dcc.Loading(id="jumb-3a"),fluid=True ,style={"padding":0})),width="auto"),
        dbc.Col(dbc.Jumbotron(dbc.Container(dcc.Loading(id="jumb-4"),fluid=True ,style={"padding":0})),width="auto")
    ])

], style=CONTENT_STYLE)


#define NN def
def getNewNN(thisDict):
    weights_pth = os.path.join('checkpoints','nn_weights.pth')
    model = MLP(input_size=5, output_size=3)
    model.load_state_dict(torch.load(weights_pth, map_location=torch.device('cpu')))
    model.eval()

    in_vector = []
    for key,value in thisDict.items():
        print(key,value)
        temp = masterMerged.loc[masterMerged[key+"_x"] == value].head(1)
        in_vector.append(temp.iloc[0][key+"_y"])

    #in_vector = [1, 589, 9, 1, 0]
    in_vector = torch.from_numpy(np.array(in_vector))
    in_vector =  Variable(in_vector).float()
    outputs = model.forward(in_vector)
    
    return outputs.tolist()

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


@app.callback(
    [Output("hidden-div","children"),Output("jumb-3a","children")],
    [Input("mode-select","value"),
    Input("ownership-radios-row","value"),
    Input("residential-type-radios-row","value"),
    Input("landed-type-radios-row","value"),
    Input("prop-type-dropdown","value"),
    Input("postcode_input","value")]
)
def intermediate_link(selected, owner, res, land,prop,postcode):
    refDict = {
        "Free_Lease_Hold_Ind":owner,
        "Property_State":postcode,
        "Property_Type":prop,
        "Residential_Type":res,
        "Landed_Type":land
    }
    
    if postcode==None:
        refDict["Property_State"] = 47301

    #convert to state
    tempDF = masterDF.copy()
    tempDF = tempDF.loc[tempDF["Postcode"]==refDict["Property_State"]].head()
    print(tempDF.head())
    refDict["Property_State"] = tempDF.iloc[0]["Property_State"]

    
    #create predicted pie chart
    labels=["SnP < OMV", "SnP = OMV" ,"SnP > OMV"]
    values = getNewNN(refDict)
    colors = ['#636EFA','#EF553B','#00CC96']
    
    fig = go.Figure(data=[go.Bar(y=labels,x=values,marker_color=colors, orientation="h",width=[0.35, 0.35 ,0.35]
                            )])
    fig.update_layout(
        template="plotly_dark", 
        paper_bgcolor="#303030",
        plot_bgcolor = "#303030",
        margin={"r":0, "t":0, "l":0, "b":0},
        width = 400,
        height = 200,
        showlegend=False,
    )
    
    fig.update_traces(textposition="inside")
    fig.update_xaxes(title_text="Confidence")

    return refDict[selected], [html.H5("Predicted Confidence Level"),html.Br(),dcc.Graph(figure=fig), html.Small("Model Accuracy: 78.6%")]

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
    print(myDF["highest_MG_SnP_OMV"].max().item()- myDF["highest_MG_SnP_OMV"].min().item())
    
    if myDF["highest_MG_SnP_OMV"].max().item()- myDF["highest_MG_SnP_OMV"].min().item()> 10**6:
        flag=True
    
    
    """
    #set axes range 
    if myDF["highest_MG_SnP_OMV"].max() > 10**6:
        axesRange = [myDF["highest_MG_SnP_OMV"].min(), 10**6]
    else:
        axesRange = [myDF["highest_MG_SnP_OMV"].min(), myDF["highest_MG_SnP_OMV"].max()]
    """

    #beautify is_SnP_higher
    myDF.rename(columns={"is_SnP_higher":"Legend","highest_MG_SnP_OMV":"Highest value between SnP/OMV"},inplace=True)

    fig = px.histogram(myDF,x="Highest value between SnP/OMV",color="Legend",log_x=False)
    fig.update_layout(
        template="plotly_dark", 
        paper_bgcolor="#303030", 
        plot_bgcolor = "#303030",
        barmode="overlay",
        margin={"r":0, "t":0, "l":0, "b":0},
        width = 570,
        height = 220,
        legend={
            "yanchor":"bottom",
            "xanchor": "right"
        },
        xaxis = dict(range=[0,10**6])
    )

    if mode == "Free_Lease_Hold_Ind":
        mode = "Ownership"
    else:
        mode = mode.replace("_"," ")
    
    print(thisInput)
    

    return [html.H5("Distribution of {}".format(thisInput)),html.Br(),dcc.Graph(figure=fig), html.Small("  ")]

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

    #calling arima
    #arimaX, arimaY = thisARIMA(sumdf["difference"].tolist(), avedf["Submission_Mth"].tolist(), avedf["average"].tolist(), 100)

    fig = px.bar(avedf,x="Submission_Mth",y="difference")
    fig.add_scatter(x=avedf["Submission_Mth"],y=avedf["average"],name="3 month average")
    #fig.add_scatter(x=arimaX,y=arimaY,name="ARIMA")
    fig.update_layout(
        template="plotly_dark", 
        paper_bgcolor="#303030", 
        plot_bgcolor="#303030",
        margin={"r":0, "t":0, "l":0, "b":0},
        width = 500,
        height = 400,
        legend={
            "yanchor":"bottom",
            "xanchor": "right"
        }
    )

    fig.update_xaxes(title_text="Submission Month")
    fig.update_yaxes(title_text="Mean SnP-OMV")

    return [html.H5("Time Series of Mean SnP-OMV for {}".format(thisInput)),html.Br(),dcc.Graph(figure=fig)]
"""
@app.callback(
    Output("jumb-3","children"),
    [Input("hidden-div","children"),
    Input("mode-select","value")]
)
def createFig3(thisInput,mode):

    #bar
    myDF = masterDF.copy()
    myDF = myDF.loc[myDF[mode]==thisInput]

    labels=["SnP < OMV", "SnP = OMV", "SnP > OMV"]
    values = [myDF.loc[myDF["is_SnP_higher"]=="SnP < OMV"]["is_SnP_higher"].count(), 
    myDF.loc[myDF["is_SnP_higher"]=="SnP = OMV"]["is_SnP_higher"].count(),
    myDF.loc[myDF["is_SnP_higher"]=="SnP > OMV"]["is_SnP_higher"].count(),]

   
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                             insidetextorientation='radial', hole=.4, sort=False
                            )])
    
    
    fig.update_layout(
        template="plotly_dark", 
        paper_bgcolor="#303030",
        margin={"r":0, "t":0, "l":0, "b":0},
        width = 200,
        height = 200,
        showlegend=False
    )
    fig.update_traces(textposition="inside")

    return [html.H5("Historical Proportion"),html.Br(),dcc.Graph(figure=fig)]
"""
@app.callback(
    Output("jumb-2","children"),
    [Input("mode-select","value")]
)
def createFig2(mode):

    myDF = masterDF.copy()

    if mode == "Free_Lease_Hold_Ind":
        myDF.rename(columns={"Free_Lease_Hold_Ind":"Ownership"},inplace=True)
        mode = "Ownership"
    else:
        myDF.rename(columns={mode:mode.replace("_"," ")},inplace=True)
        mode = mode.replace("_"," ")

    #filter exteme outliers for display
    myDF = myDF.loc[(myDF["difference"] < 10**6) & (myDF["difference"] > -10**6)]

    boxFig = go.Figure()    
    for items in myDF[mode].unique():
        boxFig.add_trace(go.Box(x=myDF["difference"][myDF[mode] == items], name=items))
        boxFig.update_traces(quartilemethod="exclusive")
    
    boxFig.update_layout(
        template="plotly_dark", 
        paper_bgcolor="#303030", 
        plot_bgcolor ="#303030" ,
        showlegend=False,
        margin={"r":0, "t":0, "l":0, "b":0},
        width = 470,
        height = 400
    )
    boxFig.update_xaxes(title_text="SnP-OMV",range=[-1.25*10**6, 1.25*10**6])

    return [html.H5("Box Plot by {}".format(mode)),html.Br(),dcc.Graph(figure=boxFig)]



#run app on local host
if __name__ == '__main__':
    app.run_server(debug=False)
