import dash
from dash import Dash, dcc, html, Input, Output, dash_table, callback
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.express as px
import pandas as pd
import os
import json

with open("./project_variables.json","r") as f:
    project_variables = json.load(f)

dash.register_page(__name__,path="/")

available_datasets = [str(file) for file in os.listdir("./app_datasets/") if file.split(".")[-1] == "csv"]

variables = {
    "SIR": "SIR",
    "SAR": "SAR",
    "SDR": "SDR",
    "Time": "separation_time",
    "Memory": "occupied_memory",
    "Main speaker": "main_source",
    "Number of speakers": "num_speakers_in_mix",
    "Mix SNR": "mix_snr_high",
    "White Noise SNR":"white_noise_snr_high"
}

models = project_variables["models"]
model_types = list(models.keys())
# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame()

model_dropdown_list = []
for model_type in models:
    for model_name in models[model_type]:
        model_dropdown_list.append(f"{model_type}: {model_name}")
        


model_comparison_layout = html.Div(children = [
    html.H2("Model comparison"),
    html.Div(id = "plot_params", children = [
        html.Div(children=[
            html.Label("Comparison type"),
            dcc.RadioItems(
                ["Multiple models","Single model"],
                "Multiple models",
                inline=False,
                id="evaluation_mode",
                labelStyle={"display":"block"}),
        ]),
        html.Br(),
        html.Div(children=[
            html.Label("Dataset"),
            dcc.Dropdown(
                available_datasets,
                id="cur_dataset",
                value = available_datasets[0],
                style = {"color":"black"}
            ),
        ]),
        html.Br(),
        html.Div(children=[
            html.Label("Number of speakers"),
            dcc.RangeSlider(
                min = 1, 
                max = 3,
                step = 1,
                value=[1,3],
                id="cur_num_speakers"
            )
        ]),
        html.Div(children=[
            html.Label("Additional Speakers SNR"),
            dcc.RangeSlider(
                min = -5, 
                max = 15,
                step = 5,
                value=[-5,15],
                id="cur_mix_snr"
            ),
            html.Label("Mix SNR can be empty (1 speaker)"),
            daq.BooleanSwitch(id="snr_can_be_empty", on=True),
        ]),
        html.Div(children=[
            html.Label("White Noise SNR"),
            dcc.RangeSlider(
                min = -5, 
                max = 15,
                step = 5,
                value=[-5,15],
                id="cur_wn_snr"
            ),
            html.Label("White noise"),
            daq.BooleanSwitch(id="white-noise",on=False),
            html.Label("White noise can be empty"),
            daq.BooleanSwitch(id="wn_can_be_empty", on=True),
        ]),
        html.Br(),
        html.Label("Models to compare:"),
        html.Div(
            dcc.Dropdown(
                model_dropdown_list, 
                multi=True,
                id="cur_model",
                style={"color":"black"}
            )
        ),
        html.Br(),
        html.Label("Variable to compare:"),
        html.Div(
            dcc.Dropdown(
                list(variables.keys()),
                id="cur_variable",
                value="SDR",
                clearable=False,
                style={"color":"black"}
            )
        ),
        html.Br(),
        html.Label("X-axis variable"),
        html.Div(
            dcc.Dropdown(
                ["Number of speakers", "Mix SNR", "White Noise SNR"],
                disabled = True,
                id="cur_xaxis",
                style={"color":"black"}
            )
        )
    ], 
    style={
        "width":"30%",
        "display":"inline-block",
        "padding":"2%",
        "background-color":"#444"
    }),

    html.Div(children = [
            dcc.Graph(
                id="main-graph"
            ),
        ],
        style={
            "width":"70%",
            "display":"inline-block",
            "padding":"5px",
            "background-color":"#303030"
        }),
    dash_table.DataTable(
        id="summary-table",
        style_header={
            'backgroundColor': 'rgb(30, 30, 30)',
            'color': 'white'
        },
        style_data={
            'backgroundColor': 'rgb(50, 50, 50)',
            'color': 'white'
        })
],className="row")


layout = html.Div(children=[
    html.H1("Evaluation of Speaker Separation Models",style={"text-align":"center"}),
    # For plotting comparisons
    html.Div(id="body", className="container scalable", children=[model_comparison_layout])
])

empty_plot = px.box(
    pd.DataFrame({"x":[],"y":[]}),
    x="x",
    y="y"
)
empty_plot.update_layout({
        "plot_bgcolor":"rgba(0,0,0,0)",
        "paper_bgcolor":"rgba(0,0,0,0)",
        "font_color":"white"}
    )
empty_plot.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor= "gray",zeroline=False)
empty_plot.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor= "gray",zeroline=False)

def update_graph_multi_eval(df,model_type_name_list,variable):
    
    if variable == "Main speaker":
        fig = px.histogram(
            df,
            x="model_name_type",
            color=variables[variable],
            barmode="group",
            labels={
                "model_name_type": "Models",
                variables[variable]: "Is estimation the main source?",
            },
            category_orders={"model_name_type":sorted(df["model_name_type"].unique())})
    else:
        fig = px.box(
            df,
            x="model_name_type",
            y=variables[variable],
            labels={
                variables[variable]: variable,
                "model_name_type": "Models"
            },
            category_orders={"model_name_type":sorted(df["model_name_type"].unique())}
            #template = "plotly_dark"
        )

    fig.update_layout({
        "plot_bgcolor":"rgba(0,0,0,0)",
        "paper_bgcolor":"rgba(0,0,0,0)",
        "font_color":"black"}
    )

    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor= "gray",zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor= "gray",zeroline=False)
    if not model_type_name_list: return fig, pd.DataFrame().to_dict("records")
    df_table = df[["model_name_type",variables[variable]]].groupby("model_name_type").describe().T
    df_table = df_table.reset_index()
    df_table = df_table.rename(columns={"level_0":"variable","level_1":"property"})
    df_table = df_table[list(df_table.columns)]
    return fig, df_table.to_dict("records")

def update_graph_single_eval(df,model_type_name,variable,xaxis):
    fig = px.box(
            df,
            x=variables[xaxis],
            y=variables[variable],
            color="model_name_type",
            labels={
                variables[variable]: variable,
                variables[xaxis]: xaxis,
                "model_name_type": "Models"
            },
            category_orders={"model_name_type":sorted(df["model_name_type"].unique())}
            #template = "plotly_dark"
        )
    fig.update_layout({
        "plot_bgcolor":"rgba(0,0,0,0)",
        "paper_bgcolor":"rgba(0,0,0,0)",
        "font_color":"black",
        "xaxis_tickformat":",d"}
    )

    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor= "gray",zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor= "gray",zeroline=False)
    return fig, pd.DataFrame().to_dict("records")

@callback(
    [Output("cur_model","multi"),Output("cur_xaxis","disabled"),Output("cur_xaxis","value")],
    [Input("evaluation_mode","value")]
)
def update_comparison_dropdowns(eval_mode):
    if eval_mode == "Multiple models":
        return True, True, None
    return True, False, None

@callback(
    [Output("main-graph","figure"), Output("summary-table","data")],
    [
    Input("evaluation_mode","value"),
    Input("cur_dataset","value"),
    Input("cur_num_speakers","value"),
    Input("cur_mix_snr","value"),
    Input("snr_can_be_empty","on"),
    Input("cur_wn_snr","value"),
    Input("white-noise","on"),
    Input("wn_can_be_empty","on"),
    Input("cur_model","value"),
    Input("cur_variable","value"),
    Input("cur_xaxis","value")]
)
def update_graph(
    eval_mode,
    data_file,
    num_speakers,
    mix_snr,
    snr_can_be_empty,
    wn_snr,
    white_noise,
    wn_can_be_empty,
    model_type_name_list,
    variable,
    xaxis):
    if not model_type_name_list: model_type_name_list = []
    if not data_file: return empty_plot, pd.DataFrame().to_dict("records")
    df = pd.read_csv(str(os.path.join("app_datasets",data_file)))
    df = df[(df["num_speakers_in_mix"] >= num_speakers[0]) & (df["num_speakers_in_mix"] <= num_speakers[1])]
    
    if snr_can_be_empty: # Consider datasets of 1 or more speakers
        df = df[(
            (df["mix_snr_low"].isnull()) | (df["mix_snr_high"].isnull())) | 
            ((df["mix_snr_low"] >= mix_snr[0]) & (df["mix_snr_high"] <= mix_snr[1]))]
        print(df)
    else: # Only consider datasets of 2 or more speakers
        df = df[(df["mix_snr_low"] >= mix_snr[0]) & (df["mix_snr_high"] <= mix_snr[1])]
    
    print(len(df))
    if not white_noise: # Consider datasets that DO NOT have white noise
        df = df[(df["white_noise_snr_low"].isnull()) & (df["white_noise_snr_low"].isnull())]
        print(df["white_noise_snr_low"])
    else:
        if wn_can_be_empty: # Consider datasets that can have or not white noise
            df = df[((df["white_noise_snr_low"].isnull()) | (df["white_noise_snr_high"].isnull())) | ((df["white_noise_snr_low"] >= wn_snr[0]) & (df["white_noise_snr_high"] <= wn_snr[1]))]
        else: # Consider only dayasets that white noise
            df = df[(df["white_noise_snr_low"] >= wn_snr[0]) & (df["white_noise_snr_high"] <= wn_snr[1])]
    df["model_name_type"] = df["model_type"] + ": " + df["model_name"]

    if eval_mode == "Multiple models":
        df = df[df["model_name_type"].isin(model_type_name_list)]
        return update_graph_multi_eval(df,model_type_name_list,variable)
    if not xaxis: return empty_plot,pd.DataFrame().to_dict("records")
    #df = df[df["model_name_type"] == model_type_name_list]
    df = df[df["model_name_type"].isin(model_type_name_list)]
    return update_graph_single_eval(df,model_type_name_list,variable,xaxis)