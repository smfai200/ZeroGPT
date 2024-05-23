from dash import dcc, html
import dash_bootstrap_components as dbc



layout=dbc.Container([
        html.Div([
            dbc.Row([
                dbc.Col(html.H2("AI CONTENT CHECKER"),
                className="header_title"
                ),
            ]),
            dbc.Row([
                dbc.Col(html.P(
                    children="""REAL MEANS HUMAN, FAKE MEANS AI GENERATED""",
                    ),
                className="header_description"),
            ]),
        ], className='header'),
        html.Div([
            dbc.Row([
                dbc.Col(dbc.Textarea(
                    id='text',
                    placeholder="Enter Text Here",
                    style={'width': '100%', 'height': 400},
                ), width={"size": 6, "offset": 3}),
            ]),
            # dbc.Row(
            # [
            #     dbc.Col([
            #         dcc.Upload(id='upload_file', children=[
            #                     'Upload File ',
            #                     html.A('(.txt, .docx, .pdf)'),
            #         ], multiple=False,
            #             className='upload-style')
            #     ], width={"size": 6, "offset": 3}),
            # ],
            # ),
            dbc.Row([
                dbc.Col(html.H5('Real', className="text-info"), width=4, style={'text-align' : 'right'}),
                dbc.Col(dcc.Loading(id='loading',children=html.Div(''),
                    type="default"),
                    width=4),
                dbc.Col(html.H5('Fake', className="text-danger"), width=4, style={'text-align' : 'left'}),
            ], style={'margin-top' : '20px'}),
            dbc.Row([
                dbc.Col(dbc.Progress(
                [
                    dbc.Progress(id="real_prog", value=50, color="info", bar=True),
                    dbc.Progress(id="fake_prog", value=50, color="danger", bar=True),
                ], style={'height' : '25px', 'font-size' : '16px'}
                ), width={"size": 6, "offset": 3}),
            ]),
        ], className='card'),
    ])