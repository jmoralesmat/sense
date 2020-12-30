import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

app = dash.Dash()


def app_layout():
    return(
        html.Div([
            dcc.Tabs(
                id='tabs',
                value=1,
                parent_className='custom-tabs',
                className='custom-tabs-container',
                children=[
                    dcc.Tab(label='Page 1', value=1, className='custom-tab'),
                    dcc.Tab(label='Page 2', value=2, className='custom-tab')
                ]
            ),
            html.Div(id='tabs-content-classes')
        ])
    )


app.layout = app_layout()


@app.callback(Output('tabs-content-classes', 'children'), [Input('tabs', 'value')])
def render_content(tab):
    data = [{
        'values': [[10, 90], [5, 95], [15, 85], [20, 80]][int(tab) - 1],
        'type': 'pie'
    }]

    return html.Div([
        dcc.Graph(
            id='graph',
            figure={
                'data': data,
                'layout': {
                    'margin': {'l': 30, 'r': 0, 'b': 30, 't': 0},
                    'legend': {'x': 0, 'y': 1}
                }
            }
        ),
        dcc.Graph(
            id='graph',
            figure={
                'data': data,
                'layout': {
                    'margin': {'l': 30, 'r': 0, 'b': 30, 't': 0},
                    'legend': {'x': 0, 'y': 1}
                }
            }
        ),
    ])


if __name__ == '__main__':
    app.server.run(debug=True)
