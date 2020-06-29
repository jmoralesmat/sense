import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

app = dash.Dash()


def app_layout():
    return(
            html.Div([
                    dcc.Tabs(
                        id="tabs-with-classes",
                        value='tab-2',
                        parent_className='custom-tabs',
                        className='custom-tabs-container',
                        children=[
                            dcc.Tab(
                                label='Pie 1',
                                value='pie-1',
                                className='custom-tab',
                                selected_className='custom-tab--pie-1'
                            ),
                            dcc.Tab(
                                label='Pie 2',
                                value='pie-2',
                                className='custom-tab',
                                selected_className='custom-tab--pie-2'
                            )
                        ]
                    ),
                    html.Div(id='tabs-content-classes')
                    ]
            )
    )


app.layout=app_layout()


@app.callback(Output('tabs-content-classes', 'children'),
              [Input('tabs-with-classes', 'value')])
def render_content(tab):
    if tab == 'pie-1':
        return html.Div([
            html.H3('Tab content 1')
        ])
    elif tab == 'pie-2':
        return html.Div([
            html.H3('Tab content 2')
        ])


if __name__ == '__main__':
    app.server.run(debug=True)
