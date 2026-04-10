#!/usr/bin/env python3
"""
NPK Experimental Data Dashboard
Interactive visualization tool for plant treatment analysis

Run with: python scripts/dashboard_app.py
Then open: http://127.0.0.1:8050
"""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

# ============================================================
# LOAD DATA
# ============================================================

DATA_DIR = Path(__file__).parent.parent
CLEANED_DATA_PATH = DATA_DIR / 'cleaned_dataset.csv'

df = pd.read_csv(CLEANED_DATA_PATH)
# experimental_code format: {species}_{part}_{treatment}_{year}_{env}
# e.g. SH_L_NPK-N100-P100-K100_21_G  →  treatment = index 2
def extract_treatment(code):
    """Format: SP_PART_TREAT_YY_ENV — tratamento é tudo entre 2º e penúltimo underscore"""
    parts = str(code).split('_')
    if len(parts) >= 5:
        # Une os elementos do meio, excluindo SP, PART, YY, ENV
        return '_'.join(parts[2:-2])
    elif len(parts) == 5:
        return parts[2]
    return 'unknown'

df['treatment_code'] = df['experimental_code'].apply(extract_treatment)

# DIAGNÓSTICO - borra esto después
print("=== EJEMPLOS experimental_code ===")
print(df['experimental_code'].head(10).tolist())
print("\n=== treatment_code únicos ===")
print(df['treatment_code'].value_counts())

# ============================================================
# CONFIGURATION
# ============================================================

VARIABLE_GROUPS = {
    'MACRONUTRIENTS': [
        'total_fat_g_100_g_dw', 'crude_protein_g_100_g_dw', 'ash_g_100_g_dw',
        'fiber_g_100_g_dw', 'carbohydrates_g_100_g_dw', 'energy_kcal_100_g_dw'
    ],
    'ORGANIC_ACIDS': [
        'oxalic_acid_mg_100_g_dw', 'quinic_acid_mg_100_g_dw', 'malic_acid_mg_100_g_dw',
        'shikinic_acid_mg_100_g_dw', 'succinic_acid_mg_100_g_dw', 'citric_acid_mg_100_g_dw',
        'total_acids_mg_100_g_dw'
    ],
    'FATTY_ACIDS': [
        'c6:0', 'c8:0', 'c10:0', 'c11:0', 'c12:0', 'c13:0', 'c14:0', 'c14:1',
        'c15:0', 'c15:1', 'c16:0', 'c16:1', 'c17:0', 'c17:1', 'c18:0', 'c18:1n9c',
        'c18:2n6c', 'c18:3n6', 'c18:3n3', 'c20:0', 'c20:2', 'c21:0', 'c22:0',
        'c20:3n3', 'c20:5n3', 'c22:2', 'c23:0', 'c24:0', 'sfa', 'mufa', 'pufa'
    ],
    'TOCOPHEROLS': [
        'α-tocopherol_mg_100_g_dw', 'β-tocopherol_mg_100_g_dw',
        'γ-tocopherol_mg_100_g_dw', 'total_tocopherols_mg_100_g_dw'
    ],
    'SUGARS': [
        'fructose_g_100_g_dw', 'glucose_g_100_g_dw', 'saccharose_g_100_g_dw',
        'trehalose_g_100_g_dw', 'raffinose_g_100_g_dw', 'total_free_sugars_g_100_g_dw'
    ],
    'PHENOLIC_COMPOUNDS': [
        'total_phenolic_acids_mg_g_of_extract', 'total_flavonoids_mg_g_of_extract',
        'total_phenolic_compounds_mg_g_of_extract'
    ],
    'ANTIOXIDANT_ACTIVITY': ['tbars_ug_ml', 'oxhlia_ug_ml']
}

# NPK fertilisation treatments — keys match treatment_code values derived from experimental_code
NPK_TREATMENTS = {
    'NPK-Ctrl':             'Control (0-0-0)',
    'NPK-N100-P100-K100':   'Low (100-100-100)',
    'NPK-N200-P100-K100':   'Medium N (200-100-100)',
    'NPK-N200-P200-K200':   'Medium N+P (200-200-200)',
    'NPK-N300-P100-K100':   'High N (300-100-100)',
    'NPK-N300-P200-K200':   'High N+P (300-200-200)',
    'NPK-N300-P300-K300':   'Maximum (300-300-300)'
}

AVAILABLE_TREATMENTS = sorted(df['treatment_code'].unique().tolist())
AVAILABLE_YEARS = sorted([str(y) for y in df['harvest_year_desc'].dropna().unique()])
AVAILABLE_SPECIES = sorted(df['species'].dropna().unique().tolist())
AVAILABLE_ENVIRONMENTS = sorted(df['environment'].dropna().unique().tolist())


print("=== PRIMERAS FILAS treatment_code ===")
print(df['treatment_code'].head(10).tolist())
print("\n=== AVAILABLE_TREATMENTS ===")
print(AVAILABLE_TREATMENTS)

# ============================================================
# DASH APP
# ============================================================

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Styles
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "300px",
    "padding": "20px",
    "background-color": "#f8f9fa",
    "overflow-y": "auto"
}

CONTENT_STYLE = {
    "margin-left": "320px",
    "padding": "20px"
}

# Sidebar
sidebar = html.Div([
    html.H2("🌱 NPK Dashboard", className="display-6", style={"color": "#2ca02c"}),
    html.Hr(),

    html.H4("📊 Treatment Selection"),
    html.Label("Select treatments:"),
    dcc.Dropdown(
        id='treatment-selector',
        options=[{'label': NPK_TREATMENTS.get(t, t), 'value': t} for t in AVAILABLE_TREATMENTS],
        value=AVAILABLE_TREATMENTS[:20],  # simplemente los primeros 4 disponibles
        multi=True,
        placeholder="Select treatments..."
    ),
    html.Br(),

    html.H4("🔀 Variable Selection"),
    html.Label("Variable group:"),
    dcc.Dropdown(
        id='group-selector',
        options=[{'label': g, 'value': g} for g in VARIABLE_GROUPS.keys()],
        value='MACRONUTRIENTS'
    ),
    html.Br(),
    html.Label("Variable:"),
    dcc.Dropdown(
        id='variable-selector',
        value='crude_protein_g_100_g_dw'
    ),
    html.Br(),

    html.H4("🔍 Filters"),
    html.Label("Species:"),
    dcc.Dropdown(
        id='species-filter',
        options=[{'label': 'All', 'value': 'ALL'}] + [{'label': s, 'value': s} for s in AVAILABLE_SPECIES],
        value='ALL',
        multi=True
    ),
    html.Br(),
    html.Label("Environment:"),
    dcc.Dropdown(
        id='env-filter',
        options=[{'label': 'All', 'value': 'ALL'}] + [{'label': e, 'value': e} for e in AVAILABLE_ENVIRONMENTS],
        value='ALL'
    ),
    html.Br(),
    html.Label("Year:"),
    dcc.Dropdown(
        id='year-filter',
        options=[{'label': 'All', 'value': 'ALL'}] + [{'label': str(y), 'value': str(y)} for y in AVAILABLE_YEARS],
        value='ALL',
        multi=True
    ),
    html.Br(),

    html.H4("⚙️ Display Options"),
    html.Label("Plot type:"),
    dcc.RadioItems(
        id='plot-type',
        options=[
            {'label': ' Box Plot', 'value': 'box'},
            {'label': ' Violin Plot', 'value': 'violin'},
            {'label': ' Bar Chart', 'value': 'bar'},
            {'label': ' Trend Line', 'value': 'line'}
        ],
        value='box',
        labelStyle={'display': 'block'}
    ),
    html.Br(),
    html.Label("Aggregation:"),
    dcc.RadioItems(
        id='aggregation',
        options=[
            {'label': ' Raw Data', 'value': 'raw'},
            {'label': ' Sample Average', 'value': 'sample_mean'},
            {'label': ' Yearly Average', 'value': 'yearly'}
        ],
        value='raw',
        labelStyle={'display': 'block'}
    ),
    html.Br(),
    html.Label("Color by:"),
    dcc.RadioItems(
        id='color-by',
        options=[
            {'label': ' Treatment', 'value': 'treatment_code'},
            {'label': ' Species', 'value': 'species'},
            {'label': ' Environment', 'value': 'environment'}
        ],
        value='treatment_code',
        labelStyle={'display': 'block'}
    ),

], style=SIDEBAR_STYLE)

# Main content
content = html.Div([
    dbc.Row([
        dbc.Col([
            html.Div(id='info-panel', className="alert alert-info")
        ])
    ]),

    html.Hr(),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='main-plot')
        ], width=8),
        dbc.Col([
            html.H4("📊 Statistics"),
            html.Div(id='stats-panel', style={"max-height": "500px", "overflow-y": "auto"})
        ], width=4)
    ]),

    html.Hr(),

    dbc.Tabs([
        dbc.Tab(label="📋 Group Comparison", children=[
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Label("Select group:"),
                    dcc.Dropdown(
                        id='group-comp-selector',
                        options=[{'label': g, 'value': g} for g in VARIABLE_GROUPS.keys()],
                        value='MACRONUTRIENTS'
                    )
                ], width=4),
                dbc.Col([
                    html.Button("Update Comparison", id='update-group-btn', className="btn btn-primary")
                ], width=4)
            ]),
            html.Br(),
            dcc.Graph(id='group-heatmap')
        ]),

        dbc.Tab(label="📈 NPK Gradient", children=[
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Label("Variable:"),
                    dcc.Dropdown(
                        id='gradient-var',
                        options=[{'label': v.replace('_', ' '), 'value': v}
                                 for v in VARIABLE_GROUPS['MACRONUTRIENTS'] + VARIABLE_GROUPS['ANTIOXIDANT_ACTIVITY']],
                        value='crude_protein_g_100_g_dw'
                    )
                ], width=4),
                dbc.Col([
                    html.Button("Analyze Gradient", id='update-gradient-btn', className="btn btn-warning")
                ], width=4)
            ]),
            html.Br(),
            dcc.Graph(id='gradient-plot')
        ]),

        dbc.Tab(label="🔬 Species Comparison", children=[
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Label("Variable:"),
                    dcc.Dropdown(
                        id='species-var',
                        options=[{'label': v.replace('_', ' '), 'value': v}
                                 for v in VARIABLE_GROUPS['MACRONUTRIENTS'] + VARIABLE_GROUPS['ANTIOXIDANT_ACTIVITY']],
                        value='crude_protein_g_100_g_dw'
                    )
                ], width=4),
                dbc.Col([
                    html.Button("Compare Species", id='update-species-btn', className="btn btn-success")
                ], width=4)
            ]),
            html.Br(),
            dcc.Graph(id='species-plot')
        ]),

        dbc.Tab(label="📉 Correlation", children=[
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Label("Group:"),
                    dcc.Dropdown(
                        id='corr-group',
                        options=[{'label': g, 'value': g} for g in VARIABLE_GROUPS.keys()],
                        value='MACRONUTRIENTS'
                    )
                ], width=4),
                dbc.Col([
                    html.Button("Show Correlation", id='update-corr-btn', className="btn btn-info")
                ], width=4)
            ]),
            html.Br(),
            dcc.Graph(id='corr-plot')
        ]),

        dbc.Tab(label="🎯 PCA", children=[
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Label("Group:"),
                    dcc.Dropdown(
                        id='pca-group',
                        options=[{'label': g, 'value': g} for g in VARIABLE_GROUPS.keys()],
                        value='MACRONUTRIENTS'
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Color by:"),
                    dcc.Dropdown(
                        id='pca-color',
                        options=[
                            {'label': 'Treatment', 'value': 'treatment_code'},
                            {'label': 'Species', 'value': 'species'},
                            {'label': 'Environment', 'value': 'environment'}
                        ],
                        value='treatment_code'
                    )
                ], width=3),
                dbc.Col([
                    html.Button("Run PCA", id='update-pca-btn', className="btn btn-danger")
                ], width=3)
            ]),
            html.Br(),
            dcc.Graph(id='pca-plot'),
            html.Div(id='pca-loadings')
        ])
    ])
], style=CONTENT_STYLE)

app.layout = html.Div([sidebar, content])

# ============================================================
# CALLBACKS
# ============================================================

@callback(
    Output('variable-selector', 'options'),
    Output('variable-selector', 'value'),
    Input('group-selector', 'value')
)
def update_variable_selector(group):
    """Update variable options when group changes."""
    options = [{'label': v.replace('_', ' ')[:30], 'value': v} for v in VARIABLE_GROUPS.get(group, [])]
    value = VARIABLE_GROUPS.get(group, [''])[0]
    return options, value


@callback(
    Output('info-panel', 'children'),
    Output('main-plot', 'figure'),
    Output('stats-panel', 'children'),
    Input('treatment-selector', 'value'),
    Input('variable-selector', 'value'),
    Input('species-filter', 'value'),
    Input('env-filter', 'value'),
    Input('year-filter', 'value'),
    Input('plot-type', 'value'),
    Input('aggregation', 'value'),
    Input('color-by', 'value')
)
def update_dashboard(treatments, variable, species_filter, env_filter, year_filter,
                     plot_type, aggregation, color_by):
    """Update main dashboard."""
    if not treatments or not variable:
        raise PreventUpdate

    # Filter data
    filtered = df.copy()
    filtered = filtered[filtered['treatment_code'].isin(treatments)]

    if species_filter and 'ALL' not in species_filter:
        filtered = filtered[filtered['species'].isin(species_filter)]

    if env_filter and env_filter != 'ALL':
        filtered = filtered[filtered['environment'] == env_filter]

    if year_filter and 'ALL' not in year_filter:
        filtered = filtered[filtered['harvest_year_desc'].astype(str).isin(year_filter)]

    # Info panel
    info = html.Span([
        f"📊 Variable: {variable.replace('_', ' ')} | ",
        f"🔬 Treatments: {len(treatments)} | ",
        f"📈 Observations: {len(filtered)}"
    ])

    if variable not in filtered.columns:
        fig = go.Figure()
        fig.add_annotation(text="Variable not found", xref="paper", yref="paper", x=0.5, y=0.5)
        return info, fig, html.Div("No data")

    plot_df = filtered.dropna(subset=[variable, 'treatment_code'])
    var_display = variable.replace('_', ' ')

    # Create plot
    if plot_type == 'box':
        fig = px.box(plot_df, x='treatment_code', y=variable, color=color_by,
                     points='all', hover_data=['experimental_code'] if 'experimental_code' in plot_df.columns else None)
    elif plot_type == 'violin':
        fig = px.violin(plot_df, x='treatment_code', y=variable, color=color_by, box=True)
    elif plot_type == 'bar':
        agg_df = plot_df.groupby('treatment_code')[variable].agg(['mean', 'std']).reset_index()
        fig = px.bar(agg_df, x='treatment_code', y='mean', error_y='std')
    elif plot_type == 'line':
        if 'harvest_year_desc' in plot_df.columns:
            trend_df = plot_df.groupby(['harvest_year_desc', 'treatment_code'])[variable].mean().reset_index()
            fig = px.line(trend_df, x='harvest_year_desc', y=variable, color='treatment_code', markers=True)
        else:
            fig = go.Figure()
            fig.add_annotation(text="Year data not available", xref="paper", yref="paper", x=0.5, y=0.5)

    fig.update_layout(
        title=f'<b>{var_display}</b> by Treatment',
        xaxis_title='Treatment',
        yaxis_title=var_display,
        height=500,
        xaxis_tickangle=-45,
        template='plotly_white'
    )

    # Statistics
    stats_list = []
    for t in treatments:
        t_data = plot_df[plot_df['treatment_code'] == t][variable]
        if len(t_data) > 0:
            stats_list.append({
                'Treatment': t,
                'N': len(t_data),
                'Mean': round(t_data.mean(), 3),
                'Std': round(t_data.std(), 3),
                'Min': round(t_data.min(), 3),
                'Max': round(t_data.max(), 3)
            })

    stats_df = pd.DataFrame(stats_list)
    stats_table = dbc.Table.from_dataframe(stats_df, striped=True, bordered=True, size='sm')

    return info, fig, stats_table


@callback(
    Output('group-heatmap', 'figure'),
    Input('update-group-btn', 'n_clicks'),
    State('group-comp-selector', 'value'),
    State('treatment-selector', 'value')
)
def update_group_heatmap(n, group, treatments):
    """Update group comparison heatmap."""
    if not n or not group or not treatments:
        raise PreventUpdate

    variables = VARIABLE_GROUPS[group]
    valid_vars = [v for v in variables if v in df.columns]

    filtered = df[df['treatment_code'].isin(treatments)]
    heatmap_data = filtered.groupby('treatment_code')[valid_vars].mean()
    heatmap_norm = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_norm.values,
        x=[v.replace('_', ' ')[:15] for v in valid_vars],
        y=heatmap_norm.index,
        colorscale='RdBu_r',
        zmid=0
    ))

    fig.update_layout(
        title=f'{group} - Treatment Comparison (Z-score)',
        height=max(400, len(treatments) * 50),
        xaxis_tickangle=-45
    )

    return fig


@callback(
    Output('gradient-plot', 'figure'),
    Input('update-gradient-btn', 'n_clicks'),
    State('gradient-var', 'value')
)
def update_gradient(n, variable):
    """Update NPK gradient plot."""
    if not n or not variable:
        raise PreventUpdate

    npk_list = list(NPK_TREATMENTS.keys())
    npk_df = df[df['treatment_code'].isin(npk_list)].copy()

    if variable not in npk_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Variable not found", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    agg_df = npk_df.groupby('treatment_code')[variable].agg(['mean', 'std']).reset_index()
    # Extract N dose directly from treatment_code: NPK-N300-P200-K200 → 300; NPK-Ctrl → 0
    def extract_n(code):
        parts = code.split('-')
        for p in parts:
            if p.startswith('N') and p[1:].isdigit():
                return int(p[1:])
        return 0
    agg_df['N'] = agg_df['treatment_code'].map(extract_n)
    agg_df = agg_df.sort_values('N')

    fig = px.bar(agg_df, x='treatment_code', y='mean', error_y='std',
                 color='N', color_continuous_scale='Viridis')

    fig.update_layout(
        title=f'{variable.replace("_", " ")} - NPK Gradient',
        xaxis_title='Treatment',
        yaxis_title=variable.replace('_', ' '),
        height=500,
        xaxis_tickangle=-45
    )

    return fig


@callback(
    Output('species-plot', 'figure'),
    Input('update-species-btn', 'n_clicks'),
    State('species-var', 'value')
)
def update_species(n, variable):
    """Update species comparison plot."""
    if not n or not variable:
        raise PreventUpdate

    plot_df = df.dropna(subset=[variable, 'species', 'treatment_code'])

    fig = px.box(plot_df, x='species', y=variable, color='treatment_code')

    fig.update_layout(
        title=f'{variable.replace("_", " ")} - Species Comparison',
        height=550,
        xaxis_tickangle=-45
    )

    return fig


@callback(
    Output('corr-plot', 'figure'),
    Input('update-corr-btn', 'n_clicks'),
    State('corr-group', 'value')
)
def update_correlation(n, group):
    """Update correlation plot."""
    if not n or not group:
        raise PreventUpdate

    variables = VARIABLE_GROUPS[group]
    valid_vars = [v for v in variables if v in df.columns]

    if len(valid_vars) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Need at least 2 variables", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    corr = df[valid_vars].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=[v.replace('_', ' ')[:15] for v in valid_vars],
        y=[v.replace('_', ' ')[:15] for v in valid_vars],
        colorscale='RdBu_r',
        zmid=0
    ))

    fig.update_layout(
        title=f'{group} - Correlation Matrix',
        height=500
    )

    return fig


@callback(
    Output('pca-plot', 'figure'),
    Output('pca-loadings', 'children'),
    Input('update-pca-btn', 'n_clicks'),
    State('pca-group', 'value'),
    State('pca-color', 'value')
)
def update_pca(n, group, color_by):
    """Update PCA plot."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    if not n or not group:
        raise PreventUpdate

    variables = VARIABLE_GROUPS[group]
    valid_vars = [v for v in variables if v in df.columns]

    pca_df = df.dropna(subset=valid_vars + [color_by]).copy()

    if len(pca_df) < 10:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig, html.Div()

    X = pca_df[valid_vars].values
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=min(len(valid_vars), 5))
    X_pca = pca.fit_transform(X_scaled)

    results = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
    results[color_by] = pca_df[color_by].values

    fig = px.scatter(results, x='PC1', y='PC2', color=color_by)

    explained_var = pca.explained_variance_ratio_[:2] * 100
    fig.update_layout(
        title=f'PCA - {group}',
        xaxis_title=f'PC1 ({explained_var[0]:.1f}%)',
        yaxis_title=f'PC2 ({explained_var[1]:.1f}%)',
        height=500
    )

    loadings = pd.DataFrame(pca.components_[:2].T, columns=['PC1', 'PC2'], index=valid_vars).round(3)
    loadings_table = dbc.Table.from_dataframe(loadings, striped=True, bordered=True, size='sm')

    return fig, html.Div([html.H5("PCA Loadings"), loadings_table])


# ============================================================
# RUN APP
# ============================================================

import os

if __name__ == "__main__":
    app.run_server(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8050)),
        debug=False
    )
