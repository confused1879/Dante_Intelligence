import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="UTR Rating History", layout="wide")

with st.sidebar:
    show_all = st.checkbox('Show All Historical Data', value=False)

@st.cache_data
def load_data():
    df = pd.read_csv('utr_history.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()
if show_all:
    df_filtered = df
    dtick = "M6"  # Show tick every 6 months for all data
else:
    latest_date = df['Date'].max()
    twelve_months_ago = latest_date - timedelta(days=365)
    df_filtered = df[df['Date'] >= twelve_months_ago]
    dtick = "M1"  # Show tick every month for 12-month view

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_filtered['Date'],
    y=df_filtered['UTR_Rating'],
    mode='lines+markers',
    name='Jannik Sinner',
    line=dict(color='#1E90FF', width=2),
    hovertemplate='Date: %{x|%Y-%m-%d}<br>UTR: %{y:.2f}<extra></extra>'
))

fig.update_layout(
    title=dict(
        text="Historical UTR Rating",
        x=0,
        y=0.95,
        font=dict(size=24)
    ),
    plot_bgcolor='white',
    height=600,
    showlegend=True,
    xaxis=dict(
        title="",
        showgrid=True,
        gridcolor='#E5E5E5',
        tickformat="%b '%y",  # Format as "MMM 'YY"
        dtick=dtick,
        tickfont=dict(size=12)
    ),
    yaxis=dict(
        title="UTR",
        showgrid=True,
        gridcolor='#E5E5E5',
        tickformat='.2f',
        tickfont=dict(size=12),
        range=[
            df_filtered['UTR_Rating'].min() - 3.25,
            df_filtered['UTR_Rating'].max() + 0.25
        ]
    )
)

fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E5E5E5')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E5E5E5')

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
### How the charts work
The blue line represents the UTR rating history. Toggle the sidebar checkbox to view all historical data or just the last 12 months.
""")