import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
import os
from utr_auth import UTRAuthManager

# Load environment variables if needed
load_dotenv()

def init_auth():
    if 'auth_manager' not in st.session_state:
        st.session_state.auth_manager = UTRAuthManager()

def save_utr_data(data, filename="utr_results.csv"):
    """
    Save UTR data to CSV.
    """
    df = pd.DataFrame([{
        'timestamp': datetime.now(),
        'id': data.get('id'),
        'displayName': f"{data.get('firstName')} {data.get('lastName')}",
        'singlesUtr': data.get('singlesUtr'),
        'doublesUtr': data.get('doublesUtr'),
        'wins': data.get('wins', 0),          # Ensure default value
        'losses': data.get('losses', 0),      # Ensure default value
        'win_rate': data.get('win_rate', "0%") # Ensure default value
    }])
    
    try:
        existing_df = pd.read_csv(filename)
        df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        pass
    
    df.to_csv(filename, index=False)

def save_matches(data, filename="matches.csv"):
    """
    Save matches data to CSV.
    """
    matches = data.get('matches', [])
    if not matches:
        return  # No matches to save
    
    # Normalize match data into a flat table
    matches_df = pd.json_normalize(matches)
    
    # Convert resultDate to datetime with error handling
    matches_df['resultDate'] = pd.to_datetime(matches_df['resultDate'], errors='coerce')
    
    # Optional: Extract additional fields if necessary
    # For example, extracting scores, opponents, etc.
    
    try:
        existing_df = pd.read_csv(filename)
        matches_df = pd.concat([existing_df, matches_df], ignore_index=True)
    except FileNotFoundError:
        pass
    
    matches_df.to_csv(filename, index=False)

def save_ratings_history(data, filename="ratings_history.csv"):
    """
    Save ratings history data to CSV.
    """
    ratings_history = data.get('extendedRatingProfile', {}).get('history', [])
    if not ratings_history:
        return  # No rating history to save
    
    ratings_df = pd.json_normalize(ratings_history)
    
    # Clean date strings by stripping whitespace and converting to datetime
    ratings_df['date'] = pd.to_datetime(ratings_df['date'].astype(str).str.strip(), errors='coerce')
    
    try:
        existing_df = pd.read_csv(filename)
        ratings_df = pd.concat([existing_df, ratings_df], ignore_index=True)
    except FileNotFoundError:
        pass
    
    ratings_df.to_csv(filename, index=False)

def create_utr_charts(df):
    """
    Create UTR Ratings Over Time chart.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['singlesUtr'],
        name='Singles UTR',
        mode='lines+markers'
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['doublesUtr'],
        name='Doubles UTR',
        mode='lines+markers',
        line=dict(dash='dash')
    ))
    fig.update_layout(
        title="UTR Ratings Over Time",
        xaxis_title="Date",
        yaxis_title="UTR Rating",
        hovermode='x unified'
    )
    return fig

def create_rating_history_chart(ratings_df):
    """
    Create Rating Over Time chart from rating history.
    """
    fig = px.line(
        ratings_df,
        x='date',
        y='rating',
        title="J. Sinner's Rating Over Time",
        labels={'date': 'Date', 'rating': 'Rating'},
        markers=True
    )
    fig.update_layout(hovermode='x unified')
    return fig

def create_match_outcomes_chart(matches_df):
    """
    Create Match Outcomes Over Time chart.
    """
    # Determine win/loss
    matches_df['outcome'] = matches_df['isWinner'].apply(lambda x: 'Win' if x else 'Loss')
    
    # Sort by date
    matches_df = matches_df.sort_values('resultDate')
    
    # Create a cumulative wins and losses
    matches_df['cumulative_wins'] = matches_df['isWinner'].cumsum()
    matches_df['cumulative_losses'] = (~matches_df['isWinner']).cumsum()
    
    # Create scatter plot for cumulative wins
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=matches_df['resultDate'],
        y=matches_df['cumulative_wins'],
        mode='lines+markers',
        name='Cumulative Wins',
        marker=dict(color='green')
    ))
    
    # Add cumulative losses on a secondary y-axis
    fig.add_trace(go.Scatter(
        x=matches_df['resultDate'],
        y=matches_df['cumulative_losses'],
        mode='lines+markers',
        name='Cumulative Losses',
        marker=dict(color='red'),
        yaxis='y2'
    ))
    
    # Update layout for dual y-axis
    fig.update_layout(
        title="Match Outcomes Over Time",
        xaxis_title="Date",
        yaxis=dict(
            title="Cumulative Wins",
            side='left'
        ),
        yaxis2=dict(
            title="Cumulative Losses",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        legend=dict(x=0.01, y=0.99),
        hovermode='x unified'
    )
    
    return fig

def create_average_rating_event_chart(matches_df):
    """
    Create Average Rating by Event bar chart.
    """
    # Check if 'eventName' and 'rating' columns exist
    if 'eventName' not in matches_df.columns or 'rating' not in matches_df.columns:
        return None
    
    # Calculate average singles UTR per event
    average_rating_event = matches_df.groupby('eventName')['rating'].mean().reset_index()
    
    fig = px.bar(
        average_rating_event,
        x='eventName',
        y='rating',
        title="Average Rating by Event",
        labels={'eventName': 'Event', 'rating': 'Average Rating'},
        hover_data={'rating': ':.2f'}
    )
    
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    return fig

def create_rating_distribution_chart(ratings_df):
    """
    Create Rating Distribution histogram.
    """
    fig = px.histogram(
        ratings_df,
        x='rating',
        nbins=30,
        title="Distribution of Ratings",
        labels={'rating': 'Rating'},
        opacity=0.75
    )
    
    fig.update_layout(bargap=0.1)
    return fig

def main():
    st.title("UTR Tracking Dashboard")
    
    # Initialize authentication
    init_auth()
    
    # Sidebar for user inputs
    with st.sidebar:
        st.header("Player Selection")
        player_id = st.number_input("Player ID", min_value=1, value=247320, step=1)
        player_name = st.text_input("Player Name", value="J. Sinner")
        if st.button("Fetch Data"):
            try:
                # Fetch data from UTR API
                data = st.session_state.auth_manager.get_player_stats(player_id)
                
                # Save data to CSVs
                save_utr_data(data)
                save_matches(data)
                save_ratings_history(data)
                
                st.success("Data fetched and saved successfully!")
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
    
    # Load UTR data
    try:
        utr_df = pd.read_csv("utr_results.csv")
        utr_df['timestamp'] = pd.to_datetime(utr_df['timestamp'], errors='coerce')
        
        # Debug: Display first few rows
        st.sidebar.write("Loaded UTR Data:")
        st.sidebar.write(utr_df.head())
        
        # Check for any unparsed dates
        if utr_df['timestamp'].isnull().any():
            st.sidebar.warning("Some UTR timestamps could not be parsed and are set to NaT.")
        
    except FileNotFoundError:
        utr_df = pd.DataFrame()
        st.sidebar.info("No UTR data available. Please fetch data for a player.")
    
    # Load matches data
    try:
        matches_df = pd.read_csv("matches.csv")
        matches_df['resultDate'] = pd.to_datetime(matches_df['resultDate'], errors='coerce')
        
        # Debug: Display first few rows
        st.sidebar.write("Loaded Matches Data:")
        st.sidebar.write(matches_df.head())
        
        # Check for any unparsed dates
        if matches_df['resultDate'].isnull().any():
            st.sidebar.warning("Some match result dates could not be parsed and are set to NaT.")
        
    except FileNotFoundError:
        matches_df = pd.DataFrame()
        st.sidebar.info("No matches data available.")
    
    # Load ratings history data
    try:
        ratings_df = pd.read_csv("ratings_history.csv")
        # **Ensure no format parameter is passed and convert to Python datetime**
        ratings_df['date'] = pd.to_datetime(ratings_df['date'], errors='coerce')
        
        # Debug: Display first few rows
        st.sidebar.write("Loaded Ratings History:")
        st.sidebar.write(ratings_df.head())
        
        # Check for any unparsed dates
        if ratings_df['date'].isnull().any():
            st.sidebar.warning("Some rating dates could not be parsed and are set to NaT.")
            # Optionally display problematic rows
            problematic_ratings = ratings_df[ratings_df['date'].isnull()]
            st.sidebar.write("Problematic Ratings Entries:")
            st.sidebar.write(problematic_ratings)
        
    except FileNotFoundError:
        ratings_df = pd.DataFrame()
        st.sidebar.info("No ratings history data available.")
    
    # Display UTR Metrics and Charts
    if not utr_df.empty:
        st.header("Latest UTR Ratings")
        latest = utr_df.iloc[-1]
        
        # **Updated Column Names: Changed 'winsCount' and 'lossesCount' to 'wins' and 'losses'**
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Singles UTR", f"{latest['singlesUtr']:.2f}")
        with col2:
            st.metric("Doubles UTR", f"{latest['doublesUtr']:.2f}")
        with col3:
            st.metric("Win Rate", latest['win_rate'])
        
        # **Added Win/Loss Metric**
        st.header("Win/Loss Record")
        st.metric("Win/Loss", f"{latest['wins']}/{latest['losses']}")
        
        # UTR Ratings Over Time Chart
        st.header("UTR Ratings Over Time")
        fig_utr = create_utr_charts(utr_df)
        st.plotly_chart(fig_utr, use_container_width=True)
    
    # Display Rating History Chart
    if not ratings_df.empty:
        st.header("Rating History")
        
        # Add a date range slider for filtering
        min_date = ratings_df['date'].min()
        max_date = ratings_df['date'].max()
        if pd.notnull(min_date) and pd.notnull(max_date):
            # **Convert pandas.Timestamp to native Python datetime.datetime**
            min_date_dt = min_date.to_pydatetime()
            max_date_dt = max_date.to_pydatetime()
            
            # Debug: Show min and max dates
            st.sidebar.write(f"Min Date: {min_date_dt}")
            st.sidebar.write(f"Max Date: {max_date_dt}")
            
            date_range = st.slider(
                "Select Date Range",
                min_value=min_date_dt,
                max_value=max_date_dt,
                value=(min_date_dt, max_date_dt),
                format="YYYY-MM-DD"
            )
            filtered_ratings = ratings_df[
                (ratings_df['date'] >= date_range[0]) & 
                (ratings_df['date'] <= date_range[1])
            ]
        else:
            filtered_ratings = ratings_df
        
        # Check for any NaT values after parsing
        if filtered_ratings['date'].isnull().any():
            st.warning("Some dates could not be parsed and are excluded from the chart.")
            filtered_ratings = filtered_ratings.dropna(subset=['date'])
        
        if not filtered_ratings.empty:
            fig_rating = create_rating_history_chart(filtered_ratings)
            st.plotly_chart(fig_rating, use_container_width=True)
        else:
            st.info("No valid rating history data to display for the selected date range.")
    
    # Display Match Outcomes Chart
    if not matches_df.empty:
        st.header("Match Outcomes Over Time")
        fig_matches = create_match_outcomes_chart(matches_df)
        st.plotly_chart(fig_matches, use_container_width=True)
        
        # Display Average Rating by Event
        st.header("Average Rating by Event")
        fig_avg_event = create_average_rating_event_chart(matches_df)
        if fig_avg_event:
            st.plotly_chart(fig_avg_event, use_container_width=True)
        else:
            st.info("Insufficient data to display Average Rating by Event.")
        
        # Display Rating Distribution
        st.header("Rating Distribution")
        if not ratings_df.empty:
            fig_rating_dist = create_rating_distribution_chart(ratings_df)
            st.plotly_chart(fig_rating_dist, use_container_width=True)
        else:
            st.info("No ratings history data available to display Rating Distribution.")
    
    # Display Data Tables
    if not utr_df.empty:
        st.header("UTR Data History")
        st.dataframe(utr_df.sort_values("timestamp", ascending=False).reset_index(drop=True))
    
    if not matches_df.empty:
        st.header("Match History")
        st.dataframe(matches_df.sort_values("resultDate", ascending=False).reset_index(drop=True))
    
    if not ratings_df.empty:
        st.header("Ratings History")
        st.dataframe(ratings_df.sort_values("date", ascending=False).reset_index(drop=True))

if __name__ == "__main__":
    main()
