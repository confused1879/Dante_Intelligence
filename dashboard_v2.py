import streamlit as st
import pandas as pd
from datetime import datetime
import os
from utr_auth import UTRAuthManager
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go

# Load environment variables from .env if needed
load_dotenv()

# Initialize authentication
def init_auth():
    if 'auth_manager' not in st.session_state:
        st.session_state.auth_manager = UTRAuthManager()

def get_player_stats(data, player_id):
    """
    Extracts player stats from the given data dictionary.

    Args:
      data: The dictionary containing the tennis data.
      player_id: The ID of the player to extract stats for.

    Returns:
      A dictionary containing the player's stats, or None if not found.
    """
    notable_match = data.get('notableMatch')
    other_stats = data
    if notable_match:
        players = notable_match.get('players')
        if players:
            for player_key in ['winner1', 'loser1']:  # Check both winner and loser
                player = players.get(player_key)
                if player and player.get('id') == str(player_id):
                    return {
                        'id': player.get('id'),
                        'firstName': player.get('firstName'),
                        'lastName': player.get('lastName'),
                        'singlesUtr': player.get('singlesUtr'),
                        'doublesUtr': player.get('doublesUtr'),
                        'recordWinPercentage': other_stats.get('recordWinPercentage'),
                        'winsCount': other_stats.get('winsCount'),
                        'lossesCount': other_stats.get('lossesCount'),
                        # Add other stats you need here
                    }
    return None  # Player not found in notable match

# Function to flatten and save player data
def save_player_data(data, filename="player_data.csv"):
    # Extract top-level player information
    player_info = {
        'timestamp': datetime.now(),
        'id': data.get('id'),
        'displayName': f"{data.get('firstName', '')} {data.get('lastName', '')}".strip(),
        'singlesUtr': data.get('singlesUtr'),
        'doublesUtr': data.get('doublesUtr'),
        'wins': data.get('winsCount'),
        'losses': data.get('lossesCount'),
        'win_rate': data.get('recordWinPercentage')
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([player_info])
    
    # Append to existing CSV if it exists
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    # Save to CSV
    df.to_csv(filename, index=False)

# Function to flatten and save matches data
def save_matches_data(matches, filename="matches.csv"):
    if not matches:
        st.warning("No match data available for this player.")
        return
    
    processed_data = []

    # Iterate through each entry in the data
    for entry in matches:
        # Extract the 'descriptions' list
        descriptions = entry.pop('descriptions')

        # Iterate through each description and create a new row
        for description in descriptions:
            # Create a copy of the original entry
            new_row = entry.copy()

            # Update the new row with the description details
            new_row.update(description)

            # Append the new row to the processed data list
            processed_data.append(new_row)

    # Create a DataFrame from the processed data
    matches_df = pd.DataFrame(processed_data)
    
    # Convert resultDate to datetime
    if 'resultDate' in matches_df.columns:
        matches_df['resultDate'] = pd.to_datetime(matches_df['resultDate'], errors='coerce')
    
    # Append to existing CSV if it exists
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        matches_df = pd.concat([existing_df, matches_df], ignore_index=True)
    
    # Save to CSV
    matches_df.to_csv(filename, index=False)

# Function to flatten and save ratings history
def save_ratings_history(ratings_history, filename="ratings_history.csv"):
    if not ratings_history:
        st.warning("No ratings history available for this player.")
        return
    
    # Normalize ratings history data
    ratings_df = pd.json_normalize(ratings_history)
    
    # Convert date to datetime
    if 'date' in ratings_df.columns:
        ratings_df['date'] = pd.to_datetime(ratings_df['date'], errors='coerce')
    
    # Append to existing CSV if it exists
    #if os.path.exists(filename):
    #    existing_df = pd.read_csv(filename)
    #    ratings_df = pd.concat([existing_df, ratings_df], ignore_index=True)
    
    # Save to CSV
    ratings_df.to_csv(filename, index=False)

# Function to flatten and save aggregate data
def save_aggregate_data(aggregate_data, filename="aggregate_data.csv"):
    if not aggregate_data:
        st.warning("No aggregate data available for this player.")
        return
    
    # Normalize aggregate data
    aggregate_df = pd.json_normalize(aggregate_data)
    
    # Append to existing CSV if it exists
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        aggregate_df = pd.concat([existing_df, aggregate_df], ignore_index=True)
    
    # Save to CSV
    aggregate_df.to_csv(filename, index=False)

# Function to flatten and save eligible results
def save_eligible_results(eligible_results, filename_prefix="eligible_results"):
    if not eligible_results:
        st.warning("No eligible results available for this player.")
        return
    
    for category, results in eligible_results.items():
        if results:
            df = pd.DataFrame(results, columns=[f"{category}_result_id"])
            filename = f"{filename_prefix}_{category}.csv"
            if os.path.exists(filename):
                existing_df = pd.read_csv(filename)
                df = pd.concat([existing_df, df], ignore_index=True)
            df.to_csv(filename, index=False)
        else:
            st.warning(f"No eligible results in category: {category}")

# Function to flatten and save rankings
def save_rankings(rankings, filename="rankings.csv"):
    if not rankings:
        st.warning("No rankings data available for this player.")
        return
    
    rankings_df = pd.json_normalize(rankings)
    
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        rankings_df = pd.concat([existing_df, rankings_df], ignore_index=True)
    
    rankings_df.to_csv(filename, index=False)

def extract_ratings_from_trend(ratings_df):
    """
    Prepares ratings data for plotting.

    Args:
        ratings_df: A pandas DataFrame with 'date' and 'rating' columns.

    Returns:
        A pandas DataFrame with 'date' and 'rating' columns.
    """
    # Ensure 'date' column is datetime
    ratings_df['date'] = pd.to_datetime(ratings_df['date'], errors='coerce')
    
    # Sort by date
    ratings_df = ratings_df.sort_values('date')
    
    return ratings_df

def plot_data(ratings_df, player_name):
    """
    Creates a Plotly line chart for UTR ratings over time.

    Args:
        ratings_df: A pandas DataFrame with 'date' and 'rating' columns.
        player_name: The name of the player for the chart title.

    Returns:
        A Plotly figure object.
    """
    if ratings_df.empty:
        st.warning("Ratings DataFrame is empty. Cannot plot.")
        return

    # Calculate min and max ratings for bands
    min_rating = ratings_df['rating'].min()
    max_rating = ratings_df['rating'].max()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ratings_df['date'],
        y=ratings_df['rating'],
        name='UTR Rating',
        mode='lines+markers',
        line=dict(color='blue')
    ))

    # Add min/max bands
    fig.add_trace(go.Scatter(
        x=ratings_df['date'],
        y=[min_rating] * len(ratings_df),
        name='Min Rating',
        line=dict(dash='dot', color='red')
    ))
    fig.add_trace(go.Scatter(
        x=ratings_df['date'],
        y=[max_rating] * len(ratings_df),
        name='Max Rating',
        line=dict(dash='dot', color='green')
    ))

    fig.update_layout(
        title=f"UTR Rating History - {player_name}",
        xaxis_title="Date",
        yaxis_title="UTR Rating",
        hovermode='x unified'
    )

    return fig

def process_search_results(response_data):
    """Process search results from UTR API into player data for dropdown"""
    if not response_data or not isinstance(response_data, dict):
        return []
    
    players = []
    for hit in response_data.get('players', {}).get('hits', []):
        source = hit.get('source', {})
        if not source:
            continue
            
        location = source.get('location', {})
        location_display = location.get('display', '') if location else ''
        
        player = {
            'id': source.get('id'),
            'first_name': source.get('firstName', ''),
            'last_name': source.get('lastName', ''),
            'utr': source.get('singlesUtr', 0),
            'location': location_display,
            'display_text': f"{source.get('firstName', '')} {source.get('lastName', '')} (UTR: {source.get('singlesUtr', 0)}) - {location_display}"
        }
        players.append(player)
    return players

def create_search_dropdown():
    st.header("Player Search")
    search_query = st.text_input("Search for player")
    
    if search_query:
        response = search_utr_players(search_query)
        if response.status_code == 200:
            players = process_search_results(response.json())
            if players:
                # Create dropdown options
                options = [p['display_text'] for p in players]
                player_map = {p['display_text']: p['id'] for p in players}
                
                selected = st.selectbox("Select player:", options)
                if selected:
                    player_id = player_map[selected]
                    if st.button("Fetch Data"):
                        return player_id
    return None


def plot_top_performances():
    st.subheader("Top Performances and Notable Matches")
    
    # Load matches data
    matches_df = pd.read_csv('matches.csv')
    
    # Ensure correct data types
    matches_df['resultDate'] = pd.to_datetime(matches_df['resultDate'], errors='coerce')
    
    # Filter out matches without necessary data
    matches_df = matches_df.dropna(subset=['rating', 'win_margin'])
    
    # Create a new column for match significance
    matches_df['match_significance'] = matches_df['is_notable'].apply(lambda x: 'Notable' if x else 'Regular')
    
    # Scatter Plot
    fig = px.scatter(
        matches_df,
        x='win_margin',
        y='rating',
        color='match_significance',
        hover_data=['details', 'eventName', 'resultDate'],
        title='UTR Rating vs. Win Margin',
        labels={'win_margin': 'Win Margin (Games)', 'rating': 'UTR Rating'},
        size='win_margin',
        size_max=15
    )
    
    fig.update_layout(
        legend_title_text='Match Significance',
        xaxis_title='Win Margin (Games)',
        yaxis_title='UTR Rating'
    )
    
    st.plotly_chart(fig)

def plot_annotated_timeline():
    # Load matches data
    matches_df = pd.read_csv('matches.csv')
    
    # Convert dates to datetime
    matches_df['resultDate'] = pd.to_datetime(matches_df['resultDate'], errors='coerce')
    
    # Sort by date
    matches_df = matches_df.sort_values('resultDate')
    
    # Create the timeline
    fig = go.Figure()
    
    # Add all matches
    fig.add_trace(go.Scatter(
        x=matches_df['resultDate'],
        y=matches_df['rating'],
        mode='markers',
        name='Matches',
        marker=dict(
            size=8,
            color='lightblue',
            opacity=0.6
        )
    ))
    
    # Add notable matches
    notable_matches = matches_df[matches_df['is_notable'] == True]
    fig.add_trace(go.Scatter(
        x=notable_matches['resultDate'],
        y=notable_matches['rating'],
        mode='markers+text',
        name='Notable Matches',
        marker=dict(
            size=12,
            color='red',
            symbol='star'
        ),
        text=notable_matches['details'],
        textposition='top center',
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title='Annotated Timeline of Matches',
        xaxis_title='Date',
        yaxis_title='UTR Rating',
        hovermode='closest'
    )
    
    st.plotly_chart(fig)

def main():
    st.title("UTR Data Fetcher")
    st.write("Enter a Player ID to fetch and save their UTR data.")
    
    init_auth()
    
    with st.sidebar:
        st.header("Player Lookup")
        player_id = st.text_input("Enter Player ID", value="", max_chars=10)
        fetch_button = st.button("Fetch Data")
    
    if fetch_button:
        if not player_id.isdigit():
            st.error("Please enter a valid numerical Player ID.")
        else:
            player_id_int = int(player_id)
            try:
                with st.spinner("Fetching data..."):
                    data = st.session_state.auth_manager.get_player_stats(player_id_int)
                
                # Fetch Player Stats
                player_stats = get_player_stats(data, player_id_int)
                if player_stats:
                    # Save Player Data
                    save_player_data(player_stats)  # Save the extracted stats
                else:
                    st.warning("Player not found in the notable match.")
                    return  # Stop execution if player stats are not found
                
                # Save Matches Data
                matches = data.get('aggregateData', [])
                save_matches_data(matches)
                
                # Save Ratings History
                ratings_history = data.get('extendedRatingProfile', {}).get('history', [])
                save_ratings_history(ratings_history)
                
                # Save Eligible Results (if present)
                eligible_results = data.get('extendedRatingProfile', {}).get('eligibleResults', {})
                save_eligible_results(eligible_results)
                
                # Save Rankings (if present)
                rankings = data.get('extendedRatingProfile', {}).get('rankings', {})
                save_rankings(rankings)
                
                st.success("Data fetched and saved successfully!")
                
                # Optional: Display saved data
                if os.path.exists("player_data.csv"):
                    latest_player = pd.read_csv("player_data.csv").iloc[-1]
                    st.subheader(latest_player['displayName'])
                    
                    # Use st.columns to display player data side-by-side
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Singles UTR", latest_player['singlesUtr'])
                    with col2:
                        st.metric("Doubles UTR", latest_player['doublesUtr'])
                    with col3:
                        st.metric("Win Rate", latest_player['win_rate'])
                    with col4:
                        st.metric("Wins", latest_player['wins'])
                    with col5:
                        st.metric("Losses", latest_player['losses'])
                
                # Display Latest Matches
                if os.path.exists("matches.csv") and not pd.read_csv("matches.csv").empty:
                    st.subheader("Latest Matches")
                    matches_df = pd.read_csv("matches.csv")
                    st.dataframe(matches_df.tail(5))
                
                # Display Latest Ratings History
                if os.path.exists("ratings_history.csv") and not pd.read_csv("ratings_history.csv").empty:
                    st.subheader("Latest Ratings History")
                    ratings_df = pd.read_csv("ratings_history.csv")
                    st.dataframe(ratings_df.tail(5))
                    
                    # Plot Ratings History
                    player_name = latest_player['displayName']
                    prepared_ratings_df = extract_ratings_from_trend(ratings_df)
                    fig = plot_data(prepared_ratings_df, player_name)
                    st.plotly_chart(fig)
                
                # Display Eligible Results
                for category in eligible_results.keys():
                    filename = f"eligible_results_{category}.csv"
                    if os.path.exists(filename) and not pd.read_csv(filename).empty and eligible_results[category]:
                        st.subheader(f"Eligible Results - {category.capitalize()}")
                        eligible_df = pd.read_csv(filename)
                        st.dataframe(eligible_df.tail(5))
                
                # Display Rankings
                if os.path.exists("rankings.csv") and not pd.read_csv("rankings.csv").empty and rankings:
                    st.subheader("Rankings")
                    rankings_df = pd.read_csv("rankings.csv")
                    st.dataframe(rankings_df.tail(5))
                
                # **Add Visualizations for Top Performances and Notable Matches**
                st.header("Insights and Visualizations")
                
                # Scatter Plot
                plot_top_performances()
                
                # Annotated Timeline
                plot_annotated_timeline()
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
