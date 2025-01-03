import streamlit as st
import pandas as pd
from datetime import datetime
import os
from utr_auth import UTRAuthManager
from player_resilience import PlayerResilience
from dotenv import load_dotenv
import plotly.graph_objects as go  # Import Plotly
import plotly.express as px
import ast


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import ast


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
    # Calculate moving average
    ratings_df['MA7'] = ratings_df['rating'].rolling(window=7).mean()
    
    fig = go.Figure()
    
    # Add main rating line
    fig.add_trace(go.Scatter(
        x=ratings_df['date'],
        y=ratings_df['rating'],
        name='UTR Rating',
        mode='lines+markers',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    # Add moving average
    fig.add_trace(go.Scatter(
        x=ratings_df['date'],
        y=ratings_df['MA7'],
        name='7-Day Average',
        line=dict(color='rgba(0,0,255,0.3)', dash='dot')
    ))
    
    # Add filled area between min/max
    fig.add_trace(go.Scatter(
        x=ratings_df['date'],
        y=[ratings_df['rating'].max()] * len(ratings_df),
        fill=None,
        mode='lines',
        line=dict(color='rgba(0,255,0,0.1)', dash='dot'),
        name='Max Rating'
    ))
    
    fig.add_trace(go.Scatter(
        x=ratings_df['date'],
        y=[ratings_df['rating'].min()] * len(ratings_df),
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(255,0,0,0.1)', dash='dot'),
        name='Min Rating',
        fillcolor='rgba(0,100,255,0.1)'
    ))
    
    # Add range annotations
    fig.add_annotation(
        x=ratings_df['date'].max(),
        y=ratings_df['rating'].max(),
        text=f"Peak: {ratings_df['rating'].max():.2f}",
        showarrow=True,
        arrowhead=1
    )
    
    fig.update_layout(
        title=f"UTR Rating History - {player_name}",
        xaxis_title="Date",
        yaxis_title="UTR Rating",
        hovermode='x unified',
        plot_bgcolor='white',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        )
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    
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



def calculate_win_margin(row):
    winner_sets = ast.literal_eval(row['winnerSets'])
    loser_sets = ast.literal_eval(row['loserSets'])
    
    # Calculate total games for winner and loser
    winner_games = sum(winner_sets)
    loser_games = sum(loser_sets)
    
    # If the player is the winner, return positive margin, otherwise negative
    margin = winner_games - loser_games
    return margin if row['isWinner'] else -margin

def calculate_strenuousness(row):
    winner_sets = ast.literal_eval(row['winnerSets'])
    loser_sets = ast.literal_eval(row['loserSets'])
    tiebreak_sets = ast.literal_eval(row['tiebreakSets'])
    
    total_sets = sum(1 for s in winner_sets if s > 0)
    tiebreaks = sum(1 for t in tiebreak_sets if t > 0)
    
    # Base score: 2 sets = 0, 3 sets = 0.5, 4+ sets = 1.0
    set_score = {2: 0, 3: 0.5, 4: 1.0}.get(total_sets, 1.0)
    
    # Add 0.25 for each tiebreak
    tiebreak_score = min(1.0, tiebreaks * 0.25)
    
    # Combine scores with weights
    final_score = (set_score * 0.7) + (tiebreak_score * 0.3)
    return final_score

def create_match_visualizations(matches_df):
    matches_df['resultDate'] = pd.to_datetime(matches_df['resultDate'])
    matches_df['win_margin'] = matches_df.apply(calculate_win_margin, axis=1)
    matches_df['strenuousness'] = matches_df.apply(calculate_strenuousness, axis=1)
    matches_df['marker_size'] = abs(matches_df['win_margin']) * 2. + 4

    # Group matches by event
    matches_df['month'] = matches_df['resultDate'].dt.strftime('%B')
    events = matches_df.groupby('eventName')
    
    fig = go.Figure()
    
    # Add event highlighting rectangles
    for event_name, event_data in events:
        if len(event_data) > 1:  # Only add for multi-match events
            start_date = event_data['resultDate'].min()
            end_date = event_data['resultDate'].max()
            
            fig.add_shape(
                type="rect",
                x0=start_date,
                x1=end_date,
                y0=matches_df['rating'].min() - 0.1,
                y1=matches_df['rating'].max() + 0.1,
                fillcolor="lightgray",
                opacity=0.2,
                layer="below",
                line_width=0,
            )
            
            # Add event annotation
            fig.add_annotation(
                x=start_date,
                y=matches_df['rating'].max() + 0.15,
                text=event_name,
                showarrow=False,
                textangle=-45
            )
    
    # Add main scatter plot
    fig.add_trace(
        go.Scatter(
            x=matches_df['resultDate'],
            y=matches_df['rating'],
            mode='markers',
            marker=dict(
                size=matches_df['marker_size'],
                color=matches_df['strenuousness'],
                colorscale='viridis',
                showscale=True,
                colorbar=dict(
                    title="Match Intensity",
                    ticktext=["2 Sets", "3 Sets", "4+ Sets/Tiebreaks"],
                    tickvals=[0, 0.5, 1],
                    tickmode="array"
                ),
                symbol=matches_df['isWinner'].map({True: 'circle', False: 'x'})
            ),
            text=matches_df.apply(lambda x: f"{x['eventName']}<br>{x['details']}<br>Game Margin: {x['win_margin']}<br>Intensity: {x['strenuousness']:.2f}", axis=1),
            hoverinfo='text',
            name='Matches'
        )
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        title_text="Match Performance Analysis"
    )
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Opponent UTR Rating")
    
    return fig

# Add to main() function:
def display_match_analysis(matches_df):
    #st.subheader("Match Performance Analysis")
    fig = create_match_visualizations(matches_df)
    st.plotly_chart(fig, use_container_width=True)




# Scatter Plot Function
def plot_top_performances():
    st.subheader("Top Performances and Notable Matches")
    
    # Load matches data
    if not os.path.exists("matches.csv"):
        st.warning("Matches data not found.")
        return
    
    matches_df = pd.read_csv("matches.csv")
    
    # Ensure correct data types
    matches_df['resultDate'] = pd.to_datetime(matches_df['resultDate'], errors='coerce')
    
    # Drop rows with missing 'rating' or 'win_margin'
    matches_df = matches_df.dropna(subset=['rating', 'win_margin'])
    
    # Create a new column for match significance
    matches_df['match_significance'] = matches_df['is_notable'].apply(lambda x: 'Notable' if x else 'Regular')
    
    # Create a new column for absolute win margin
    matches_df['abs_win_margin'] = matches_df['win_margin'].abs()
    
    # Scatter Plot
    fig = px.scatter(
        matches_df,
        x='abs_win_margin',
        y='rating',
        color='isWinner',  # Use 'isWinner' to color code
        hover_data=['details', 'eventName', 'resultDate'],
        title='UTR Rating vs. Win Margin',
        labels={
            'abs_win_margin': 'Win Margin (Absolute Games)',
            'rating': 'UTR Rating',
            'isWinner': 'Match Outcome'
        },
        size='abs_win_margin',
        size_max=15,
        symbol='isWinner',  # Different symbols for wins and losses
        category_orders={'isWinner': [True, False]},  # Ensure consistent color mapping
        color_discrete_map={True: 'green', False: 'red'}  # Green for wins, red for losses
    )
    
    fig.update_layout(
        legend_title_text='Match Outcome',
        xaxis_title='Win Margin (Absolute Games)',
        yaxis_title='UTR Rating'
    )
    
    st.plotly_chart(fig, use_container_width=True)


# Annotated Timeline Function
def plot_annotated_timeline():
    st.subheader("Annotated Timeline of Matches")
    
    # Load matches data
    matches_df = load_matches_data()
    
    # Convert dates to datetime (already done in load_matches_data)
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
        ),
        hoverinfo='none'  # Disable hover for regular matches
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
        hoverinfo='text',
        hovertext=notable_matches['details']
    ))
    
    fig.update_layout(
        title='Annotated Timeline of Matches',
        xaxis_title='Date',
        yaxis_title='UTR Rating',
        hovermode='closest'
    )
    
    st.plotly_chart(fig)


def calculate_total_games(winner_sets, loser_sets):
    if isinstance(winner_sets, list) and isinstance(loser_sets, list):
        return sum(winner_sets) - sum(loser_sets)
    return 0


def load_matches_data():
    df = pd.read_csv('matches.csv')
    df['resultDate'] = pd.to_datetime(df['resultDate'], errors='coerce')
    # Calculate win_margin if not already present
    if 'win_margin' not in df.columns:
        df['win_margin'] = df.apply(
            lambda row: calculate_total_games(eval(row['winnerSets']), eval(row['loserSets'])),
            axis=1
        )
    # Identify notable matches if not already present
    if 'is_notable' not in df.columns:
        df['is_notable'] = df.apply(identify_notable_matches, axis=1)
    return df

def identify_notable_matches(row, notable_events, high_margin_threshold):
    if row['eventName'] in notable_events:
        return True
    if row['win_margin'] >= high_margin_threshold:
        return True
    return False

def save_matches_data(matches, filename="matches.csv"):
    try:
        processed_data = []
        notable_events = ['US Open', 'Australian Open', 'French Open', 'Wimbledon']
        high_margin_threshold = 10
        
        for entry in matches:
            descriptions = entry.get('descriptions', [])
            if not isinstance(descriptions, list):
                st.warning("Descriptions format is incorrect. Skipping this entry.")
                continue
            
            for description in descriptions:
                new_row = {k: v for k, v in entry.items() if k != 'descriptions'}
                new_row.update(description)
                win_margin = calculate_total_games(description.get('winnerSets', []), description.get('loserSets', []))
                new_row['win_margin'] = win_margin
                new_row['is_notable'] = identify_notable_matches(new_row, notable_events, high_margin_threshold)
                processed_data.append(new_row)
        
        if not processed_data:
            st.warning("No valid match descriptions found.")
            return
        
        matches_df = pd.DataFrame(processed_data)
        
        # Convert resultDate to datetime
        matches_df['resultDate'] = pd.to_datetime(matches_df['resultDate'], errors='coerce')
        
        # Save to CSV
        #if os.path.exists(filename):
        #    existing_df = pd.read_csv(filename)
        #    matches_df = pd.concat([existing_df, matches_df], ignore_index=True)
        
        matches_df.to_csv(filename, index=False)
    except Exception as e:
        st.error(f"Error saving matches data: {e}")

def main():
    st.title("UTR Data Fetcher")
    init_auth()
    
    with st.sidebar:
        method = st.radio("Search Method", ["Search by Name", "Enter ID"])
        
        if method == "Search by Name":
            search_query = st.text_input("Search for player")
            player_id = None
            if search_query:
                players = process_search_results(st.session_state.auth_manager.search_players(search_query))
                if players:
                    options = [p['display_text'] for p in players]
                    player_map = {p['display_text']: p['id'] for p in players}
                    selected = st.selectbox("Select player:", options)
                    if selected:
                        player_id = player_map[selected]
        else:
            player_id = st.text_input("Enter Player ID", value="", max_chars=10)

        fetch_button = st.button("Fetch Data")

    if fetch_button and player_id:
        if not str(player_id).isdigit():
            st.error("Please enter a valid numerical Player ID.")
            return

        try:
            with st.spinner("Fetching data..."):
                data = st.session_state.auth_manager.get_player_stats(int(player_id))
            
            # Move display to main area
            player_stats = get_player_stats(data, player_id)
            if player_stats:
                save_player_data(player_stats)
            else:
                st.warning("Player not found in the notable match.")
                return

            # Save all other data
            matches = data.get('victoryMarginChart', {}).get('results', [])
            #st.write(matches)
            save_matches_data(matches)
            
           
            
            ratings_history = data.get('extendedRatingProfile', {}).get('history', [])
            save_ratings_history(ratings_history)
            
            eligible_results = data.get('extendedRatingProfile', {}).get('eligibleResults', {})
            save_eligible_results(eligible_results)
            
            rankings = data.get('extendedRatingProfile', {}).get('rankings', {})
            save_rankings(rankings)
            
            st.success("Data fetched and saved successfully!")

            # Display data in main area
            if os.path.exists("player_data.csv"):
                latest_player = pd.read_csv("player_data.csv").iloc[-1]
                st.subheader(latest_player['displayName'])
                
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

            # Add tabs for main views
            tab1, tab2, tab3 = st.tabs(["Performance Analysis", "Notable Matches", "Mental Resilience"])

            with tab1:
                if os.path.exists("matches.csv") and matches:
                    #st.subheader("Latest Matches")
                    matches_df = pd.read_csv("matches.csv")
                    #st.dataframe(matches_df)
                    display_match_analysis(matches_df)

                if os.path.exists("ratings_history.csv") and ratings_history:
                    #st.subheader("Latest Ratings History")
                    ratings_df = pd.read_csv("ratings_history.csv")
                    fig = plot_data(ratings_df, latest_player['displayName'])
                    st.plotly_chart(fig)
                
                
                for category in eligible_results.keys():
                    filename = f"eligible_results_{category}.csv"
                    if os.path.exists(filename) and eligible_results[category]:
                        #st.subheader(f"Eligible Results - {category.capitalize()}")
                        eligible_df = pd.read_csv(filename)
                        #st.dataframe(eligible_df.tail(5))
                
                if os.path.exists("rankings.csv") and rankings:
                    #st.subheader("Rankings")
                    rankings_df = pd.read_csv("rankings.csv")
                    #st.dataframe(rankings_df.tail(5))
            
            with tab2:
                # **Add Visualizations for Top Performances and Notable Matches**
                st.header("Insights and Visualizations")
                
                # Scatter Plot for Top Performances
                plot_top_performances()
                
                # Annotated Timeline for Notable Matches
                plot_annotated_timeline()
                
                # **Mental Resilience Over Time**
                #plot_mental_resilience_over_time()

            with tab3:
                st.header("Mental Resilience Analysis")
                if os.path.exists("matches.csv"):
                    matches_df = pd.read_csv("matches.csv")
                    
                    resilience = PlayerResilience(matches_df)
                    
                    # Overall analysis
                    analysis = resilience.get_detailed_analysis()
                    #st.write(analysis)
                    st.metric("Overall Resilience", f"{analysis['overall_resilience']:.2f}")
                    
                    # Time-based analysis
                    st.subheader("Resilience Over Time")
                    fig = resilience.plot_scores_over_time()
                    st.plotly_chart(fig, use_container_width=True)
                

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
