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
import numpy as np


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
    try:
        winner_sets = eval(row['winnerSets'])
        loser_sets = eval(row['loserSets'])
        winner_total = sum(x for x in winner_sets if x is not None and isinstance(x, (int, float)))
        loser_total = sum(x for x in loser_sets if x is not None and isinstance(x, (int, float)))
        return winner_total - loser_total
    except:
        return 0

def calculate_strenuousness(row):
    try:
        winner_sets = eval(row['winnerSets'])
        loser_sets = eval(row['loserSets'])
        tiebreak_sets = eval(row['tiebreakSets'])
        
        # Count completed sets
        total_sets = sum(1 for s in winner_sets if s > 0)
        # Count tiebreaks
        tiebreaks = sum(1 for t in tiebreak_sets if t is not None and t > 0)
        
        # Base score based on sets played
        set_score = {2: 0, 3: 0.5, 4: 1.0}.get(total_sets, 1.0)
        # Add score for tiebreaks
        tiebreak_score = min(1.0, tiebreaks * 0.25)
        
        return (set_score * 0.7) + (tiebreak_score * 0.3)
    except:
        return 0



def create_match_visualizations(matches_df, player_id):
    """
    Creates a Plotly visualization of the player's match performance against opponent UTRs.

    :param matches_df: DataFrame containing match data.
    :param player_id: ID of the player to visualize matches for.
    :return: Plotly Figure object.
    """
    # Display the DataFrame for debugging purposes
    #st.write("### Processed Matches Data", matches_df)

    # Ensure 'win_margin' exists
    if 'win_margin' not in matches_df.columns:
        st.error("'win_margin' column is missing from the DataFrame.")
        return

    # Calculate 'strenuousness' based on 'win_margin'
    def calculate_strenuousness(row):
        """
        Calculates match strenuousness based on win margin.
        Higher margins indicate less strenuous matches.
        """
        margin = abs(row['win_margin'])
        # Define thresholds for strenuousness
        if margin <= 3:
            return 1.0  # Most strenuous
        elif margin <= 6:
            return 0.7
        else:
            return 0.4  # Least strenuous


    matches_df['strenuousness'] = matches_df.apply(calculate_strenuousness, axis=1)

    # Calculate marker size based on 'win_margin'
    matches_df['marker_size'] = matches_df['win_margin'].abs() * 2 + 4

    # Create 'opponent_utr' by averaging loser UTRs for doubles or using single loser UTR for singles
    matches_df['opponent_utr'] = matches_df.apply(
        lambda row: row['loser1_utr'] if pd.isna(row['loser2_utr']) else (row['loser1_utr'] + row['loser2_utr']) / 2,
        axis=1
    )

    # Create 'opponent_name' by concatenating loser names for doubles or using single loser name for singles
    matches_df['opponent_name'] = matches_df.apply(
        lambda row: row['loser1_name'] if pd.isna(row['loser2_name']) else f"{row['loser1_name']} & {row['loser2_name']}",
        axis=1
    )

    # Create 'opponent_nationality' similarly
    matches_df['opponent_nationality'] = matches_df.apply(
        lambda row: row['loser1_nationality'] if pd.isna(row['loser2_nationality']) else f"{row['loser1_nationality']} & {row['loser2_nationality']}",
        axis=1
    )

    # Convert date columns to datetime
    date_columns = ['resultDate', 'eventStartDate', 'eventEndDate']
    for col in date_columns:
        if col in matches_df.columns:
            matches_df[col] = pd.to_datetime(matches_df[col], errors='coerce')

    # Initialize Plotly figure
    fig = go.Figure()

    # Group matches by event to add event boundaries and annotations
    events = matches_df.groupby(['eventId', 'eventName', 'eventStartDate', 'eventEndDate'])

    for (event_id, event_name, start_date, end_date), event_data in events:
        if len(event_data) > 1 and pd.notnull(start_date) and pd.notnull(end_date):
            y_min = matches_df['opponent_utr'].min()
            y_max = matches_df['opponent_utr'].max()

            # Add shaded region for the event duration
            fig.add_shape(
                type="rect",
                x0=start_date,
                x1=end_date,
                y0=y_min - 0.5,
                y1=y_max + 0.5,
                fillcolor="lightgray",
                opacity=0.2,
                layer="below",
                line_width=0,
            )

            # Add event name annotation
            fig.add_annotation(
                x=start_date,
                y=y_max + 0.5,
                text=event_name,
                showarrow=False,
                textangle=-45,
                font=dict(color="black")
            )

    # Define hover text formatting
    def format_hover_text(row):
        date_str = row['resultDate'].strftime('%Y-%m-%d') if pd.notnull(row['resultDate']) else 'Unknown'
        utr_str = f"{row['opponent_utr']:.2f}" if pd.notnull(row['opponent_utr']) else 'Unknown'
        return (
            f"Event: {row['eventName']}<br>"
            f"Round: {row['round']}<br>"
            f"Date: {date_str}<br>"
            f"Opponent: {row['opponent_name']}<br>"
            f"Opponent UTR: {utr_str}<br>"
            f"Game Margin: {row['win_margin']}<br>"
            f"Intensity: {row['strenuousness']:.2f}"
        )

    # Apply hover text
    hover_text = matches_df.apply(format_hover_text, axis=1)

    # Add scatter trace for matches
    fig.add_trace(
        go.Scatter(
            x=matches_df['resultDate'],
            y=matches_df['opponent_utr'],
            mode='markers',
            marker=dict(
                size=matches_df['marker_size'],
                color=matches_df['strenuousness'],
                symbol=matches_df['isWinner'].map({True: 'circle', False: 'x'}),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Match Intensity",
                    tickmode="array",
                    tickvals=[0, 0.5, 1],
                    ticktext=["High", "Medium", "Low"],
                    titleside="top"
                )
            ),
            text=hover_text,
            hoverinfo='text',
            name='Matches'
        )
    )

    # Update layout
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Match Performance Analysis vs Opponent UTR",
        xaxis_title="Date",
        yaxis_title="Opponent UTR Rating",
        hovermode="closest",
        template="plotly_white"
    )

    return fig

def display_match_analysis(matches_df, player_id):
    """
    Displays the match analysis visualization in Streamlit.

    :param matches_df: DataFrame containing match data.
    :param player_id: ID of the player to visualize matches for.
    """
    fig = create_match_visualizations(matches_df, player_id)
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
    matches_df = matches_df.dropna(subset=['opponent_utr', 'win_margin'])
    
    # Create a new column for match significance
    matches_df['match_significance'] = matches_df['is_notable'].apply(lambda x: 'Notable' if x else 'Regular')
    
    # Create a new column for absolute win margin
    matches_df['abs_win_margin'] = matches_df['win_margin'].abs()
    
    # Scatter Plot
    fig = px.scatter(
        matches_df,
        x='abs_win_margin',
        y='opponent_utr',
        color='isWinner',  # Use 'isWinner' to color code
        hover_data=['eventName', 'resultDate'],
        title='UTR Rating vs. Win Margin',
        labels={
            'abs_win_margin': 'Win Margin (Absolute Games)',
            'opponent_utr': 'UTR Rating',
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
        y=matches_df['opponent_utr'],
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
        y=notable_matches['opponent_utr'],
        mode='markers+text',
        name='Notable Matches',
        marker=dict(
            size=12,
            color='red',
            symbol='star'
        ),
        text=notable_matches['eventName'],
        textposition='top center',
        hoverinfo='text',
        hovertext=notable_matches['score_details']
    ))
    
    fig.update_layout(
        title='Annotated Timeline of Matches',
        xaxis_title='Date',
        yaxis_title='UTR Rating',
        hovermode='closest'
    )
    
    st.plotly_chart(fig)


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

def calculate_total_games(winner_sets, loser_sets):
    winner_total = sum(x for x in winner_sets if x is not None and isinstance(x, (int, float)))
    loser_total = sum(x for x in loser_sets if x is not None and isinstance(x, (int, float)))
    return winner_total - loser_total



import pandas as pd
import numpy as np
import streamlit as st  # Ensure Streamlit is imported for st.warning and st.error

def save_matches_data(matches_data, player_id, filename="matches.csv"):
    """
    Saves match data involving the specified player to a CSV file.

    :param matches_data: Dictionary containing match data.
    :param player_id: ID of the player to filter matches.
    :param filename: Name of the CSV file to save the data.
    """
    def calculate_total_games(winner_sets, loser_sets):
        """
        Calculate the total game margin for the player.
        Positive value indicates a win margin, negative indicates a loss margin.
        """
        total_winner = sum(winner_sets)
        total_loser = sum(loser_sets)
        return total_winner - total_loser

    def identify_notable_matches(match_row, notable_events, high_margin_threshold):
        """
        Identify if a match is notable based on event and win margin.
        """
        event = match_row.get('eventName', '')
        win_margin = match_row.get('win_margin', 0)

        if event in notable_events or abs(win_margin) >= high_margin_threshold:
            return True
        return False

    try:
        processed_data = []
        notable_events = ['US Open', 'Australian Open', 'French Open', 'Wimbledon']
        high_margin_threshold = 10

        participant_id = str(player_id)

        # Iterate through each event
        for event in matches_data.get('events', []):
            event_id = event.get('id', '')
            event_name = event.get('name', '')
            event_start = event.get('startDate')
            event_end = event.get('endDate')

            # Iterate through each draw within the event
            for draw in event.get('draws', []):
                draw_name = draw.get('name', '')
                results = draw.get('results', [])

                # Iterate through each match within the draw
                for match in results:
                    if not match or not match.get('players'):
                        continue  # Skip if match data is incomplete

                    try:
                        # Get 'winner' and 'loser' dicts, ensure they are dicts
                        winner = match.get('winner', {}) or {}
                        loser = match.get('loser', {}) or {}
                        players = match.get('players', {}) or {}

                        # Validate 'players' is a dictionary
                        if not isinstance(players, dict):
                            match_id = match.get('id', 'Unknown')
                            st.warning(f"Invalid players data for match ID {match_id}. Skipping match.")
                            continue

                        # Collect all winner and loser player IDs
                        winner_ids = []
                        if 'winner1' in players and players['winner1']:
                            winner_ids.append(str(players['winner1'].get('id', '')))
                        if 'winner2' in players and players['winner2']:
                            winner_ids.append(str(players['winner2'].get('id', '')))

                        loser_ids = []
                        if 'loser1' in players and players['loser1']:
                            loser_ids.append(str(players['loser1'].get('id', '')))
                        if 'loser2' in players and players['loser2']:
                            loser_ids.append(str(players['loser2'].get('id', '')))

                        # Determine if the player is a winner or a loser
                        is_winner = participant_id in winner_ids
                        is_loser = participant_id in loser_ids

                        if not (is_winner or is_loser):
                            continue  # Player not involved in this match

                        # Extract set scores
                        winner_sets = [winner.get(f'set{i}', 0) or 0 for i in range(1, 7)]
                        loser_sets = [loser.get(f'set{i}', 0) or 0 for i in range(1, 7)]

                        # Extract tiebreak sets
                        tiebreak_sets = []
                        for i in range(1, 7):
                            tiebreak_winner = winner.get(f'tiebreakerSet{i}', None)
                            tiebreak_loser = loser.get(f'tiebreakerSet{i}', None)
                            if tiebreak_winner is not None:
                                tiebreak_sets.append(tiebreak_winner)
                            elif tiebreak_loser is not None:
                                tiebreak_sets.append(tiebreak_loser)
                            else:
                                tiebreak_sets.append(None)

                        # Extract player details
                        # Winner1
                        winner1 = players.get('winner1', {}) or {}
                        winner1_id = str(winner1.get('id', '')) if winner1 else ''
                        winner1_name = f"{winner1.get('firstName', '')} {winner1.get('lastName', '')}".strip() if winner1 else ''
                        winner1_utr = float(winner1.get('singlesUtr')) if winner1.get('singlesUtr') else np.nan
                        winner1_nationality = winner1.get('nationality', '') if winner1 else ''

                        # Winner2
                        winner2 = players.get('winner2') or {}
                        winner2_id = str(winner2.get('id', '')) if winner2 else ''
                        winner2_name = f"{winner2.get('firstName', '')} {winner2.get('lastName', '')}".strip() if winner2 else ''
                        winner2_utr = float(winner2.get('singlesUtr')) if winner2.get('singlesUtr') else np.nan
                        winner2_nationality = winner2.get('nationality', '') if winner2 else ''

                        # Loser1
                        loser1 = players.get('loser1', {}) or {}
                        loser1_id = str(loser1.get('id', '')) if loser1 else ''
                        loser1_name = f"{loser1.get('firstName', '')} {loser1.get('lastName', '')}".strip() if loser1 else ''
                        loser1_utr = float(loser1.get('singlesUtr')) if loser1.get('singlesUtr') else np.nan
                        loser1_nationality = loser1.get('nationality', '') if loser1 else ''

                        # Loser2
                        loser2 = players.get('loser2') or {}
                        loser2_id = str(loser2.get('id', '')) if loser2 else ''
                        loser2_name = f"{loser2.get('firstName', '')} {loser2.get('lastName', '')}".strip() if loser2 else ''
                        loser2_utr = float(loser2.get('singlesUtr')) if loser2.get('singlesUtr') else np.nan
                        loser2_nationality = loser2.get('nationality', '') if loser2 else ''

                        # Determine opponent UTR based on player's role
                        if is_winner:
                            # Player is the winner; opponents are losers
                            if not np.isnan(loser2_utr):
                                opponent_utr = (loser1_utr + loser2_utr) / 2
                            else:
                                opponent_utr = loser1_utr
                        elif is_loser:
                            # Player is the loser; opponents are winners
                            if not np.isnan(winner2_utr):
                                opponent_utr = (winner1_utr + winner2_utr) / 2
                            else:
                                opponent_utr = winner1_utr
                        else:
                            opponent_utr = np.nan  # Should not occur

                        # Handle cases where opponent UTR is NaN
                        if np.isnan(opponent_utr):
                            opponent_utr = np.nan

                        # Determine if the match is a doubles match
                        is_doubles = not (pd.isna(winner2_id) and pd.isna(loser2_id))

                        # Determine outcome
                        outcome = 'W' if is_winner else 'L'

                        # Build new_row with separate winner1 and winner2 columns
                        new_row = {
                            'eventId': event_id,
                            'eventName': event_name,
                            'eventStartDate': event_start,
                            'eventEndDate': event_end,
                            'drawName': draw_name,
                            'resultDate': match.get('date'),
                            'round': match.get('round', {}).get('name', '') if match.get('round') else '',
                            'roundCode': match.get('round', {}).get('code', '') if match.get('round') else '',
                            'matchId': match.get('id'),
                            'winnerSets': winner_sets,
                            'loserSets': loser_sets,
                            'tiebreakSets': tiebreak_sets,
                            'sourceType': match.get('sourceType', ''),
                            'outcome': outcome,
                            'score_details': match.get('score', {}),
                            'isWinner': is_winner,
                            'win_margin': calculate_total_games(winner_sets, loser_sets),
                            'opponent_utr': opponent_utr,  # Adding opponent UTR
                            'is_doubles': is_doubles,      # Adding is_doubles flag
                            'is_notable': identify_notable_matches({}, notable_events, high_margin_threshold)  # Placeholder, updated below
                        }

                        # Add separate winner1 and winner2 details
                        new_row.update({
                            'winner1_id': winner1_id,
                            'winner1_name': winner1_name,
                            'winner1_utr': winner1_utr,
                            'winner1_nationality': winner1_nationality,
                            'winner2_id': winner2_id,
                            'winner2_name': winner2_name,
                            'winner2_utr': winner2_utr,
                            'winner2_nationality': winner2_nationality,
                            'loser1_id': loser1_id,
                            'loser1_name': loser1_name,
                            'loser1_utr': loser1_utr,
                            'loser1_nationality': loser1_nationality,
                            'loser2_id': loser2_id,
                            'loser2_name': loser2_name,
                            'loser2_utr': loser2_utr,
                            'loser2_nationality': loser2_nationality
                        })

                        # Update 'is_notable' with the correct row data
                        new_row['is_notable'] = identify_notable_matches(new_row, notable_events, high_margin_threshold)

                        # Append to processed_data
                        processed_data.append(new_row)

                    except Exception as e:
                        # Include match ID in the warning for easier debugging
                        match_id = match.get('id', 'Unknown')
                        st.warning(f"Error processing match ID {match_id}: {e}")
                        continue

        if processed_data:
            matches_df = pd.DataFrame(processed_data)

            # Convert date columns to datetime
            for date_col in ['resultDate', 'eventStartDate', 'eventEndDate']:
                if date_col in matches_df.columns:
                    matches_df[date_col] = pd.to_datetime(matches_df[date_col], errors='coerce')

            # Convert UTR columns to float (ensure numerical types)
            utr_columns = ['winner1_utr', 'winner2_utr', 'loser1_utr', 'loser2_utr', 'opponent_utr']
            for col in utr_columns:
                if col in matches_df.columns:
                    matches_df[col] = pd.to_numeric(matches_df[col], errors='coerce')

            # Convert 'score_details' to JSON string if needed
            # Here, we'll keep it as a string representation
            matches_df['score_details'] = matches_df['score_details'].apply(lambda x: str(x))

            # Reorder columns for better readability (optional)
            columns_order = [
                'eventId', 'eventName', 'eventStartDate', 'eventEndDate',
                'drawName', 'resultDate', 'round', 'roundCode', 'matchId',
                'winner1_id', 'winner1_name', 'winner1_utr', 'winner1_nationality',
                'winner2_id', 'winner2_name', 'winner2_utr', 'winner2_nationality',
                'loser1_id', 'loser1_name', 'loser1_utr', 'loser1_nationality',
                'loser2_id', 'loser2_name', 'loser2_utr', 'loser2_nationality',
                'opponent_utr', 'is_doubles',
                'winnerSets', 'loserSets', 'tiebreakSets',
                'sourceType', 'outcome', 'score_details',
                'isWinner', 'win_margin', 'is_notable'
            ]

            # Ensure all columns are present
            columns_order = [col for col in columns_order if col in matches_df.columns]

            matches_df = matches_df[columns_order]

            # Check for duplicate matches
            duplicate_matches = matches_df[matches_df.duplicated(subset=['matchId'], keep=False)]
            if not duplicate_matches.empty:
                st.warning("Duplicate match entries found. These will be removed.")
                matches_df = matches_df.drop_duplicates(subset=['matchId'])

            # Verify 'isWinner' flags are correctly set
            # For example, ensure that 'win_margin' is positive for wins and negative for losses
            inconsistent_flags = matches_df[
                ((matches_df['isWinner'] == True) & (matches_df['win_margin'] < 0)) |
                ((matches_df['isWinner'] == False) & (matches_df['win_margin'] > 0))
            ]
            if not inconsistent_flags.empty:
                st.warning("Inconsistent 'isWinner' flags found. These matches will be removed.")
                matches_df = matches_df.drop(inconsistent_flags.index)

            # Save to CSV with appropriate data types
            matches_df.to_csv(filename, index=False)
            st.success(f"Match data successfully saved to {filename}.")
        else:
            st.warning("No valid matches found involving the player.")

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
            #matches = data.get('victoryMarginChart', {}).get('results', [])
            matches = st.session_state.auth_manager.get_player_results(int(player_id))
            #st.write(matches)
            save_matches_data(matches, player_id)
            
           
            
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
                    display_match_analysis(matches_df, player_id)

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
