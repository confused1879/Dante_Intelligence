import streamlit as st
import pandas as pd
from datetime import datetime
import os
from utr_auth import UTRAuthManager
from dotenv import load_dotenv
import plotly.graph_objects as go  # Import Plotly
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
    #if os.path.exists(filename):
    #    existing_df = pd.read_csv(filename)
    #    matches_df = pd.concat([existing_df, matches_df], ignore_index=True)
    
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


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import ast

def calculate_win_margin(row):
    # Convert string lists to actual lists
    winner_sets = ast.literal_eval(row['winnerSets'])
    loser_sets = ast.literal_eval(row['loserSets'])
    
    # Calculate games won by each player
    winner_games = sum(winner_sets)
    loser_games = sum(loser_sets)
    
    # Calculate margin based on whether the player of interest won or lost
    if row['isWinner']:
        return winner_games - loser_games
    else:
        return loser_games - winner_games

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


def calculate_comeback_wins(matches_df):
    """
    Calculate the percentage of matches won after losing the first set.
    """
    comeback_wins = 0
    total_first_set_losses = 0
    
    for _, row in matches_df.iterrows():
        try:
            winner_sets = ast.literal_eval(row['winnerSets'])
            loser_sets = ast.literal_eval(row['loserSets'])
            is_winner = row['isWinner']
            
            if len(winner_sets) >= 1 and len(loser_sets) >= 1:
                first_set_winner = 'player' if winner_sets[0] > loser_sets[0] else 'opponent'
                if first_set_winner == 'opponent':
                    total_first_set_losses += 1
                    if is_winner:
                        comeback_wins += 1
        except Exception as e:
            st.warning(f"Error processing match for comeback wins: {e}")
            continue
    
    if total_first_set_losses > 0:
        comeback_rate = (comeback_wins / total_first_set_losses) * 100
    else:
        comeback_rate = 0.0
    
    return comeback_rate

def calculate_tiebreak_performance(matches_df):
    """
    Calculate the tiebreak win rate.
    """
    total_tiebreaks = 0
    tiebreaks_won = 0
    
    for _, row in matches_df.iterrows():
        try:
            tiebreak_sets = ast.literal_eval(row['tiebreakSets'])
            winner_sets = ast.literal_eval(row['winnerSets'])
            loser_sets = ast.literal_eval(row['loserSets'])
            is_winner = row['isWinner']
            
            for idx, tiebreak in enumerate(tiebreak_sets):
                if tiebreak > 0:
                    total_tiebreaks += 1
                    winner_score = winner_sets[idx]
                    loser_score = loser_sets[idx]
                    if (winner_score > loser_score and is_winner) or (loser_score > winner_score and not is_winner):
                        tiebreaks_won += 1
        except Exception as e:
            st.warning(f"Error processing match for tiebreak performance: {e}")
            continue
    
    if total_tiebreaks > 0:
        tiebreak_win_rate = (tiebreaks_won / total_tiebreaks) * 100
    else:
        tiebreak_win_rate = 0.0
    
    return tiebreak_win_rate

def calculate_deciding_set_performance(matches_df):
    """
    Calculate the percentage of matches won in deciding sets.
    Assumes best-of-three sets.
    """
    deciding_set_matches = 0
    deciding_set_wins = 0
    
    for _, row in matches_df.iterrows():
        try:
            winner_sets = ast.literal_eval(row['winnerSets'])
            loser_sets = ast.literal_eval(row['loserSets'])
            is_winner = row['isWinner']
            
            total_sets = max(len(winner_sets), len(loser_sets))
            if total_sets >= 3:  # Best-of-three
                deciding_set_matches += 1
                player_final_set_games = winner_sets[-1] if is_winner else loser_sets[-1]
                opponent_final_set_games = loser_sets[-1] if is_winner else winner_sets[-1]
                
                if player_final_set_games > opponent_final_set_games:
                    deciding_set_wins += 1
        except Exception as e:
            st.warning(f"Error processing match for deciding set performance: {e}")
            continue
    
    if deciding_set_matches > 0:
        deciding_set_win_rate = (deciding_set_wins / deciding_set_matches) * 100
    else:
        deciding_set_win_rate = 0.0
    
    return deciding_set_win_rate

def calculate_close_match_performance(matches_df, margin_threshold=3):
    """
    Calculate the win rate in close matches where win margin is <= margin_threshold.
    """
    try:
        matches_df['abs_win_margin'] = matches_df['win_margin'].abs()
        close_matches = matches_df[matches_df['abs_win_margin'] <= margin_threshold]
        total_close_matches = len(close_matches)
        close_matches_won = close_matches['isWinner'].sum()
        
        if total_close_matches > 0:
            close_match_win_rate = (close_matches_won / total_close_matches) * 100
        else:
            close_match_win_rate = 0.0
    except Exception as e:
        st.warning(f"Error calculating close match performance: {e}")
        close_match_win_rate = 0.0
    
    return close_match_win_rate

def calculate_performance_consistency(ratings_df):
    """
    Calculate the standard deviation of UTR ratings to assess performance consistency.
    Lower std dev indicates higher consistency.
    """
    try:
        if not ratings_df.empty and 'rating' in ratings_df.columns:
            rating_std_dev = ratings_df['rating'].std()
        else:
            rating_std_dev = 0.0
    except Exception as e:
        st.warning(f"Error calculating performance consistency: {e}")
        rating_std_dev = 0.0
    
    return rating_std_dev

def calculate_metrics_over_time(matches_df, ratings_df, window='M'):
    """
    Calculate mental resilience metrics over time using a specified rolling window.
    """
    try:
        # Ensure resultDate is datetime
        matches_df['resultDate'] = pd.to_datetime(matches_df['resultDate'], errors='coerce')
        ratings_df['date'] = pd.to_datetime(ratings_df['date'], errors='coerce')
        
        # Sort data by date
        matches_df = matches_df.sort_values('resultDate')
        ratings_df = ratings_df.sort_values('date')
        
        # Set index for resampling
        matches_df.set_index('resultDate', inplace=True)
        
        # Resample matches monthly
        monthly_matches = matches_df.resample(window)
        
        # Initialize lists to store metrics
        periods = []
        comeback_rates = []
        tiebreak_rates = []
        deciding_set_rates = []
        close_match_rates = []
        performance_consistencies = []
        
        # Loop through each monthly window
        for period, group in monthly_matches:
            if group.empty:
                continue
            
            # Calculate metrics
            comeback_rate = calculate_comeback_wins(group)
            tiebreak_rate = calculate_tiebreak_performance(group)
            deciding_set_rate = calculate_deciding_set_performance(group)
            close_match_rate = calculate_close_match_performance(group)
            
            # For consistency, use the ratings up to the current period
            current_ratings = ratings_df[ratings_df['date'] <= period]
            performance_consistency = calculate_performance_consistency(current_ratings)
            
            # Append to lists
            periods.append(period)
            comeback_rates.append(comeback_rate)
            tiebreak_rates.append(tiebreak_rate)
            deciding_set_rates.append(deciding_set_rate)
            close_match_rates.append(close_match_rate)
            performance_consistencies.append(performance_consistency)
        
        # Create a DataFrame with metrics
        metrics_df = pd.DataFrame({
            'Period': periods,
            'Comeback Win Rate (%)': comeback_rates,
            'Tiebreak Win Rate (%)': tiebreak_rates,
            'Deciding Set Win Rate (%)': deciding_set_rates,
            'Close Match Win Rate (%)': close_match_rates,
            'Performance Consistency (Std Dev of UTR)': performance_consistencies
        })
        
        return metrics_df
    except Exception as e:
        st.error(f"Error calculating metrics over time: {e}")
        return pd.DataFrame()

def visualize_mental_resilience_over_time(metrics_df):
    """
    Visualize mental resilience metrics over time.
    """
    st.subheader("Mental Resilience Over Time")
    
    if metrics_df.empty:
        st.warning("No metrics data available to display.")
        return
    
    # Melt the DataFrame for easier plotting with Plotly
    metrics_melted = metrics_df.melt(id_vars=['Period'], 
                                     value_vars=[
                                         'Comeback Win Rate (%)', 
                                         'Tiebreak Win Rate (%)', 
                                         'Deciding Set Win Rate (%)', 
                                         'Close Match Win Rate (%)'
                                     ],
                                     var_name='Metric',
                                     value_name='Value')
    
    # Line Chart for Win Rates
    fig = px.line(metrics_melted, x='Period', y='Value', color='Metric',
                  title='Mental Resilience Metrics Over Time',
                  labels={'Period': 'Time Period', 'Value': 'Percentage (%)'},
                  markers=True)
    
    fig.update_layout(xaxis_title='Time Period (Monthly)', yaxis_title='Percentage (%)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Line Chart for Performance Consistency
    fig_consistency = px.line(metrics_df, x='Period', y='Performance Consistency (Std Dev of UTR)',
                              title='Performance Consistency Over Time',
                              labels={'Period': 'Time Period (Monthly)', 
                                      'Performance Consistency (Std Dev of UTR)': 'Std Dev of UTR'})
    fig_consistency.update_layout(yaxis=dict(range=[0, metrics_df['Performance Consistency (Std Dev of UTR)'].max() + 1]))
    st.plotly_chart(fig_consistency, use_container_width=True)

def plot_mental_resilience_over_time():
    st.header("Mental Resilience Analysis")
    
    # Load matches and ratings data
    if not (os.path.exists("matches.csv") and os.path.exists("ratings_history.csv")):
        st.warning("Required data files not found.")
        return
    
    matches_df = pd.read_csv("matches.csv")
    ratings_df = pd.read_csv("ratings_history.csv")
    
    # Calculate metrics over time
    metrics_df = calculate_metrics_over_time(matches_df, ratings_df, window='M')  # 'M' for monthly
    
    # Visualize metrics
    visualize_mental_resilience_over_time(metrics_df)

# Scatter Plot Function
def plot_top_performances():
    st.subheader("Top Performances and Notable Matches")
    
    # Load matches data
    matches_df = load_matches_data()
    
    # Filter out matches without necessary data
    matches_df = matches_df.dropna(subset=['rating', 'win_margin'])
    
    # Create a new column for match significance
    matches_df['match_significance'] = matches_df['is_notable'].apply(lambda x: 'Notable' if x else 'Regular')
    
    # Sidebar filters
    st.sidebar.header("Scatter Plot Filters")
    min_rating = st.sidebar.slider("Minimum UTR Rating", 
                                   min_value=float(matches_df['rating'].min()), 
                                   max_value=float(matches_df['rating'].max()), 
                                   value=float(matches_df['rating'].min()))
    max_rating = st.sidebar.slider("Maximum UTR Rating", 
                                   min_value=float(matches_df['rating'].min()), 
                                   max_value=float(matches_df['rating'].max()), 
                                   value=float(matches_df['rating'].max()))
    selected_significance = st.sidebar.multiselect("Match Significance", 
                                                   options=['Notable', 'Regular'], 
                                                   default=['Notable', 'Regular'])
    
    # Apply filters
    filtered_df = matches_df[
        (matches_df['rating'] >= min_rating) &
        (matches_df['rating'] <= max_rating) &
        (matches_df['match_significance'].isin(selected_significance))
    ]
    
    # Scatter Plot
    fig = px.scatter(
        filtered_df,
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

@st.cache(allow_output_mutation=True)
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
                    st.subheader(f"Eligible Results - {category.capitalize()}")
                    eligible_df = pd.read_csv(filename)
                    st.dataframe(eligible_df.tail(5))
            
            if os.path.exists("rankings.csv") and rankings:
                st.subheader("Rankings")
                rankings_df = pd.read_csv("rankings.csv")
                st.dataframe(rankings_df.tail(5))
            
            # **Add Visualizations for Top Performances and Notable Matches**
            st.header("Insights and Visualizations")
            
            # Scatter Plot for Top Performances
            plot_top_performances()
            
            # Annotated Timeline for Notable Matches
            plot_annotated_timeline()
            
            # **Mental Resilience Over Time**
            plot_mental_resilience_over_time()

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
