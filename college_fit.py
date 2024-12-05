import streamlit as st
import pandas as pd
import json
from utr_auth import UTRAuthManager
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import logging
import time

class CollegeFitDashboard:
    def __init__(self):
        self.utr = UTRAuthManager()
        # Initialize the geocoder with a user agent
        self.geolocator = Nominatim(user_agent="college_fit_dashboard")
        
        # Initialize auth_manager in session state if not already present
        if 'auth_manager' not in st.session_state:
            st.session_state.auth_manager = self.utr
        
        # Configure logging
        logging.basicConfig(
            filename='college_fit_dashboard.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def search_utr_players(self, name):
        """
        Search for players by name using UTR API.
        """
        headers = self.utr.get_headers()
        url = "https://api.utrsports.net/v2/search/players"
        
        params = {
            'query': name,
            'limit': 10
        }
        
        try:
            response = self.utr.session.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error searching for players: {e}")
            logging.error(f"Error searching for players: {e}")
            return {}
    
    def process_search_results(self, data):
        """
        Process search results from UTR API into player data for dropdown.
        """
        if not data or not isinstance(data, dict):
            return []
        
        players = []
        for hit in data.get('hits', []):
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
    
    def create_search_section(self):
        """
        Create a search section allowing users to search by name or enter ID.
        """
        method = st.radio("Search Method", ["Search by Name", "Enter ID"], key="search_method")
        
        player_utr = None  # Initialize player_id
        
        if method == "Search by Name":
            search_query = st.text_input("Search for player", key="search_query")
            if search_query:
                # Search players using UTR API
                search_results = self.search_utr_players(search_query)
                
                players = self.process_search_results(search_results)
                if players:
                    options = [p['display_text'] for p in players]
                    player_map = {p['display_text']: p['utr'] for p in players}
                    selected = st.selectbox("Select player:", options, key="select_player")
                    if selected:
                        player_utr = player_map[selected]
        else:
            player_utr = st.text_input("Enter Player ID", value="", max_chars=10, key="enter_player_id")
        
        
        return player_utr
    
    def save_player_data(self, player_stats, filename="player_data.csv"):
        """
        Save player statistics to a CSV file.
        """
        try:
            df = pd.json_normalize(player_stats)
            df.to_csv(filename, index=False)
            st.success(f"Player data successfully saved to {filename}.")
            logging.info(f"Player data saved to {filename}.")
        except Exception as e:
            st.error(f"Error saving player data: {e}")
            logging.error(f"Error saving player data: {e}")
    
    def search_colleges(self, utr_rating, position=6, top=100):
        """
        Search colleges using UTR API based on fit rating.
        """
        headers = self.utr.get_headers()
        url = "https://api.utrsports.net/v2/search/colleges"
        
        params = {
            'top': top,
            'skip': 0,
            'utrType': 'verified',
            'utrTeamType': 'singles',
            'utrFitRating': utr_rating,
            'utrFitPosition': position,
            'schoolClubSearch': 'true',
            'sort': 'school.power6:desc'
        }
        
        try:
            response = self.utr.session.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error fetching college data: {e}")
            logging.error(f"Error fetching college data: {e}")
            return {}
    
    def process_college_data(self, data):
        """
        Process raw API response into a pandas DataFrame, capturing all available data fields.
        """
        if not data.get('hits'):
            st.error("No college data found in API response.")
            logging.error("No college data found in API response.")
            return pd.DataFrame()
            
        processed_data = []
        
        for hit in data.get('hits', []):
            try:
                source = hit.get('source', {})
                school = source.get('school', {})
                location = source.get('location', {})
                conference = school.get('conference', {})
                division = conference.get('division', {}) if conference else {}
                
                # Determine if the match is a doubles match
                # Adjust based on actual data structure
                # Here, assuming 'winner2Id' and 'loser2Id' are fields
                winner2_id = school.get('winner2Id', None)
                loser2_id = school.get('loser2Id', None)
                is_doubles = not (pd.isna(winner2_id) and pd.isna(loser2_id))
                
                entry = {
                    # School Basic Info
                    'College': school.get('displayName', ''),
                    'Short Name': school.get('shortName', ''),
                    'Nickname': school.get('nickname', ''),
                    'Alt Name 1': school.get('altName1', ''),
                    'Alt Name 2': school.get('altName2', ''),
                    'Alt Name 3': school.get('altName3', ''),
                    'Alt Name 4': school.get('altName4', ''),
                    'Alt Nickname 1': school.get('altNickname1', ''),
                    'Alt Nickname 2': school.get('altNickname2', ''),
                    'Lady Nickname': school.get('ladyNickname', ''),
                    'Private': 'Yes' if school.get('private', False) else 'No',
                    'School Type': school.get('type', ''),
                    
                    # UTR and Power Rankings
                    'Power 6': round(school.get('power6', 0), 2),
                    'Power 6 Avg': round(school.get('power6Avg', 0), 2),
                    'Power 6 High': round(school.get('power6High', 0), 2),
                    'Power 6 Low': round(school.get('power6Low', 0), 2),
                    'Power 6 Men': school.get('power6Men', np.nan),
                    'Power 6 Men High': school.get('power6MenHigh', np.nan),
                    'Power 6 Men Low': school.get('power6MenLow', np.nan),
                    'Power 6 Women': school.get('power6Women', np.nan),
                    'Power 6 Women High': school.get('power6WomenHigh', np.nan),
                    'Power 6 Women Low': school.get('power6WomenLow', np.nan),
                    
                    # Conference and Division
                    'Conference ID': conference.get('id', ''),
                    'Conference Name': conference.get('conferenceName', ''),
                    'Conference Short': conference.get('shortName', ''),
                    'Division ID': division.get('id', ''),
                    'Division Name': division.get('divisionName', ''),
                    'Division Short': division.get('shortName', ''),
                    
                    # Location Details
                    'City': location.get('cityName', ''),
                    'City Abbr': location.get('cityAbbr', ''),
                    'State': location.get('stateAbbr', ''),
                    'State Full': location.get('stateName', ''),
                    'Country': location.get('countryName', ''),
                    'Country Code': location.get('countryCode2', ''),
                    'Location Display': location.get('display', ''),
                    'Latitude': location.get('latLng', [None, None])[0],
                    'Longitude': location.get('latLng', [None, None])[1],
                    'Street Address': location.get('streetAddress', ''),
                    'City State Zip': location.get('cityStateZip', ''),
                    
                    # Club/Team Details
                    'Club ID': source.get('id', ''),
                    'Team Name': source.get('name', ''),
                    'Team Gender': source.get('gender', ''),
                    'Member Count': source.get('memberCount', 0),
                    'Event Count': source.get('eventCount', 0),
                    'Is Private': source.get('private', False),
                    'Is College': source.get('isCollege', False),
                    'Is High School': source.get('isHighSchool', False),
                    'Team Website': source.get('url', ''),
                    'Sanctioned': source.get('sanctioned', False),
                    'Can Run Events': source.get('canRunEvents', False),
                    'Tier Type ID': source.get('tierTypeId', ''),
                    'Club Sub Type ID': source.get('clubSubTypeId', ''),
                    
                    # Image URLs
                    'Profile Photo URL': source.get('profilePhotoUrl', ''),
                    'Banner URL': source.get('bannerUrl', ''),
                    
                    # Additional School Info
                    'Roster Count': school.get('rosterCount', 0),
                    'Roster Year': school.get('rosterYear', ''),
                    'Roster Has Unclaimed Players': school.get('rosterHasUnclaimedPlayers', False),
                    
                    # Sort Value from API
                    'API Sort Value': hit.get('sorts', [None])[0],
                    
                    # Doubles Flag
                    'is_doubles': is_doubles
                }
                
                # Append to processed_data
                processed_data.append(entry)
                
            except Exception as e:
                match_id = hit.get('id', 'Unknown')
                st.error(f"Error processing college entry (Match ID: {match_id}): {e}")
                logging.error(f"Error processing college entry (Match ID: {match_id}): {e}")
                continue
        
        df = pd.DataFrame(processed_data)
        
        # Add debug information
        st.write("Number of colleges processed:", len(df))
        logging.info(f"Number of colleges processed: {len(df)}")
        if len(df) == 0:
            st.write("Raw API response:", data)
            logging.warning("Processed DataFrame is empty. Displaying raw API response.")
            
        return df
    
    def normalize(self, series):
        """
        Normalize a pandas Series to a 0-1 range.
        """
        if series.max() == series.min():
            return series.apply(lambda x: 0.5)  # Avoid division by zero; assign neutral score
        return (series - series.min()) / (series.max() - series.min())
    
    def geocode_address(self, address):
        """
        Geocode an address to (latitude, longitude).
        """
        retries = 3
        for attempt in range(retries):
            try:
                location = self.geolocator.geocode(address, timeout=10)
                if location:
                    return (location.latitude, location.longitude)
                else:
                    st.error("Geocoding failed: Address not found.")
                    return None
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                st.warning(f"Geocoding attempt {attempt + 1} failed: {e}")
                logging.warning(f"Geocoding attempt {attempt + 1} failed: {e}")
                time.sleep(1)  # Wait before retrying
        st.error("Geocoding failed after multiple attempts.")
        logging.error("Geocoding failed after multiple attempts.")
        return None
    
    def calculate_fit_score(self, df, athlete_utr, player_location, weights):
        """
        Calculate the Fit Score for each college based on defined criteria and weights.
        """
        # Normalize weights internally
        total_weight = weights['Athletic'] + weights['Location'] + weights['Culture']
        if total_weight == 0:
            st.warning("All weights are set to zero. Assigning equal weights to all criteria.")
            logging.warning("All weights are zero. Assigning equal weights to all criteria.")
            weights_normalized = {'Athletic': 1/3, 'Location': 1/3, 'Culture': 1/3}
        else:
            weights_normalized = {k: v / total_weight for k, v in weights.items()}
        
        # Display normalized weights for user information
        st.sidebar.markdown("### **Normalized Weights**")
        st.sidebar.write(f"Athletic Development: {weights_normalized['Athletic']*100:.1f}%")
        st.sidebar.write(f"Geographic Location: {weights_normalized['Location']*100:.1f}%")
        st.sidebar.write(f"Campus Culture: {weights_normalized['Culture']*100:.1f}%")
        
        # Athletic Score based on Power 6
        df['Athletic Score'] = self.normalize(df['Power 6'])
        
        # Adjust Athletic Score based on Player's UTR Fit
        df['UTR Difference Low'] = abs(athlete_utr - df['Power 6 Low'])
        df['UTR Difference High'] = abs(athlete_utr - df['Power 6 High'])
        avg_utr_diff = (df['UTR Difference Low'] + df['UTR Difference High']) / 2
        df['UTR Fit Score'] = 1 - self.normalize(avg_utr_diff)
        df['Athletic Score'] = (df['Athletic Score'] + df['UTR Fit Score']) / 2  # Average the two scores
        
        # Location Score based on distance from player's home
        def calculate_distance(row):
            college_location = (row['Latitude'], row['Longitude'])
            if None in college_location:
                return None
            return geodesic(player_location, college_location).miles
        
        df['Distance'] = df.apply(calculate_distance, axis=1)
        # Handle missing distances by assigning a high distance (low score)
        max_distance = df['Distance'].max()
        df['Distance'].fillna(max_distance, inplace=True)
        df['Location Score'] = 1 - self.normalize(df['Distance'])  # Closer colleges score higher
        
        # Culture Score based on School Type
        # Example: Assign 1 for private, 0 for public
        df['Culture Score'] = df['Private'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        # Combine scores based on normalized weights
        df['Fit Score'] = (
            df['Athletic Score'] * weights_normalized['Athletic'] +
            df['Location Score'] * weights_normalized['Location'] +
            df['Culture Score'] * weights_normalized['Culture']
        )
        
        # Handle any NaN Fit Scores by assigning 0
        df['Fit Score'].fillna(0, inplace=True)
        
        return df
    
    def calculate_total_games(self, winner_sets, loser_sets):
        """
        Calculate the total game margin for the player.
        Positive value indicates a win margin, negative indicates a loss margin.
        """
        total_winner = sum(winner_sets)
        total_loser = sum(loser_sets)
        return total_winner - total_loser
    
    def create_power6_chart(self, df):
        """
        Create a customized bar chart for Power 6 ranges using Plotly.
        """
        if df.empty:
            st.warning("No data available to display in Power 6 Analysis.")
            logging.warning("No data available to display in Power 6 Analysis.")
            return
        
        # Sort data by Power 6 High
        chart_data = df.sort_values('Power 6 High', ascending=False).head(15)
        
        fig = go.Figure()
        
        # Add the Power 6 Low bars
        fig.add_trace(go.Bar(
            name='Power 6 Low',
            x=chart_data['College'],
            y=chart_data['Power 6 Low'],
            marker_color='lightblue'
        ))
        
        # Add the Power 6 High - Low difference
        fig.add_trace(go.Bar(
            name='Range',
            x=chart_data['College'],
            y=chart_data['Power 6 High'] - chart_data['Power 6 Low'],
            base=chart_data['Power 6 Low'],
            marker_color='royalblue'
        ))
        
        # Update the layout
        fig.update_layout(
            title='Team Power 6 Ranges',
            barmode='stack',
            showlegend=True,
            xaxis_tickangle=-45,
            yaxis=dict(
                title='Power 6 Rating',
                range=[8, 16],  # Set fixed range for UTR scale
                dtick=1,  # Show tick marks for every 1.0
            ),
            height=500,
            margin=dict(b=150)  # Increased bottom margin for rotated labels
        )
        
        return fig
    
    def create_college_map(self, df, player_location):
        """
        Create an interactive map of colleges and player's home using Plotly.
        """
        if df.empty:
            st.warning("No data available to display on the College Map.")
            logging.warning("No data available to display on the College Map.")
            return
        
        # Remove entries with missing coordinates
        df_map = df.dropna(subset=['Latitude', 'Longitude'])
        
        if df_map.empty:
            st.warning("No college locations available to display on the map.")
            logging.warning("No college locations available to display on the map.")
            return
        
        # Normalize 'Power 6 Avg' for marker sizes
        size_min = 2
        size_max = 15
        df_map['Size'] = self.normalize(df_map['Power 6 Avg']) * (size_max - size_min) + size_min
        
        # Create hover text for colleges
        df_map['hover_text'] = df_map.apply(lambda row: f"""
            <b>{row['College']}</b><br>
            Power 6: {row['Power 6']:.2f}<br>
            Range: {row['Power 6 Low']:.2f} - {row['Power 6 High']:.2f}<br>
            Conference: {row['Conference Name']}<br>
            Division: {row['Division Name']}<br>
            Fit Score: {row['Fit Score']:.2f}
        """, axis=1)
        
        # Create color scale based on Fit Score
        fig = px.scatter_mapbox(
            df_map,
            lat='Latitude',
            lon='Longitude',
            hover_name='College',
            hover_data={
                'Latitude': False,
                'Longitude': False,
                'Power 6': ':.2f',
                'Conference Name': True,
                'Division Name': True,
                'Fit Score': ':.2f'
            },
            color='Fit Score',
            color_continuous_scale='viridis',
            size='Size',  # Size proportional to Power 6 Avg
            size_max=size_max,
            zoom=3,
            center={'lat': 39.8283, 'lon': -98.5795},  # Center of USA
            title='College Tennis Programs by Fit Score'
        )
        
        # Add player's home location as a separate trace
        if player_location:
            home_hover_text = "Your Home Location"
            home_lat, home_lon = player_location
            fig.add_trace(go.Scattermapbox(
                lat=[home_lat],
                lon=[home_lon],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=15,
                    color='red',
                    symbol='star'  # Different symbol
                ),
                hoverinfo='text',
                hovertext=home_hover_text,
                name='Home Location'
            ))
        
        # Update map style and layout
        fig.update_layout(
            mapbox_style='carto-positron',
            height=600,
            margin={'r': 0, 'l': 0, 'b': 0, 't': 50},
            showlegend=True
        )
        
        return fig
    
    def save_matches_data(self, matches_data, player_id, filename="matches.csv"):
        """
        Saves match data involving the specified player to a CSV file.
        """
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
                                logging.warning(f"Invalid players data for match ID {match_id}. Skipping match.")
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
                                if not pd.isna(loser2_utr):
                                    opponent_utr = (loser1_utr + loser2_utr) / 2
                                else:
                                    opponent_utr = loser1_utr
                            elif is_loser:
                                # Player is the loser; opponents are winners
                                if not pd.isna(winner2_utr):
                                    opponent_utr = (winner1_utr + winner2_utr) / 2
                                else:
                                    opponent_utr = winner1_utr
                            else:
                                opponent_utr = np.nan  # Should not occur

                            # Handle cases where opponent UTR is NaN
                            if pd.isna(opponent_utr):
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
                                'win_margin': self.calculate_total_games(winner_sets, loser_sets),
                                'opponent_utr': opponent_utr,  # Adding opponent UTR
                                'is_doubles': is_doubles,      # Adding is_doubles flag
                                'is_notable': False  # Placeholder, can be updated based on criteria
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

                            # Determine if the match is notable
                            is_notable = False
                            notable_events = ['US Open', 'Australian Open', 'French Open', 'Wimbledon']
                            high_margin_threshold = 10
                            if new_row['eventName'] in notable_events or abs(new_row['win_margin']) >= high_margin_threshold:
                                is_notable = True
                            new_row['is_notable'] = is_notable

                            # Append to processed_data
                            processed_data.append(new_row)

                        except Exception as e:
                            # Include match ID in the warning for easier debugging
                            match_id = match.get('id', 'Unknown')
                            st.error(f"Error processing match ID {match_id}: {e}")
                            logging.error(f"Error processing match ID {match_id}: {e}")
                            continue
        except Exception as e:
            st.error(f"Error retrieving player UTR: {e}")
            logging.error(f"Error retrieving player UTR: {e}")
            return None

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
                logging.warning("Duplicate match entries found. Removing duplicates.")
                matches_df = matches_df.drop_duplicates(subset=['matchId'])

            # Verify 'isWinner' flags are correctly set
            # For example, ensure that 'win_margin' is positive for wins and negative for losses
            inconsistent_flags = matches_df[
                ((matches_df['isWinner'] == True) & (matches_df['win_margin'] < 0)) |
                ((matches_df['isWinner'] == False) & (matches_df['win_margin'] > 0))
            ]
            if not inconsistent_flags.empty:
                st.warning("Inconsistent 'isWinner' flags found. These matches will be removed.")
                logging.warning("Inconsistent 'isWinner' flags found. Removing inconsistent matches.")
                matches_df = matches_df.drop(inconsistent_flags.index)

            # Save to CSV with appropriate data types
            try:
                matches_df.to_csv(filename, index=False)
                st.success(f"Match data successfully saved to {filename}.")
                logging.info(f"Match data saved to {filename}.")
            except Exception as e:
                st.error(f"Error saving match data to CSV: {e}")
                logging.error(f"Error saving match data to CSV: {e}")
        else:
            st.warning("No valid matches found involving the player.")
            logging.warning("No valid matches found involving the player.")

    def run(self):
        """
        Main method to run the Streamlit app.
        """
        st.title("College Tennis Fit Analysis")
        
        # Sidebar for athlete inputs and preferences
        with st.sidebar:
            st.header("Athlete Information")
            
            # Create search section
            player_utr = self.create_search_section()
            athlete_utr = player_utr
            
            # Desired Position
            desired_position = st.slider(
                "Desired Position on Team",
                min_value=1,
                max_value=8,
                value=6,
                help="1 = Top of lineup, 8 = Bottom of lineup"
            )
            
            # Player's Home Location Input as Address
            st.subheader("Your Home Address")
            home_address = st.text_input(
                "Enter your home address",
                value="1600 Amphitheatre Parkway, Mountain View, CA",
                help="Enter your full home address (e.g., 1600 Amphitheatre Parkway, Mountain View, CA)"
            )
            st.header("Filters")
            
            # Conference Filter
            conferences = [
                "All Conferences",
                "Atlantic Coast Conference",
                "Big 12 Conference",
                "Big Ten Conference",
                "Pac-12 Conference",
                "Southeastern Conference"
            ]
            conference_filter = st.selectbox(
                "Filter by Conference",
                conferences
            )
            
            # Private/Public Filter
            school_type = st.radio(
                "School Type",
                ["All", "Private Only", "Public Only"]
            )
            
            # State Filter
            state_filter = st.text_input(
                "Filter by State (optional)",
                help="Enter state abbreviation (e.g., CA, NY)"
            )
            
            # Weight Inputs
            st.subheader("Fit Score Weights")
            athletic_weight = st.slider(
                "Athletic Development Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
            location_weight = st.slider(
                "Geographic Location Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1
            )
            culture_weight = st.slider(
                "Campus Culture Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1
            )
            
            # Option to include/exclude doubles matches
            include_doubles = st.checkbox("Include Doubles Matches", value=True)
        
            # Button to find matches
            find_matches = st.button("Find Matches")
        
        if find_matches:
            if player_utr:
                # User searched by name and selected a player
                pass  # Proceed as below
            else:
                # User entered player ID manually
                pass  # Proceed as below
            
            # Validate inputs
            if not home_address.strip():
                st.error("Please enter a valid home address.")
                return
            
            try:
                with st.spinner("Processing..."):
                    # Geocode the home address
                    player_location = self.geocode_address(home_address)
                    
                    if player_location is None:
                        st.error("Unable to geocode the provided address. Please check and try again.")
                        return
                    
                    # Get college data from UTR API
                    colleges_data = self.search_colleges(
                        utr_rating=athlete_utr,
                        position=desired_position
                    )
                    
                    # Process data
                    df = self.process_college_data(colleges_data)
                    
                    # Apply filters
                    if conference_filter != "All Conferences":
                        df = df[df['Conference Name'] == conference_filter]
                    
                    if state_filter:
                        df = df[df['State'].str.contains(state_filter.upper(), na=False)]
                        
                    if school_type == "Private Only":
                        df = df[df['Private'] == 'Yes']
                    elif school_type == "Public Only":
                        df = df[df['Private'] == 'No']
                    
                    if df.empty:
                        st.warning("No college data found for the given UTR rating and position.")
                        return
                    
                    # Calculate Fit Scores
                    weights = {
                        'Athletic': athletic_weight,
                        'Location': location_weight,
                        'Culture': culture_weight
                    }
                    df = self.calculate_fit_score(df, athlete_utr, player_location, weights)
                    
                    # Filter based on doubles matches
                    if not include_doubles:
                        df = df[df['is_doubles'] == False]
                        st.write("### Note: Doubles matches are excluded from the visualization.")
                    
                    # Drop entries with Fit Score of 0 (optional)
                    df = df[df['Fit Score'] > 0]
                    
                    # Sort by Fit Score
                    df = df.sort_values(by='Fit Score', ascending=False)
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Matches", len(df))
                    with col2:
                        avg_fit_score = df['Fit Score'].mean()
                        st.metric("Average Fit Score", f"{avg_fit_score:.2f}")
                    with col3:
                        top_fit = df['Fit Score'].max()
                        st.metric("Highest Fit Score", f"{top_fit:.2f}")
                    
                    # Display results in tabs
                    tab1, tab2, tab3 = st.tabs(["Top Matches", "College Map", "Power 6 Analysis"])
                    
                    with tab1:
                        st.subheader("Top College Matches")
                        # Display top 10 matches
                        st.dataframe(
                            df[['College', 'Fit Score', 'Power 6', 'City', 'State']].head(10).reset_index(drop=True)
                        )
                        
                        # Optional: Allow users to download the top matches
                        st.download_button(
                            label="Download Top Matches as CSV",
                            data=df[['College', 'Fit Score', 'Power 6', 'City', 'State']].head(10).to_csv(index=False),
                            file_name='top_college_matches.csv',
                            mime='text/csv',
                        )
                    
                    with tab2:
                        st.subheader("College Locations Map")
                        st.plotly_chart(self.create_college_map(df, player_location), use_container_width=True)
                    
                    with tab3:
                        st.subheader("Team UTR Analysis")
                        st.plotly_chart(self.create_power6_chart(df), use_container_width=True)
        
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                logging.error(f"Unexpected error: {e}")
    
    def get_player_utr(self, player_id):
        """
        Retrieve the UTR rating for the given player ID.
        """
        headers = st.session_state.auth_manager.get_headers()
        url = f"https://api.utrsports.net/v4/player/{player_id}/results"
        
        try:
            response = st.session_state.auth_manager.session.get(url, headers=headers)
            response.raise_for_status()
            player_data = response.json()
            
            # Adjust the key based on actual API response structure
            # For example, if 'singlesUtr' is nested
            singles_utr = player_data.get('singlesUtr', None)
            
            if singles_utr is None:
                st.warning(f"No 'singlesUtr' found for player ID {player_id}.")
                logging.warning(f"No 'singlesUtr' found for player ID {player_id}.")
            
            return singles_utr
        
        except Exception as e:
            st.error(f"Error retrieving player UTR: {e}")
            logging.error(f"Error retrieving player UTR for player ID {player_id}: {e}")
            return None


if __name__ == "__main__":
    dashboard = CollegeFitDashboard()
    dashboard.run()
