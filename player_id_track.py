import streamlit as st
import pandas as pd
from utr_auth import UTRAuthManager
import json
from datetime import datetime
import os
from dotenv import load_dotenv
import sqlite3
from sqlite3 import Error
import pydeck as pdk
import numpy as np
import logging

# Load environment variables from .env if needed
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure Streamlit page
st.set_page_config(page_title="Tennis Dashboard", layout="wide")


class TennisDashboard:
    def __init__(self):
        self.utr = UTRAuthManager()
        self.countries = self.load_countries()
        self.db_path = "players.db"
        self.setup_database()
        # Initialize auth_manager in session state
        if 'auth_manager' not in st.session_state:
            st.session_state.auth_manager = self.utr

    def load_countries(self):
        """Loads country data from the countries.json file."""
        try:
            with open("countries.json", "r", encoding="utf-8") as f:
                countries = json.load(f)
            return countries
        except FileNotFoundError:
            st.error("The 'countries.json' file was not found.")
            return []
        except json.JSONDecodeError as e:
            st.error(f"Error decoding JSON: {e}")
            return []

    def setup_database(self):
        """Sets up the SQLite database and creates the players table if it doesn't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS players (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    utr_id TEXT UNIQUE,
                    name TEXT,
                    country_code TEXT,
                    college TEXT,
                    division TEXT,
                    conference TEXT,
                    utr FLOAT,
                    last_active TEXT,
                    last_updated TEXT
                )
            """)
            conn.commit()
            conn.close()
        except Error as e:
            st.error(f"Database error: {e}")
            logging.error(f"Database error: {e}")

    def get_college_locations(self, top=100):
        """
        Fetches all colleges from the UTR API and extracts their geographical coordinates.
        """
        colleges = []
        skip = 0
        
        while True:
            try:
                params = {
                    'top': top,
                    'skip': skip,
                    'primaryTags': 'College',
                    'utrType': 'verified',
                    'utrTeamType': 'singles',
                    'schoolClubSearch': 'true',
                    'searchOrigin': 'searchPage'
                }
                response = self.utr.session.get("https://api.utrsports.net/v2/search/colleges", 
                                            headers=self.utr.get_headers(), 
                                            params=params)
                response.raise_for_status()
                data = response.json()
                hits = data.get('hits', [])
                if not hits:
                    break

                for hit in hits:
                    try:
                        source = hit.get('source')
                        if not source:
                            continue
                            
                        location = source.get('location')
                        if not location:
                            continue
                            
                        school = source.get('school')
                        if not school:
                            continue
                        
                        lat_lng = location.get('latLng')
                        if not lat_lng or len(lat_lng) != 2:
                            continue
                            
                        college_name = school.get('displayName')
                        if not college_name:
                            continue
                        
                        colleges.append({
                            'College': college_name.strip(),
                            'Latitude': lat_lng[0],
                            'Longitude': lat_lng[1],
                            'City': location.get('cityName', ''),
                            'State': location.get('stateAbbr', '')
                        })
                        
                    except Exception as e:
                        logging.error(f"Error processing college entry: {e}")
                        continue
                        
                skip += top
                
                # Break if we've processed all available colleges
                if len(hits) < top:
                    break
                    
            except Exception as e:
                logging.error(f"Error fetching college locations: {e}")
                break
                
        df_colleges = pd.DataFrame(colleges)
        logging.info(f"Total colleges fetched: {len(df_colleges)}")
        return df_colleges

    def fetch_country_players(self, nationality, top=100):
        """
        Fetches player data from the UTR platform for a specific country and merges it with college locations.
        
        Args:
            nationality (str): ISO 3166-1 alpha-3 country code (e.g., 'ARG')
            top (int): Maximum number of players to fetch
        
        Returns:
            pd.DataFrame: DataFrame containing player information along with college coordinates
        """
        try:
            players_response = st.session_state.auth_manager.search_players_advanced(
                top=top,
                skip=0,
                primary_tags='College',
                nationality=nationality,
                utr_type="verified",
                utr_team_type="singles",
                show_tennis=True,
                show_pickleball=False
            )

            hits = players_response.get('hits', [])
            if not hits:
                return pd.DataFrame()

            player_data = []
            for hit in hits:
                source = hit.get('source', {})
                player_info = {
                    'UTR ID': source.get('id', 'N/A'),  # Unique identifier
                    'Name': f"{source.get('firstName', '')} {source.get('lastName', '')}".strip(),
                    'UTR': source.get('singlesUtr', 'N/A'),
                    'College': source.get('playerCollege', {}).get('name', 'N/A'),
                    'Division': source.get('playerCollege', {}).get('conference', {}).get('shortName', 'N/A'),
                    'Conference': source.get('playerCollege', {}).get('conference', {}).get('conferenceName', 'N/A'),
                    'Last Active': datetime.fromtimestamp(source.get('lastActive', 0)).strftime('%Y-%m-%d') if source.get('lastActive') else 'N/A'
                }
                player_data.append(player_info)

            df_players = pd.DataFrame(player_data)

            # Fetch college locations
            df_colleges = self.get_college_locations(top=100)

            # Merge player data with college locations
            df_merged = pd.merge(df_players, df_colleges, on='College', how='left')

            # Convert UTR to numeric for aggregation and visualization
            df_merged['UTR_numeric'] = pd.to_numeric(df_merged['UTR'], errors='coerce')

            # Update the database with merged data (including lat/lon if necessary)
            self.update_database(df_merged, nationality)

            return df_merged

        except Exception as e:
            st.error(f"Error fetching data: {e}")
            logging.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    def update_database(self, df, country_code):
        """Inserts or updates player data in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            for _, row in df.iterrows():
                cursor.execute("""
                    INSERT INTO players (utr_id, name, country_code, college, division, conference, utr, last_active, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(utr_id) DO UPDATE SET
                        name=excluded.name,
                        country_code=excluded.country_code,
                        college=excluded.college,
                        division=excluded.division,
                        conference=excluded.conference,
                        utr=excluded.utr,
                        last_active=excluded.last_active,
                        last_updated=excluded.last_updated
                """, (
                    row.get('UTR ID', 'N/A'),
                    row.get('Name', 'N/A'),
                    country_code,
                    row.get('College', 'N/A'),
                    row.get('Division', 'N/A'),
                    row.get('Conference', 'N/A'),
                    float(row.get('UTR', 0)) if row.get('UTR', 'N/A') != 'N/A' else None,
                    row.get('Last Active', 'N/A'),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))
            conn.commit()
            conn.close()
        except Error as e:
            st.error(f"Database update error: {e}")
            logging.error(f"Database update error: {e}")

    def get_stored_data(self, country_code):
        """Retrieves stored player data from the database for a specific country."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name, utr, college, division, conference, last_active, last_updated
                FROM players
                WHERE country_code=?
            """, (country_code,))
            rows = cursor.fetchall()
            conn.close()
            df = pd.DataFrame(rows, columns=['Name', 'UTR', 'College', 'Division', 'Conference', 'Last Active', 'Last Updated'])
            return df
        except Error as e:
            st.error(f"Database retrieval error: {e}")
            logging.error(f"Database retrieval error: {e}")
            return pd.DataFrame()

    def display_players_table(self, df):
        """Displays player data in a table with metrics and export functionality."""
        if df.empty:
            st.warning("No player data available.")
        else:
            st.markdown("### Player Rankings")
            st.dataframe(df.sort_values(by='UTR_numeric', ascending=False).reset_index(drop=True))

            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Players", len(df))
            with col2:
                # Handle 'N/A' values for UTR
                average_utr = df['UTR_numeric'].mean()
                average_utr_display = f"{average_utr:.2f}" if pd.notna(average_utr) else 'N/A'
                st.metric("Average UTR", average_utr_display)
            with col3:
                st.metric("Active Colleges", df['College'].nunique())

            # Sorting options
            st.markdown("### Player List")
            sort_by = st.selectbox("Sort by", options=['Name', 'UTR', 'College', 'Division', 'Conference', 'Last Active'], key='sort_by_players')
            ascending = st.checkbox("Ascending", True, key='ascending_players')

            sorted_df = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
            st.dataframe(sorted_df)

            # Export to CSV
            csv = sorted_df.to_csv(index=False).encode('utf-8')
            filename = f"{df['College'].iloc[0][:3].upper()}_players_{datetime.now().strftime('%Y%m%d')}.csv" if not df.empty else "players.csv"

            st.download_button(
                label="Export to CSV",
                data=csv,
                file_name=filename,
                mime="text/csv",
                key='download-csv-players'
            )
            st.success("Data exported successfully!")

    def display_players_map(self, df):
        """
        Displays players on a map with markers sized by the number of players at each college
        and colored by average UTR.
        
        Args:
            df (pd.DataFrame): DataFrame containing player information along with college coordinates
        """
        if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
            st.error("DataFrame missing 'Latitude' or 'Longitude' columns.")
            return

        # Drop rows with missing coordinates
        df = df.dropna(subset=['Latitude', 'Longitude'])

        if df.empty:
            st.warning("No player data with valid location information to display on the map.")
            return

        # Aggregate by College, Latitude, Longitude
        aggregated_df = df.groupby(['College', 'Latitude', 'Longitude']).agg(
            Player_Count=('Name', 'count'),
            Average_UTR=('UTR_numeric', 'mean'),
            City=('City', 'first'),
            State=('State', 'first')
        ).reset_index()

        # Normalize UTR for color scaling
        min_utr = aggregated_df['Average_UTR'].min()
        max_utr = aggregated_df['Average_UTR'].max()
        aggregated_df['UTR_normalized'] = (aggregated_df['Average_UTR'] - min_utr) / (max_utr - min_utr) if max_utr != min_utr else 0.5

        # Function to map normalized UTR to RGB color
        def get_color(utr_norm):
            """
            Maps a normalized UTR value (0 to 1) to a color.
            0 -> Red, 1 -> Green, intermediate values interpolate between.
            """
            red = int((1 - utr_norm) * 255)
            green = int(utr_norm * 255)
            blue = 0
            return [red, green, blue]

        aggregated_df['Color'] = aggregated_df['UTR_normalized'].apply(get_color)

        # Define the layer
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=aggregated_df,
            get_position='[Longitude, Latitude]',
            auto_highlight=True,
            get_radius='Player_Count * 10',  # Adjust the multiplier as needed
            get_fill_color='Color',
            pickable=True
        )

        # Set the viewport location
        initial_view_state = pdk.ViewState(
            latitude=aggregated_df['Latitude'].mean(),
            longitude=aggregated_df['Longitude'].mean(),
            zoom=3,
            pitch=0
        )

        # Define the map
        r = pdk.Deck(
            layers=[layer],
            initial_view_state=initial_view_state,
            tooltip={
                "html": "<b>College:</b> {College}<br/>"
                        "<b>Player Count:</b> {Player_Count}<br/>"
                        "<b>Average UTR:</b> {Average_UTR:.2f}<br/>"
                        "<b>Location:</b> {City}, {State}",
                "style": {"color": "white"}
            }
        )

        st.pydeck_chart(r)

    def search_colleges(self, utr_rating, position=6, top=100):
        """
        Search colleges using UTR API based on fit rating.
        
        Args:
            utr_rating (float): Minimum UTR fit rating to filter colleges
            position (int): Position parameter for the search
            top (int): Number of records to fetch per API call
        
        Returns:
            dict: JSON response from the API
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
        
        Args:
            data (dict): JSON response from the API
        
        Returns:
            pd.DataFrame: DataFrame containing processed college data
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

                # Extract necessary fields
                college_name = school.get('displayName', '').strip()
                student_count = source.get('memberCount', 0)
                utr = school.get('power6Avg', 0)
                latitude, longitude = location.get('latLng', [None, None])

                # Append only if location data is available
                if college_name and latitude and longitude:
                    entry = {
                        'College': college_name,
                        'Student Count': student_count,
                        'UTR': utr,
                        'Latitude': latitude,
                        'Longitude': longitude,
                        'City': location.get('cityName', ''),
                        'State': location.get('stateAbbr', ''),
                        'Country': location.get('countryName', '')
                    }
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

    def display_players_table(self, df):
        """Displays player data in a table with metrics and export functionality."""
        if df.empty:
            st.warning("No player data available.")
        else:
            st.markdown("### Player Rankings")
            st.dataframe(df.sort_values(by='UTR_numeric', ascending=False).reset_index(drop=True))

            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Players", len(df))
            with col2:
                # Handle 'N/A' values for UTR
                average_utr = df['UTR_numeric'].mean()
                average_utr_display = f"{average_utr:.2f}" if pd.notna(average_utr) else 'N/A'
                st.metric("Average UTR", average_utr_display)
            with col3:
                st.metric("Active Colleges", df['College'].nunique())

            # Sorting options
            st.markdown("### Player List")
            sort_by = st.selectbox("Sort by", options=['Name', 'UTR', 'College', 'Division', 'Conference', 'Last Active'], key='sort_by_players')
            ascending = st.checkbox("Ascending", True, key='ascending_players')

            sorted_df = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
            st.dataframe(sorted_df)

            # Export to CSV
            csv = sorted_df.to_csv(index=False).encode('utf-8')
            filename = f"{df['College'].iloc[0][:3].upper()}_players_{datetime.now().strftime('%Y%m%d')}.csv" if not df.empty else "players.csv"

            st.download_button(
                label="Export to CSV",
                data=csv,
                file_name=filename,
                mime="text/csv",
                key='download-csv-players'
            )
            st.success("Data exported successfully!")

    def display_players_map(self, df):
        """
        Displays players on a map with markers sized by the number of players at each college
        and colored by average UTR.
        
        Args:
            df (pd.DataFrame): DataFrame containing player information along with college coordinates
        """
        if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
            st.error("DataFrame missing 'Latitude' or 'Longitude' columns.")
            return

        # Drop rows with missing coordinates
        df = df.dropna(subset=['Latitude', 'Longitude'])

        if df.empty:
            st.warning("No player data with valid location information to display on the map.")
            return

        # Aggregate by College, Latitude, Longitude
        aggregated_df = df.groupby(['College', 'Latitude', 'Longitude']).agg(
            Player_Count=('Name', 'count'),
            Average_UTR=('UTR_numeric', 'mean'),
            City=('City', 'first'),
            State=('State', 'first')
        ).reset_index()

        # Normalize UTR for color scaling
        min_utr = aggregated_df['Average_UTR'].min()
        max_utr = aggregated_df['Average_UTR'].max()
        aggregated_df['UTR_normalized'] = (aggregated_df['Average_UTR'] - min_utr) / (max_utr - min_utr) if max_utr != min_utr else 0.5

        # Function to map normalized UTR to RGB color
        def get_color(utr_norm):
            """
            Maps a normalized UTR value (0 to 1) to a color.
            0 -> Red, 1 -> Green, intermediate values interpolate between.
            """
            red = int((1 - utr_norm) * 255)
            green = int(utr_norm * 255)
            blue = 0
            return [red, green, blue]

        aggregated_df['Color'] = aggregated_df['UTR_normalized'].apply(get_color)

        # Define the layer
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=aggregated_df,
            get_position='[Longitude, Latitude]',
            auto_highlight=True,
            get_radius='Player_Count * 10',  # Adjust the multiplier as needed
            get_fill_color='Color',
            pickable=True
        )

        # Set the viewport location
        initial_view_state = pdk.ViewState(
            latitude=aggregated_df['Latitude'].mean(),
            longitude=aggregated_df['Longitude'].mean(),
            zoom=3,
            pitch=0
        )

        # Define the map
        r = pdk.Deck(
            layers=[layer],
            initial_view_state=initial_view_state,
            tooltip={
                "html": "<b>College:</b> {College}<br/>"
                        "<b>Player Count:</b> {Player_Count}<br/>"
                        "<b>Average UTR:</b> {Average_UTR:.2f}<br/>"
                        "<b>Location:</b> {City}, {State}",
                "style": {"color": "white"}
            }
        )

        st.pydeck_chart(r)

    def search_colleges(self, utr_rating, position=6, top=100):
        """
        Search colleges using UTR API based on fit rating.
        
        Args:
            utr_rating (float): Minimum UTR fit rating to filter colleges
            position (int): Position parameter for the search
            top (int): Number of records to fetch per API call
        
        Returns:
            dict: JSON response from the API
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
        
        Args:
            data (dict): JSON response from the API
        
        Returns:
            pd.DataFrame: DataFrame containing processed college data
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

                # Extract necessary fields
                college_name = school.get('displayName', '').strip()
                student_count = source.get('memberCount', 0)
                utr = school.get('power6Avg', 0)
                latitude, longitude = location.get('latLng', [None, None])

                # Append only if location data is available
                if college_name and latitude and longitude:
                    entry = {
                        'College': college_name,
                        'Student Count': student_count,
                        'UTR': utr,
                        'Latitude': latitude,
                        'Longitude': longitude,
                        'City': location.get('cityName', ''),
                        'State': location.get('stateAbbr', ''),
                        'Country': location.get('countryName', '')
                    }
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

    def display_players_table(self, df):
        """Displays player data in a table with metrics and export functionality."""
        if df.empty:
            st.warning("No player data available.")
        else:
            st.markdown("### Player Rankings")
            st.dataframe(df.sort_values(by='UTR_numeric', ascending=False).reset_index(drop=True))

            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Players", len(df))
            with col2:
                # Handle 'N/A' values for UTR
                average_utr = df['UTR_numeric'].mean()
                average_utr_display = f"{average_utr:.2f}" if pd.notna(average_utr) else 'N/A'
                st.metric("Average UTR", average_utr_display)
            with col3:
                st.metric("Active Colleges", df['College'].nunique())

            # Sorting options
            st.markdown("### Player List")
            sort_by = st.selectbox("Sort by", options=['Name', 'UTR', 'College', 'Division', 'Conference', 'Last Active'], key='sort_by_players')
            ascending = st.checkbox("Ascending", True, key='ascending_players')

            sorted_df = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
            st.dataframe(sorted_df)

            # Export to CSV
            csv = sorted_df.to_csv(index=False).encode('utf-8')
            filename = f"{df['College'].iloc[0][:3].upper()}_players_{datetime.now().strftime('%Y%m%d')}.csv" if not df.empty else "players.csv"

            st.download_button(
                label="Export to CSV",
                data=csv,
                file_name=filename,
                mime="text/csv",
                key='download-csv-players'
            )
            st.success("Data exported successfully!")

    def display_players_map(self, df):
        """
        Displays players on a map with markers sized by the number of players at each college
        and colored by average UTR.
        
        Args:
            df (pd.DataFrame): DataFrame containing player information along with college coordinates
        """
        if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
            st.error("DataFrame missing 'Latitude' or 'Longitude' columns.")
            return

        # Drop rows with missing coordinates
        df = df.dropna(subset=['Latitude', 'Longitude'])

        if df.empty:
            st.warning("No player data with valid location information to display on the map.")
            return

        # Aggregate by College, Latitude, Longitude
        aggregated_df = df.groupby(['College', 'Latitude', 'Longitude']).agg(
            Player_Count=('Name', 'count'),
            Average_UTR=('UTR_numeric', 'mean'),
            City=('City', 'first'),
            State=('State', 'first')
        ).reset_index()

        # Normalize UTR for color scaling
        min_utr = aggregated_df['Average_UTR'].min()
        max_utr = aggregated_df['Average_UTR'].max()
        aggregated_df['UTR_normalized'] = (aggregated_df['Average_UTR'] - min_utr) / (max_utr - min_utr) if max_utr != min_utr else 0.5

        # Function to map normalized UTR to RGB color
        def get_color(utr_norm):
            """
            Maps a normalized UTR value (0 to 1) to a color.
            0 -> Red, 1 -> Green, intermediate values interpolate between.
            """
            red = int((1 - utr_norm) * 255)
            green = int(utr_norm * 255)
            blue = 0
            return [red, green, blue]

        aggregated_df['Color'] = aggregated_df['UTR_normalized'].apply(get_color)

        # Define the layer
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=aggregated_df,
            get_position='[Longitude, Latitude]',
            auto_highlight=True,
            get_radius='Player_Count * 10',  # Adjust the multiplier as needed
            get_fill_color='Color',
            pickable=True
        )

        # Set the viewport location
        initial_view_state = pdk.ViewState(
            latitude=aggregated_df['Latitude'].mean(),
            longitude=aggregated_df['Longitude'].mean(),
            zoom=3,
            pitch=0
        )

        # Define the map
        r = pdk.Deck(
            layers=[layer],
            initial_view_state=initial_view_state,
            tooltip={
                "html": "<b>College:</b> {College}<br/>"
                        "<b>Player Count:</b> {Player_Count}<br/>"
                        "<b>Average UTR:</b> {Average_UTR:.2f}<br/>"
                        "<b>Location:</b> {City}, {State}",
                "style": {"color": "white"}
            }
        )

        st.pydeck_chart(r)

    def run(self):
        """Runs the Streamlit dashboard."""
        st.title("Tennis Dashboard")

        # Sidebar for user inputs
        with st.sidebar:
            st.header("Filters")
            # Country Selection
            country_options = [f"{country['flag']} {country['name']}" for country in self.countries]
            country_dict = {f"{country['flag']} {country['name']}": country['code'] for country in self.countries}
            selected_country = st.selectbox("Select Country of Origin", options=country_options)

            # View Selection
            view_mode = st.radio("View Mode", options=["Table", "Map"])

        country_code = country_dict.get(selected_country, "")

        if st.sidebar.button("Load Players"):
            if country_code:
                if view_mode == "Table":
                    with st.spinner("Fetching player data..."):
                        df_players = self.fetch_country_players(country_code)
                        if not df_players.empty:
                            self.display_players_table(df_players)
                        else:
                            st.warning("No players found for the selected country.")
                elif view_mode == "Map":
                    with st.spinner("Fetching player data for map..."):
                        df_players = self.fetch_country_players(country_code)
                        if df_players.empty:
                            st.warning("No players found for the selected country.")
                        else:
                            self.display_players_map(df_players)
            else:
                st.sidebar.error("Please select a valid country.")

        st.markdown("---")

        # Additional Features (e.g., Alerts, Notifications) can be added here


# Main Streamlit App
def main():
    dashboard = TennisDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
