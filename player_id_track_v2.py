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
import time
import plotly.graph_objects as go
import plotly.express as px

# Load environment variables from .env if needed
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure Streamlit page
st.set_page_config(page_title="Player Identification and Tracking", layout="wide")


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
        """Fetches college locations with file caching."""
        cache_file = "college_locations.csv"
        
        # Try to load from cache
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file)

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
                response = self.utr.session.get(
                    "https://api.utrsports.net/v2/search/colleges", 
                    headers=self.utr.get_headers(), 
                    params=params
                )
                response.raise_for_status()
                hits = response.json().get('hits', [])
                if not hits:
                    break

                for hit in hits:
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
                        
                skip += top
                if len(hits) < top:
                    break
                    
            except Exception as e:
                break
        
        df = pd.DataFrame(colleges)
        df.to_csv(cache_file, index=False)
        return df

    def fetch_country_players(self, nationality=None, divisions=None, conference_id=None, college_id=None):
        """
        Fetches player data across multiple divisions.
        
        Args:
            nationality: Optional country code
            divisions: List of division IDs (defaults to all divisions 1-9)
            conference_id: Optional conference ID
            college_id: Optional college ID
        """
        try:
            start_time = time.time()
            all_player_data = []
            divisions = divisions or range(1, 10)
            
            for division in divisions:
                api_start = time.time()
                players_response = st.session_state.auth_manager.search_players_advanced(
                    top=100,
                    skip=0,
                    primary_tags='College',
                    nationality=nationality,
                    division_id=division,
                    conference_id=conference_id,
                    college_id=college_id,
                    utr_type="verified",
                    utr_team_type="singles"
                )
                
                api_time = time.time() - api_start
                logging.info(f"Players API fetch time (division={division}): {api_time:.2f}s")

                hits = players_response.get('hits', [])
                if not hits:
                    continue

                for hit in hits:
                    source = hit.get('source', {})
                    college_details = source.get('playerCollegeDetails', {})
                    
                    player_info = {
                        'UTR ID': source.get('id', 'N/A'),
                        'Name': f"{source.get('firstName', '')} {source.get('lastName', '')}".strip(),
                        'UTR': source.get('singlesUtr', 'N/A'),
                        'College': source.get('playerCollege', {}).get('name', 'N/A'),
                        'Division': source.get('playerCollege', {}).get('conference', {}).get('shortName', 'N/A'),
                        'Conference': source.get('playerCollege', {}).get('conference', {}).get('conferenceName', 'N/A'),
                        'Last Active': datetime.fromtimestamp(source.get('lastActive', 0)).strftime('%Y-%m-%d') if source.get('lastActive') else 'N/A',
                        'Age Range': source.get('ageRange', 'N/A'),
                        'Graduation Year': college_details.get('gradYear', 'N/A').split('T')[0] if college_details.get('gradYear') else 'N/A',
                        'Year': college_details.get('gradClassName', 'N/A'),
                        'ATP Rank': next((ranking['rank'] for ranking in source.get('thirdPartyRankings', []) 
                                        if ranking.get('source') == 'ATP' and ranking.get('type') == 'Singles'), 'N/A'),
                        'Is Pro': source.get('isPro', False),
                        'Location': source.get('location', {}).get('display', 'N/A'),
                        'Nationality': source.get('nationality', 'N/A'),
                        'Current UTR': source.get('singlesUtrDisplay', 'N/A'),
                        'Three Month UTR': source.get('threeMonthRating', 'N/A'),
                        'UTR Status': source.get('ratingStatusSingles', 'N/A'),
                        'UTR Progress': source.get('ratingProgressSingles', 'N/A')
                    }
                    all_player_data.append(player_info)

                st.write(f"Division {division}: Found {len(hits)} players")
                logging.info(f"Division {division}: Found {len(hits)} players")

            if not all_player_data:
                return pd.DataFrame()

            df_players = pd.DataFrame(all_player_data)
            df_players = df_players.drop_duplicates(subset=['UTR ID'], keep='first')
            
            college_start = time.time()
            df_colleges = self.get_college_locations(top=100)
            college_time = time.time() - college_start
            logging.info(f"College locations fetch time: {college_time:.2f}s")

            df_merged = pd.merge(df_players, df_colleges, on='College', how='left')
            df_merged['UTR_numeric'] = pd.to_numeric(df_merged['UTR'], errors='coerce')

            df_merged = df_merged.drop_duplicates(subset=['UTR ID'], keep='first')
            
            total_time = time.time() - start_time
            logging.info(f"Total execution time: {total_time:.2f}s")
            logging.info(f"Total players fetched: {len(df_merged)}")

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
        """Create interactive map of colleges using Plotly with marker sizes based on player count."""
        # Ensure the required columns are present
        required_columns = ['College', 'Latitude', 'Longitude', 'UTR_numeric', 'City', 'State']
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Missing required column: {col}")
                return

        # Drop rows with missing location data
        df_map = df.dropna(subset=['Latitude', 'Longitude'])

        if df_map.empty:
            st.warning("No college locations available.")
            return

        # Aggregate data: count number of players per college
        aggregation = df_map.groupby(
            ['College', 'Latitude', 'Longitude', 'City', 'State']
        ).agg(
            Num_Players=pd.NamedAgg(column='Name', aggfunc='count'),
            Avg_UTR=pd.NamedAgg(column='UTR_numeric', aggfunc='mean')
        ).reset_index()

        # Optionally, apply a scaling factor or logarithmic transformation to marker sizes
        # This helps in cases where there's a large variance in player counts
        aggregation['Marker_Size'] = aggregation['Num_Players'].apply(lambda x: np.log(x + 1))  # Example scaling

        # Create the scatter mapbox plot
        fig = px.scatter_mapbox(
            aggregation,
            lat='Latitude',
            lon='Longitude',
            hover_name='College',
            hover_data={
                'Latitude': False,
                'Longitude': False,
                'Avg_UTR': ':.2f',
                'City': True,
                'State': True,
                'Num_Players': True
            },
            color='Avg_UTR',
            color_continuous_scale='viridis',
            size='Marker_Size',
            size_max=15,  # Adjust based on your scaling
            zoom=3,
            center={'lat': 39.8283, 'lon': -98.5795},  # Center of the US
            title='College Tennis Programs'
        )

        # Update layout for better aesthetics
        fig.update_layout(
            mapbox_style='carto-positron',
            height=600,
            margin={'r': 0, 'l': 0, 'b': 0, 't': 50},
            coloraxis_colorbar=dict(
                title="Average UTR",
                thickness=20,
                len=0.75,
                yanchor="top",
                y=0.95,
                ticks="outside"
            )
        )

        # Display the map in Streamlit
        st.plotly_chart(fig, use_container_width=True)

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
        """Displays player data in a table with metrics."""
        if df.empty:
            st.warning("No player data available.")
            return

        if 'df_players' not in st.session_state:
            st.session_state.df_players = df

        st.markdown("### Player Rankings")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Players", len(df))
        with col2:
            average_utr = df['UTR_numeric'].mean()
            average_utr_display = f"{average_utr:.2f}" if pd.notna(average_utr) else 'N/A'
            st.metric("Average UTR", average_utr_display)
        with col3:
            st.metric("Active Colleges", df['College'].nunique())

        # Sorting using session state
        sort_by = st.selectbox("Sort by", options=['Name', 'UTR', 'College', 'Division', 'Conference', 'Last Active'], key='sort_by_players')
        ascending = st.checkbox("Ascending", True, key='ascending_players')
        
        sorted_df = st.session_state.df_players.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
        st.dataframe(sorted_df)

        # Export functionality
        csv = sorted_df.to_csv(index=False).encode('utf-8')
        filename = f"{sorted_df['College'].iloc[0][:3].upper()}_players_{datetime.now().strftime('%Y%m%d')}.csv" if not df.empty else "players.csv"
        
        st.download_button(
            label="Export to CSV",
            data=csv,
            file_name=filename,
            mime="text/csv",
            key='download-csv-players'
        )

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
        if 'df_players' not in st.session_state:
            st.session_state.df_players = df

        st.markdown("### Player List")
        
        # Get sorting preferences
        #sort_by = st.selectbox("Sort by", options=['Name', 'UTR ID', 'UTR', 'College', 'Division', 'Conference'], key='sort_by')
        #ascending = st.checkbox("Ascending", True, key='ascending')
        ascending = True
        sort_by = 'Name'
        
        # Use session state data for sorting
        sorted_df = st.session_state.df_players.sort_values(by=sort_by, ascending=ascending)
        st.dataframe(sorted_df)

        # Export to CSV
        csv = sorted_df.to_csv(index=False).encode('utf-8')
        filename = f"tennis_players_{datetime.now().strftime('%Y%m%d')}.csv"
        
        st.download_button(
            label="Export to CSV",
            data=csv,
            file_name=filename,
            mime="text/csv"
        )
        st.success("Data exported successfully!")

    def run(self):
        st.title("")

        # Sidebar setup
        with st.sidebar:
            st.header("Filters")
            country_options = [f"{country['flag']} {country['name']}" for country in self.countries]
            country_dict = {f"{country['flag']} {country['name']}": country['code'] for country in self.countries}
            selected_country = st.selectbox("Select Country of Origin", options=country_options)
            country_code = country_dict.get(selected_country, "")

        if st.sidebar.button("Load Players"):
            if not country_code:
                st.sidebar.error("Please select a valid country.")
                return

            with st.spinner("Fetching player data..."):
                df_players = self.fetch_country_players(country_code)
                if df_players.empty:
                    st.warning("No players found for the selected country.")
                    return
                    
                st.session_state.df_players = df_players
                
                # Display metrics above tabs
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Players", len(df_players))
                with col2:
                    average_utr = df_players['UTR_numeric'].mean()
                    average_utr_display = f"{average_utr:.2f}" if pd.notna(average_utr) else 'N/A'
                    st.metric("Average UTR", average_utr_display)
                with col3:
                    st.metric("Active Colleges", df_players['College'].nunique())

                # Create tabs for different views
                tab1, tab2 = st.tabs(["Map View", "Table View"])
                
                with tab1:
                    self.display_players_map(df_players)
                with tab2:
                    self.display_players_table(df_players)
        
        st.markdown("---")

        # Additional Features (e.g., Alerts, Notifications) can be added here


# Main Streamlit App
def main():
    dashboard = TennisDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
