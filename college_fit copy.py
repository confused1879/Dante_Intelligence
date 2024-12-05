import streamlit as st
import pandas as pd
from utr_auth import UTRAuthManager
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import plotly.graph_objects as go
import plotly.express as px
import time

class CollegeFitDashboard:
    def __init__(self):
        self.utr = UTRAuthManager()
        # Initialize the geocoder with a user agent
        self.geolocator = Nominatim(user_agent="college_fit_dashboard")
        
    def search_colleges(self, utr_rating, position=6, top=100):
        """
        Search colleges using UTR API based on fit rating
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
        
        response = self.utr.session.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    
    def process_college_data(self, data):
        """
        Process raw API response into a pandas DataFrame, capturing all available data fields
        """
        if not data.get('hits'):
            st.error("No college data found in API response")
            return pd.DataFrame()
            
        processed_data = []
        
        for hit in data.get('hits', []):
            try:
                source = hit.get('source', {})
                school = source.get('school', {})
                location = source.get('location', {})
                conference = school.get('conference', {})
                division = conference.get('division', {}) if conference else {}
                
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
                    'Power 6 Men': school.get('power6Men'),
                    'Power 6 Men High': school.get('power6MenHigh'),
                    'Power 6 Men Low': school.get('power6MenLow'),
                    'Power 6 Women': school.get('power6Women'),
                    'Power 6 Women High': school.get('power6WomenHigh'),
                    'Power 6 Women Low': school.get('power6WomenLow'),
                    
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
                    'API Sort Value': hit.get('sorts', [None])[0]
                }
                
                processed_data.append(entry)
                
            except Exception as e:
                st.error(f"Error processing college entry: {str(e)}")
                continue
        
        df = pd.DataFrame(processed_data)
        
        # Add debug information
        st.write("Number of colleges processed:", len(df))
        if len(df) == 0:
            st.write("Raw API response:", data)
            
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
        try:
            location = self.geolocator.geocode(address, timeout=10)
            if location:
                return (location.latitude, location.longitude)
            else:
                st.error("Geocoding failed: Address not found.")
                return None
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            st.error(f"Geocoding error: {e}")
            return None
    
    def calculate_fit_score(self, df, athlete_utr, player_location, weights):
        """
        Calculate the Fit Score for each college based on defined criteria and weights.
        """
        # Normalize weights internally
        total_weight = weights['Athletic'] + weights['Location'] + weights['Culture']
        if total_weight == 0:
            st.warning("All weights are set to zero. Assigning equal weights to all criteria.")
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
    
    def run(self):
        st.title("College Tennis Fit Analysis")
        
        # Sidebar for athlete inputs and preferences
        with st.sidebar:
            st.header("Athlete Information")
            
            # UTR Input
            athlete_utr = st.number_input(
                "Your UTR Rating",
                min_value=1.0,
                max_value=16.0,
                value=11.0,
                step=0.1,
                help="Your current Universal Tennis Rating"
            )
            
            # Desired Position
            desired_position = st.slider(
                "Desired Position on Team",
                min_value=1,
                max_value=8,
                value=6,
                help="1 = Top of lineup, 6 = Bottom of lineup"
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
            
            st.header("Preference Weighting")
            st.markdown("Assign weights to prioritize different criteria. These weights will be normalized automatically.")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                athletic_weight = st.slider(
                    "Athletic Development",
                    min_value=0,
                    max_value=100,
                    value=50,
                    step=1,
                    help="Importance of Athletic Development"
                )
            with col_b:
                location_weight = st.slider(
                    "Geographic Location",
                    min_value=0,
                    max_value=100,
                    value=30,
                    step=1,
                    help="Importance of Geographic Location"
                )
            with col_c:
                culture_weight = st.slider(
                    "Campus Culture",
                    min_value=0,
                    max_value=100,
                    value=20,
                    step=1,
                    help="Importance of Campus Culture"
                )
            
            # Display the total weight for user awareness (optional)
            total_input_weight = athletic_weight + location_weight + culture_weight
            st.markdown(f"**Total Weight Input:** {total_input_weight}")
            st.markdown("_Weights will be normalized automatically._")
            
            # Display a button to find matches
            find_matches = st.button("Find Matches")
        
        def create_power6_chart(df):
            """
            Create a customized bar chart for Power 6 ranges using Plotly
            """
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
        Create an interactive map of colleges and player's home using Plotly
        """
        # Remove entries with missing coordinates
        df_map = df.dropna(subset=['Latitude', 'Longitude'])
        
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
        
        # Create hover text for player's home
        if player_location:
            home_hover_text = "Your Home Location"
            home_lat, home_lon = player_location
        else:
            home_hover_text = None
        
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

    # Continue with the rest of the class methods...

    def run(self):
        st.title("College Tennis Fit Analysis")
        
        # Sidebar for athlete inputs and preferences
        with st.sidebar:
            st.header("Athlete Information")
            
            # UTR Input
            athlete_utr = st.number_input(
                "Your UTR Rating",
                min_value=1.0,
                max_value=16.0,
                value=11.0,
                step=0.1,
                help="Your current Universal Tennis Rating"
            )
            
            # Desired Position
            desired_position = st.slider(
                "Desired Position on Team",
                min_value=1,
                max_value=8,
                value=6,
                help="1 = Top of lineup, 6 = Bottom of lineup"
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
            
            st.header("Preference Weighting")
            st.markdown("Assign weights to prioritize different criteria. These weights will be normalized automatically.")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                athletic_weight = st.slider(
                    "Athletic Development",
                    min_value=0,
                    max_value=100,
                    value=50,
                    step=1,
                    help="Importance of Athletic Development"
                )
            with col_b:
                location_weight = st.slider(
                    "Geographic Location",
                    min_value=0,
                    max_value=100,
                    value=30,
                    step=1,
                    help="Importance of Geographic Location"
                )
            with col_c:
                culture_weight = st.slider(
                    "Campus Culture",
                    min_value=0,
                    max_value=100,
                    value=20,
                    step=1,
                    help="Importance of Campus Culture"
                )
            
            # Display the total weight for user awareness (optional)
            total_input_weight = athletic_weight + location_weight + culture_weight
            st.markdown(f"**Total Weight Input:** {total_input_weight}")
            st.markdown("_Weights will be normalized automatically._")
            
            # Display a button to find matches
            find_matches = st.button("Find Matches")
        
        def create_power6_chart(df):
            """
            Create a customized bar chart for Power 6 ranges using Plotly
            """
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
            Create an interactive map of colleges and player's home using Plotly
            """
            # Remove entries with missing coordinates
            df_map = df.dropna(subset=['Latitude', 'Longitude'])
            
            # Normalize 'Power 6 Avg' for marker sizes
            size_min = 10
            size_max = 50
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
            
            # Create hover text for player's home
            if player_location:
                home_hover_text = "Your Home Location"
                home_lat, home_lon = player_location
            else:
                home_hover_text = None
            
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
        
        # Main content
        if find_matches:
            try:
                with st.spinner("Searching colleges..."):
                    # Geocode the home address
                    player_location = self.geocode_address(home_address)
                    
                    if player_location is None:
                        st.error("Unable to geocode the provided address. Please check and try again.")
                    else:
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
                        
                        # Calculate Fit Scores
                        weights = {
                            'Athletic': athletic_weight,
                            'Location': location_weight,
                            'Culture': culture_weight
                        }
                        df = self.calculate_fit_score(df, athlete_utr, player_location, weights)
                        
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
                            st.plotly_chart(create_power6_chart(df), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error fetching or processing college data: {str(e)}")

if __name__ == "__main__":
    dashboard = CollegeFitDashboard()
    dashboard.run()
