import streamlit as st
import pandas as pd
from utr_auth import UTRAuthManager

class CollegeFitDashboard:
    def __init__(self):
        self.utr = UTRAuthManager()
        
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
    
    
    
    
        
    def run(self):
        st.title("College Tennis Fit Analysis")
        
        # Sidebar for athlete inputs
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
        
        def create_power6_chart(df):
            """
            Create a customized bar chart for Power 6 ranges using Plotly
            """
            import plotly.graph_objects as go
            
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
                margin=dict(b=100)  # Add bottom margin for rotated labels
            )
            
            return fig

        def create_college_map(df):
            """
            Create an interactive map of colleges using Plotly
            """
            import plotly.express as px
            
            # Create hover text
            df['hover_text'] = df.apply(lambda row: f"""
                <b>{row['College']}</b><br>
                Power 6: {row['Power 6']:.2f}<br>
                Range: {row['Power 6 Low']:.2f} - {row['Power 6 High']:.2f}<br>
                Conference: {row['Conference Name']}<br>
                Division: {row['Division Name']}
            """, axis=1)
            
            # Create color scale based on Power 6 rating
            fig = px.scatter_mapbox(
                df,
                lat='Latitude',
                lon='Longitude',
                hover_name='College',
                hover_data={
                    'Latitude': False,
                    'Longitude': False,
                    'Power 6': ':.2f',
                    'Conference Name': True,
                    'Division Name': True
                },
                color='Power 6',
                color_continuous_scale='viridis',
                size=[20] * len(df),  # Constant size for all markers
                zoom=3,
                center={'lat': 39.8283, 'lon': -98.5795},  # Center of USA
                title='College Tennis Programs'
            )
            
            # Update map style and layout
            fig.update_layout(
                mapbox_style='carto-positron',
                height=600,
                margin={'r': 0, 'l': 0, 'b': 0, 't': 30}
            )
            
            return fig
        
        # Main content
        if st.sidebar.button("Find Matches"):
            try:
                with st.spinner("Searching colleges..."):
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
                    
                    # Create metrics columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Matches", len(df))
                    with col2:
                        st.metric("Average Power 6 Score", f"{df['Power 6'].mean():.2f}")
                    with col3:
                        st.metric("Highest Team Power 6", f"{df['Power 6 High'].max():.2f}")
                    
                    # Display results in tabs
                    tab1, tab2 = st.tabs(["College List", "Team UTR Analysis"])
                    
                    with tab1:
                        st.plotly_chart(create_college_map(df), use_container_width=True)
                    
                    with tab2:
                        st.plotly_chart(create_power6_chart(df), use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error fetching college data: {str(e)}")

if __name__ == "__main__":
    dashboard = CollegeFitDashboard()
    dashboard.run()