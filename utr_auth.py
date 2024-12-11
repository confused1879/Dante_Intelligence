import requests
import jwt
import time
from datetime import datetime
import os
from dotenv import load_dotenv

class UTRAuthManager:
    def __init__(self):
        load_dotenv()
        self.email = os.getenv('UTR_EMAIL')
        self.password = os.getenv('UTR_PASSWORD')
        self.token = None
        self.newrelic = None
        self.session = requests.Session()

    def login(self):
        url = "https://app.utrsports.net/api/v1/auth/login"
        response = self.session.post(url, json={
            'email': self.email,
            'password': self.password
        })
        response.raise_for_status()
        self.token = response.headers.get('Authorization')
        self.newrelic = response.headers.get('newrelic')

    def get_headers(self):
        if not self.token or self._is_token_expired():
            self.login()
            
        return {
            'Authorization': self.token,
            'newrelic': self.newrelic,
            'origin': 'https://app.utrsports.net',
            'referer': 'https://app.utrsports.net/',
            'x-client-name': 'buildId - 96461'
        }

    def _is_token_expired(self):
        if not self.token:
            return True
        try:
            token = self.token.split(' ')[1]
            payload = jwt.decode(token, options={"verify_signature": False})
            return datetime.fromtimestamp(payload['exp']) <= datetime.now()
        except:
            return True

    def get_player_stats(self, player_id):
        headers = self.get_headers()
        url = f"https://api.utrsports.net/v4/player/{player_id}/all-stats"
        params = {
            'type': 'singles',
            'resultType': 'verified',
            'months': 12,
            'fetchAllResults': 'false'
        }
        response = self.session.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    
    def search_players(self, name, top=10, skip=0):
        """
        Search for players by name using the UTR API.
        
        Args:
            name (str): Name to search for
            top (int): Maximum number of results to return
            skip (int): Number of results to skip for pagination
            
        Returns:
            dict: Search results from the API
        """
        headers = self.get_headers()
        url = "https://api.utrsports.net/v2/search"
        
        params = {
            'schoolClubSearch': 'true',
            'query': name,
            'top': top,
            'skip': skip
        }
        
        response = self.session.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def get_player_results(self, player_id, limit=None):
        """
        Get match results for a specific player.
        
        Args:
            player_id (int): Player's UTR ID
            limit (int, optional): Maximum number of results to return
            
        Returns:
            dict: Player's match results from the API
        """
        headers = self.get_headers()
        url = f"https://api.utrsports.net/v4/player/{player_id}/results"
        
        params = {}
        if limit:
            params['limit'] = limit
            
        response = self.session.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def search_players_advanced(self, top=100, skip=0, primary_tags=None, nationality=None, 
                          division_id=None, conference_id=None, college_id=None,
                          utr_type="verified", utr_team_type="singles", 
                          show_tennis=True, show_pickleball=False):
        headers = self.get_headers()
        url = "https://api.utrsports.net/v2/search/players"
        
        # Base parameters
        params = {
            'top': top,
            'skip': skip,
            'utrType': utr_type,
            'utrTeamType': utr_team_type,
            'showTennisContent': str(show_tennis).lower(),
            'showPickleballContent': str(show_pickleball).lower(),
            'searchOrigin': 'searchPage'
        }
        
        # Optional parameters - only add if not None
        optional_params = {
            'primaryTags': primary_tags,
            'nationality': nationality,
            'divisionId': division_id,
            'conferenceId': conference_id,
            'collegeId': college_id
        }
        
        # Add optional parameters if they have values
        params.update({k: v for k, v in optional_params.items() if v is not None})
            
        response = self.session.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def search_colleges(self, utr_rating, position=6, top=100):
        """
        Search colleges using UTR API based on fit rating.
        
        Args:
            utr_rating (float): Player's UTR rating for fit calculation
            position (int): Position in lineup (default: 6)
            top (int): Maximum number of results to return
            
        Returns:
            dict: College search results from the API
        """
        headers = self.get_headers()
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
            response = self.session.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching college data: {e}")
            return {}