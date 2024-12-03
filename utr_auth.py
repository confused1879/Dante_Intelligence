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