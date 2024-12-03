import pandas as pd
import numpy as np
from datetime import datetime
import ast
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PlayerResilience:
    def __init__(self, matches_data):
        self.matches_df = pd.DataFrame(matches_data)
        self.matches_df['resultDate'] = pd.to_datetime(self.matches_df['resultDate'])
        self.matches_df = self.matches_df.sort_values('resultDate')
        
    def calculate_scores_over_time(self):
        """Calculate resilience scores for each match individually."""
        scores_list = []
        
        for _, match in self.matches_df.iterrows():
            # Parse match data
            winner_sets = ast.literal_eval(match['winnerSets'])
            loser_sets = ast.literal_eval(match['loserSets'])
            is_winner = match['isWinner']
            event_name = match['eventName']
            
            # Calculate individual scores
            comeback_score = self._calculate_match_comeback(is_winner, winner_sets, loser_sets)
            tiebreak_score = self._calculate_match_tiebreak_score(match)
            pressure_score = self._calculate_match_pressure(is_winner, winner_sets, loser_sets, event_name)
            consistency_score = 1 if is_winner else 0
            recovery_score = self._calculate_match_recovery(match)
            
            scores_list.append({
                'date': match['resultDate'],
                'comeback_score': comeback_score,
                'tiebreak_score': tiebreak_score,
                'pressure_score': pressure_score,
                'consistency_score': consistency_score,
                'recovery_score': recovery_score
            })
        
        scores_df = pd.DataFrame(scores_list)
        return scores_df


    def _calculate_match_comeback(self, is_winner, winner_sets, loser_sets):
        player_lost_first = (is_winner and winner_sets[0] < loser_sets[0]) or \
                        (not is_winner and loser_sets[0] < winner_sets[0])
        if player_lost_first:
            return 1 if is_winner else 0
        return None

    def _calculate_match_tiebreak(self, is_winner, winner_sets, loser_sets, tiebreak_sets):
        has_tiebreak = any(tb == 1 for tb in tiebreak_sets)
        if has_tiebreak:
            for i, is_tiebreak in enumerate(tiebreak_sets):
                if is_tiebreak == 1:
                    if (is_winner and winner_sets[i] > loser_sets[i]) or \
                    (not is_winner and loser_sets[i] > winner_sets[i]):
                        return 1
                    return 0
        return None

    def _calculate_match_pressure(self, is_winner, winner_sets, loser_sets, event_name):
        is_deciding_set = sum(1 for s in winner_sets if s > 0) == 3
        is_important_event = event_name in ['US Open', 'Australian Open', 'Wimbledon', 'French Open']
        
        if is_deciding_set or is_important_event:
            return 1 if is_winner else 0
        return None

    def _calculate_match_recovery(self, match):
        match_date = match['resultDate']
        next_match = self.matches_df[self.matches_df['resultDate'] > match_date].iloc[0] if not self.matches_df[self.matches_df['resultDate'] > match_date].empty else None
        
        if next_match is not None:
            days_until_next = (next_match['resultDate'] - match_date).days
            if days_until_next <= 2:
                return 1 if match['isWinner'] else 0
        return None

    def calculate_resilience_score(self):
        # Calculate each component's average score
        scores_df = self.calculate_scores_over_time()
        
        scores = {
            'comeback_score': scores_df['comeback_score'].mean(),
            'tiebreak_score': scores_df['tiebreak_score'].mean(),
            'pressure_score': scores_df['pressure_score'].mean(),
            'consistency_score': scores_df['consistency_score'].mean(),
            'recovery_score': scores_df['recovery_score'].mean()
        }
        
        # Replace NaN values with 0.5
        scores = {k: v if pd.notnull(v) else 0.5 for k, v in scores.items()}
        
        # Weighted average
        weights = {
            'comeback_score': 0.25,
            'tiebreak_score': 0.20,
            'pressure_score': 0.25,
            'consistency_score': 0.15,
            'recovery_score': 0.15
        }
        
        total_score = sum(score * weights[metric] for metric, score in scores.items())
        return total_score, scores
    
    def plot_scores_over_time(self):
        scores_df = self.calculate_scores_over_time()
        
        # Remove NaN values before plotting
        scores_df = scores_df.fillna(method='ffill').fillna(0.5)
        
        fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True,
                        subplot_titles=('Match Performance', 'Consistency Metrics'))
        
        # Top subplot: Pressure, Comeback, and Tiebreak scores
        metrics_colors = {
            'pressure_score': '#1f77b4',
            'comeback_score': '#ff7f0e',
            'tiebreak_score': '#2ca02c'
        }
        
        for metric, color in metrics_colors.items():
            # Add both raw data points and smoothed line
            fig.add_trace(
                go.Scatter(x=scores_df['date'], 
                        y=scores_df[metric],
                        mode='markers',
                        name=f"{metric.replace('_', ' ').title()}",
                        marker=dict(color=color, size=6),
                        opacity=0.5),
                row=1, col=1)
                
            fig.add_trace(
                go.Scatter(x=scores_df['date'], 
                        y=scores_df[metric].rolling(window=5, min_periods=1).mean(),
                        name=f"{metric.replace('_', ' ').title()} (Trend)",
                        line=dict(color=color, width=2)),
                row=1, col=1)
        
        # Bottom subplot: Consistency and Recovery
        bottom_metrics_colors = {
            'consistency_score': '#d62728',
            'recovery_score': '#9467bd'
        }
        
        for metric, color in bottom_metrics_colors.items():
            fig.add_trace(
                go.Scatter(x=scores_df['date'],
                        y=scores_df[metric],
                        mode='markers',
                        name=f"{metric.replace('_', ' ').title()}",
                        marker=dict(color=color, size=6),
                        opacity=0.5),
                row=2, col=1)
                
            fig.add_trace(
                go.Scatter(x=scores_df['date'],
                        y=scores_df[metric].rolling(window=5, min_periods=1).mean(),
                        name=f"{metric.replace('_', ' ').title()} (Trend)",
                        line=dict(color=color, width=2)),
                row=2, col=1)
        
        fig.update_layout(
            height=700,
            title='Mental Resilience Analysis',
            showlegend=True,
            yaxis=dict(range=[0, 1], title='Score'),
            yaxis2=dict(range=[0, 1], title='Score'),
            xaxis2_title='Date'
        )
        
        return fig

    def _calculate_comeback_score_single(self, match):
        winner_sets = ast.literal_eval(match['winnerSets'])
        loser_sets = ast.literal_eval(match['loserSets'])
        
        player_lost_first = (match['isWinner'] and winner_sets[0] < loser_sets[0]) or \
                            (not match['isWinner'] and loser_sets[0] < winner_sets[0])
        
        if player_lost_first:
            return 1 if match['isWinner'] else 0
        return None  # No comeback opportunity
        
    def _calculate_match_tiebreak_score(self, match):
        """
        Calculate the tiebreak performance score for a single match.
        """
        try:
            tiebreak_sets = ast.literal_eval(match['tiebreakSets'])
            winner_sets = ast.literal_eval(match['winnerSets'])
            loser_sets = ast.literal_eval(match['loserSets'])
            is_winner = match['isWinner']
            
            tiebreaks_played = 0
            tiebreaks_won = 0
            
            for i, is_tiebreak in enumerate(tiebreak_sets):
                if is_tiebreak == 1:
                    tiebreaks_played += 1
                    player_score = winner_sets[i] if is_winner else loser_sets[i]
                    opponent_score = loser_sets[i] if is_winner else winner_sets[i]
                    
                    if player_score > opponent_score:
                        tiebreaks_won += 1
                        
            if tiebreaks_played > 0:
                tiebreak_win_rate = tiebreaks_won / tiebreaks_played
                return tiebreak_win_rate
            else:
                return None  # No tiebreaks played in this match
        except Exception as e:
            # Handle parsing errors or unexpected data
            print(f"Error calculating tiebreak score for match on {match['resultDate']}: {e}")
            return None

    
    def _calculate_pressure_score(self):
        pressure_matches = 0
        pressure_wins = 0
        
        for _, match in self.matches_df.iterrows():
            winner_sets = ast.literal_eval(match['winnerSets'])
            loser_sets = ast.literal_eval(match['loserSets'])
            
            # Define pressure situations
            is_deciding_set = sum(1 for s in winner_sets if s > 0) == 3  # Match went to final set
            is_important_event = match['eventName'] in ['US Open', 'Australian Open', 'Wimbledon', 'French Open']
            
            if is_deciding_set or is_important_event:
                pressure_matches += 1
                if match['isWinner']:
                    pressure_wins += 1
        
        return (pressure_wins / pressure_matches) if pressure_matches > 0 else 0.5
    
    def _calculate_consistency_score(self):
        recent_matches = self.matches_df.sort_values('resultDate').tail(10)
        wins = sum(recent_matches['isWinner'])
        return wins / len(recent_matches) if not recent_matches.empty else 0.5
    
    def _calculate_recovery_score(self):
        self.matches_df['next_match_days'] = self.matches_df['resultDate'].diff().dt.days * -1
        
        quick_turnaround_matches = self.matches_df[self.matches_df['next_match_days'] <= 2]
        if quick_turnaround_matches.empty:
            return 0.5
            
        return sum(quick_turnaround_matches['isWinner']) / len(quick_turnaround_matches)
    
    def get_detailed_analysis(self):
        total_score, component_scores = self.calculate_resilience_score()
        
        analysis = {
            'overall_resilience': total_score,
            'components': component_scores,
            'interpretation': self._get_score_interpretation(total_score),
            'strengths': self._identify_strengths(component_scores),
            'areas_for_improvement': self._identify_weaknesses(component_scores)
        }
        
        return analysis
    
    def _get_score_interpretation(self, score):
        if score >= 0.8: return "Elite mental resilience"
        elif score >= 0.7: return "Very strong mental resilience"
        elif score >= 0.6: return "Good mental resilience"
        elif score >= 0.5: return "Average mental resilience"
        else: return "Below average mental resilience"
    
    def _identify_strengths(self, scores):
        return [k for k, v in scores.items() if v >= 0.7]
    
    def _identify_weaknesses(self, scores):
        return [k for k, v in scores.items() if v < 0.5]