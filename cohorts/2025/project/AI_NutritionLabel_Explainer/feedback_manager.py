"""
Feedback Manager for AI Nutrition Label Explainer
------------------------------------------------
Handles storage and retrieval of user interactions and feedback data using JSON files.
Supports monitoring dashboard with comprehensive metrics.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd


class FeedbackManager:
    """Manages user feedback and interaction data storage in JSON format."""
    
    def __init__(self, data_dir: str = "data_ingestion"):
        """Initialize feedback manager with data directory."""
        self.data_dir = data_dir
        self.interactions_file = os.path.join(data_dir, "interactions.json")
        self.feedback_file = os.path.join(data_dir, "user_feedback.json")
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize empty files if they don't exist
        self._ensure_files_exist()
    
    def _ensure_files_exist(self):
        """Create empty JSON files if they don't exist."""
        for file_path in [self.interactions_file, self.feedback_file]:
            if not os.path.exists(file_path):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
    
    def save_interaction(self, interaction_id: str, query: str, answer: str, 
                       context: str, method: str, latency: float, 
                       num_docs: int, error: Optional[str] = None) -> bool:
        """
        Save user interaction data.
        
        Args:
            interaction_id: Unique identifier for the interaction
            query: User's question
            answer: AI assistant's response
            context: Retrieved context used for answer
            method: Retrieval method used (Dense, Hybrid, Hybrid + Re-rank)
            latency: Response time in milliseconds
            num_docs: Number of documents retrieved
            error: Error message if any
            
        Returns:
            bool: Success status
        """
        try:
            interaction_data = {
                "interaction_id": interaction_id,
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "answer": answer,
                "context": context,
                "method": method,
                "latency": latency,
                "num_docs": num_docs,
                "error": error
            }
            
            # Load existing data
            with open(self.interactions_file, 'r', encoding='utf-8') as f:
                interactions = json.load(f)
            
            # Add new interaction
            interactions.append(interaction_data)
            
            # Save back to file
            with open(self.interactions_file, 'w', encoding='utf-8') as f:
                json.dump(interactions, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error saving interaction: {e}")
            return False
    
    def save_user_feedback(self, interaction_id: str, thumbs_up: Optional[bool] = None,
                          star_rating: Optional[int] = None, comment: Optional[str] = None) -> bool:
        """
        Save user feedback for an interaction.
        
        Args:
            interaction_id: ID of the interaction being rated
            thumbs_up: True for thumbs up, False for thumbs down, None for no rating
            star_rating: Star rating from 1-5, None for no rating
            comment: Optional text comment
            
        Returns:
            bool: Success status
        """
        try:
            feedback_data = {
                "interaction_id": interaction_id,
                "timestamp": datetime.now().isoformat(),
                "thumbs_up": thumbs_up,
                "star_rating": star_rating,
                "comment": comment
            }
            
            # Load existing feedback
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                feedback = json.load(f)
            
            # Check if feedback already exists for this interaction
            existing_idx = None
            for i, fb in enumerate(feedback):
                if fb["interaction_id"] == interaction_id:
                    existing_idx = i
                    break
            
            if existing_idx is not None:
                # Update existing feedback
                feedback[existing_idx] = feedback_data
            else:
                # Add new feedback
                feedback.append(feedback_data)
            
            # Save back to file
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(feedback, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error saving feedback: {e}")
            return False
    
    def get_interactions_data(self) -> List[Dict]:
        """Load all interaction data."""
        try:
            with open(self.interactions_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading interactions: {e}")
            return []
    
    def get_feedback_data(self) -> List[Dict]:
        """Load all feedback data."""
        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading feedback: {e}")
            return []
    
    def get_combined_data(self) -> pd.DataFrame:
        """Get combined interactions and feedback data as DataFrame."""
        interactions = self.get_interactions_data()
        feedback = self.get_feedback_data()
        
        if not interactions:
            return pd.DataFrame()
        
        # Convert to DataFrames
        df_interactions = pd.DataFrame(interactions)
        df_feedback = pd.DataFrame(feedback)
        
        # Merge on interaction_id
        if not df_feedback.empty:
            df_combined = df_interactions.merge(df_feedback, on='interaction_id', how='left', suffixes=('', '_feedback'))
        else:
            df_combined = df_interactions.copy()
            df_combined['thumbs_up'] = None
            df_combined['star_rating'] = None
            df_combined['comment'] = None
        
        # Convert timestamp to datetime (use the main timestamp column)
        if 'timestamp' in df_combined.columns:
            df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])
        
        return df_combined
    
    def get_metrics(self) -> Dict:
        """Calculate key metrics for dashboard."""
        df = self.get_combined_data()
        
        if df.empty:
            return {
                "total_queries": 0,
                "avg_satisfaction": 0,
                "avg_latency": 0,
                "error_rate": 0,
                "feedback_rate": 0,
                "avg_star_rating": 0
            }
        
        # Basic metrics
        total_queries = len(df)
        
        # Error rate
        error_rate = (df['error'].notna().sum() / total_queries * 100) if total_queries > 0 else 0
        
        # Latency metrics
        avg_latency = df['latency'].mean() if total_queries > 0 else 0
        
        # Feedback metrics
        feedback_data = df[df['thumbs_up'].notna()]
        feedback_rate = (len(feedback_data) / total_queries * 100) if total_queries > 0 else 0
        
        # Satisfaction metrics
        thumbs_up_count = feedback_data[feedback_data['thumbs_up'] == True].shape[0]
        avg_satisfaction = (thumbs_up_count / len(feedback_data) * 100) if len(feedback_data) > 0 else 0
        
        # Star rating metrics
        star_data = df[df['star_rating'].notna()]
        avg_star_rating = star_data['star_rating'].mean() if len(star_data) > 0 else 0
        
        return {
            "total_queries": total_queries,
            "avg_satisfaction": round(avg_satisfaction, 1),
            "avg_latency": round(avg_latency, 1),
            "error_rate": round(error_rate, 1),
            "feedback_rate": round(feedback_rate, 1),
            "avg_star_rating": round(avg_star_rating, 1)
        }
    
    def get_feedback_trends(self, period: str = 'day') -> pd.DataFrame:
        """Get feedback trends over time."""
        df = self.get_combined_data()
        
        if df.empty:
            return pd.DataFrame()
        
        # Filter feedback data
        feedback_df = df[df['thumbs_up'].notna()].copy()
        
        if feedback_df.empty:
            return pd.DataFrame()
        
        # Group by time period
        if period == 'hour':
            feedback_df['time_group'] = feedback_df['timestamp'].dt.floor('H')
        else:  # day
            feedback_df['time_group'] = feedback_df['timestamp'].dt.date
        
        # Calculate satisfaction rate by time period
        trends = feedback_df.groupby('time_group').agg({
            'thumbs_up': ['count', lambda x: (x == True).sum()]
        }).round(2)
        
        trends.columns = ['total_feedback', 'thumbs_up_count']
        trends['satisfaction_rate'] = (trends['thumbs_up_count'] / trends['total_feedback'] * 100).round(1)
        
        return trends.reset_index()
    
    def get_latency_analysis(self) -> Dict:
        """Get latency analysis by retrieval method."""
        df = self.get_combined_data()
        
        if df.empty:
            return {}
        
        latency_stats = df.groupby('method')['latency'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(1)
        
        # Calculate percentiles
        percentiles = {}
        for method in df['method'].unique():
            method_data = df[df['method'] == method]['latency']
            percentiles[method] = {
                'p50': method_data.quantile(0.5),
                'p90': method_data.quantile(0.9),
                'p99': method_data.quantile(0.99)
            }
        
        return {
            'stats': latency_stats.to_dict(),
            'percentiles': percentiles
        }
    
    def get_query_volume(self, period: str = 'hour') -> pd.DataFrame:
        """Get query volume over time."""
        df = self.get_combined_data()
        
        if df.empty:
            return pd.DataFrame()
        
        # Group by time period
        if period == 'hour':
            df['time_group'] = df['timestamp'].dt.floor('H')
        else:  # day
            df['time_group'] = df['timestamp'].dt.date
        
        volume = df.groupby('time_group').size().reset_index(name='query_count')
        
        return volume
    
    def get_method_usage(self) -> pd.DataFrame:
        """Get retrieval method usage distribution."""
        df = self.get_combined_data()
        
        if df.empty:
            return pd.DataFrame()
        
        method_counts = df['method'].value_counts().reset_index()
        method_counts.columns = ['method', 'count']
        method_counts['percentage'] = (method_counts['count'] / method_counts['count'].sum() * 100).round(1)
        
        return method_counts
    
    def get_top_products(self, top_n: int = 10) -> pd.DataFrame:
        """Extract top queried products/categories from context."""
        df = self.get_combined_data()
        
        if df.empty:
            return pd.DataFrame()
        
        # Extract product names from context
        product_counts = {}
        
        for context in df['context'].dropna():
            # Simple extraction - look for "Product: " pattern
            lines = context.split('\n')
            for line in lines:
                if line.strip().startswith('Product:'):
                    product = line.split('Product:')[1].split('.')[0].strip()
                    if product and product != 'Unknown':
                        product_counts[product] = product_counts.get(product, 0) + 1
        
        # Convert to DataFrame
        if product_counts:
            top_products = pd.DataFrame([
                {'product': product, 'count': count} 
                for product, count in sorted(product_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
            ])
            return top_products
        
        return pd.DataFrame()
    
    def get_satisfaction_distribution(self) -> Dict:
        """Get satisfaction score distribution."""
        df = self.get_combined_data()
        
        if df.empty:
            return {}
        
        # Star rating distribution
        star_data = df[df['star_rating'].notna()]
        star_distribution = star_data['star_rating'].value_counts().sort_index().to_dict()
        
        # Satisfaction by method
        satisfaction_by_method = {}
        for method in df['method'].unique():
            method_feedback = df[(df['method'] == method) & (df['thumbs_up'].notna())]
            if len(method_feedback) > 0:
                satisfaction_rate = (method_feedback['thumbs_up'] == True).sum() / len(method_feedback) * 100
                satisfaction_by_method[method] = round(satisfaction_rate, 1)
        
        return {
            'star_distribution': star_distribution,
            'satisfaction_by_method': satisfaction_by_method
        }
