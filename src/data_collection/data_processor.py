"""
Data Processor for FinTech Application
Handles data cleaning, merging, and feature engineering
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and prepare data for modeling"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def merge_market_and_news_data(self, market_data, news_data, symbol):
        """
        Merge market data with news sentiment data
        
        Args:
            market_data: DataFrame with market data
            news_data: DataFrame with news data
            symbol: Stock/crypto symbol
            
        Returns:
            Merged DataFrame
        """
        try:
            if news_data.empty:
                self.logger.warning("No news data to merge")
                return market_data
            
            # Ensure date columns are datetime
            market_data['Date'] = pd.to_datetime(market_data['Date'])
            news_data['date'] = pd.to_datetime(news_data['date'])
            
            # Aggregate news sentiment by date
            daily_sentiment = news_data.groupby('date').agg({
                'sentiment_score': ['mean', 'std', 'count'],
                'relevance_score': 'mean' if 'relevance_score' in news_data.columns else lambda x: 1.0
            }).reset_index()
            
            # Flatten column names
            daily_sentiment.columns = ['Date', 'avg_sentiment', 'sentiment_std', 
                                      'news_count', 'avg_relevance']
            
            # Merge with market data
            merged_data = pd.merge(market_data, daily_sentiment, 
                                  on='Date', how='left')
            
            # Fill missing sentiment values
            merged_data['avg_sentiment'].fillna(0, inplace=True)
            merged_data['sentiment_std'].fillna(0, inplace=True)
            merged_data['news_count'].fillna(0, inplace=True)
            merged_data['avg_relevance'].fillna(0, inplace=True)
            
            self.logger.info(f"Merged market and news data: {len(merged_data)} records")
            return merged_data
            
        except Exception as e:
            self.logger.error(f"Error merging data: {str(e)}")
            return market_data
    
    def clean_and_validate_data(self, data, symbol):
        """
        Clean and validate dataset
        
        Args:
            data: DataFrame to clean
            symbol: Stock/crypto symbol
            
        Returns:
            Cleaned DataFrame
        """
        try:
            self.logger.info(f"Cleaning data for {symbol}")
            
            # Remove duplicates
            initial_len = len(data)
            data = data.drop_duplicates(subset=['Date'], keep='last')
            self.logger.info(f"Removed {initial_len - len(data)} duplicate records")
            
            # Sort by date
            data = data.sort_values('Date').reset_index(drop=True)
            
            # Handle missing values in critical columns
            critical_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            for col in critical_cols:
                if col in data.columns:
                    # Forward fill then backward fill
                    data[col].fillna(method='ffill', inplace=True)
                    data[col].fillna(method='bfill', inplace=True)
            
            # Remove rows with any remaining NaN in critical columns
            before_drop = len(data)
            data = data.dropna(subset=critical_cols, how='any')
            self.logger.info(f"Removed {before_drop - len(data)} rows with missing critical values")
            
            # Validate price data
            data = data[(data['Open'] > 0) & (data['High'] > 0) & 
                       (data['Low'] > 0) & (data['Close'] > 0) & 
                       (data['Volume'] >= 0)]
            
            # Validate high/low relationships
            data = data[(data['High'] >= data['Low']) & 
                       (data['High'] >= data['Open']) & 
                       (data['High'] >= data['Close']) &
                       (data['Low'] <= data['Open']) & 
                       (data['Low'] <= data['Close'])]
            
            self.logger.info(f"Data validation complete: {len(data)} records remaining")
            return data
            
        except Exception as e:
            self.logger.error(f"Error cleaning data: {str(e)}")
            return data
    
    def create_feature_summary(self, data):
        """
        Create summary of features in dataset
        
        Args:
            data: DataFrame
            
        Returns:
            Dictionary with feature summary
        """
        summary = {
            'total_records': len(data),
            'total_features': len(data.columns),
            'date_range': {
                'start': str(data['Date'].min()) if 'Date' in data.columns else 'Unknown',
                'end': str(data['Date'].max()) if 'Date' in data.columns else 'Unknown'
            },
            'missing_values': int(data.isnull().sum().sum()),
            'feature_categories': {}
        }
        
        # Categorize features
        price_features = [col for col in data.columns if col in ['Open', 'High', 'Low', 'Close']]
        technical_features = [col for col in data.columns if any(x in col for x in ['SMA', 'EMA', 'RSI', 'MACD', 'BB'])]
        sentiment_features = [col for col in data.columns if 'sentiment' in col.lower() or 'news' in col.lower()]
        
        summary['feature_categories'] = {
            'price_features': len(price_features),
            'technical_indicators': len(technical_features),
            'sentiment_features': len(sentiment_features),
            'other_features': len(data.columns) - len(price_features) - len(technical_features) - len(sentiment_features)
        }
        
        return summary
    
    def get_minimal_feature_set(self, data):
        """
        Get minimal set of features for modeling
        
        Returns:
            List of essential feature names
        """
        essential_features = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Add key technical indicators if available
        optional_features = ['SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 
                           'RSI', 'MACD', 'BB_Upper', 'BB_Lower', 
                           'Volatility_20d', 'Daily_Return']
        
        minimal_set = essential_features.copy()
        
        for feature in optional_features:
            if feature in data.columns:
                minimal_set.append(feature)
        
        return minimal_set
    
    def save_to_csv(self, data, symbol, exchange, news_data=None):
        """
        Save data to CSV files
        
        Args:
            data: Market data DataFrame
            symbol: Stock/crypto symbol
            exchange: Exchange name
            news_data: News data DataFrame (optional)
            
        Returns:
            Dictionary of saved file paths
        """
        try:
            from config.config import PROCESSED_DATA_DIR
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f"{symbol}_{exchange}_{timestamp}"
            
            saved_files = {}
            
            # Save market data
            market_file = PROCESSED_DATA_DIR / f"{base_filename}_market.csv"
            data.to_csv(market_file, index=False)
            saved_files['market_data'] = str(market_file)
            self.logger.info(f"Saved market data to {market_file}")
            
            # Save news data if available
            if news_data is not None and not news_data.empty:
                news_file = PROCESSED_DATA_DIR / f"{base_filename}_news.csv"
                news_data.to_csv(news_file, index=False)
                saved_files['news_data'] = str(news_file)
                self.logger.info(f"Saved news data to {news_file}")
            
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Error saving CSV files: {str(e)}")
            return {}
    
    def save_to_json(self, data, symbol, exchange, news_data=None):
        """
        Save data to JSON files
        
        Args:
            data: Market data DataFrame
            symbol: Stock/crypto symbol
            exchange: Exchange name
            news_data: News data DataFrame (optional)
            
        Returns:
            Dictionary of saved file paths
        """
        try:
            from config.config import PROCESSED_DATA_DIR
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f"{symbol}_{exchange}_{timestamp}"
            
            saved_files = {}
            
            # Save market data
            market_file = PROCESSED_DATA_DIR / f"{base_filename}_market.json"
            data.to_json(market_file, orient='records', date_format='iso', indent=2)
            saved_files['market_data_json'] = str(market_file)
            self.logger.info(f"Saved market data to {market_file}")
            
            # Save news data if available
            if news_data is not None and not news_data.empty:
                news_file = PROCESSED_DATA_DIR / f"{base_filename}_news.json"
                news_data.to_json(news_file, orient='records', date_format='iso', indent=2)
                saved_files['news_data_json'] = str(news_file)
                self.logger.info(f"Saved news data to {news_file}")
            
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Error saving JSON files: {str(e)}")
            return {}