"""
Database Manager for FinTech Forecasting Application
Supports SQLite, MongoDB, and other databases
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manager for database operations"""
    
    def __init__(self, db_path='data/fintech.db'):
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database and create tables"""
        try:
            # Ensure directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._create_tables()
            logger.info(f"Database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def _create_tables(self):
        """Create necessary database tables"""
        cursor = self.conn.cursor()
        
        # Market data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                technical_indicators TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, exchange, date)
            )
        ''')
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                model_name TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                forecast_horizon INTEGER,
                predicted_value REAL,
                confidence_lower REAL,
                confidence_upper REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Model performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                model_name TEXT NOT NULL,
                rmse REAL,
                mae REAL,
                mape REAL,
                r2 REAL,
                training_date TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # News data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                headline TEXT,
                summary TEXT,
                source TEXT,
                link TEXT,
                sentiment_score REAL,
                sentiment_label TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
        logger.info("Database tables created successfully")
    
    def save_market_data(self, df, symbol, exchange):
        """
        Save market data to database
        
        Args:
            df: DataFrame with market data
            symbol: Stock/crypto symbol
            exchange: Exchange name
        """
        try:
            # Prepare data
            records = []
            for _, row in df.iterrows():
                # Extract technical indicators
                tech_indicators = {}
                for col in df.columns:
                    if col not in ['Date', 'Symbol', 'Exchange', 'Open', 'High', 'Low', 'Close', 'Volume']:
                        tech_indicators[col] = float(row[col]) if pd.notna(row[col]) else None
                
                record = (
                    symbol,
                    exchange,
                    str(row.get('Date', row.name)),
                    float(row['Open']) if pd.notna(row['Open']) else None,
                    float(row['High']) if pd.notna(row['High']) else None,
                    float(row['Low']) if pd.notna(row['Low']) else None,
                    float(row['Close']) if pd.notna(row['Close']) else None,
                    int(row['Volume']) if pd.notna(row['Volume']) else None,
                    json.dumps(tech_indicators)
                )
                records.append(record)
            
            # Insert data
            cursor = self.conn.cursor()
            cursor.executemany('''
                INSERT OR REPLACE INTO market_data 
                (symbol, exchange, date, open, high, low, close, volume, technical_indicators)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', records)
            
            self.conn.commit()
            logger.info(f"Saved {len(records)} market data records for {symbol}")
            
        except Exception as e:
            logger.error(f"Error saving market data: {str(e)}")
            raise
    
    def get_market_data(self, symbol, exchange, start_date=None, end_date=None):
        """
        Retrieve market data from database
        
        Args:
            symbol: Stock/crypto symbol
            exchange: Exchange name
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            DataFrame with market data
        """
        try:
            query = '''
                SELECT * FROM market_data 
                WHERE symbol = ? AND exchange = ?
            '''
            params = [symbol, exchange]
            
            if start_date:
                query += ' AND date >= ?'
                params.append(start_date)
            
            if end_date:
                query += ' AND date <= ?'
                params.append(end_date)
            
            query += ' ORDER BY date ASC'
            
            df = pd.read_sql_query(query, self.conn, params=params)
            
            # Parse technical indicators
            if 'technical_indicators' in df.columns:
                for idx, row in df.iterrows():
                    if row['technical_indicators']:
                        indicators = json.loads(row['technical_indicators'])
                        for key, value in indicators.items():
                            df.at[idx, key] = value
                
                df = df.drop('technical_indicators', axis=1)
            
            logger.info(f"Retrieved {len(df)} market data records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving market data: {str(e)}")
            return pd.DataFrame()
    
    def save_predictions(self, predictions_df, symbol, exchange, model_name):
        """
        Save model predictions to database
        
        Args:
            predictions_df: DataFrame with predictions
            symbol: Stock/crypto symbol
            exchange: Exchange name
            model_name: Name of the model
        """
        try:
            records = []
            for _, row in predictions_df.iterrows():
                record = (
                    symbol,
                    exchange,
                    model_name,
                    str(row.get('date', row.get('Date', ''))),
                    int(row.get('horizon', 1)),
                    float(row.get('predicted_value', 0)),
                    float(row.get('confidence_lower', 0)),
                    float(row.get('confidence_upper', 0))
                )
                records.append(record)
            
            cursor = self.conn.cursor()
            cursor.executemany('''
                INSERT INTO predictions 
                (symbol, exchange, model_name, prediction_date, forecast_horizon, 
                 predicted_value, confidence_lower, confidence_upper)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', records)
            
            self.conn.commit()
            logger.info(f"Saved {len(records)} predictions for {symbol} using {model_name}")
            
        except Exception as e:
            logger.error(f"Error saving predictions: {str(e)}")
            raise
    
    def get_predictions(self, symbol, exchange, model_name=None):
        """
        Retrieve predictions from database
        
        Args:
            symbol: Stock/crypto symbol
            exchange: Exchange name
            model_name: Model name (optional)
            
        Returns:
            DataFrame with predictions
        """
        try:
            query = '''
                SELECT * FROM predictions 
                WHERE symbol = ? AND exchange = ?
            '''
            params = [symbol, exchange]
            
            if model_name:
                query += ' AND model_name = ?'
                params.append(model_name)
            
            query += ' ORDER BY prediction_date DESC, forecast_horizon ASC'
            
            df = pd.read_sql_query(query, self.conn, params=params)
            logger.info(f"Retrieved {len(df)} predictions for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving predictions: {str(e)}")
            return pd.DataFrame()
    
    def save_model_performance(self, symbol, exchange, model_name, metrics):
        """
        Save model performance metrics
        
        Args:
            symbol: Stock/crypto symbol
            exchange: Exchange name
            model_name: Model name
            metrics: Dictionary of performance metrics
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO model_performance 
                (symbol, exchange, model_name, rmse, mae, mape, r2, training_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                exchange,
                model_name,
                metrics.get('RMSE', 0),
                metrics.get('MAE', 0),
                metrics.get('MAPE', 0),
                metrics.get('R2', 0),
                datetime.now().strftime('%Y-%m-%d')
            ))
            
            self.conn.commit()
            logger.info(f"Saved performance metrics for {model_name}")
            
        except Exception as e:
            logger.error(f"Error saving model performance: {str(e)}")
            raise
    
    def get_model_performance(self, symbol=None, exchange=None):
        """
        Retrieve model performance metrics
        
        Args:
            symbol: Stock/crypto symbol (optional)
            exchange: Exchange name (optional)
            
        Returns:
            DataFrame with performance metrics
        """
        try:
            query = 'SELECT * FROM model_performance WHERE 1=1'
            params = []
            
            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)
            
            if exchange:
                query += ' AND exchange = ?'
                params.append(exchange)
            
            query += ' ORDER BY created_at DESC'
            
            df = pd.read_sql_query(query, self.conn, params=params)
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving model performance: {str(e)}")
            return pd.DataFrame()
    
    def save_news_data(self, news_df, symbol):
        """Save news data to database"""
        try:
            records = []
            for _, row in news_df.iterrows():
                record = (
                    symbol,
                    str(row.get('date', '')),
                    str(row.get('headline', '')),
                    str(row.get('summary', '')),
                    str(row.get('source', '')),
                    str(row.get('link', '')),
                    float(row.get('sentiment_score', 0)),
                    str(row.get('sentiment_label', ''))
                )
                records.append(record)
            
            cursor = self.conn.cursor()
            cursor.executemany('''
                INSERT OR REPLACE INTO news_data 
                (symbol, date, headline, summary, source, link, sentiment_score, sentiment_label)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', records)
            
            self.conn.commit()
            logger.info(f"Saved {len(records)} news records for {symbol}")
            
        except Exception as e:
            logger.error(f"Error saving news data: {str(e)}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")