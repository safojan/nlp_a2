"""
Main Streamlit Application for FinTech Forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import *
from src.data_collection.market_data_scraper import MarketDataScraper
from src.data_collection.news_scraper import NewsScraper
from src.data_collection.data_processor import DataProcessor
from src.models.traditional_models import ARIMAModel, MovingAverageModel, TraditionalModelEnsemble
from src.models.neural_models import LSTMModel, GRUModel, TransformerModel
from src.database.db_manager import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


class ForecastingApp:
    """Main application class"""
    
    def __init__(self):
        self.db = DatabaseManager(DB_PATH)
        self.market_scraper = MarketDataScraper()
        self.news_scraper = NewsScraper()
        self.processor = DataProcessor()
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'predictions_made' not in st.session_state:
            st.session_state.predictions_made = False
        if 'market_data' not in st.session_state:
            st.session_state.market_data = None
        if 'predictions' not in st.session_state:
            st.session_state.predictions = {}
    
    def render_sidebar(self):
        """Render sidebar with input controls"""
        st.sidebar.title(f"{APP_ICON} FinTech Forecasting")
        st.sidebar.markdown("---")
        
        # Symbol selection
        st.sidebar.subheader("📊 Select Instrument")
        
        exchange = st.sidebar.selectbox(
            "Exchange",
            options=list(SUPPORTED_EXCHANGES.keys()),
            help="Select the exchange or asset type"
        )
        
        # Symbol input with suggestions
        if exchange == 'CRYPTO':
            suggested_symbols = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD']
        else:
            suggested_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        symbol = st.sidebar.selectbox(
            "Symbol",
            options=suggested_symbols,
            help="Select or enter a symbol"
        )
        
        custom_symbol = st.sidebar.text_input(
            "Or enter custom symbol",
            help="Enter a custom symbol if not in the list"
        )
        
        if custom_symbol:
            symbol = custom_symbol.upper()
        
        st.sidebar.markdown("---")
        
        # Data collection settings
        st.sidebar.subheader("⚙️ Data Settings")
        
        period = st.sidebar.selectbox(
            "Historical Period",
            options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
            index=3,
            help="Amount of historical data to fetch"
        )
        
        interval = st.sidebar.selectbox(
            "Data Interval",
            options=['1h', '1d', '1wk'],
            index=1,
            help="Time interval for data points"
        )
        
        st.sidebar.markdown("---")
        
        # Forecast settings
        st.sidebar.subheader("🔮 Forecast Settings")
        
        forecast_horizon = st.sidebar.selectbox(
            "Forecast Horizon",
            options=['1 Hour', '3 Hours', '24 Hours', '72 Hours'],
            index=2,
            help="How far ahead to predict"
        )
        
        # Model selection
        st.sidebar.subheader("🤖 Model Selection")
        
        use_traditional = st.sidebar.checkbox("Traditional Models", value=True)
        if use_traditional:
            traditional_models = st.sidebar.multiselect(
                "Select Traditional Models",
                options=['ARIMA', 'Moving Average', 'Ensemble'],
                default=['ARIMA', 'Moving Average']
            )
        else:
            traditional_models = []
        
        use_neural = st.sidebar.checkbox("Neural Models", value=True)
        if use_neural:
            neural_models = st.sidebar.multiselect(
                "Select Neural Models",
                options=['LSTM', 'GRU', 'Transformer'],
                default=['LSTM', 'GRU']
            )
        else:
            neural_models = []
        
        st.sidebar.markdown("---")
        
        # Action buttons
        load_data = st.sidebar.button("📥 Load Data", use_container_width=True)
        train_predict = st.sidebar.button("🚀 Train & Predict", use_container_width=True, 
                                         disabled=not st.session_state.data_loaded)
        
        return {
            'symbol': symbol,
            'exchange': exchange,
            'period': period,
            'interval': interval,
            'forecast_horizon': forecast_horizon,
            'traditional_models': traditional_models,
            'neural_models': neural_models,
            'load_data': load_data,
            'train_predict': train_predict
        }
    
    def load_data(self, symbol, exchange, period, interval):
        """Load market data"""
        try:
            with st.spinner(f"Loading data for {symbol}..."):
                # Check if data exists in database
                db_data = self.db.get_market_data(symbol, exchange)
                
                if not db_data.empty and len(db_data) > 100:
                    st.success("✅ Loaded data from database")
                    market_data = db_data
                else:
                    # Fetch new data
                    if exchange == 'CRYPTO':
                        market_data = self.market_scraper.get_crypto_data(symbol, period, interval)
                    else:
                        market_data = self.market_scraper.get_stock_data(symbol, exchange, period, interval)
                    
                    if market_data is None or market_data.empty:
                        st.error("❌ Failed to load data. Please check the symbol and try again.")
                        return None
                    
                    # Save to database
                    self.db.save_market_data(market_data, symbol, exchange)
                    st.success(f"✅ Loaded {len(market_data)} records from API")
                
                # Store in session state
                st.session_state.market_data = market_data
                st.session_state.data_loaded = True
                
                return market_data
                
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            logger.error(f"Error loading data: {str(e)}")
            return None
    
    def create_candlestick_chart(self, data, predictions=None, title="Price Chart"):
        """Create interactive candlestick chart with predictions"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(title, 'Volume')
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data['Date'] if 'Date' in data.columns else data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # Add moving averages if available
        if 'SMA_20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['Date'] if 'Date' in data.columns else data.index,
                    y=data['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        if 'SMA_50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['Date'] if 'Date' in data.columns else data.index,
                    y=data['SMA_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
        
        # Add predictions if available
        if predictions:
            for model_name, pred_data in predictions.items():
                if 'dates' in pred_data and 'values' in pred_data:
                    fig.add_trace(
                        go.Scatter(
                            x=pred_data['dates'],
                            y=pred_data['values'],
                            mode='lines+markers',
                            name=f'{model_name} Prediction',
                            line=dict(dash='dash', width=2),
                            marker=dict(size=8)
                        ),
                        row=1, col=1
                    )
        
        # Volume bars
        colors = ['red' if row['Close'] < row['Open'] else 'green' 
                 for _, row in data.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=data['Date'] if 'Date' in data.columns else data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=700,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            template='plotly_dark',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    def train_and_predict(self, data, traditional_models, neural_models, horizon_hours):
        """Train models and make predictions"""
        predictions = {}
        metrics = {}
        
        try:
            # Prepare data
            close_prices = data['Close'].values
            train_size = int(len(close_prices) * TRAIN_TEST_SPLIT)
            train_data = close_prices[:train_size]
            test_data = close_prices[train_size:]
            
            # Convert horizon to steps
            steps = horizon_hours  # Simplified - adjust based on interval
            
            # Train Traditional Models
            if traditional_models:
                st.subheader("🔧 Training Traditional Models")
                
                for model_name in traditional_models:
                    with st.spinner(f"Training {model_name}..."):
                        try:
                            if model_name == 'ARIMA':
                                model = ARIMAModel(order=(5, 1, 0))
                                model.fit(train_data)
                                preds = model.predict(steps=steps)
                                metrics[model_name] = model.evaluate(test_data[:steps])
                            
                            elif model_name == 'Moving Average':
                                model = MovingAverageModel(window=20)
                                model.fit(train_data)
                                preds = model.predict(steps=steps)
                                metrics[model_name] = model.evaluate(test_data[:steps])
                            
                            elif model_name == 'Ensemble':
                                ensemble = TraditionalModelEnsemble()
                                ensemble.add_model('ARIMA', ARIMAModel(order=(5, 1, 0)))
                                ensemble.add_model('MA', MovingAverageModel(window=20))
                                ensemble.fit(train_data)
                                preds = ensemble.predict(steps=steps)
                                metrics[model_name] = ensemble.evaluate(test_data[:steps])
                            
                            # Store predictions
                            last_date = pd.to_datetime(data['Date'].iloc[-1] if 'Date' in data.columns else data.index[-1])
                            pred_dates = pd.date_range(start=last_date + timedelta(hours=1), 
                                                      periods=steps, freq='H')
                            
                            predictions[model_name] = {
                                'dates': pred_dates,
                                'values': preds
                            }
                            
                            st.success(f"✅ {model_name} trained successfully")
                            
                        except Exception as e:
                            st.warning(f"⚠️ Failed to train {model_name}: {str(e)}")
                            logger.error(f"Error training {model_name}: {str(e)}")
            
            # Train Neural Models
            if neural_models:
                st.subheader("🧠 Training Neural Models")
                
                for model_name in neural_models:
                    with st.spinner(f"Training {model_name}... (this may take a while)"):
                        try:
                            if model_name == 'LSTM':
                                model = LSTMModel(sequence_length=SEQUENCE_LENGTH,
                                                units=[64, 32], dropout=0.2)
                                model.fit(train_data, epochs=50, batch_size=32, 
                                        validation_split=VALIDATION_SPLIT)
                                preds = model.predict(close_prices, steps=steps)
                                metrics[model_name] = model.evaluate(close_prices)
                            
                            elif model_name == 'GRU':
                                model = GRUModel(sequence_length=SEQUENCE_LENGTH,
                                               units=[64, 32], dropout=0.2)
                                model.fit(train_data, epochs=50, batch_size=32,
                                        validation_split=VALIDATION_SPLIT)
                                preds = model.predict(close_prices, steps=steps)
                                metrics[model_name] = model.evaluate(close_prices)
                            
                            elif model_name == 'Transformer':
                                model = TransformerModel(sequence_length=SEQUENCE_LENGTH,
                                                       num_heads=4, hidden_dim=128)
                                model.fit(train_data, epochs=50, batch_size=32,
                                        validation_split=VALIDATION_SPLIT)
                                preds = model.predict(close_prices, steps=steps)
                                metrics[model_name] = model.evaluate(close_prices)
                            
                            # Store predictions
                            last_date = pd.to_datetime(data['Date'].iloc[-1] if 'Date' in data.columns else data.index[-1])
                            pred_dates = pd.date_range(start=last_date + timedelta(hours=1),
                                                      periods=steps, freq='H')
                            
                            predictions[model_name] = {
                                'dates': pred_dates,
                                'values': preds
                            }
                            
                            st.success(f"✅ {model_name} trained successfully")
                            
                        except Exception as e:
                            st.warning(f"⚠️ Failed to train {model_name}: {str(e)}")
                            logger.error(f"Error training {model_name}: {str(e)}")
            
            return predictions, metrics
            
        except Exception as e:
            st.error(f"❌ Error in training: {str(e)}")
            logger.error(f"Error in training: {str(e)}")
            return {}, {}
    
    def display_metrics(self, metrics):
        """Display model performance metrics"""
        if not metrics:
            return
        
        st.subheader("📊 Model Performance Comparison")
        
        # Create metrics dataframe
        metrics_df = pd.DataFrame(metrics).T
        metrics_df = metrics_df.round(4)
        
        # Display as table
        st.dataframe(metrics_df, use_container_width=True)
        
        # Create comparison chart
        fig = go.Figure()
        
        for metric in ['RMSE', 'MAE', 'MAPE']:
            if metric in metrics_df.columns:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=metrics_df.index,
                    y=metrics_df[metric],
                    text=metrics_df[metric].round(2),
                    textposition='auto',
                ))
        
        fig.update_layout(
            title="Model Performance Metrics Comparison",
            xaxis_title="Model",
            yaxis_title="Metric Value",
            barmode='group',
            height=400,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Main application loop"""
        # Header
        st.title(f"{APP_ICON} FinTech Forecasting Dashboard")
        st.markdown("Real-time stock and cryptocurrency price forecasting using traditional and neural models")
        st.markdown("---")
        
        # Render sidebar and get settings
        settings = self.render_sidebar()
        
        # Load data button
        if settings['load_data']:
            self.load_data(
                settings['symbol'],
                settings['exchange'],
                settings['period'],
                settings['interval']
            )
        
        # Main content area
        if st.session_state.data_loaded and st.session_state.market_data is not None:
            data = st.session_state.market_data
            
            # Display data summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Symbol", settings['symbol'])
            with col2:
                st.metric("Records", len(data))
            with col3:
                latest_price = data['Close'].iloc[-1]
                st.metric("Latest Price", f"${latest_price:.2f}")
            with col4:
                price_change = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
                st.metric("Period Change", f"{price_change:+.2f}%")
            
            st.markdown("---")
            
            # Display candlestick chart
            st.subheader("📈 Price Chart with Technical Indicators")
            chart = self.create_candlestick_chart(
                data,
                predictions=st.session_state.predictions if st.session_state.predictions_made else None,
                title=f"{settings['symbol']} - {settings['exchange']}"
            )
            st.plotly_chart(chart, use_container_width=True)
            
            # Train and predict button
            if settings['train_predict']:
                horizon_map = {'1 Hour': 1, '3 Hours': 3, '24 Hours': 24, '72 Hours': 72}
                horizon_hours = horizon_map[settings['forecast_horizon']]
                
                predictions, metrics = self.train_and_predict(
                    data,
                    settings['traditional_models'],
                    settings['neural_models'],
                    horizon_hours
                )
                
                if predictions:
                    st.session_state.predictions = predictions
                    st.session_state.predictions_made = True
                    
                    # Display metrics
                    self.display_metrics(metrics)
                    
                    # Display predictions
                    st.subheader("🔮 Predictions")
                    pred_df = pd.DataFrame({
                        model: pred_data['values']
                        for model, pred_data in predictions.items()
                    })
                    st.dataframe(pred_df.round(2), use_container_width=True)
                    
                    st.success("✅ Predictions generated successfully!")
        
        else:
            # Welcome message
            st.info("👈 Please select a symbol and load data to get started")
            
            st.markdown("""
            ### Features:
            - 📊 Real-time market data collection
            - 🤖 Multiple forecasting models (ARIMA, LSTM, GRU, Transformer)
            - 📈 Interactive candlestick charts
            - 🔮 Multi-horizon predictions
            - 💾 Database storage for historical data
            - 📉 Model performance comparison
            
            ### How to use:
            1. Select an exchange and symbol from the sidebar
            2. Click "Load Data" to fetch historical data
            3. Choose your preferred models
            4. Click "Train & Predict" to generate forecasts
            """)


if __name__ == "__main__":
    app = ForecastingApp()
    app.run()