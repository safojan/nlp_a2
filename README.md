# FinTech Forecasting Application

A comprehensive stock and cryptocurrency forecasting application built with Streamlit, featuring both traditional (ARIMA, Moving Averages) and neural network models (LSTM, GRU, Transformer).

## Features

- 📊 **Real-time Data Collection**: Fetch historical market data using yfinance API
- 🤖 **Multiple Forecasting Models**: 
  - Traditional: ARIMA, VAR, Moving Averages, Exponential Smoothing
  - Neural: LSTM, GRU, Transformer
- 📈 **Interactive Visualizations**: Candlestick charts with technical indicators
- 🔮 **Multi-horizon Predictions**: Forecast 1hr, 3hrs, 24hrs, or 72hrs ahead
- 💾 **Database Storage**: SQLite database for historical data and predictions
- 📉 **Performance Metrics**: RMSE, MAE, MAPE, R² scores
- 📰 **News Sentiment Analysis**: Integrate news sentiment into predictions

## Installation

### 1. Clone or Setup Repository

```bash
# Create project directory
mkdir fintech-forecasting
cd fintech-forecasting
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env
# Edit .env with your settings
```

## Project Structure

```
fintech-forecasting/
├── .env                          # Environment variables
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── config/
│   └── config.py                 # Configuration settings
│
├── data/
│   ├── raw/                      # Raw data
│   ├── processed/                # Processed data
│   └── predictions/              # Predictions
│
├── src/
│   ├── data_collection/
│   │   ├── market_data_scraper.py
│   │   ├── news_scraper.py
│   │   └── data_processor.py
│   │
│   ├── models/
│   │   ├── traditional_models.py
│   │   ├── neural_models.py
│   │   └── ensemble_model.py
│   │
│   ├── database/
│   │   ├── db_manager.py
│   │   └── schema.py
│   │
│   └── utils/
│       └── visualization.py
│
├── app/
│   └── streamlit_app.py          # Main Streamlit application
│
└── logs/                          # Application logs
```

## Usage

### 1. Run the Streamlit Application

```bash
streamlit run app/streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

### 2. Using the Application

1. **Select an Instrument**:
   - Choose an exchange (NASDAQ, NYSE, CRYPTO, etc.)
   - Select or enter a symbol (e.g., AAPL, BTC-USD)

2. **Configure Data Settings**:
   - Historical period (1mo, 3mo, 6mo, 1y, etc.)
   - Data interval (1h, 1d, 1wk)

3. **Load Data**:
   - Click "Load Data" to fetch historical data
   - Data will be cached in the database

4. **Select Models**:
   - Choose traditional models (ARIMA, Moving Average, Ensemble)
   - Choose neural models (LSTM, GRU, Transformer)

5. **Generate Predictions**:
   - Select forecast horizon (1hr, 3hrs, 24hrs, 72hrs)
   - Click "Train & Predict"
   - View predictions overlaid on candlestick chart

### 3. Command Line Data Collection (Optional)

You can also collect data using the command line:

```bash
# Collect stock data
python main.py --symbol AAPL --exchange NASDAQ --period 1y

# Collect crypto data
python main.py --symbol BTC-USD --exchange CRYPTO --period 6mo

# Collect with news data
python main.py --symbol TSLA --exchange NASDAQ --period 1y --news-days 30

# Skip news collection
python main.py --symbol ETH-USD --exchange CRYPTO --no-news
```

## Models Implemented

### Traditional Models

1. **ARIMA (AutoRegressive Integrated Moving Average)**
   - Good for univariate time series
   - Captures trend and seasonality
   - Parameters: order=(5, 1, 0)

2. **VAR (Vector Autoregression)**
   - Multivariate time series forecasting
   - Captures relationships between variables
   - Parameters: maxlags=15

3. **Moving Average**
   - Simple and weighted moving averages
   - Good baseline model
   - Parameters: window=20

4. **Exponential Smoothing**
   - Holt-Winters method
   - Handles seasonality
   - Parameters: seasonal='add', periods=12

### Neural Network Models

1. **LSTM (Long Short-Term Memory)**
   - Captures long-term dependencies
   - Architecture: [64, 32] units
   - Dropout: 0.2

2. **GRU (Gated Recurrent Unit)**
   - More efficient than LSTM
   - Architecture: [64, 32] units
   - Dropout: 0.2

3. **Transformer**
   - State-of-the-art architecture
   - Multi-head attention mechanism
   - Parameters: 4 heads, 128 hidden dim

## Performance Metrics

All models are evaluated using:
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: R-squared score

## Database Schema

### Market Data Table
```sql
CREATE TABLE market_data (
    id INTEGER PRIMARY KEY,
    symbol TEXT,
    exchange TEXT,
    date TEXT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    technical_indicators TEXT,
    created_at TEXT
);
```

### Predictions Table
```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    symbol TEXT,
    model_name TEXT,
    prediction_date TEXT,
    forecast_horizon INTEGER,
    predicted_value REAL,
    confidence_lower REAL,
    confidence_upper REAL,
    created_at TEXT
);
```

## Configuration

Key configuration settings in `config/config.py`:

```python
# Database
DB_TYPE = 'sqlite'  # or 'mongodb'
DB_PATH = 'data/fintech.db'

# Model Settings
SEQUENCE_LENGTH = 60
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.1

# Prediction Horizons
PREDICTION_HORIZONS = [1, 3, 24, 72]  # hours

# Technical Indicators
TECHNICAL_INDICATORS = {
    'sma_periods': [5, 10, 20, 50, 200],
    'ema_periods': [12, 26],
    'rsi_period': 14,
    'bollinger_period': 20
}
```

## Technical Indicators Included

- **Moving Averages**: SMA (5, 10, 20, 50, 200), EMA (12, 26)
- **Momentum**: RSI (14-day)
- **Trend**: MACD (12, 26, 9)
- **Volatility**: Bollinger Bands (20-day, 2 std)
- **Volume**: Volume ratios and moving averages
- **Returns**: Daily returns, log returns

## Troubleshooting

### Common Issues

1. **Module not found errors**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Database locked error**:
   - Close other connections to the database
   - Delete `data/fintech.db` and restart

3. **TensorFlow/Keras errors**:
   ```bash
   pip install tensorflow==2.15.0 keras==2.15.0
   ```

4. **Memory errors with neural models**:
   - Reduce batch size in config
   - Reduce sequence length
   - Use fewer epochs

### GitHub Codespaces

When running in GitHub Codespaces:

```bash
# Make port 8501 public
# In Codespaces: Ports tab → Forward port 8501 → Make Public

# Run the app
streamlit run app/streamlit_app.py
```

## Testing

Run unit tests:

```bash
pytest tests/
```

Run specific test:

```bash
pytest tests/test_models.py -v
```

## Development

### Adding New Models

1. Create model class in `src/models/`
2. Implement `fit()`, `predict()`, and `evaluate()` methods
3. Add to model selection in `streamlit_app.py`

### Adding New Data Sources

1. Create scraper in `src/data_collection/`
2. Follow the pattern from existing scrapers
3. Update `config.py` with new source settings

## Assignment Requirements Checklist

- ✅ Front-end web interface (Streamlit)
- ✅ Database backend (SQLite with MongoDB support)
- ✅ Traditional forecasting models (ARIMA, Moving Averages, VAR)
- ✅ Neural network models (LSTM, GRU, Transformer)
- ✅ Candlestick chart visualization with predictions
- ✅ Multiple forecast horizons (1hr, 3hrs, 24hrs, 72hrs)
- ✅ Performance metrics (RMSE, MAE, MAPE, R²)
- ✅ Modular code structure
- ✅ Version control ready
- ✅ Requirements file
- ✅ Documentation

## Report Generation

For your assignment report, the application tracks:
- Model training times
- Prediction accuracy
- Performance comparisons
- Architecture diagrams can be generated from the code structure

## License

This project is for educational purposes as part of CS4063 - Natural Language Processing course.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the code documentation
3. Check logs in `logs/app.log`

## Future Enhancements

- Real-time data streaming
- More advanced ensemble methods
- Sentiment analysis from social media
- Portfolio optimization
- Risk management features
- Docker containerization
- API endpoints for predictions