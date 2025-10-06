"""
Neural Network Models for Time Series Forecasting
Implements LSTM, GRU, and Transformer architectures
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import os

logger = logging.getLogger(__name__)


class LSTMModel:
    """LSTM Model for time series forecasting"""
    
    def __init__(self, sequence_length=60, units=[64, 32], dropout=0.2, 
                 learning_rate=0.001):
        self.sequence_length = sequence_length
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = MinMaxScaler()
        self.history = None
        
    def build_model(self, input_shape):
        """Build LSTM architecture"""
        model = keras.Sequential()
        
        # First LSTM layer
        model.add(layers.LSTM(
            self.units[0],
            return_sequences=True if len(self.units) > 1 else False,
            input_shape=input_shape
        ))
        model.add(layers.Dropout(self.dropout))
        
        # Additional LSTM layers
        for i, units in enumerate(self.units[1:]):
            return_seq = i < len(self.units) - 2
            model.add(layers.LSTM(units, return_sequences=return_seq))
            model.add(layers.Dropout(self.dropout))
        
        # Output layer
        model.add(layers.Dense(1))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_data(self, data, is_training=True):
        """
        Prepare data for LSTM
        
        Args:
            data: Time series data
            is_training: Whether preparing training data
            
        Returns:
            X, y arrays
        """
        if is_training:
            scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        else:
            scaled_data = self.scaler.transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def fit(self, train_data, validation_split=0.1, epochs=100, batch_size=32,
            model_path=None):
        """
        Train LSTM model
        
        Args:
            train_data: Training data
            validation_split: Validation split ratio
            epochs: Number of training epochs
            batch_size: Batch size
            model_path: Path to save best model
        """
        try:
            logger.info("Preparing data for LSTM training")
            X_train, y_train = self.prepare_data(train_data, is_training=True)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            
            logger.info(f"Building LSTM model with architecture: {self.units}")
            self.model = self.build_model((X_train.shape[1], 1))
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ]
            
            if model_path:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                callbacks.append(ModelCheckpoint(model_path, save_best_only=True))
            
            logger.info("Training LSTM model")
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=0
            )
            
            logger.info("LSTM model training completed")
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {str(e)}")
            raise
    
    def predict(self, data, steps=1):
        """
        Make predictions
        
        Args:
            data: Input data for prediction
            steps: Number of steps to predict
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        try:
            # Prepare last sequence
            scaled_data = self.scaler.transform(data.reshape(-1, 1))
            last_sequence = scaled_data[-self.sequence_length:]
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(steps):
                # Reshape for prediction
                input_seq = current_sequence.reshape(1, self.sequence_length, 1)
                
                # Predict next value
                pred_scaled = self.model.predict(input_seq, verbose=0)[0, 0]
                
                # Inverse transform
                pred = self.scaler.inverse_transform([[pred_scaled]])[0, 0]
                predictions.append(pred)
                
                # Update sequence
                current_sequence = np.append(current_sequence[1:], [[pred_scaled]], axis=0)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error in LSTM prediction: {str(e)}")
            raise
    
    def evaluate(self, test_data):
        """Evaluate model on test data"""
        X_test, y_test = self.prepare_data(test_data, is_training=False)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        predictions = self.model.predict(X_test, verbose=0)
        predictions = self.scaler.inverse_transform(predictions)
        y_test = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        r2 = r2_score(y_test, predictions)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        }


class GRUModel:
    """GRU Model for time series forecasting"""
    
    def __init__(self, sequence_length=60, units=[64, 32], dropout=0.2, 
                 learning_rate=0.001):
        self.sequence_length = sequence_length
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = MinMaxScaler()
        self.history = None
        
    def build_model(self, input_shape):
        """Build GRU architecture"""
        model = keras.Sequential()
        
        # First GRU layer
        model.add(layers.GRU(
            self.units[0],
            return_sequences=True if len(self.units) > 1 else False,
            input_shape=input_shape
        ))
        model.add(layers.Dropout(self.dropout))
        
        # Additional GRU layers
        for i, units in enumerate(self.units[1:]):
            return_seq = i < len(self.units) - 2
            model.add(layers.GRU(units, return_sequences=return_seq))
            model.add(layers.Dropout(self.dropout))
        
        # Output layer
        model.add(layers.Dense(1))
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_data(self, data, is_training=True):
        """Prepare data for GRU"""
        if is_training:
            scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        else:
            scaled_data = self.scaler.transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def fit(self, train_data, validation_split=0.1, epochs=100, batch_size=32,
            model_path=None):
        """Train GRU model"""
        try:
            logger.info("Preparing data for GRU training")
            X_train, y_train = self.prepare_data(train_data, is_training=True)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            
            logger.info(f"Building GRU model with architecture: {self.units}")
            self.model = self.build_model((X_train.shape[1], 1))
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ]
            
            if model_path:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                callbacks.append(ModelCheckpoint(model_path, save_best_only=True))
            
            logger.info("Training GRU model")
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=0
            )
            
            logger.info("GRU model training completed")
            
        except Exception as e:
            logger.error(f"Error training GRU model: {str(e)}")
            raise
    
    def predict(self, data, steps=1):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        try:
            scaled_data = self.scaler.transform(data.reshape(-1, 1))
            last_sequence = scaled_data[-self.sequence_length:]
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(steps):
                input_seq = current_sequence.reshape(1, self.sequence_length, 1)
                pred_scaled = self.model.predict(input_seq, verbose=0)[0, 0]
                pred = self.scaler.inverse_transform([[pred_scaled]])[0, 0]
                predictions.append(pred)
                current_sequence = np.append(current_sequence[1:], [[pred_scaled]], axis=0)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error in GRU prediction: {str(e)}")
            raise
    
    def evaluate(self, test_data):
        """Evaluate model on test data"""
        X_test, y_test = self.prepare_data(test_data, is_training=False)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        predictions = self.model.predict(X_test, verbose=0)
        predictions = self.scaler.inverse_transform(predictions)
        y_test = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        r2 = r2_score(y_test, predictions)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        }


class TransformerModel:
    """Transformer Model for time series forecasting"""
    
    def __init__(self, sequence_length=60, num_heads=4, hidden_dim=128, 
                 num_layers=2, dropout=0.1, learning_rate=0.001):
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = MinMaxScaler()
        self.history = None
    
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        """Transformer encoder block"""
        # Multi-head attention
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs
        
        # Feed forward
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res
    
    def build_model(self, input_shape):
        """Build Transformer architecture"""
        inputs = keras.Input(shape=input_shape)
        x = inputs
        
        # Transformer encoder blocks
        for _ in range(self.num_layers):
            x = self.transformer_encoder(
                x,
                head_size=self.hidden_dim // self.num_heads,
                num_heads=self.num_heads,
                ff_dim=self.hidden_dim,
                dropout=self.dropout
            )
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        
        # Dense layers
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        outputs = layers.Dense(1)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_data(self, data, is_training=True):
        """Prepare data for Transformer"""
        if is_training:
            scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        else:
            scaled_data = self.scaler.transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def fit(self, train_data, validation_split=0.1, epochs=100, batch_size=32,
            model_path=None):
        """Train Transformer model"""
        try:
            logger.info("Preparing data for Transformer training")
            X_train, y_train = self.prepare_data(train_data, is_training=True)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            
            logger.info("Building Transformer model")
            self.model = self.build_model((X_train.shape[1], 1))
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)
            ]
            
            if model_path:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                callbacks.append(ModelCheckpoint(model_path, save_best_only=True))
            
            logger.info("Training Transformer model")
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=0
            )
            
            logger.info("Transformer model training completed")
            
        except Exception as e:
            logger.error(f"Error training Transformer model: {str(e)}")
            raise
    
    def predict(self, data, steps=1):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        try:
            scaled_data = self.scaler.transform(data.reshape(-1, 1))
            last_sequence = scaled_data[-self.sequence_length:]
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(steps):
                input_seq = current_sequence.reshape(1, self.sequence_length, 1)
                pred_scaled = self.model.predict(input_seq, verbose=0)[0, 0]
                pred = self.scaler.inverse_transform([[pred_scaled]])[0, 0]
                predictions.append(pred)
                current_sequence = np.append(current_sequence[1:], [[pred_scaled]], axis=0)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error in Transformer prediction: {str(e)}")
            raise
    
    def evaluate(self, test_data):
        """Evaluate model on test data"""
        X_test, y_test = self.prepare_data(test_data, is_training=False)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        predictions = self.model.predict(X_test, verbose=0)
        predictions = self.scaler.inverse_transform(predictions)
        y_test = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        r2 = r2_score(y_test, predictions)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        }