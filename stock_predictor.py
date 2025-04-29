import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import seaborn as sns



# These are helper functions for the fallback calculations
def calculate_rsi(series, period=14):

    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df, period=14):

    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

# I set random seeds for reproducibility
np.random.seed(42)

class StockPricePredictor:

    def __init__(self, ticker, start_date, end_date, seq_length=60, test_size=0.2):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.seq_length = seq_length
        self.test_size = test_size
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()


    def fetch_data(self):
        print(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}")
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        if self.data.empty:
            raise ValueError("No data was fetched. Please check ticker symbol and date range.")
        print(f"Data fetched: {len(self.data)} rows")
        return self.data



    def prepare_features(self):
        df = self.data.copy()
        df['RSI'] = calculate_rsi(df['Close'], 14)
        df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        df['ATR'] = calculate_atr(df, 14)
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volatility'] = df['Close'].rolling(window=20).std()
        df['Price_Volume_Ratio'] = df['Close'] / df['Volume']
        df.dropna(inplace=True)
        self.feature_data = df
        print(f"Features prepared: {len(self.feature_data)} rows")
        return self.feature_data


    def create_sequences(self, scaled_data):
        X, y = [], []
        for i in range(len(scaled_data) - self.seq_length):
            X.append(scaled_data[i:(i + self.seq_length), :])
            y.append(scaled_data[i + self.seq_length, 0])
        return np.array(X), np.array(y)


    def prepare_data(self):
        features = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'EMA_9', 
                    'SMA_20', 'SMA_50', 'ATR', 'Price_Change', 'Volume_Change',
                    'Volume_MA_5', 'Volatility']
        self.all_closing_prices = self.feature_data['Close'].values.reshape(-1, 1)
        feature_set = self.feature_data[features].values
        scaled_features = self.feature_scaler.fit_transform(feature_set)
        X, y = self.create_sequences(scaled_features)
        split_idx = int(len(X) * (1 - self.test_size))
        self.X_train, self.X_test = X[:split_idx], X[split_idx:]
        self.y_train, self.y_test = y[:split_idx], y[split_idx:]
        self.test_set_prices = self.all_closing_prices[-len(self.y_test):]
        self.test_dates = self.feature_data.index[-len(self.y_test):]
        print(f"Training set size: {len(self.X_train)}")
        print(f"Testing set size: {len(self.X_test)}")
        return self.X_train, self.y_train, self.X_test, self.y_test


    def build_model(self, units=100, dropout_rate=0.2, learning_rate=0.001):
        n_features = self.X_train.shape[2]
        model = Sequential([
            GRU(units, return_sequences=True, input_shape=(self.seq_length, n_features)),
            Dropout(dropout_rate),
            GRU(units),
            Dropout(dropout_rate),
            Dense(units//2, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
        self.model = model
        print(model.summary())
        return model

    def train_model(self, epochs=100, batch_size=32, patience=20):
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)
        checkpoint = ModelCheckpoint(f"{self.ticker}_gru_model.h5", monitor='val_loss', save_best_only=True, verbose=1)
        history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size,
                                 validation_split=0.1, callbacks=[early_stopping, checkpoint], verbose=1)
        self.history = history
        return history

    def predict(self):
        scaled_predictions = self.model.predict(self.X_test)
        temp_array = np.zeros((len(scaled_predictions), self.X_train.shape[2]))
        temp_array[:, 0] = scaled_predictions.flatten()
        predictions_inverse = self.feature_scaler.inverse_transform(temp_array)[:, 0]
        self.predictions = predictions_inverse
        return self.predictions



    def evaluate_model(self):
        actual_prices = self.test_set_prices.flatten()
        predicted_prices = self.predictions
        mse = mean_squared_error(actual_prices, predicted_prices)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_prices, predicted_prices)
        r2 = r2_score(actual_prices, predicted_prices)
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
        metrics = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}
        self.metrics = metrics
        return metrics



    def plot_predictions(self):
        plt.figure(figsize=(14,7))
        plt.plot(self.test_dates, self.test_set_prices, label='Actual Price', color='blue')
        plt.plot(self.test_dates, self.predictions, label='Predicted Price', color='red')
        plt.title(f'{self.ticker} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Stock Price ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



    def plot_training_history(self):
        plt.figure(figsize=(14,7))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss During Training')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



    def run_complete_pipeline(self, plot=True):
        self.fetch_data()
        self.prepare_features()
        self.prepare_data()
        self.build_model()
        self.train_model()
        self.predict()
        metrics = self.evaluate_model()
        if plot:
            self.plot_training_history()
            self.plot_predictions()
        print("\nModel Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        return metrics


# This is the main Execution
if __name__ == "__main__":
    TICKER = "AAPL"
    START_DATE = "2015-01-01"
    END_DATE = "2023-11-13"
    SEQ_LENGTH = 60
    

    
    predictor = StockPricePredictor(ticker=TICKER, start_date=START_DATE, end_date=END_DATE, seq_length=SEQ_LENGTH)
    metrics = predictor.run_complete_pipeline()
    print(f"\n\n Final RMSE for {TICKER}: {metrics['RMSE']:.4f}\n\n")
