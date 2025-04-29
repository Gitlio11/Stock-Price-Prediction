# Stock Price Prediction using GRU Neural Networks
This project implements a stock price prediction system using Gated Recurrent Unit(GRU) neural networks. The system analyzes historical stock data, trains a deep learning model, and generates price predictions.



## Overview
The stock predictor uses Recurrent Neural Networks (RNNs) to identify patterns in historical stock data and make predictions about future price movements. The implementation specifically uses GRUs, which are efficient variants of recurrent neural networks that address the vanishing gradient problem common in traditional RNNs.

## Features

- Data fetching from Yahoo Finance API
- Comprehensive feature engineering for technical indicators
- Sequence-based data preparation for time series analysis
- GRU neural network model implementation
- Model evaluation with multiple metrics
- Visualization of predictions and model performance


## Requirements

The project requires the following dependencies:

```
numpy
pandas
matplotlib
seaborn
yfinance
scikit-learn
tensorflow
ta-lib
```

You can install all of the requirements by using pip:

```bash
pip install -r requirements.txt
```

## Project Structure

- `stock_predictor.py`: The main class implementing the StockPricePredictor
- `run_prediction.py`: Script to run the prediction pipeline
- `requirements.txt`: List of required Python packages

## Usage

To run the model:

```bash
python run_prediction.py
```

This will run Apple (AAPL) stock price prediction by default and use data from 2015 to 2023.


## How It Works

The prediction pipeline follows these steps:

1. **Data Fetching**: Historical stock data is downloaded from Yahoo Finance.
2. **Feature Engineering**: Technical indicators are calculated, including:
   - Relative Strength Index (RSI)
   - Moving Averages (EMA, SMA)
   - Moving Average Convergence Divergence (MACD)
   - Average True Range (ATR)
   - Price and Volume changes
   - Volatility measures

3. **Sequence Preparation**: Data is transformed into sequences for time-series analysis.
4. **Model Building**: A GRU neural network is constructed with the following architecture:
   - First GRU layer with return sequences
   - Dropout for regularization
   - Second GRU layer
   - Additional dropout
   - Dense layers for final prediction

5. **Training**: The model is trained with early stopping to prevent overfitting.
6. **Prediction**: Forecasts are generated for the test period.
7. **Evaluation**: Multiple metrics assess model performance:
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - R-squared (R2)
   - Mean Absolute Percentage Error (MAPE)

8. **Visualization**: Results are plotted for easy interpretation.

## Configuration

You can customize the prediction parameters in the `run_prediction.py` file:

- `TICKER`: Stock symbol (default: "AAPL")
- `START_DATE`: Beginning of historical data range (default: "2015-01-01")
- `END_DATE`: End of historical data range (default: "2023-11-13")
- `SEQ_LENGTH`: Number of historical days to use for each prediction (default: 60)

## Model Architecture

The GRU model consists of:
- Input shape based on sequence length and number of features
- First GRU layer with 100 units and return sequences
- 20% dropout for regularization
- Second GRU layer with 100 units
- Additional 20% dropout
- Dense layer with half the units of GRU layers and ReLU activation
- Output dense layer with linear activation

## Feature Importance Analysis

The model includes functionality to analyze which features contribute most to the predictions. This can help identify the most important technical indicators for a particular stock.

## Extending the Model

To use this project with other stocks or different parameters:

1. Modify the `TICKER`, `START_DATE`, `END_DATE`, and `SEQ_LENGTH` variables in `run_prediction.py`
2. Run the script to generate predictions for your custom configuration

You can also extend the feature set by modifying the `prepare_features` method in the `StockPricePredictor` class.

## Advanced Usage

The `StockPricePredictor` class includes a `run_complete_pipeline` method that executes all steps in sequence, making it easy to test different configurations. You can import this class into your own scripts to customize the prediction process further.

```python
from stock_predictor import StockPricePredictor

# Create custom predictor
predictor = StockPricePredictor(
    ticker="MSFT", 
    start_date="2018-01-01", 
    end_date="2023-04-30", 
    seq_length=30
)

# Run entire pipeline
metrics = predictor.run_complete_pipeline(plot=True)
print(f"RMSE: {metrics['RMSE']:.4f}")
```

## Limitations

- Stock price prediction is inherently challenging due to market volatility and external factors
- The model is trained on historical data and may not account for unexpected market events
- Performance will vary depending on the specific stock and time period analyzed

## Future Improvements

Potential enhancements to the model could include:
- Incorporating sentiment analysis from news and social media
- Implementing ensemble methods with multiple model types
- Adding attention mechanisms to better capture long-term dependencies
- Hyperparameter optimization for different stocks

## License

MIT License

## Acknowledgments

This project implements concepts from recurrent neural networks, particularly GRUs, for time series prediction in financial markets.