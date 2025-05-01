from stock_predictor import StockPricePredictor



def main():

    # I defined the parameters here
    TICKER = "AAPL"  # This is Apple
    START_DATE = "2015-01-01"
    END_DATE = "2023-11-13"
    SEQ_LENGTH = 60  # This is 60 days of historical data window
    
    print("Starting prediction for", TICKER)
    
    # I created the predictor here
    predictor = StockPricePredictor(
        ticker=TICKER,
        start_date=START_DATE,
        end_date=END_DATE,
        seq_length=SEQ_LENGTH
    )
    
    print("Running pipeline stages:")

    # This part fetches and prepares the data
    print("1. Fetching data...")
    predictor.fetch_data()
    
    print("2. Preparing features...")
    predictor.prepare_features()
    
    print("3. Preparing sequences...")
    predictor.prepare_data()
    
    print("4. Building model...")
    predictor.build_model()
    
    print("5. Training model...")
    predictor.train_model()
    
    print("6. Making predictions...")
    predictor.predict()
    
    print("7. Evaluating model...")
    metrics = predictor.evaluate_model()
    
    print("8. Plotting results...")
    predictor.plot_predictions()
    predictor.plot_training_history()
    predictor.feature_importance_analysis()
    
    
    print("\nModel Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()