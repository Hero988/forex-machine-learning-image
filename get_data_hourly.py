import MetaTrader5 as mt5
import mplfinance as mpf
from dotenv import load_dotenv
import pandas as pd
import os
from datetime import datetime, timedelta
import shutil
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates

# Load environment variables for MT5 login credentials
load_dotenv()
login = int(os.getenv('MT5_LOGIN'))  # Replace with your login ID
password = os.getenv('MT5_PASSWORD')  # Replace with your password
server = os.getenv('MT5_SERVER')  # Replace with your server name

# Initialize MetaTrader 5 connection
if not mt5.initialize():
    if not mt5.initialize(login=login, password=password, server=server):
        print("Failed to initialize MT5, error code:", mt5.last_error())
        quit()

# Define the list of symbols to be processed
symbols = ["EURUSD"]  # Add more symbols as needed
timeframe = mt5.TIMEFRAME_H1  # 1-hour timeframe for each candlestick
interval = timedelta(hours=5)  # 12 hours of data per image

# Convert to integer hours
hours = int(interval.total_seconds() / 3600)

# Define start and end dates
end_date = datetime.now()  # End date is the current date and time
start_date = end_date - timedelta(days=365)  # Start date is 365 days before the end date
output_dir = "output"  # Main output directory
screenshots_dir = os.path.join(output_dir, "market_screenshots")  # Directory to save images
future_rates_dir = os.path.join(output_dir, "future_rates")  # Directory to save future rates images
debug_log_file = os.path.join(output_dir, "debug_log.txt")  # File to log debug information
price_threshold = 0.00  # Price change threshold for labeling
prediction_window = 1  # Set to either 1 for 1-hour or 5 for 5-hour prediction

# Create output directories if they don't exist
os.makedirs(screenshots_dir, exist_ok=True)
os.makedirs(future_rates_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")

def get_label_based_on_future(current_close, future_close, swing_low, swing_high):
    price_change = (future_close - current_close) / current_close
    if price_change > price_threshold and future_close > swing_low:
        return "buy"
    elif price_change < -price_threshold and future_close < swing_high:
        return "sell"
    else:
        return "hold"

def save_combined_chart(df, future_df, filename):
    """
    Save a custom candlestick chart from DataFrame `df` and `future_df` combined with a highlighted box.
    """
    # Ensure the index is datetime for both DataFrames
    df.index = pd.to_datetime(df.index, errors='coerce')
    future_df.index = pd.to_datetime(future_df.index, errors='coerce')
    
    # Combine both DataFrames, keeping the index as datetime
    combined_df = pd.concat([df, future_df])
    combined_df = combined_df[combined_df.index.notnull()]  # Drop rows with NaT in the index

    # Convert required columns to numeric, dropping rows with NaNs
    for col in ['open', 'high', 'low', 'close']:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    combined_df.dropna(inplace=True)

    # Ensure the index is timezone-naive
    combined_df.index = combined_df.index.tz_localize(None)

    # Plot setup
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each candlestick manually
    for i in range(len(combined_df)):
        open_price = combined_df['open'].iloc[i]
        close_price = combined_df['close'].iloc[i]
        high_price = combined_df['high'].iloc[i]
        low_price = combined_df['low'].iloc[i]
        time = combined_df.index[i]
        
        # Use identical color scheme as in mplfinance
        color = '#00A600' if close_price >= open_price else '#FF0000'
        
        # Draw the wick
        ax.plot([mdates.date2num(time), mdates.date2num(time)], [low_price, high_price], color=color, linewidth=1)
        
        # Draw the body
        width = 0.02  # Adjust width to match mplfinance output closely
        rect = patches.Rectangle(
            (mdates.date2num(time) - width / 2, min(open_price, close_price)), 
            width=width, 
            height=abs(close_price - open_price), 
            color=color
        )
        ax.add_patch(rect)
    
    # Highlight the future candle with a blue box if future data exists
    if not future_df.empty:
        future_time = future_df.index[0]
        future_low = future_df['low'].iloc[0]
        future_high = future_df['high'].iloc[0]
        
        # Draw a rectangle around the future candle
        future_width = width  # Use the same width for the box
        rect = patches.Rectangle(
            (mdates.date2num(future_time) - future_width / 2, future_low),
            width=future_width,
            height=(future_high - future_low),
            linewidth=1,
            edgecolor='blue',
            facecolor='none'
        )
        ax.add_patch(rect)

    # Set limits to match the axis range
    ax.set_ylim(combined_df['low'].min() * 0.999, combined_df['high'].max() * 1.001)
    ax.set_xlim(mdates.date2num(combined_df.index.min()) - 0.02, mdates.date2num(combined_df.index.max()) + 0.02)
    
    # Remove the x and y axes and spines
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Save the plot as an image
    plt.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_candlestick_chart(df, filename):
    """
    Save a candlestick chart from DataFrame `df` to `filename` with axes and spines removed.
    """
    # Ensure that essential columns ('open', 'high', 'low', 'close') are numeric and drop rows with NaN values
    df = df[['open', 'high', 'low', 'close']].apply(pd.to_numeric, errors='coerce').dropna()
    
    # Create the figure and axis
    fig, ax = plt.subplots()
    
    # Plot the candlestick chart
    try:
        mpf.plot(df, type='candle', style='charles', ax=ax)
    except Exception as e:
        print("Error during plotting with mplfinance:", e)
        return  # Exit if there's an error in plotting

    # Remove the x and y axes and spines
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Save the plot as an image
    plt.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()

# Function to retrieve rates data from MetaTrader5 within the specified range
def get_rates(symbol, timeframe, start, end):
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    if rates is None or len(rates) == 0:
        with open(debug_log_file, "a") as log:
            log.write(f"No data for {symbol} in the specified range from {start} to {end}.\n")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

# Loop through each symbol
for symbol in symbols:
    print(f"Processing symbol: {symbol}")
    
    # Main loop through each interval for the current symbol
    current_date = start_date
    while current_date <= end_date:
        next_date = current_date + interval
        main_filename = os.path.join(screenshots_dir, f"{current_date.strftime('%Y%m%d%H%M')}.png")
        
        # Retrieve the 12-hour data period
        df = get_rates(symbol, timeframe, current_date, next_date)
        if df is not None and len(df) >= hours:
            save_candlestick_chart(df, main_filename)  # Save the main chart image here
            
            current_close = df['close'].iloc[-1]
            swing_low = df['low'].min()
            swing_high = df['high'].max()
            
            # Retrieve the future rates for the prediction window
            future_date = next_date + timedelta(hours=prediction_window)
            future_rates = mt5.copy_rates_from(symbol, timeframe, future_date, prediction_window)
            
            if future_rates is not None and len(future_rates) > 0:
                future_df = pd.DataFrame(future_rates)
                future_df['time'] = pd.to_datetime(future_df['time'], unit='s')
                future_df.set_index('time', inplace=True)
                future_close = future_df['close'].iloc[0]
                label = get_label_based_on_future(current_close, future_close, swing_low, swing_high)
                
                # Log label decision
                with open(debug_log_file, "a") as log:
                    log.write(f"{symbol} Date: {current_date} to {next_date}, Label: {label}, Current Close: {current_close}, "
                              f"Future Close: {future_close}, Swing Low: {swing_low}, Swing High: {swing_high}\n")
                
                # Save the future rates image with combined data and highlighted future candle
                if label in ["buy", "sell"]:
                    destination_dir = train_dir if random.random() < 0.8 else test_dir
                    label_dir = os.path.join(destination_dir, label)
                    os.makedirs(label_dir, exist_ok=True)
                    
                    dest_path = os.path.join(label_dir, os.path.basename(main_filename))
                    shutil.move(main_filename, dest_path)
                    future_filename = os.path.join(future_rates_dir, f"future_{current_date.strftime('%Y%m%d%H%M')}.png")
                    save_combined_chart(df, future_df, future_filename)
            else:
                with open(debug_log_file, "a") as log:
                    log.write(f"{symbol} Missing future data for prediction at {future_date} after {current_date} to {next_date}.\n")
        else:
            with open(debug_log_file, "a") as log:
                log.write(f"{symbol} No valid data for interval {current_date} to {next_date}.\n")
        
        # Move to the next interval
        current_date = next_date

# Shut down MetaTrader5 connection
mt5.shutdown()