
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
import datetime
from datetime import date, timedelta

@st.cache_data
def load_data(stock, from_date=None, to_date=None):
    """
    Load historical data for a given cryptocurrency pair.

    Parameters:
    - stock (str): The cryptocurrency pair ('BTC-USDT', 'ETH-USDC', 'ETH-BTC').

    Returns:
    - DataFrame: A pandas DataFrame containing historical data with selected columns.

    Notes: 
    - Data for different pairs can be downloaded from 
    Example:
    >>> btc_data = load_data('BTC-USDT')
    >>> print(btc_data.head())
    """

    try:
        # Load data based on the selected cryptocurrency pair
        print(f"Loading {stock} Data")
        data = pd.read_csv(f"data/Binance_{stock.replace('-', '')}_d.csv")
        
        # Reverse the order of data
        data = data[::-1]

        # Convert 'Date' column to datetime and set it as index
        data['Date'] = data['Date'].astype('datetime64[s]')
        s = pd.to_datetime(data['Date'], unit='s')
        data.index = s

        # Select relevant columns
        selected_columns = ['Open', 'High', 'Low', 'Close']
        data = data[selected_columns]

        if from_date is not None: 
            print(f'Before from_date filtering: {data.shape} rows')        
            data = data[from_date:]
            print(f'After from_date filtering @ {from_date}, leaves {data.shape} rows')

        if to_date is not None:
            print(f'Before to_date filtering: {data.shape} rows')  
            data = data[:to_date]
            print(f'After to_date filtering @ {to_date}, leaves {data.shape} rows')
        return data
    except Exception as ex:
        print(ex)
        return None

def predict_trend(df, strategy='MA', lookback=30, selected_price='Close', threshold=0.25):
    """
    Predicts the trend of a cryptocurrency based on historical data.

    Parameters:
    - df (DataFrame): Historical data DataFrame.
    - strategy (str): The strategy for moving average calculation ('MA' or 'EMA', default: 'MA').
    - lookback (int): The number of days to consider for the moving average (default: 30).
    - selected_price (str): The price attribute to use for analysis (default: 'Close').
    - threshold (float): The threshold value for trend determination (default: 0.25).

    Returns:
    - DataFrame: A pandas DataFrame containing historical data with additional columns for trend prediction.

    Example:
    >>> df = pd.DataFrame(...)  # Input historical data DataFrame
    >>> df_result = predict_trend_ma(df, strategy='EMA', lookback=20, selected_price='Close', threshold=0.2)
    >>> print(df_result.head())
    """

    # Initialize 'Trend' column with 'Sideways'
    df['Trend'] = 'Sideways'

    # Calculate moving average based on the selected strategy
    if strategy.upper() == 'EMA':
        df['MA'] = df[selected_price].ewm(span=lookback, adjust=False).mean()
    else:
        df['MA'] = df[selected_price].rolling(window=lookback).mean()

    # Calculate the difference between the selected price and the moving average
    df['Difference'] = df[selected_price] - df['MA']
    df = df.dropna()
    df = df.copy()
    # Set the 'Trend' column based on the specified threshold
    df['Cutoff'] = threshold * df[selected_price]
    df.loc[df['Difference'] >= df['Cutoff'], 'Trend'] = 'Up'
    df.loc[df['Difference'] <= -df['Cutoff'], 'Trend'] = 'Down'

    # Set the index to date for better visualization
    df.index = df.index.date

    return df

def visualize(selected_pool, data, lookback, selected_price):
    """
    Visualizes trend predictions based on historical data.

    Parameters:
    - selected_pool (str): The cryptocurrency pair (e.g., 'BTC-USDT').
    - data (DataFrame): Historical data with trend predictions.
    - lookback (int): Number of historical values considered for trend prediction.
    - selected_price (str): The price attribute for visualization (e.g., 'Close').

    Returns:
    - None: Displays an interactive plot.

    Example:
    >>> visualize(selected_pool='BTC-USDT', data=df, lookback=30, selected_price='Close')
    """

    # Define mapping for trend labels to numerical values
    trend_mapping = {'Up': 1, 'Down': -1, 'Sideways': 0}

    # Set your custom color map
    color_map = {'Up': 'green', 'Down': 'red', 'Sideways': 'blue'}

    # Map the trend labels to numerical values for plotting
    data['Trend_numeric'] = data['Trend'].map(trend_mapping)

    # Create a reference line for the original close prices
    reference_line = go.Scatter(x=data.index,
                                y=data[selected_price],
                                mode="lines",
                                line=go.scatter.Line(color="gray"),
                                showlegend=True, name=f"{selected_price} Price")
    
    ma_line = go.Scatter(x=data.index,
                                y=data.MA,
                                mode="lines",
                                line=go.scatter.Line(color="orange"),
                                showlegend=True, name='Moving Average')

    # Create an interactive scatter plot for trend visualization
    fig = (px.scatter(data, x=data.index, y=data[selected_price], 
                hover_data=[data.Difference, data.Cutoff], color=data.Trend, color_discrete_map=color_map,               
                title=f"Trend Predictions based on historic {lookback} values")
    .update_layout(title_font_size=24)
    .update_xaxes(showgrid=True)
    .update_traces(
        line=dict(dash="dot", width=4),
        selector=dict(type="scatter", mode="lines"))
    )

    # Add the reference line to the plot
    fig.add_trace(reference_line, row=1, col=1)
    fig.add_trace(ma_line, row=1, col=1)
    
    # Update layout
    fig.update_layout(
        xaxis=dict(title='Date'),
        yaxis=dict(title=selected_price),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    # Show the interactive plot
    #fig.show()
    st.plotly_chart(fig)


if __name__ == '__main__':
    START = None
    END = None
    min_lookback_period = 7

    st.title("Trend Prediction APP")
    pools = ("BTC-USDT", "ETH-USDC", "ETH-BTC", "ETH-USDT")
    pool_start_date=(datetime.date(2017, 8, 17), datetime.date(2018, 12, 15), datetime.date(2017, 7, 14), datetime.date(2017, 8, 17))
    prices = ("Close", "Open", "High", "Low")
    strategy = ("MA", "EMA")

    #selected_pool = 0
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        selected_pool = st.selectbox("Select Pair", pools, index=0)
        #symbol = st.selectbox('Choose stock symbol', options=['AAPL', 'MSFT', 'GOOG', 'AMZN'], index=1)
    with c2:
        if selected_pool is None:
            idx = 0
        else:
            idx = pools.index(selected_pool)  # Use pools.index to get the index of selected_pool

        START = st.date_input("Start-Date", min_value=pool_start_date[idx], value=pool_start_date[idx])
    with c3:
        if selected_pool is None:
            idx = 0
        else:
            idx = pools.index(selected_pool)  # Use pools.index to get the index of selected_pool

        END = st.date_input("End-Date", min_value=pool_start_date[idx]+timedelta(days=min_lookback_period), max_value=datetime.date(2024, 2, 29), value=datetime.date(2024, 2, 29))


    st.markdown('---')

    st.sidebar.subheader('Settings')
    st.sidebar.caption('Adjust charts settings and then press apply')

    with st.sidebar.form('settings_form'):        
        strategy = st.selectbox("Select Strategy", strategy, index=0)
        
        selected_price = st.selectbox("Select Price", prices, index=0)

        lookback = st.slider("Lookback days (Long Term)", min_value=7, max_value=365, value=15)

        threshold = st.slider("Threshold", min_value=0.0, max_value=0.3, value=0.05)

        show_data = st.checkbox('Show data table', False)
        st.form_submit_button('Apply')
    
    # Load historical data
    df = load_data(selected_pool, from_date=START, to_date=END)
    

    # # Perform trend prediction
    if df is not None:
        df_result = predict_trend(df, strategy=strategy, lookback=lookback, selected_price=selected_price, threshold=threshold)
        #print(df_result.head())
        visualize(selected_pool=selected_pool, data = df_result, lookback=lookback, selected_price=selected_price, )

    else:
        print('Data not available for the selected pool')
    
    if show_data:
        st.markdown('---')
        st.dataframe(df_result.tail(50))