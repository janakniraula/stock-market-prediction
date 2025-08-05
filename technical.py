# import streamlit as st
# import talib as ta
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import matplotlib.pyplot as plt

# class Technical:
#     def calculate_indicator(data):
#         data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
#         data['SMA_5'] = ta.SMA(data['Close'],5)
#         data['SMA_10'] = ta.SMA(data['Close'],10)
#         data['SMA_15'] = ta.SMA(data['Close'],15)
#         data['SMA_20'] = ta.SMA(data['Close'],20)
#         data['SMA_50'] = ta.SMA(data['Close'],50)
#         data['SMA_200'] = ta.SMA(data['Close'],200)
#         data['EMA_5'] = ta.EMA(data['Close'],5)
#         data['EMA_10'] = ta.EMA(data['Close'],10)
#         data['EMA_15'] = ta.EMA(data['Close'],15)
#         data['EMA_20'] = ta.EMA(data['Close'],20)
#         data['EMA_50'] = ta.EMA(data['Close'],50)
#         data['EMA_200'] = ta.EMA(data['Close'],200)
#         BBU, BBM, BBL = ta.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
#         data['BBU'] = BBU
#         data['BBM'] = BBM
#         data['BBL'] = BBL

#         # Momentum indicators
#         macd, macds,macdh = ta.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
#         data['MACD'] = macd
#         data['MACD_Signal'] = macds
#         data['MACD_Hist'] = macdh
#         data['RSI'] = ta.RSI(data['Close'], timeperiod=14)

#         return(data)
    
#     def main(full_data,selected_df):
#         sma, ema, macd, rsi, bollinger = st.columns(5)

#         with sma:
#             sma_checkbox = st.checkbox("SMA", key="sma")

#         with ema:
#             ema_checkbox = st.checkbox("EMA", key="ema")

#         with macd:
#             macd_checkbox = st.checkbox("MACD", key="macd")

#         with rsi:
#             rsi_checkbox = st.checkbox("RSI", key="rsi")

#         with bollinger:
#             bollinger_checkbox = st.checkbox("Bollinger Bands", key="bollinger")
        
#         st.subheader("Adjust Slider (Long Term/Short Term)")
#         term = st.slider("SMA", 3, 60, 3)

#         if term <=15:
#             st.info("Very Short Term Analysis")
#         elif term > 15 and term <= 30:
#             st.info("Short Term Analysis")
#         else:
#             st.info("Long Term Analysis")
        
#         if sma_checkbox:
#             if term <=15:
#                 term_high = term + 7 
#             elif term > 15 and term <= 30:
#                 term_high = term + 20
#             else:
#                 term_high = 200
            
#             # df_sma = pd.DataFrame(full_data)
#             full_data[f'SMA_{term}'] = ta.SMA(full_data['Close'],term)
#             full_data[f'SMA_{term_high}'] = ta.SMA(full_data['Close'],term_high)
            
#             # Generate signals
#             full_data['Signal'] = np.where(full_data[f'SMA_{term}'] > full_data[f'SMA_{term_high}'], 1, 0)
#             # Generate trading orders
#             full_data['Position'] = full_data['Signal'].diff()

#             df = pd.DataFrame()
#             df = pd.merge(full_data[['Date', f'SMA_{term}', f'SMA_{term_high}',f'Signal',f'Position']], selected_df, on='Date', how='inner')
#             df.set_index('Date', inplace=True)
            
#         if sma_checkbox:
#             # Determine term_high based on selected term
#             if term <= 15:
#                 term_high = term + 7
#             elif term > 15 and term <= 30:
#                 term_high = term + 20
#             else:
#                 term_high = 200

#             # Calculate SMAs using rolling mean
#             full_data[f'SMA_{term}'] = full_data['Close'].rolling(window=term).mean()
#             full_data[f'SMA_{term_high}'] = full_data['Close'].rolling(window=term_high).mean()

#             # Generate trading signals
#             full_data['Signal'] = np.where(full_data[f'SMA_{term}'] > full_data[f'SMA_{term_high}'], 1, 0)
#             full_data['Position'] = full_data['Signal'].diff()

#             # Merge with selected_df on Date
#             df = pd.merge(
#                 full_data[['Date', f'SMA_{term}', f'SMA_{term_high}', 'Signal', 'Position']],
#                 selected_df,
#                 on='Date',
#                 how='inner'
#             )
#             df.set_index('Date', inplace=True)

#             # Drop NaNs only from Close
#             df.dropna(subset=['Close'], inplace=True)

#             # Fill SMA columns with fallback values (optional: can use Close)
#             df[f"SMA_{term}"].fillna(df['Close'], inplace=True)
#             df[f"SMA_{term_high}"].fillna(df['Close'], inplace=True)

#             # Optional: warn user
#             if df[[f"SMA_{term}", f"SMA_{term_high}"]].isna().all().any():
#                 st.warning("Not enough data for reliable SMA calculation. Using fallback values (Close price) for analysis.")


#             # Plotting the SMA and signals
#             fig = go.Figure()
#             fig.add_trace(go.Scatter(x=df.index, y=df[f'SMA_{term}'], mode='lines',
#                                     name=f'SMA_{term}', line=dict(color='orange', width=1.5)))
#             fig.add_trace(go.Scatter(x=df.index, y=df[f'SMA_{term_high}'], mode='lines',
#                                     name=f'SMA_{term_high}', line=dict(color='blue', width=1.5)))

#             # Buy signals
#             buy_signals = df[df['Position'] == 1]
#             fig.add_trace(go.Scatter(
#                 x=buy_signals.index,
#                 y=buy_signals[f'SMA_{term}'],
#                 mode='markers',
#                 marker=dict(symbol='triangle-up', size=10, color='cyan'),
#                 name='BUY Signal'
#             ))

#             # Sell signals
#             sell_signals = df[df['Position'] == -1]
#             fig.add_trace(go.Scatter(
#                 x=sell_signals.index,
#                 y=sell_signals[f'SMA_{term}'],
#                 mode='markers',
#                 marker=dict(symbol='triangle-down', size=10, color='yellow'),
#                 name='SELL Signal'
#             ))

#             # Finalize chart
#             fig.update_layout(
#                 title=f'Simple Moving Average (SMA)',
#                 xaxis_title='Date',
#                 yaxis_title='Price',
#                 height=600,
#                 width=900,
#                 xaxis_rangeslider_visible=False
#             )
#             st.plotly_chart(fig)

#             # Get final SMA and Close values
#             end_sma_low = df[f"SMA_{term}"].iloc[-1]
#             end_sma_high = df[f"SMA_{term_high}"].iloc[-1]
#             end_close = df['Close'].iloc[-1]

#             # Signal interpretation
#             if end_sma_high > end_sma_low:
#                 if end_close > end_sma_high:
#                     st.success("Buy")
#                     st.info(f"The current SMA {term} is lower than the current market price, and the higher SMA {term_high} is higher. This indicates a potential uptrend. Consider buying and holding.")
#                 elif end_close < end_sma_high:
#                     st.warning("Hold")
#                     st.info(f"The current SMA {term} is lower than the current market price, but the higher SMA {term_high} is higher. Consider holding and observing the trend.")
#             else:
#                 if end_close > end_sma_low:
#                     st.warning("Hold")
#                     st.info(f"The current SMA {term} is higher than the market price, but the higher SMA {term_high} is lower. Consider holding and monitoring the trend.")
#                 elif end_close < end_sma_low:
#                     st.error("Sell")
#                     st.info(f"The current SMA {term} is higher than the market price, and the higher SMA {term_high} is lower. This may indicate a downtrend. Consider selling or waiting for a better entry point.")


#         if ema_checkbox:
#             if term <=15:
#                 term_high = term + 7
#             elif term > 15 and term <= 30:
#                 term_high = term + 20
#             else:
#                 term_high = 200
            
#             # df_sma = pd.DataFrame(full_data)
#             full_data[f'EMA_{term}'] = ta.SMA(full_data['Close'],term)
#             full_data[f'EMA_{term_high}'] = ta.SMA(full_data['Close'],term_high)
            
#             # Generate signals
#             full_data['Signal'] = np.where(full_data[f'EMA_{term}'] > full_data[f'EMA_{term_high}'], 1, 0)
#             # Generate trading orders
#             full_data['Position'] = full_data['Signal'].diff()

#             df = pd.DataFrame()
#             df = pd.merge(full_data[['Date', f'EMA_{term}', f'EMA_{term_high}',f'Signal',f'Position']], selected_df, on='Date', how='inner')
#             df.set_index('Date', inplace=True)
            
#             # SMA line graph
#             fig = go.Figure()
#             fig.add_trace(go.Candlestick(x=df.index,
#                                          open=df['Open'],
#                             high=df['High'],
#                             low=df['Low'],
#                             close=df['Close'],
#                             increasing_line_color='green',  # Color for increasing candlesticks
#                             decreasing_line_color='red',    # Color for decreasing candlesticks
#                             line_width=1.2,                 # Width of the candlestick lines
#                             whiskerwidth=0.2,               # Width of the candlestick whiskers
#                             opacity=0.85,                   # Opacity of the candlesticks
#                             showlegend=False))
#             # fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
#             fig.add_trace(go.Scatter(x=df.index, y=df[f'EMA_{term}'], mode='lines', name=f'EMA_{term}',line=dict(color='orange', width=1.5)))
#             fig.add_trace(go.Scatter(x=df.index, y=df[f'EMA_{term_high}'], mode='lines', name=f'EMA_{term_high}',line=dict(color='blue', width=1.5)))
            
#             # Plot Buy signals
#             buy_signals = df[df['Position'] == 1]
#             fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals[f'EMA_{term}'],
#                                     mode='markers',
#                                     marker=dict(symbol='triangle-up', size=10, color='cyan'),
#                                     name='BUY Signal'))

#             # Plot Sell signals
#             sell_signals = df[df['Position'] == -1]
#             fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals[f'EMA_{term}'],
#                                     mode='markers',
#                                     marker=dict(symbol='triangle-down', size=10, color='yellow'),
#                                     name='SELL Signal'))
            
#             # Update layout for better visibility
#             fig.update_layout(title=f'Exponential Moving Average (EMA)',
#                             xaxis_title='Date',
#                             yaxis_title='Price',
#                             height=600,
#                             width=900,
#                             xaxis_rangeslider_visible=False)
                            
#             # Show the plot
#             st.plotly_chart(fig)

#             end_ema_low = df[f"EMA_{term}"].iloc[-1]
#             end_ema_high = df[f"EMA_{term_high}"].iloc[-1]
#             end_close = df['Close'].iloc[-1]

#             if end_ema_high > end_ema_low:
#                 if end_close > end_ema_high:
#                     st.success("Buy")
#                     st.info(f'The current EMA {term} is lower than current market price, but the higher EMA {term_high} is higher. This indicates a potential up-trend.\
#                         It is recommended to buy and hold. It is best to sell when the lower EMA {term} goes higher than current market price.')

#                 elif end_close < end_ema_high:
#                     st.warning("Hold")
#                     st.info(f'The current EMA {term} is lower than current market price, but the higher EMA {term_high} is higher. It is recommended to hold and observe.')

#             else:
#                 if end_close > end_ema_low:
#                     st.warning("Hold")
#                     st.info(f'The current EMA {term} is higher than current market price, but the higher EMA {term_high} is lower. It is recommended to hold and observe.')

#                 elif end_close < end_ema_low:
#                     st.error("Sell")
#                     st.info(f'The current EMA {term} is higher than current market price, and the higher EMA {term_high} is lower. This indicates a down-trend.\
#                         It is recommended to watch and sell. It is best to buy when the lower EMA {term} goes higher than current market price.')
        
#         if macd_checkbox:
#             macd, macds,macdh = ta.MACD(full_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
#             full_data['MACD'] = macd
#             full_data['MACD_Signal'] = macds
#             full_data['MACD_Hist'] = macdh

#             df = pd.DataFrame()
#             df = pd.merge(full_data[['Date', 'MACD', 'MACD_Signal','MACD_Hist']], selected_df, on='Date', how='inner')
#             df.set_index('Date', inplace=True)

#             fig = go.Figure()
#             fig.add_trace(go.Candlestick(x=df.index,
#                                          open=df['Open'],
#                             high=df['High'],
#                             low=df['Low'],
#                             close=df['Close'],
#                             increasing_line_color='green',  # Color for increasing candlesticks
#                             decreasing_line_color='red',    # Color for decreasing candlesticks
#                             line_width=1.2,                 # Width of the candlestick lines
#                             whiskerwidth=0.2,               # Width of the candlestick whiskers
#                             opacity=0.85,                   # Opacity of the candlesticks
#                             showlegend=False))
#             fig1 = go.Figure()
#             fig1.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD',line=dict(color='blue', width=1.5)))
#             fig1.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='MACD Signal',line=dict(color='orange', width=1.5)))
#             fig1.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Histogram'))
            
#             # Update layout for better visibility
#             fig.update_layout(title=f'Moving Average Convergence Divergence (MACD)',
#                             xaxis_title='Date',
#                             yaxis_title='Price',
#                             height=600,
#                             width=900,
#                             xaxis_rangeslider_visible=False)
#             fig1.update_layout(title=f'Moving Average Convergence Divergence (MACD)',
#                             height=300,
#                             width=1100,
#                             xaxis_rangeslider_visible=False)
                            
#             # Show the plot
#             st.plotly_chart(fig)
#             st.plotly_chart(fig1)

#             end_macd = df['MACD'].iloc[-1]
#             end_macd_signal = df['MACD_Signal'].iloc[-1]
#             end_macd_hist = df['MACD_Hist'].iloc[-1]

#             st.subheader("Current Strategy")

#             if end_macd > end_macd_signal:
#                 if end_macd_hist > 0:
#                     st.error("Sell\
#                             \n(strong signal when MACD goes lower than MACD Signal)")
#                     st.info(f'The current MACD is higher than the MACD Signal. This indicates a potential up-trend.\
#                         It is recommended to hold and sell when the MACD goes lower than the MACD Signal.')
#                 else:
#                     st.warning("Hold")
#                     # st.info(f'The current MACD is lower than the MACD Signal. It is recommended to hold and observe.')
#             elif end_macd < end_macd_signal:
#                 if end_macd_hist < 0:
#                     st.success("Buy\
#                                 \n(strong signal when MACD goes higher than MACD Signal)")
#                     st.info(f'The current MACD is lower than the MACD Signal. This indicates a down-trend.\
#                         It is recommended to watch and buy when the MACD goes higher than the MACD Signal.')
#                 else:
#                     st.warning("Hold")
#                     # st.info(f'The current MACD is lower than the MACD Signal. It is recommended to hold and observe.')

#         if rsi_checkbox:
#             rsi = ta.RSI(full_data['Close'], timeperiod=14)
#             full_data['RSI'] = rsi

#             df = pd.DataFrame()
#             df = pd.merge(full_data[['Date', 'RSI']], selected_df, on='Date', how='inner')
#             df.set_index('Date', inplace=True)

#             fig = go.Figure()
#             fig.add_trace(go.Candlestick(x=df.index,
#                                          open=df['Open'],
#                             high=df['High'],
#                             low=df['Low'],
#                             close=df['Close'],
#                             increasing_line_color='green',  # Color for increasing candlesticks
#                             decreasing_line_color='red',    # Color for decreasing candlesticks
#                             line_width=1.2,                 # Width of the candlestick lines
#                             whiskerwidth=0.2,               # Width of the candlestick whiskers
#                             opacity=0.85,                   # Opacity of the candlesticks
#                             showlegend=False))
            
#             fig1 = go.Figure()
#             fig1.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI',line=dict(color='orange', width=1.5)))
#             fig1.add_trace(go.Scatter(x=df.index, y=[70]*len(df), mode='lines', name='Overbought',line=dict(color='red', width=1.8,dash='dash')))
#             fig1.add_trace(go.Scatter(x=df.index, y=[30]*len(df), mode='lines', name='Oversold',line=dict(color='green', width=1.8,dash='dash')))
#             # Update layout for better visibility
#             fig.update_layout(title=f'Relative Strength Index (RSI)',
#                             xaxis_title='Date',
#                             yaxis_title='Price',
#                             height=600,
#                             width=900,
#                             xaxis_rangeslider_visible=False)
            
#             fig1.update_layout(title=f'Relative Strength Index (RSI)',
#                             height=350,
#                             width=1050,
#                             xaxis_rangeslider_visible=False)
                            
#             # Show the plot
#             st.plotly_chart(fig)
#             st.plotly_chart(fig1)

#             end_rsi = df['RSI'].iloc[-1]

#             if end_rsi > 70:
#                 st.error("Sell")
#                 st.info(f'The current RSI is {end_rsi}. This indicates that the stock is overbought.\
#                     It is recommended to sell and wait for the RSI to go below 70 before buying again.')
#             elif end_rsi < 30:
#                 st.success("Buy")
#                 st.info(f'The current RSI is {end_rsi}. This indicates that the stock is oversold.\
#                     It is recommended to buy and wait for the RSI to go above 30 before selling again.')
#             else:
#                 st.warning("Hold")
#                 st.info(f'The current RSI is {end_rsi}. It is recommended to hold and observe.')

#         if bollinger_checkbox:
#             BBU, BBM, BBL = ta.BBANDS(full_data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
#             full_data['BBU'] = BBU
#             full_data['BBM'] = BBM
#             full_data['BBL'] = BBL

#             df = pd.DataFrame()
#             df = pd.merge(full_data[['Date', 'BBU', 'BBM','BBL']], selected_df, on='Date', how='inner')
#             df.set_index('Date', inplace=True)

#             fig = go.Figure()
#             fig.add_trace(go.Candlestick(x=df.index,
#                             open=df['Open'],
#                             high=df['High'],
#                             low=df['Low'],
#                             close=df['Close'],
#                             increasing_line_color='green',  # Color for increasing candlesticks
#                             decreasing_line_color='red',    # Color for decreasing candlesticks
#                             line_width=1.2,                 # Width of the candlestick lines
#                             whiskerwidth=0.2,               # Width of the candlestick whiskers
#                             opacity=0.85,                   # Opacity of the candlesticks
#                             showlegend=False))
#             fig.add_trace(go.Scatter(x=df.index, y=df['BBU'], mode='lines', name='BBU',line=dict(color='blue', width=1.5)))
#             fig.add_trace(go.Scatter(x=df.index, y=df['BBM'], mode='lines', name='BBM',line=dict(color='orange', width=1.5)))
#             fig.add_trace(go.Scatter(x=df.index, y=df['BBL'], mode='lines', name='BBL',line=dict(color='blue', width=1.5)))

#             # Shade between upper and lower Bollinger Bands
#             fig.add_trace(go.Scatter(x=df.index, y=df['BBU'], fill=None, mode='lines', line=dict(color='blue', width=0), showlegend=False))
#             fig.add_trace(go.Scatter(x=df.index, y=df['BBL'], fill='tonexty',fillcolor='rgba(0, 0, 255, 0.1)', mode='lines', line=dict(color='blue', width=0), name='Bollinger Bands'))

#             # Update layout for better visibility
#             fig.update_layout(title=f'Bollinger Bands',
#                             xaxis_title='Date',
#                             yaxis_title='Price',
#                             height=600,
#                             width=900,
#                             xaxis_rangeslider_visible=False)
            
#             # Show the plot
#             st.plotly_chart(fig)

#             end_bbu = df['BBU'].iloc[-1]
#             end_bbm = df['BBM'].iloc[-1]
#             end_bbl = df['BBL'].iloc[-1]
#             end_close = df['Close'].iloc[-1]
            
#             if end_close > end_bbu:
#                 st.error("Sell")
#                 st.info(f'The current market price is higher than the upper Bollinger Band. This indicates a potential down-trend.\
#                     It is recommended to sell and wait for the market price to go below the upper Bollinger Band before buying again.')
#             elif end_close < end_bbl:
#                 st.success("Buy")
#                 st.info(f'The current market price is lower than the lower Bollinger Band. This indicates a potential up-trend.\
#                     It is recommended to buy and wait for the market price to go above the lower Bollinger Band before selling again.')
#             else:
#                 st.warning("Hold")
#                 st.info(f'The current market price is within the Bollinger Bands. It is recommended to hold and observe.')
import pandas as pd
import numpy as np
import talib as ta
import plotly.graph_objects as go
import plotly.io as pio

class Technical:
    @staticmethod
    def calculate_indicator(data):
        data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')
        data['SMA_5'] = ta.SMA(data['Close'], 5)
        data['SMA_10'] = ta.SMA(data['Close'], 10)
        data['EMA_5'] = ta.EMA(data['Close'], 5)
        data['RSI'] = ta.RSI(data['Close'], timeperiod=14)
        macd, macds, macdh = ta.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        data['MACD'] = macd
        data['MACD_Signal'] = macds
        data['MACD_Hist'] = macdh
        return data

    @staticmethod
    def sma_strategy(full_data: pd.DataFrame, selected_df: pd.DataFrame, term: int):
        # 1. Calculate term_high automatically
        term_high = 200 if term > 30 else term + 7 if term <= 15 else term + 20

        # 2. Calculate SMAs using TA-Lib
        full_data[f'SMA_{term}'] = ta.SMA(full_data['Close'], timeperiod=term)
        full_data[f'SMA_{term_high}'] = ta.SMA(full_data['Close'], timeperiod=term_high)

        # 3. Generate signals
        full_data['Signal'] = np.where(full_data[f'SMA_{term}'] > full_data[f'SMA_{term_high}'], 1, 0)
        full_data['Position'] = full_data['Signal'].diff()

        # 4. Merge for plotting
        df = pd.merge(
            full_data[['Date', f'SMA_{term}', f'SMA_{term_high}', 'Signal', 'Position']],
            selected_df,
            on='Date',
            how='inner'
        )
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Optional fill
        df[[f"SMA_{term}", f"SMA_{term_high}"]] = df[[f"SMA_{term}", f"SMA_{term_high}"]].fillna(method='bfill')

        # 5. Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[f'SMA_{term}'], mode='lines', name=f'SMA {term}', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df.index, y=df[f'SMA_{term_high}'], mode='lines', name=f'SMA {term_high}', line=dict(color='blue')))

        # Buy/Sell Markers
        buy_signals = df[df['Position'] == 1]
        sell_signals = df[df['Position'] == -1]

        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals[f'SMA_{term}'], mode='markers',
                                marker=dict(color='green', symbol='triangle-up', size=10), name='Buy Signal'))

        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals[f'SMA_{term}'], mode='markers',
                                marker=dict(color='red', symbol='triangle-down', size=10), name='Sell Signal'))

        fig.update_layout(title='SMA Strategy', xaxis_title='Date', yaxis_title='Price', height=600, width=900)

        # 6. Interpret strategy
        end_sma_low = df[f"SMA_{term}"].iloc[-1]
        end_sma_high = df[f"SMA_{term_high}"].iloc[-1]
        end_close = df['Close'].iloc[-1]

        messages = []

        if end_sma_high > end_sma_low:
            if end_close > end_sma_high:
                messages.append("📈 <strong>Buy:</strong> Price is above both short and long SMA. Trend is up.")
            else:
                messages.append("⚠️ <strong>Hold:</strong> Price is under long SMA but short-term shows strength.")
        else:
            if end_close < end_sma_low:
                messages.append("📉 <strong>Sell:</strong> Both SMAs are above current price. Likely downtrend.")
            else:
                messages.append("⚠️ <strong>Hold:</strong> Crossover isn't strong enough to act yet.")

        return {
            "chart": pio.to_html(fig, full_html=False),
            "messages": messages
        }
    # def sma_strategy(df, term=10):
    #     term_high = 200 if term > 30 else term + 7 if term <= 15 else term + 20

    #     df[f'SMA_{term}'] = ta.SMA(df['Close'], timeperiod=term)
    #     df[f'SMA_{term_high}'] = ta.SMA(df['Close'], timeperiod=term_high)
    #     df['Signal'] = np.where(df[f'SMA_{term}'] > df[f'SMA_{term_high}'], 1, 0)
    #     df['Position'] = df['Signal'].diff()

    #     fig = go.Figure()
    #     fig.add_trace(go.Scatter(x=df['Date'], y=df[f'SMA_{term}'], mode='lines', name=f'SMA_{term}', line=dict(color='orange')))
    #     fig.add_trace(go.Scatter(x=df['Date'], y=df[f'SMA_{term_high}'], mode='lines', name=f'SMA_{term_high}', line=dict(color='blue')))

    #     buy_signals = df[df['Position'] == 1]
    #     sell_signals = df[df['Position'] == -1]

    #     fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals[f'SMA_{term}'], mode='markers',
    #                              marker=dict(color='green', symbol='triangle-up', size=10), name='Buy Signal'))
    #     fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals[f'SMA_{term}'], mode='markers',
    #                              marker=dict(color='red', symbol='triangle-down', size=10), name='Sell Signal'))

    #     fig.update_layout(title='SMA Strategy', xaxis_title='Date', yaxis_title='Price', height=600, width=900)
    #     return pio.to_html(fig, full_html=False)

    @staticmethod
    def macd_strategy(full_data: pd.DataFrame, selected_df: pd.DataFrame):
    # Step 1: Compute MACD
        macd, macds, macdh = ta.MACD(full_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        full_data['MACD'] = macd
        full_data['MACD_Signal'] = macds
        full_data['MACD_Hist'] = macdh

        # Step 2: Merge with selected range
        df = pd.merge(full_data[['Date', 'MACD', 'MACD_Signal', 'MACD_Hist']], selected_df, on='Date', how='inner')
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Step 3: Plot candlestick chart
        fig_price = go.Figure()
        fig_price.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color='green',
            decreasing_line_color='red',
            line_width=1.2,
            whiskerwidth=0.2,
            opacity=0.85,
            showlegend=False
        ))
        fig_price.update_layout(
            title='MACD with Candlestick Chart',
            xaxis_title='Date',
            yaxis_title='Price',
            height=600,
            width=900,
            xaxis_rangeslider_visible=False
        )

        # Step 4: Plot MACD indicator separately
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue', width=1.5)))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='MACD Signal', line=dict(color='orange', width=1.5)))
        fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Histogram'))

        fig_macd.update_layout(
            title='MACD Indicator',
            height=300,
            width=1100,
            xaxis_rangeslider_visible=False
        )

        # Step 5: Generate strategy recommendation
        end_macd = df['MACD'].iloc[-1]
        end_signal = df['MACD_Signal'].iloc[-1]
        end_hist = df['MACD_Hist'].iloc[-1]

        messages = []
        if end_macd > end_signal:
            if end_hist > 0:
                messages.append("📉 <strong>Sell Signal:</strong> MACD is above Signal line but histogram is positive. Indicates short-term strength, but expect reversal.")
            else:
                messages.append("⚠️ <strong>Hold:</strong> MACD is above Signal line but histogram is weakening.")
        elif end_macd < end_signal:
            if end_hist < 0:
                messages.append("📈 <strong>Buy Signal:</strong> MACD is below Signal line but histogram is negative. Indicates downtrend may be bottoming out.")
            else:
                messages.append("⚠️ <strong>Hold:</strong> MACD is below Signal line but histogram is improving.")

        return {
            "candlestick_chart": pio.to_html(fig_price, full_html=False),
            "macd_chart": pio.to_html(fig_macd, full_html=False),
            "messages": messages
        }

    @staticmethod
    def rsi_strategy(df):
        df['RSI'] = ta.RSI(df['Close'], timeperiod=14)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df['Date'], y=[70]*len(df), mode='lines', name='Overbought', line=dict(dash='dash', color='red')))
        fig.add_trace(go.Scatter(x=df['Date'], y=[30]*len(df), mode='lines', name='Oversold', line=dict(dash='dash', color='green')))

        fig.update_layout(title='RSI Indicator', xaxis_title='Date', height=350, width=900)
        return pio.to_html(fig, full_html=False)
    @staticmethod
    def ema_strategy(full_data: pd.DataFrame, selected_df: pd.DataFrame, term: int):
        # 1. Determine long-term EMA period
        if term <= 15:
            term_high = term + 7
        elif term <= 30:
            term_high = term + 20
        else:
            term_high = 200

        # 2. Calculate EMAs
        full_data[f'EMA_{term}'] = ta.EMA(full_data['Close'], term)
        full_data[f'EMA_{term_high}'] = ta.EMA(full_data['Close'], term_high)

        # 3. Generate signals
        full_data['Signal'] = np.where(full_data[f'EMA_{term}'] > full_data[f'EMA_{term_high}'], 1, 0)
        full_data['Position'] = full_data['Signal'].diff()

        # 4. Merge with filtered range
        df = pd.merge(
            full_data[['Date', f'EMA_{term}', f'EMA_{term_high}', 'Signal', 'Position']],
            selected_df,
            on='Date',
            how='inner'
        )
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # 5. Plot candlestick + EMAs + signals
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color='green',
            decreasing_line_color='red',
            line_width=1.2,
            whiskerwidth=0.2,
            opacity=0.85,
            showlegend=False
        ))

        fig.add_trace(go.Scatter(x=df.index, y=df[f'EMA_{term}'], name=f'EMA {term}', mode='lines', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df.index, y=df[f'EMA_{term_high}'], name=f'EMA {term_high}', mode='lines', line=dict(color='blue')))

        # Buy and Sell Markers
        buy_signals = df[df['Position'] == 1]
        sell_signals = df[df['Position'] == -1]

        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals[f'EMA_{term}'], mode='markers',
                                marker=dict(symbol='triangle-up', size=10, color='cyan'), name='BUY Signal'))
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals[f'EMA_{term}'], mode='markers',
                                marker=dict(symbol='triangle-down', size=10, color='yellow'), name='SELL Signal'))

        fig.update_layout(
            title='Exponential Moving Average (EMA) Strategy',
            xaxis_title='Date',
            yaxis_title='Price',
            height=600,
            width=900,
            xaxis_rangeslider_visible=False
        )

        # 6. Strategy recommendation
        end_ema_low = df[f'EMA_{term}'].iloc[-1]
        end_ema_high = df[f'EMA_{term_high}'].iloc[-1]
        end_close = df['Close'].iloc[-1]

        messages = []

        if end_ema_high > end_ema_low:
            if end_close > end_ema_high:
                messages.append(f"📈 <strong>Buy:</strong> EMA {term} is below price, and EMA {term_high} is higher. Uptrend forming.<br>Hold and consider selling when EMA {term} > price.")
            else:
                messages.append(f"⚠️ <strong>Hold:</strong> Price is under EMA {term_high}. Market may be reversing. Watch closely.")
        else:
            if end_close > end_ema_low:
                messages.append(f"⚠️ <strong>Hold:</strong> Price is above EMA {term}, but trend is weak. Watch for crossover.")
            else:
                messages.append(f"📉 <strong>Sell:</strong> EMA {term} is above price and EMA {term_high} is low. Downtrend confirmed. Consider selling.")

        return {
            "chart": pio.to_html(fig, full_html=False),
            "strategy_notes": messages
        }
    @staticmethod
    def run_analysis(df, indicators, term):
        """
        Parameters:
            df: full DataFrame with OHLC data
            indicators: list like ['SMA', 'MACD', 'RSI']
            term: int - short-term period
        Returns:
            Dict with {indicator_name: chart_html or list of outputs}
        """
        results = {}

        # Subset date range if needed (here it's full df)
        selected_df = df.copy()

        if 'SMA' in indicators:
            results['SMA'] = Technical.sma_strategy(df.copy(), selected_df.copy(), term)

        if 'EMA' in indicators:
            results['EMA'] = Technical.ema_strategy(df.copy(), selected_df.copy(), term)

        if 'MACD' in indicators:
            results['MACD'] = Technical.macd_strategy(df.copy(), selected_df.copy())

        if 'RSI' in indicators:
            results['RSI'] = Technical.rsi_strategy(df.copy())

        return results
