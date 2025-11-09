# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# from sklearn.cluster import KMeans
# import streamlit as st

# class Strategy:
#     def main(full_data):
#         df=pd.DataFrame()
#         df = full_data.copy()
#         df.set_index('Date', inplace=True)

#         def find_support_resistance(data, no_cluster):
#             support = []
#             resistance = []
#             k = no_cluster
#             kmeans = KMeans(n_clusters=k).fit(data[['Close']])
#             clusters = kmeans.predict(data[['Close']])

#             min_max_values = []
#             for i in range(k):
#                 min_max_values.append([np.inf,-np.inf])

#             for i in range(len(data['Close'])):
#                 cluster = clusters[i]

#                 if  data['Close'][i]< min_max_values[cluster][0]:
#                     min_max_values[cluster][0] = data['Close'][i]
                
#                 if  data['Close'][i]> min_max_values[cluster][1]:
#                     min_max_values[cluster][1] = data['Close'][i]

#             output = []

#             s = sorted(min_max_values, key=lambda x: x[0])

#             for i, (_min,_max) in enumerate(s):
#                 if i == 0:
#                     output.append(_min)
                
#                 if i == len(min_max_values) - 1:
#                     output.append(_max)
#                 else:
#                     output.append(sum([_max, s[i+1][0]])/2)
            
#             return output
                
#         num_clusters = 50 # number of clusters
#         supportResistance = find_support_resistance(df, num_clusters)
#         print("S/R: ",supportResistance)

#         def filter_sr(sr_array, close, difference):
#             filtered_sr = []
#             for value in sr_array:
#                 if abs(value - close) <= difference:
#                     filtered_sr.append(value)
#             return filtered_sr

#         supportResistance = filter_sr(supportResistance, df['Close'].iloc[-1], 50)
#         print("Filtered S/R: ",supportResistance)
#         print("Close: ",df['Close'].iloc[-1])
#         data_range=80
#         fig = go.Figure()
#         fig.add_trace(go.Candlestick(x=df.index[-data_range:],
#                                     open=df['Open'][-data_range:],
#                                     high=df['High'][-data_range:],
#                                     low=df['Low'][-data_range:],
#                                     close=df['Close'][-data_range:],
#                                     increasing_line_color='green',  # Color for increasing candlesticks
#                                     decreasing_line_color='red',    # Color for decreasing candlesticks
#                                     line_width=1.2,                 # Width of the candlestick lines
#                                     whiskerwidth=0.2,               # Width of the candlestick whiskers
#                                     opacity=0.85,                   # Opacity of the candlesticks
#                                     name='Candlesticks'))
#         supports = []
#         resistances=[]
#         for point in supportResistance:
#             if point < df['Close'].iloc[-1]:
#                 color = 'red'
#                 supports.append(point)
#             else:
#                 resistances.append(point)
#                 color = 'green'

#             fig.add_shape(type="line",
#                         x0=df.index[-data_range],
#                         y0=point,
#                         x1=df.index[-1],
#                         y1=point,
#                         line=dict(color=color, width=1),
#                         opacity=0.7)

#         fig.update_layout(title='Stock Price with Significant Support and Resistance Levels',
#                         xaxis=dict(title='Date',showgrid=False,zeroline=False),
#                         yaxis=dict(title='Price',showgrid=False,zeroline=False),
#                         showlegend=False,
#                         height=800,
#                         width=1000,)

#         st.plotly_chart(fig)

#         st.subheader("Current Strategy")

#         current_support = max(supports)
#         current_resistance = min(resistances)

#         if df['Close'].iloc[-1] < current_resistance:
#             # Place sell order at resistance
#             sell_target_1, sell_target_2 = current_resistance - 2, current_resistance + 2
#             strategy = "Sell at Resistance"
#             st.error(f"Sell at Resistance\
#                     \nSell @ Rs. {sell_target_1} to {sell_target_2}")

#         elif df['Close'].iloc[-1] > current_resistance:
#             # Place buy order at resistance
#             buy_target_1, buy_target_2 = current_resistance - 2, current_resistance + 2
#             strategy = "Buy at Support"
#             st.success(f"Buy at Support\
#                     \nBuy @ Rs. {buy_target_1} to {buy_target_2}\
#                     \nStop Loss if price below Rs. {current_resistance - 2}")


#         # Support level logic
#         if df['Close'].iloc[-1] < current_support:
#             # Place sell order for stop loss
#             stop_loss_target_1, stop_loss_target_2 = current_support - 2, current_support + 2
#             strategy = "Watch and Hold"
#             st.warning(f"{strategy}\
#                     \nStop Loss if price below Rs. {stop_loss_target_1}")

#         elif df['Close'].iloc[-1] > current_support:
#             # Place buy order at support
#             buy_target_1, buy_target_2 = current_support - 2, current_support + 2
#             strategy = "Buy at Support"
#             st.success(f"{strategy}\
#                     \nBuy @ Rs. {buy_target_1} to {buy_target_2}\
#                     \nStop Loss if price below Rs. {buy_target_1}")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import plotly.io as pio

class Strategy:
    def __init__(self, df):
        self.df = df.copy()
        self.df.columns = self.df.columns.str.strip()  # fix BOM
        self.df.set_index("Date", inplace=True)

    def kmeans_clustering(self, n_clusters=6, random_state=42):
        self.df["avg_price"] = (self.df["High"] + self.df["Low"] + self.df["Close"]) / 3.0
        prices = self.df["avg_price"].values.reshape(-1, 1)

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        centers = sorted(kmeans.fit(prices).cluster_centers_.flatten())

        # remove clusters too close (<1%)
        filtered = [centers[0]]
        for c in centers[1:]:
            if abs(c - filtered[-1]) / filtered[-1] > 0.01:
                filtered.append(c)

        return filtered

    def run(self, start_date=None, end_date=None):
        df = self.df.copy()

        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        if df.empty:
            raise ValueError("No data available for selected date range.")

        sr_levels = self.kmeans_clustering()
        current_price = df["Close"].iloc[-1]

        supports = [lvl for lvl in sr_levels if lvl < current_price]
        resistances = [lvl for lvl in sr_levels if lvl > current_price]

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            increasing_line_color="green", decreasing_line_color="red"
        ))

        # Plot Support (green)
        for lvl in supports:
            fig.add_shape(type="line", x0=df.index.min(), y0=lvl,
                        x1=df.index.max(), y1=lvl,
                        line=dict(color="green", width=1.4, dash="dot"))
            # Offset annotation to the left
            fig.add_annotation(x=df.index.min(), y=lvl, text=f"S {round(lvl,2)}",
                            xanchor='right', yanchor='bottom',
                            showarrow=False, font=dict(color="green"))

        # Plot Resistance (red)
        for lvl in resistances:
            fig.add_shape(type="line", x0=df.index.min(), y0=lvl,
                        x1=df.index.max(), y1=lvl,
                        line=dict(color="red", width=1.4, dash="dot"))
            # Offset annotation to the left
            fig.add_annotation(x=df.index.min(), y=lvl, text=f"R {round(lvl,2)}",
                            xanchor='right', yanchor='bottom',
                            showarrow=False, font=dict(color="red"))


        fig.update_layout(title=f"Support & Resistance ({start_date} → {end_date})",
                          xaxis_rangeslider_visible=False,
                          height=800)

        chart_html = pio.to_html(fig, full_html=False)

        # ✅ Strategy Messages
        messages = []
        if supports:
            s = max(supports)
            messages.append(
                f"📈 <b>Buy Near Support:</b> Rs. {round(s,2)}<br>"
                f"🔻 Stop Loss: <b>Rs. {round(s-2,2)}</b>"
            )

        if resistances:
            r = min(resistances)
            messages.append(
                f"📉 <b>Sell Near Resistance:</b> Rs. {round(r,2)}<br>"
                f"🎯 Target: <b>Rs. {round(r+2,2)}</b>"
            )

        return {"chart_html": chart_html, "strategy_notes": messages}

