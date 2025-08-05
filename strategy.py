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
    @staticmethod
    def run(full_data):
        df = full_data.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.set_index('Date', inplace=True)

        def find_support_resistance(data, no_cluster):
            kmeans = KMeans(n_clusters=no_cluster, n_init='auto').fit(data[['Close']])
            clusters = kmeans.predict(data[['Close']])

            min_max_values = [[np.inf, -np.inf] for _ in range(no_cluster)]
            for i, price in enumerate(data['Close']):
                cluster = clusters[i]
                if price < min_max_values[cluster][0]:
                    min_max_values[cluster][0] = price
                if price > min_max_values[cluster][1]:
                    min_max_values[cluster][1] = price

            sorted_ranges = sorted(min_max_values, key=lambda x: x[0])
            output = []
            for i, (_min, _max) in enumerate(sorted_ranges):
                if i == 0:
                    output.append(_min)
                elif i == len(sorted_ranges) - 1:
                    output.append(_max)
                else:
                    output.append((_max + sorted_ranges[i + 1][0]) / 2)
            return output

        def filter_sr(sr_array, close, threshold):
            return [val for val in sr_array if abs(val - close) <= threshold]

        num_clusters = 50
        sr_levels = find_support_resistance(df, num_clusters)
        sr_levels = filter_sr(sr_levels, df['Close'].iloc[-1], 50)

        data_range = 80
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index[-data_range:],
            open=df['Open'][-data_range:],
            high=df['High'][-data_range:],
            low=df['Low'][-data_range:],
            close=df['Close'][-data_range:],
            increasing_line_color='green',
            decreasing_line_color='red',
            line_width=1.2,
            whiskerwidth=0.2,
            opacity=0.85,
            name='Candlesticks'
        ))

        supports, resistances = [], []
        for point in sr_levels:
            color = 'red' if point < df['Close'].iloc[-1] else 'green'
            if point < df['Close'].iloc[-1]:
                supports.append(point)
            else:
                resistances.append(point)
            fig.add_shape(type="line",
                          x0=df.index[-data_range],
                          y0=point,
                          x1=df.index[-1],
                          y1=point,
                          line=dict(color=color, width=1),
                          opacity=0.7)

        fig.update_layout(
            title='Stock Price with Support and Resistance',
            xaxis_title='Date',
            yaxis_title='Price',
            height=800,
            width=1000
        )

        chart_html = pio.to_html(fig, full_html=False)

        # STRATEGY MESSAGES
        messages = []
        current_price = df['Close'].iloc[-1]
        strategy = "None"

        current_support = max(supports) if supports else None
        current_resistance = min(resistances) if resistances else None

        if current_resistance:
            if current_price < current_resistance:
                sell_t1 = round(current_resistance - 2, 2)
                sell_t2 = round(current_resistance + 2, 2)
                strategy = "Sell at Resistance"
                messages.append(f"📉 <strong>{strategy}</strong>: Sell between <strong>Rs. {sell_t1}</strong> and <strong>Rs. {sell_t2}</strong>")

            elif current_price > current_resistance:
                buy_t1 = round(current_resistance - 2, 2)
                buy_t2 = round(current_resistance + 2, 2)
                strategy = "Buy at Resistance"
                messages.append(f"📈 <strong>{strategy}</strong>: Buy between <strong>Rs. {buy_t1}</strong> and <strong>Rs. {buy_t2}</strong><br>Stop loss below <strong>Rs. {buy_t1}</strong>")

        if current_support:
            if current_price < current_support:
                stop_loss = round(current_support - 2, 2)
                strategy = "Watch and Hold"
                messages.append(f"🕒 <strong>{strategy}</strong>: Stop loss below <strong>Rs. {stop_loss}</strong>")
            elif current_price > current_support:
                buy_t1 = round(current_support - 2, 2)
                buy_t2 = round(current_support + 2, 2)
                strategy = "Buy at Support"
                messages.append(f"📈 <strong>{strategy}</strong>: Buy between <strong>Rs. {buy_t1}</strong> and <strong>Rs. {buy_t2}</strong><br>Stop loss below <strong>Rs. {buy_t1}</strong>")

        return {
            "chart_html": chart_html,
            "strategy_notes": messages
        }
