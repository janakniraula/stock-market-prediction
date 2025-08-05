import pandas as pd
import numpy as np
from pymongo import MongoClient
import talib as ta
import statsmodels.tsa.arima.model as ARIMA
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization, Input
import xgboost as xgb
from xgboost import plot_importance
from sklearn.decomposition import PCA
# import streamlit as st

class Preprocess:
    def getData(symbol):
        # conn_str = "mongodb+srv://pranavdahal:NW0NR8zXe85nwCYs@cluster0.cqakhhi.mongodb.net/"
        # client = MongoClient(conn_str)
        # db = client["nepse"]
        # collection = db["todayPrice"]

        # data = pd.DataFrame(list(collection.find({"symbol":symbol})))
        file_path = f"data/{symbol}.csv"
        # Read CSV data
        data = pd.read_csv(file_path)
        df = pd.DataFrame(columns=['Date', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = data['businessDate']
        df['symbol'] = data['symbol']
        df['Open'] = data['openPrice']
        df['High'] = data['highPrice']
        df['Low'] = data['lowPrice']
        df['Close'] = data['closePrice']
        df['Volume'] = data['totalTradedQuantity']
        df.set_index('Date', inplace=True)
        return df 
    @staticmethod
    def minMaxNormalization(input):
        normalized = (input - np.min(input)) / (np.max(input) - np.min(input))
        return normalized
    @staticmethod
    def minMaxDeNormalization(input, original):
        # Denormalize the data
        min_val = np.min(original)
        max_val = np.max(original)
        denormalized = input * (max_val - min_val) + min_val
        return denormalized
    @staticmethod
    def get_technical_indicators(nepse_data):
        data = nepse_data.copy()
        # data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
        # Moving Averages
        data['SMA_7'] = ta.SMA(data['Close'],7)
        data['SMA_21'] = ta.SMA(data['Close'],21)

        # MACD
        macd, macds,macdh = ta.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        data['MACD'] = macd
        data['MACD_signal'] = macds

        # Bollinger Bands
        BBU, BBM, BBL = ta.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        data['upper_band'] = BBU
        data['20sd'] = BBM
        data['lower_band'] = BBL

        
        
        # Create Exponential moving average
        data['EMA_7'] = ta.EMA(data['Close'],7)
        data['EMA_21'] = ta.EMA(data['Close'],21)
        
        # Create Momentum
        data['momentum'] = data['Close']-1
        data['log_momentum'] = np.log(data['momentum'])

        # data['MA_200'] = ta.MA(data['Close'],200)
        data['RSI'] = ta.RSI(data['Close'], timeperiod=14)

        # 0: buy, 1: sell, 2: hold
        data['label'] = np.where((data['Close'] > data['SMA_7']) & (data['MACD'] > data['MACD_signal']), 1,
                            np.where((data['Close'] < data['SMA_7']) & (data['MACD'] < data['MACD_signal']), 0,
                                    2))
        
        test_df = data.copy()

        normalized_data = pd.DataFrame()

        # Normalization
        for col in ['Open','High','Low','Close','Volume','SMA_7','SMA_21','MACD','MACD_signal','upper_band','20sd','lower_band','EMA_7','EMA_21','momentum','log_momentum']:
            normalized_data[col] = Preprocess.minMaxNormalization(data[col])

        return normalized_data, data['label'],test_df
    @staticmethod
    def get_fourier(data):
        """
        Apply Fourier Transform to extract frequency domain features
        """
        try:
            # Check if data is empty or None
            if data is None or data.empty:
                print("Warning: Empty data provided to get_fourier")
                return data
            
            # Data types to apply FFT to (assuming these are your OHLCV columns)
            data_types = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Filter to only existing columns
            available_types = [col for col in data_types if col in data.columns]
            
            if not available_types:
                print("Warning: No OHLCV columns found for FFT")
                return data
            
            for data_type in available_types:
                try:
                    # Get the column data and ensure it's not empty
                    column_data = data[data_type].dropna()
                    
                    if len(column_data) == 0:
                        print(f"Warning: No valid data for column {data_type}, skipping FFT")
                        continue
                    
                    # Convert to numpy array
                    data_array = np.asarray(column_data.tolist())
                    
                    # Check if we have enough data points for meaningful FFT
                    if len(data_array) < 2:
                        print(f"Warning: Insufficient data points ({len(data_array)}) for FFT on {data_type}")
                        continue
                    
                    # Perform FFT
                    data_fft = np.fft.fft(data_array)
                    
                    # Extract magnitude and phase
                    fft_magnitude = np.abs(data_fft)
                    fft_phase = np.angle(data_fft)
                    
                    # Create feature names
                    magnitude_col = f'{data_type}_fft_magnitude'
                    phase_col = f'{data_type}_fft_phase'
                    
                    # Pad or truncate to match original data length
                    original_length = len(data)
                    
                    if len(fft_magnitude) > original_length:
                        # Truncate
                        fft_magnitude = fft_magnitude[:original_length]
                        fft_phase = fft_phase[:original_length]
                    elif len(fft_magnitude) < original_length:
                        # Pad with zeros
                        pad_length = original_length - len(fft_magnitude)
                        fft_magnitude = np.pad(fft_magnitude, (0, pad_length), mode='constant')
                        fft_phase = np.pad(fft_phase, (0, pad_length), mode='constant')
                    
                    # Add to dataframe
                    data[magnitude_col] = fft_magnitude
                    data[phase_col] = fft_phase
                    
                except Exception as e:
                    print(f"Error processing FFT for {data_type}: {str(e)}")
                    continue
            
            return data
            
        except Exception as e:
            print(f"Error in get_fourier: {str(e)}")
            return data  # Return original data if FFT fails
    @staticmethod
    def get_arima_features(data,order=(2,1,3)):
        for data_type in ['Open','High','Low','Close']:
            X = data[data_type].values
            size = int(len(X) * 0.8)
            train, test = X[0:size], X[size:len(X)]
            history = [x for x in train]
            predictions = list()
            for t in range(len(test)):
                model = ARIMA.ARIMA(history, order=order)
                model_fit = model.fit(method_kwargs={"warn_convergence": False})
                output = model_fit.forecast()
                yhat = output[0]
                predictions.append(yhat)
                obs = test[t]
                history.append(obs)
            
            arima = [x for x in train]
            for pred in predictions:
                arima.append(pred)
            data[f'arima{order}_{data_type}'] = arima
            print(f'ARIMA{order}_{data_type} Generated..')
        
        return data
        
    # Auto Encoder features
    @staticmethod
    def autoEncoderFeatures(data, encoding_dim):
        for data_type in ['Open','High','Low','Close']:
            input_dim = 1
            hidden_dim = [int(len(data)/2), int(len(data)/4), int(len(data)/8)]

            # Encoder
            encoder = Sequential([
                Input(shape=(input_dim,)),
                Dense(hidden_dim[0], activation='relu'),
                Dense(hidden_dim[1], activation='relu'),
                Dense(hidden_dim[2], activation='relu'),
                Dense(encoding_dim, activation='relu')
            ])

            high_level_features = encoder.predict(data[data_type])
            col_names = [f'{data_type}_enc'+str(i) for i in range(encoding_dim)]
            enc_df = pd.DataFrame(high_level_features,columns=col_names)
            data = pd.concat([data, enc_df], axis = 1)
        return data

    def get_PCA(data,n_components):
        pca = PCA(n_components=n_components)
        pca.fit(data)
        principalComponents = pca.transform(data)
        principalDf = pd.DataFrame(data = principalComponents)

        data = pd.concat([data, principalDf], axis = 1)
        return data

    def preprocessData(data):
        # data['Date'] = pd.to_datetime(data['Date'])

        # technical indicators
        normalized,label,test_df = Preprocess.get_technical_indicators(data)

        # fourier transform
        normalized = Preprocess.get_fourier(normalized)

        # arima features
        # get_arima_features(normalized,order=(2,1,3))

        # auto encoder features
        normalized = Preprocess.autoEncoderFeatures(normalized,8)

        return normalized, label, test_df

    @staticmethod
    def get_important_features(xg_data,days=1):
        xg_data = xg_data.dropna()
        X = xg_data
        y = xg_data['Close'].shift(-days)
        y = y.dropna()
        X = xg_data[:-days]

        eval_set = [(X, y)]
        model = xgb.XGBRegressor(gamma=0.02, learning_rate= 0.08, max_depth= 15, n_estimators= 100, random_state=42, objective='reg:squarederror')
        model.fit(X, y,eval_set=eval_set, verbose=False)

        # plot_importance(model, height=0.5)

        importance_dict = model.get_booster().get_score(importance_type='weight')

        importance_list = [(feature, importance) for feature, importance in importance_dict.items()]

        importance_list = sorted(importance_list, key=lambda x: x[1], reverse=True)

        feature_score = pd.DataFrame(importance_list, columns=['feature', 'score'])

        col_names = list(feature_score['feature'].values)
        selected_feature_df = X[col_names]

        return selected_feature_df, feature_score
    
    @staticmethod
    def get_imp_features_signal(xg_data,label):
        temp = pd.concat([xg_data,label],axis=1)
        temp = temp.dropna()
        xg_data = xg_data.dropna()
        date = xg_data['Date']
        xg_data = xg_data.drop(['Date'],axis=1)
        X = xg_data
        y = temp['label']

        eval_set = [(X, y)]
        model = xgb.XGBRegressor(gamma=0.02, learning_rate= 0.08, max_depth= 15, n_estimators= 100, random_state=42, objective='reg:squarederror')
        model.fit(X, y,eval_set=eval_set, verbose=False)

        importance_dict = model.get_booster().get_score(importance_type='weight')

        importance_list = [(feature, importance) for feature, importance in importance_dict.items()]

        importance_list = sorted(importance_list, key=lambda x: x[1], reverse=True)

        feature_score = pd.DataFrame(importance_list, columns=['feature', 'score'])
        feature_score = feature_score[feature_score['score'] > 10]
        # st.dataframe(feature_score)

        col_names = list(feature_score['feature'].values)
        selected_feature_df = X[col_names]
        selected_feature_df['Date'] = date.values
        selected_feature_df['label'] = y.values

        return selected_feature_df, feature_score
