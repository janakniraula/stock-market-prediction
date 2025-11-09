# import streamlit as st
# import datetime
# import pandas as pd
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# from utils import Preprocess 
# from technical import Technical
# from predictor import MarketPredictor
# from strategy import Strategy
# import csv

# # json_data = []
# # symbols = []
# # csv_file_path = 'data/companyName.csv'

# # with open(csv_file_path, 'r') as csv_file:
# #     csv_reader = csv.DictReader(csv_file)
# #     for row in csv_reader:
# #         json_data.append({
# #             "symbol": row["StockSymbol"],
# #             "companyName": row["CompanyName"],
# #         })
# #         symbols.append(row["StockSymbol"])
# json_data = [
#     {"symbol": "NABIL", "companyName": "Nepal Investment Bank Ltd"},
#     {"symbol": "ADBL", "companyName": "Agricultural Development Bank Ltd"},
#     {"symbol": "NICA", "companyName": "NIC Asia Bank Ltd"},
#     {"symbol": "SHL", "companyName": "Soaltee Hotel Ltd"},
#     {"symbol": "EBL", "companyName": "Everest Bank Ltd"},
#     {"symbol": "HIDCL", "companyName": "Hydroelectricity Investment & Dev Co Ltd"},
# ]

# symbols = [item["symbol"] for item in json_data]


# st.set_page_config(page_title="Stock Action Predictor", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# symbols.sort()
# # SIDEBAR #
# st.sidebar.header('Select Stock')
# stocklist = st.sidebar.selectbox('Select a stock symbol',symbols)
# min_date = datetime.datetime.strptime("2018-03-29", '%Y-%m-%d')
# today = "2024-02-20"
# today = datetime.datetime.strptime(today, '%Y-%m-%d')
# before = today - datetime.timedelta(days=90)
# start_date = st.sidebar.date_input('Start date', before, max_value=today,min_value=min_date)
# end_date = st.sidebar.date_input('End date', today, min_value=min_date, max_value=today)

# # Convert the dates to the 'dd-mm-yyyy' format
# s1 = start_date.strftime("%d-%m-%Y")
# e1 = end_date.strftime("%d-%m-%Y")

# st.sidebar.header('Displaying:')
# if start_date < end_date:
#     st.sidebar.success('Start date:  `%s`\n\nEnd date:  `%s`' % (s1, e1))
# else:
#     st.sidebar.error('Error: End date must fall after the start date.')

# st.sidebar.warning('Disclaimer: This app is for educational purposes only and is not intended to be trading advice. Use at your own risk.')

# def get_company_name(symbol):
#     for item in json_data:
#         if item['symbol'] == symbol:
#             return item['companyName']  
#     return symbol

# # DASHBOARD #
# stock_name = get_company_name(stocklist)
# data = pd.read_csv('data/' + stocklist + '.csv')

# # Attempt to convert the 'Date' column to the desired format, and if that fails, try an alternative format
# try:
#     data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
# except ValueError:
#     data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')

# # Filter DataFrame based on the date range
# df = data[data['Date'].between(s1, e1)]
# print(f"Length: {s1}, {e1}",len(df))
# # Candlestick Chart
# fig = go.Figure(data=[go.Candlestick(x=df['Date'],
#                 open=df['Open'],
#                 high=df['High'],
#                 low=df['Low'],
#                 close=df['Close'],
#                 increasing_line_color='green',  # Color for increasing candlesticks
#                 decreasing_line_color='red',    # Color for decreasing candlesticks
#                 line_width=1.2,                 # Width of the candlestick lines
#                 whiskerwidth=0.2,               # Width of the candlestick whiskers
#                 opacity=0.85,                   # Opacity of the candlesticks
#                 showlegend=False)])             

# fig.update_layout(
#     xaxis_rangeslider_visible=True,
#     height=700, 
#     width=1000,                      
#     paper_bgcolor='rgba(0,0,0,0.1)',  
#     plot_bgcolor='rgba(0,0,0,0.1)',    
#     font=dict(color='white'),      
#     margin=dict(l=10, r=10, t=50, b=10), 
#     title=f'Stock Daily Price Movement',  
#     title_font=dict(size=20),        
#     title_x=0.4,                     
# )

# # MAIN PAGE #

# # Display stock name and symbol
# st.header(stocklist + ' (' + stock_name + ')')

# # Display daily price movement candlestick chart
# st.plotly_chart(fig)

# st.subheader("Choose an option")

# option = st.selectbox('Select an option',('-----','Analyze Technical Indicator', 'Predict Trading Action using GAN', 'Predict Strategy based on Support and Resistance'))

# # Analyze Technical Indicator #
# if option=="Analyze Technical Indicator":
#     st.info("A technical indicator in the stock market is a mathematical tool derived from historical market data,\
#             helping traders analyze price movements and forecast future trends. These indicators encompass various types,\
#             such as trend-following, momentum, volatility, and oscillators, each serving specific purposes in market analysis.")
#     Technical.main(data,df)

# # # Button 2: Predict price movement
# # if option=="Predict Future Price Movement":
# #     st.info("Development in progress")
# #     # Movement.main(data)

# # Button 3: Predict trading action using GAN
# if option=="Predict Trading Action using GAN":
#     MarketPredictor.main(data,df)

# # Button 4: Predict strategy based on support and resistance
# if option=="Predict Strategy based on Support and Resistance":
#     Strategy.main(data)



from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify
import pandas as pd
import datetime
import plotly.graph_objs as go
import plotly.io as pio
import os
os.path.exists('predictor.py')
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from utils import Preprocess
from technical import Technical
from predictor import MarketPredictor
from strategy import Strategy
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8', errors='replace')
import uuid
from sqlalchemy.dialects.postgresql import UUID
from functools import wraps
from scrapperflask import update_csv


app = Flask(__name__)
# Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://root:1234@localhost:5432/stock-market-prediction'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'supersecretkey' 

db = SQLAlchemy(app)

class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
class Stock(db.Model):
    __tablename__ = 'stocks'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    company_name = db.Column(db.String(255), nullable=False)
    company_code = db.Column(db.String(50), unique=True, nullable=False)
    
def get_company_name(symbol):
    stock = Stock.query.filter_by(company_code=symbol).first()
    return stock.company_name if stock else symbol

predictor = MarketPredictor()
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to continue.')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function
def get_sidebar_data():
    """Prepare sidebar data dynamically from DB."""
    stocks = Stock.query.order_by(Stock.company_code).all()
    symbols = [stock.company_code for stock in stocks]

    sidebar_data = {
        "symbols": symbols
    }
    return sidebar_data


@app.context_processor
def inject_sidebar_data():
    """
    Context processor to inject sidebar data into ALL templates automatically
    """
    return {
        'sidebar_data': get_sidebar_data()
    }

@app.route("/stock/<symbol>")
def stock_data(symbol):
    """Read CSV data for a given stock and return from/to dates."""
    csv_path = os.path.join("data", f"{symbol}.csv")

    if not os.path.exists(csv_path):
        return {"error": f"No CSV found for {symbol}"}, 404

    # Read CSV
    df = pd.read_csv(csv_path)

    # Normalize column names to match even if lowercase
    df.columns = [col.strip().capitalize() for col in df.columns]

    required_cols = {"Date", "Symbol", "Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(df.columns):
        return {"error": "CSV missing required columns"}, 400

    # Sort by date to ensure order
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Correct order: first value = from_date, last value = to_date
    from_date = str(df.iloc[0]["Date"].date())
    to_date = str(df.iloc[-1]["Date"].date())

    return {
        "symbol": symbol,
        "from_date": from_date,
        "to_date": to_date
    }

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip().lower()
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Check if passwords match
        if password != confirm_password:
            flash('Passwords do not match!')
            return redirect(url_for('register'))

        # Check if username or email already exist
        if User.query.filter_by(username=username).first():
            flash('Username already exists!')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered!')
            return redirect(url_for('register'))

        hashed_pw = generate_password_hash(password)
        new_user = User(username=username, email=email, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()

        flash('Registered successfully! Please log in.')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash('Login successful!')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.')

    return render_template('login.html')
@app.route('/scrape', methods=['POST'])
@login_required
def scrape():
    symbol = request.form.get('symbol', '').upper()
    if not symbol:
        flash("Please enter a stock symbol.")
        return redirect(url_for('home'))

    try:
        result = update_csv(symbol)

        # Load the scraped CSV into a DataFrame
        df = pd.read_csv(result['file'])
        table_html = df.tail(20).to_html(classes="table table-striped", index=False)  # show last 20 rows

        return render_template(
            'scrape_result.html',
            symbol=symbol,
            message=result['message'],
            date_range=result['date_range'],
            rows=result['rows'],
            table_html=table_html
        )

    except Exception as e:
        flash(f"Error: {e}")
        return redirect(url_for('home'))


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logged out successfully!')
    return redirect(url_for('login'))

@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    symbols = [s.company_code for s in Stock.query.order_by(Stock.company_code).all()]
    # today = datetime.datetime.strptime("2024-02-20", "%Y-%m-%d")
    today = datetime.date.today()
    default_start = (today - datetime.timedelta(days=90)).strftime("%Y-%m-%d")

    if request.method == "POST":
        # stocklist = request.form["  symbol"]
        # start_date = request.form["start_date"]
        # end_date = request.form["end_date"]
        # option = request.form["option"]

        # try:
        #     df = pd.read_csv(f"data/{stocklist}.csv")
        #     df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        #     mask = (df["Date"] >= start_date) & (df["Date"] <= end_date)
        #     df = df[mask]
        # except Exception as e:
        #     return f"Error loading data: {e}"

        # # Candlestick chart
        # fig = go.Figure(data=[go.Candlestick(
        #     x=df["Date"],
        #     open=df["Open"],
        #     high=df["High"],
        #     low=df["Low"],
        #     close=df["Close"],
        #     increasing_line_color='green',
        #     decreasing_line_color='red',
        # )])
        # fig.update_layout(
        #     xaxis_rangeslider_visible=True,
        #     title="Stock Daily Price Movement",
        #     height=600
        # )
        # chart_html = pio.to_html(fig, full_html=False)

        # # Handle user option
        # result_html = ""
        # if option == "Analyze Technical Indicator":
        #     result_html = Technical.main(df)
        # elif option == "Predict Trading Action using GAN":
        #     result_html = MarketPredictor.main(df)
        # elif option == "Predict Strategy based on Support and Resistance":
        #     result_html = Strategy.main(df)

        # return render_template("chart.html", symbol=stocklist,
        #                        company=get_company_name(stocklist),
        #                        chart_html=chart_html,
        #                        result_html=result_html,
        #                        start_date=start_date,
        #                        end_date=end_date,
        #                        option=option)
        pass
    # GET request renders initial form
    return render_template("index.html", symbols=symbols, today=today.strftime("%Y-%m-%d"), default_start=default_start)

# @app.route("/predict", methods=["POST"])
# def predictor_view():
#     symbol = request.form.get("symbol")
#     start_date = request.form.get("start_date")
#     end_date = request.form.get("end_date")
#     print(f"Processing: {symbol}, {start_date}, {end_date}")
    
#     try:
#         # Load the full dataset
#         full_data = pd.read_csv(f"data/{symbol}.csv")
#         print(f"Loaded CSV with {len(full_data)} rows")
        
#         # Fix date parsing warning by being explicit
#         full_data['Date'] = pd.to_datetime(full_data['Date'], errors='coerce', dayfirst=False)
#         full_data = full_data.dropna(subset=['Date'])
#         print(f"After date parsing: {len(full_data)} rows")
        
#         # Ensure minimum data requirement
#         if len(full_data) < 100:
#             return f"Error: Insufficient data for symbol {symbol}. Need at least 100 rows, found {len(full_data)}"
        
#         # Create filtered dataset if date range provided
#         selected_df = None
#         if start_date and end_date:
#             # Convert string dates to datetime for comparison
#             start_dt = pd.to_datetime(start_date)
#             end_dt = pd.to_datetime(end_date)
            
#             selected_df = full_data[(full_data['Date'] >= start_dt) & (full_data['Date'] <= end_dt)].copy()
#             print(f"Filtered data: {len(selected_df)} rows")
            
#             # If filtered data is too small, use full dataset but warn user
#             if len(selected_df) < 50:
#                 print(f"Warning: Filtered data too small ({len(selected_df)} rows), using full dataset")
#                 selected_df = None
        
#         # Ensure we have required columns
#         required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
#         missing_columns = [col for col in required_columns if col not in full_data.columns]
#         if missing_columns:
#             return f"Error: Missing required columns: {missing_columns}"
        
#         # Call run_model with proper parameters
#         result = MarketPredictor.run_model(full_data, selected_df)

#         return render_template(
#             'predictor.html',
#             symbol=symbol,
#             chart=result.get('price_chart'),
#             confusion_matrix=result.get('confusion_matrix'),
#             metrics=result.get('metrics')
#         )

#     except FileNotFoundError:
#         return f"Error: Data file for symbol {symbol} not found"
#     except ValueError as e:
#         return f"Data error: {str(e)}"
#     except Exception as e:
#         print(f"Unexpected error: {str(e)}")
#         return f"An unexpected error occurred during prediction: {str(e)}"

@app.route("/predict", methods=["POST"])
@login_required
def predictor_view():
    symbol = request.form.get("symbol")
    start_date = request.form.get("start_date")
    end_date = request.form.get("end_date")
    # print(f"Processing: {symbol}, {start_date}, {end_date}")
    
    try:
        # Load the full dataset
        full_data = pd.read_csv(f"data/{symbol}.csv")
        # print(f"Loaded CSV with {len(full_data)} rows")
        
        # Fix date parsing warning by being explicit
        full_data['Date'] = pd.to_datetime(full_data['Date'], errors='coerce', dayfirst=False)
        full_data = full_data.dropna(subset=['Date'])
        # print(f"After date parsing: {len(full_data)} rows")
        
        # Ensure minimum data requirement
        if len(full_data) < 100:
            # return f"Error: Insufficient data for symbol {symbol}. Need at least 100 rows, found {len(full_data)}"
         return render_template('error.html',
                                 error_message=f"Insufficient data for symbol {symbol}. Need at least 100 rows, found {len(full_data)}",
                                 symbol=symbol,
                                 company_name=get_company_name(symbol))
        # Create filtered dataset if date range provided
        selected_df = None
        if start_date and end_date:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            selected_df = full_data[(full_data['Date'] >= start_dt) & (full_data['Date'] <= end_dt)].copy()
            # print(f"Filtered data: {len(selected_df)} rows")
            
            if len(selected_df) < 50:
                # print(f"Warning: Filtered data too small ({len(selected_df)} rows), using full dataset")
                selected_df = None
        
        # Ensure we have required columns
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in full_data.columns]
        if missing_columns:
            # return f"Error: Missing required columns: {missing_columns}"
            return render_template('error.html',
                                 error_message=f"Missing required columns: {missing_columns}",
                                 symbol=symbol,
                                 company_name=get_company_name(symbol))
        
        # print("About to call MarketPredictor.run_model...")
        
        # Call run_model with proper parameters
        result = MarketPredictor.run_model(full_data, selected_df)
        
        # print(f"Model completed. Result keys: {list(result.keys())}")
        
        # Try to render template with all 4 charts
        return render_template(
            'prediction_result.html',
            symbol=symbol,
            initial_chart=result.get('initial_chart'),           # Chart 1: Initial zones
            feature_importance=result.get('feature_importance'),  # Chart 2: Feature importance
            final_chart=result.get('price_chart'),               # Chart 3: Final predictions
            confusion_matrix=result.get('confusion_matrix'),     # Chart 4: Confusion matrix
            metrics=result.get('metrics')
        )
        
    except Exception as e:
        print(f"Exception occurred: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"An unexpected error occurred: {str(e)}"
    except Exception as e:
        print(f"Exception occurred: {type(e).__name__}: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        return f"An unexpected error occurred: {str(e)}"


@app.route("/technical", methods=["POST"])
@login_required
def technical_analysis():
    symbol = request.form.get("symbol")
    indicators = request.form.getlist("indicators")  # checkbox values
    term = int(request.form.get("term"))
    start_date = request.form.get("start_date")
    end_date = request.form.get("end_date")
    try:
        df = pd.read_csv(f"data/{symbol}.csv")
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)

        results = Technical.run_analysis(df, indicators, term, start_date, end_date)

        return render_template(
            "technical_result.html",
            symbol=symbol,
            company_name=get_company_name(symbol),
            charts=results,
            indicators=indicators,
            term=term,
            start_date=start_date,
            end_date=end_date
        )
    except Exception as e:
            return render_template('error.html',
                             error_message=f"Technical analysis error: {str(e)}",
                             symbol=symbol,
                             company_name=get_company_name(symbol))


@app.route("/strategy", methods=["POST"])
@login_required
def strategy_view():
    symbol = request.form.get("symbol")
    start_date = request.form.get("start_date")
    end_date = request.form.get("end_date")

    try:
        df = pd.read_csv(f"data/{symbol}.csv", encoding="utf-8-sig")
        df.columns = df.columns.str.strip()  # fix BOM whitespace
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.dropna(subset=["Date"], inplace=True)

        strategy = Strategy(df)  # instantiate strategy with df
        result = strategy.run(start_date=start_date, end_date=end_date)

        return render_template(
            "strategy_result.html",
            symbol=symbol,
            company_name=get_company_name(symbol),
            chart=result["chart_html"],
            notes=result["strategy_notes"]
        )

    except Exception as e:
        return render_template(
            "error.html",
            error_message=f"Strategy analysis error: {str(e)}",
            symbol=symbol,
            company_name=get_company_name(symbol)
        )

@app.route('/utils', methods=['GET', 'POST'])
@login_required
def utils():
    error = None
    feature_score_html = None
    symbol = None

    if request.method == 'POST':
        symbol = request.form.get('symbol')
        if not symbol:
            error = "Please enter a stock symbol."
        else:
            data = Preprocess.getData(symbol)
            if data.empty:
                error = f"No data found for symbol '{symbol}'."
            else:
                normalized, label, _ = Preprocess.preprocessData(data)
                _, feature_score = Preprocess.get_important_features(normalized, days=1)
                feature_score_html = feature_score.to_html(classes="table table-striped", index=False)

    return render_template('utils.html', error=error, feature_score_table=feature_score_html, symbol=symbol)

if __name__ == "__main__":
    app.run(debug=True)

