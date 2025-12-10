# Stock Market Prediction Application

A comprehensive Flask-based web application for analyzing and predicting stock market trends in the Nepali stock market using advanced machine learning techniques, technical indicators, and GAN (Generative Adversarial Network) models.

## 🌟 Features

- **User Authentication**: Secure registration and login system with password hashing
- **Stock Data Management**: Real-time scraping of stock data from Merolagani
- **Technical Analysis**: Multiple technical indicators including:
  - Simple Moving Average (SMA)
  - Exponential Moving Average (EMA)
  - MACD (Moving Average Convergence Divergence)
  - RSI (Relative Strength Index)
  - Bollinger Bands
- **AI-Powered Predictions**: GAN-based model for buy/sell/hold signal generation
- **Strategy Analysis**: Support and resistance-based trading strategies
- **Interactive Visualizations**: Dynamic charts using Plotly
- **Model Caching**: Intelligent caching system to speed up repeated predictions

## 📋 Prerequisites

- **Python 3.10** (Required for TA-Lib compatibility)
- PostgreSQL Database
- MongoDB (Optional, for certain features)
- Windows 64-bit OS (for pre-compiled TA-Lib)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd smp
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac
```

### 3. Install TA-Lib (Windows)

The project includes a pre-compiled TA-Lib wheel for Python 3.10:

```bash
pip install ta_lib-0.6.3-cp310-cp310-win_amd64.whl
```

**For other platforms**, install TA-Lib following [official instructions](https://github.com/mrjbq7/ta-lib#installation).

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install Playwright Browsers

```bash
playwright install chromium
```

### 6. Database Setup

#### PostgreSQL Setup

1. Install PostgreSQL if not already installed
2. Create a database:

```sql
CREATE DATABASE "stock-market-prediction";
CREATE USER root WITH PASSWORD '1234';
GRANT ALL PRIVILEGES ON DATABASE "stock-market-prediction" TO root;
```

3. Update the database URI in `app.py` if needed:

```python
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://root:1234@localhost:5432/stock-market-prediction'
```

4. Initialize the database tables:

```python
from app import app, db
with app.app_context():
    db.create_all()
```

5. Populate stocks table with company data:

```sql
INSERT INTO stocks (company_name, company_code) VALUES
('Nabil Bank Ltd', 'NABIL'),
('NIC Asia Bank Ltd', 'NICA'),
('Everest Bank Ltd', 'EBL');
-- Add more stocks as needed
```

## 📁 Project Structure

```
smp/
├── app.py                 # Main Flask application
├── predictor.py          # GAN model for predictions
├── technical.py          # Technical indicator analysis
├── strategy.py           # Trading strategy implementation
├── utils.py              # Data preprocessing utilities
├── scrapper.py           # Stock data scraper
├── scrapperflask.py      # Flask-integrated scraper
├── requirements.txt      # Python dependencies
├── data/                 # Stock CSV files
├── data2/                # Scraped data storage
├── model_cache/          # Cached ML models
├── static/               # CSS, JS, images
└── templates/            # HTML templates
    ├── index.html
    ├── login.html
    ├── register.html
    ├── prediction_result.html
    ├── technical_result.html
    └── strategy_result.html
```

## 🎯 Usage

### 1. Start the Application

```bash
python app.py
```

The application will run on `http://127.0.0.1:5000`

### 2. Register & Login

- Navigate to `/register` to create an account
- Login at `/login` with your credentials

### 3. Scrape Stock Data

- From the home page, enter a stock symbol (e.g., NABIL)
- Click "Scrape Data" to fetch latest stock information
- Data is saved to `data/` directory

### 4. Analyze Stocks

#### Technical Analysis
- Select a stock symbol and date range
- Choose indicators (SMA, EMA, MACD, RSI, Bollinger Bands)
- Adjust term slider for short/long-term analysis
- View interactive charts and strategy recommendations

#### AI Predictions
- Select stock and date range
- Click "Predict Trading Action"
- View:
  - Initial buy/sell/hold zones
  - Feature importance chart
  - AI model predictions
  - Confusion matrix and metrics

#### Strategy Analysis
- Select stock and date range
- Analyze support/resistance levels
- Get trading strategy recommendations

## 🔧 Configuration

### Environment Variables (Optional)

Create a `.env` file for configuration:

```env
DATABASE_URL=postgresql://root:1234@localhost:5432/stock-market-prediction
SECRET_KEY=your-secret-key-here
FLASK_ENV=development
```

### Database Configuration

Update in `app.py`:

```python
app.config['SQLALCHEMY_DATABASE_URI'] = 'your-database-uri'
app.config['SECRET_KEY'] = 'your-secret-key'
```

## 🤖 Model Details

### GAN Architecture

- **Generator**: Bidirectional LSTM with dropout and batch normalization
- **Discriminator**: CNN with LeakyReLU activation
- **Training**: Early stopping with 100 epochs max, patience=15
- **Output**: 3-class classification (Buy/Sell/Hold)

### Technical Indicators

- **SMA/EMA**: Configurable periods (3-200 days)
- **MACD**: 12/26/9 standard parameters
- **RSI**: 14-day period
- **Bollinger Bands**: 20-day period, ±2 standard deviations

## 📊 Data Format

Stock CSV files should have the following columns:

```
Date,Symbol,Open,High,Low,Close,Volume
2024-01-01,NABIL,850.0,870.0,845.0,865.0,15000
```

## ⚠️ Important Notes

1. **TA-Lib Installation**: The provided wheel is for Python 3.10 on Windows 64-bit only
2. **Database**: Ensure PostgreSQL is running before starting the app
3. **Scraping**: Manual interaction required - browser will open for you to click "Price History"
4. **Model Cache**: Predictions are cached daily; clear `model_cache/` to force retraining
5. **Data Requirements**: Minimum 100 rows needed for predictions

## 🐛 Troubleshooting

### TA-Lib Import Error
- Ensure you're using Python 3.10
- Install the provided wheel file
- For other Python versions, compile TA-Lib from source

### Database Connection Error
- Verify PostgreSQL is running
- Check database credentials
- Ensure database exists

### Playwright Browser Error
- Run `playwright install chromium`
- Check internet connection for initial browser download

### Model Training Errors
- Ensure sufficient data (100+ rows)
- Check for NaN values in CSV files
- Verify all required columns are present

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is for educational purposes only and is not intended as financial advice.

## 🔗 Resources

- [TA-Lib Documentation](https://ta-lib.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Plotly Documentation](https://plotly.com/python/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)

## 👥 Support

For issues and questions:
- Check the troubleshooting section
- Review existing issues on GitHub
- Create a new issue with detailed information

---

**Disclaimer**: This application is for educational and research purposes only. Do not use it as the sole basis for making investment decisions. Always conduct your own research and consult with financial professionals before investing.
