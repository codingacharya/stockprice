import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.title('Stock Price Predictions')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
st.sidebar.info("For Educational Purposes Only")

scaler = StandardScaler()

@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df


# Sidebar for input method
st.sidebar.subheader("Choose Data Source")
data_source = st.sidebar.radio("Data Source", ["Manual CSV Upload", "Fetch from Yahoo Finance"])

# Initialize data variable
data = None
uploaded_file = None

if data_source == "Manual CSV Upload":
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file, parse_dates=True)
        if 'Date' in data.columns:
            data.set_index('Date', inplace=True)
elif data_source == "Fetch from Yahoo Finance":
    option = st.sidebar.text_input('Enter a Stock Symbol', value='INFY').upper()
    today = datetime.date.today()
    duration = st.sidebar.number_input('Enter duration (days)', value=3000)
    before = today - datetime.timedelta(days=duration)
    start_date = st.sidebar.date_input('Start Date', value=before)
    end_date = st.sidebar.date_input('End date', today)
    if st.sidebar.button('Fetch Data'):
        if start_date < end_date:
            st.sidebar.success(f'Start: `{start_date}`  End: `{end_date}`')
            data = download_data(option, start_date, end_date)
        else:
            st.sidebar.error('End date must be after start date')

def main():
    if data is None:
        st.warning("Please upload a CSV or fetch data from Yahoo Finance.")
        return

    option = st.sidebar.selectbox('Choose an Action', ['Visualize','Recent Data', 'Predict'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    else:
        predict()

def tech_indicators():
    st.header('Technical Indicators')
    option = st.radio('Choose Technical Indicator', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

    bb_indicator = BollingerBands(data['Close'])
    bb = data.copy()
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    bb = bb[['Close', 'bb_h', 'bb_l']]
    macd = MACD(data['Close']).macd()
    rsi = RSIIndicator(data['Close']).rsi()
    sma = SMAIndicator(data['Close'], window=14).sma_indicator()
    ema = EMAIndicator(data['Close']).ema_indicator()

    if option == 'Close':
        st.line_chart(data['Close'])
    elif option == 'BB':
        st.line_chart(bb)
    elif option == 'MACD':
        st.line_chart(macd)
    elif option == 'RSI':
        st.line_chart(rsi)
    elif option == 'SMA':
        st.line_chart(sma)
    elif option == 'EMA':
        st.line_chart(ema)

def dataframe():
    st.header('Recent Data')
    st.dataframe(data.tail(10))

def predict():
    model = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'])
    num = st.number_input('How many days to forecast?', value=5)
    num = int(num)

    if st.button('Predict'):
        engine = None
        if model == 'LinearRegression':
            engine = LinearRegression()
        elif model == 'RandomForestRegressor':
            engine = RandomForestRegressor()
        elif model == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
        elif model == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
        else:
            engine = XGBRegressor()
        
        model_engine(engine, num)

def model_engine(model, num):
    df = data[['Close']].copy()
    df['preds'] = df['Close'].shift(-num)

    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    x_forecast = x[-num:]
    x = x[:-num]
    y = df['preds'].values[:-num]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    st.text(f'r2_score: {r2_score(y_test, preds):.4f} \nMAE: {mean_absolute_error(y_test, preds):.4f}')

    forecast_pred = model.predict(x_forecast)
    for i, val in enumerate(forecast_pred, start=1):
        st.text(f'Day {i}: {val:.2f}')

if __name__ == '__main__':
    main()
