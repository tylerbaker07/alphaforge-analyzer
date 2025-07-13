import streamlit as st
import yfinance as yf
import pandas as pd

# Your full code from before (functions, constants, founder_led list, etc.) goes here
# ... (omit for brevity, but include normalize_high, normalize_low, calculate_dcf, weights, etc.)

@st.cache_data
def load_data():
    # Your data fetching and DF creation code here
    # ... (tickers fetch, loop for data append, score calculations, composite score)
    return df

st.title("AlphaForge Analyzer - Stock Screener")
st.markdown("Screen S&P 500 stocks with 12 metrics, sector grouping, and DCF valuation. Use filters below.")

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
min_rev_growth = st.sidebar.slider("Min Revenue Growth", 0.0, 1.0, 0.0, 0.01)
min_composite = st.sidebar.slider("Min Composite Score", 0.0, 10.0, 0.0, 0.1)
selected_sector = st.sidebar.selectbox("Sector", ["All"] + list(df['Sector'].unique()))

# Apply filters
filtered_df = df[df['Rev Growth'] >= min_rev_growth]
filtered_df = filtered_df[filtered_df['Composite Score'] >= min_composite]
if selected_sector != "All":
    filtered_df = filtered_df[filtered_df['Sector'] == selected_sector]

# Top 10 display
st.subheader("Top 10 Stocks Overall")
top_10 = filtered_df.sort_values('Composite Score', ascending=False).head(10)
st.dataframe(top_10[['Ticker', 'Sector', 'Price', 'Composite Score', 'DCF Value']])

# Top by sector
st.subheader("Top 5 per Sector")
grouped = filtered_df.groupby('Sector')
for sector, group in grouped:
    if sector != 'Unknown':
        with st.expander(f"Sector: {sector}"):
            top_in_sector = group.sort_values('Composite Score', ascending=False).head(5)
            st.dataframe(top_in_sector[['Ticker', 'Price', 'Composite Score', 'DCF Value']])

# Stock details
ticker_input = st.text_input("Enter Ticker for Details")
if ticker_input:
    stock_data = df[df['Ticker'] == ticker_input.upper()]
    if not stock_data.empty:
        stock = stock_data.iloc[0]
        st.subheader(f"Details for {ticker_input.upper()}")
        st.write(f"Sector: {stock['Sector']}")
        st.write(f"Price: ${stock['Price']:.2f}")
        st.write(f"Composite Score: {stock['Composite Score']:.2f}")
        # Add metric table from your show_stock_details function, adapted to st.dataframe

# Exports
if st.button("Export Full Data"):
    st.download_button("Download CSV", df.to_csv(index=False), "full_data.csv")
