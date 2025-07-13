import streamlit as st
import yfinance as yf
import pandas as pd

# Predefined list of founder-led S&P 500 companies (based on 2025 data)
founder_led = ['NVDA', 'TSLA', 'META', 'ABNB', 'DELL', 'CRM', 'PINS']

# Constants for DCF calculation
RISK_FREE_RATE = 0.04  # Approximate 10-year Treasury yield
MARKET_RISK_PREMIUM = 0.05  # Historical average
PERPETUAL_GROWTH_RATE = 0.03  # Long-term GDP growth approximation
FORECAST_YEARS = 5  # Number of years for explicit forecast

# Function to normalize high-desired metrics (higher is better)
def normalize_high(series):
    if series.max() - series.min() == 0:
        return series * 0  # Avoid division by zero
    return (series - series.min()) / (series.max() - series.min()) * 10

# Function to normalize low-desired metrics (lower is better)
def normalize_low(series):
    if series.max() - series.min() == 0:
        return series * 0
    return 10 - normalize_high(series)

# Simple DCF calculation function
def calculate_dcf(info):
    try:
        fcf = info.get('freeCashflow', 0)
        if fcf <= 0:
            return 0  # Skip if no positive FCF

        growth_rate = info.get('earningsGrowth', 0)
        beta = info.get('beta', 1.0)
        wacc = RISK_FREE_RATE + beta * MARKET_RISK_PREMIUM  # Approximate WACC using CAPM (ignoring debt for simplicity)

        if wacc <= PERPETUAL_GROWTH_RATE:
            return 0  # Avoid invalid terminal value

        # Project FCF for forecast years
        projected_fcfs = [fcf * (1 + growth_rate) ** (i + 1) for i in range(FORECAST_YEARS)]

        # Discount projected FCFs
        discounted_fcfs = [projected_fcfs[i] / (1 + wacc) ** (i + 1) for i in range(FORECAST_YEARS)]

        # Terminal value at end of forecast
        terminal_fcf = projected_fcfs[-1] * (1 + PERPETUAL_GROWTH_RATE)
        terminal_value = terminal_fcf / (wacc - PERPETUAL_GROWTH_RATE)
        discounted_terminal = terminal_value / (1 + wacc) ** FORECAST_YEARS

        # Enterprise value
        enterprise_value = sum(discounted_fcfs) + discounted_terminal

        # Equity value (subtract net debt)
        net_debt = info.get('totalDebt', 0) - info.get('totalCash', 0)
        equity_value = enterprise_value - net_debt

        # Intrinsic value per share
        shares_outstanding = info.get('sharesOutstanding', 1)  # Avoid division by zero
        intrinsic_value = max(equity_value / shares_outstanding, 0)

        return intrinsic_value
    except:
        return 0

# Fetch S&P 500 tickers
@st.cache_data
def load_data():
    tickers_df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers = tickers_df['Symbol'].tolist()[:100]  # Limit to 100 for performance; adjust as needed

    data = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Extract metrics
            rev_growth = info.get('revenueGrowth', 0)
            eps_growth = info.get('earningsGrowth', 0)
            market_cap = info.get('marketCap', 1e12)  # High default for low score
            fcf = info.get('freeCashflow', 0)
            roe = info.get('returnOnEquity', 0)
            d2e = info.get('debtToEquity', 100)
            pe = info.get('trailingPE', 100)
            peg = info.get('pegRatio', 5)
            gross_margin = info.get('grossMargins', 0)
            total_rev = info.get('totalRevenue', 1e10)
            price = info.get('currentPrice', 0)
            sector = info.get('sector', 'Unknown')  # Add sector for grouping

            # Calculate ROCE
            income_stmt = stock.income_stmt
            ebit = income_stmt.loc['EBIT'].iloc[0] if 'EBIT' in income_stmt.index else 0
            balance_sheet = stock.balance_sheet
            total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else 0
            current_liab = balance_sheet.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in balance_sheet.index else 0
            capital_employed = total_assets - current_liab
            roce = ebit / capital_employed if capital_employed != 0 else 0

            # Binary flags
            is_founder_led = 1 if ticker in founder_led else 0
            is_pre_revenue = 1 if total_rev < 1e6 else 0

            # Calculate DCF intrinsic value
            dcf_value = calculate_dcf(info)

            data.append({
                'Ticker': ticker,
                'Price': price,
                'Sector': sector,
                'Rev Growth': rev_growth,
                'EPS Growth': eps_growth,
                'Market Cap': market_cap,
                'FCF': fcf,
                'ROE': roe,
                'D/E': d2e,
                'P/E': pe,
                'PEG': peg,
                'ROCE': roce,
                'Gross Margin': gross_margin,
                'Founder Led': is_founder_led,
                'Pre Revenue': is_pre_revenue,
                'DCF Value': dcf_value
            })
        except Exception:
            pass

    df = pd.DataFrame(data)

    # Calculate individual scores
    df['Score Rev Growth'] = normalize_high(df['Rev Growth'])
    df['Score EPS Growth'] = normalize_high(df['EPS Growth'])
    df['Score Market Cap'] = normalize_low(df['Market Cap'])
    df['Score FCF'] = (df['FCF'] > 0).astype(int) * 10  # Binary for positive FCF
    df['Score ROE'] = normalize_high(df['ROE'])
    df['Score D/E'] = normalize_low(df['D/E'])
    df['Score P/E'] = normalize_low(df['P/E'])
    df['Score PEG'] = normalize_low(df['PEG'])
    df['Score ROCE'] = normalize_high(df['ROCE'])
    df['Score Gross Margin'] = normalize_high(df['Gross Margin'])
    df['Score Founder Led'] = df['Founder Led'] * 10
    df['Score Pre Revenue'] = df['Pre Revenue'] * 10

    # Define weights for composite score (total should sum to 1.0)
    weights = {
        'Score Rev Growth': 0.15,       # High weight on growth
        'Score EPS Growth': 0.15,
        'Score Market Cap': 0.05,       # Lower weight on size
        'Score FCF': 0.10,
        'Score ROE': 0.10,
        'Score D/E': 0.08,
        'Score P/E': 0.08,
        'Score PEG': 0.08,
        'Score ROCE': 0.08,
        'Score Gross Margin': 0.05,
        'Score Founder Led': 0.05,
        'Score Pre Revenue': 0.03
    }

    # Ensure weights sum to approximately 1 (handle floating point precision)
    weight_sum = sum(weights.values())
    assert abs(weight_sum - 1.0) < 0.001, f"Weights must sum to 1, got {weight_sum}"

    # Weighted composite score
    score_columns = list(weights.keys())
    df['Composite Score'] = sum(df[col] * weights[col] for col in score_columns)

    # Add DCF-based score (e.g., if DCF > Price, higher score)
    df['Score DCF'] = normalize_high((df['DCF Value'] - df['Price']) / df['Price'])  # Relative undervaluation

    # Optional: Include DCF score in composite (adjust weights accordingly if needed)
    df['Composite Score'] += df['Score DCF'] * 0.1  # Add 10% weight for DCF; adjust as needed

    return df

# Streamlit app structure
st.title("AlphaForge Analyzer - Stock Screener")
st.markdown("Interactive tool to screen S&P 500 stocks based on 12 metrics, with sector grouping and DCF valuation.")

# Load data with caching for performance
df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
min_rev_growth = st.sidebar.slider("Minimum Revenue Growth", 0.0, 1.0, 0.0)
min_composite = st.sidebar.slider("Minimum Composite Score", 0.0, 10.0, 0.0)
selected_sector = st.sidebar.selectbox("Sector", ["All"] + sorted(df['Sector'].unique()))

# Apply filters
filtered_df = df[df['Rev Growth'] >= min_rev_growth]
filtered_df = filtered_df[filtered_df['Composite Score'] >= min_composite]
if selected_sector != "All":
    filtered_df = filtered_df[filtered_df['Sector'] == selected_sector]

# Display top 10
st.subheader("Top 10 Stocks Overall by Composite Score")
top_10 = filtered_df.sort_values('Composite Score', ascending=False).head(10)
st.dataframe(top_10[['Ticker', 'Sector', 'Price', 'Composite Score', 'DCF Value']])

# Display top by sector
st.subheader("Top Stocks by Sector (Top 5 per Sector)")
grouped = filtered_df.groupby('Sector')
for sector, group in grouped:
    if sector == 'Unknown':
        continue
    st.write(f"**Sector: {sector}**")
    top_in_sector = group.sort_values('Composite Score', ascending=False).head(5)
    st.dataframe(top_in_sector[['Ticker', 'Price', 'Composite Score', 'DCF Value']])

# Detailed stock analysis
st.sidebar.header("Stock Details")
ticker_input = st.sidebar.text_input("Enter Ticker for Details (e.g., ABNB)")
if ticker_input:
    stock_data = filtered_df[filtered_df['Ticker'] == ticker_input.upper()]
    if not stock_data.empty:
        stock = stock_data.iloc[0]
        st.subheader(f"Detailed Analysis for {ticker_input.upper()}")
        st.write(f"Sector: {stock['Sector']}")
        st.write(f"Current Price: ${stock['Price']:.2f}")
        st.write(f"Composite Score: {stock['Composite Score']:.2f}/10")
        st.write(f"DCF Value: ${stock['DCF Value']:.2f}")

        # Display metric breakdown
        st.write("**Metric Breakdown:**")
        metrics_df = pd.DataFrame({
            'Metric': ['Revenue Growth', 'EPS Growth', 'Market Cap', 'Free Cash Flow', 'ROE', 'D/E', 'P/E', 'PEG', 'ROCE', 'Gross Margin', 'Founder Led', 'Pre Revenue'],
            'Value': [f"{stock['Rev Growth']*100:.1f}%", f"{stock['EPS Growth']*100:.1f}%", f"${stock['Market Cap']/1e9:.1f}B", f"${stock['FCF']/1e9:.1f}B", f"{stock['ROE']*100:.1f}%", f"{stock['D/E']:.2f}", f"{stock['P/E']:.2f}", f"{stock['PEG']:.2f}", f"{stock['ROCE']*100:.1f}%", f"{stock['Gross Margin']*100:.1f}%", "Yes" if stock['Founder Led'] == 1 else "No", "Yes" if stock['Pre Revenue'] == 1 else "No"],
            'Score': [stock['Score Rev Growth'], stock['Score EPS Growth'], stock['Score Market Cap'], stock['Score FCF'], stock['Score ROE'], stock['Score D/E'], stock['Score P/E'], stock['Score PEG'], stock['Score ROCE'], stock['Score Gross Margin'], stock['Score Founder Led'], stock['Score Pre Revenue']]
        })
        st.dataframe(metrics_df)

# Export buttons
if st.button("Export Full Data to CSV"):
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "stock_analysis_full.csv", "text/csv")

if st.button("Export Top 10 to CSV"):
    csv = top_10.to_csv(index=False)
    st.download_button("Download CSV", csv, "top_10_stocks.csv", "text/csv")
