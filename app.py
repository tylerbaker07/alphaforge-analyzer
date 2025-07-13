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
tickers_df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers = tickers_df['Symbol'].tolist()[:100]  # Limit to 100 for performance; adjust as needed

# Collect data
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

# Create DataFrame
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

# Function to display detailed stock analysis
def show_stock_details(ticker):
    stock_data = df[df['Ticker'] == ticker.upper()]
    if stock_data.empty:
        print(f"Stock {ticker} not found.")
        return
    
    stock = stock_data.iloc[0]
    print(f"\n{'='*50}")
    print(f"DETAILED ANALYSIS FOR {ticker.upper()}")
    print(f"{'='*50}")
    print(f"Sector: {stock['Sector']}")
    print(f"Current Price: ${stock['Price']:.2f}")
    print(f"Composite Score: {stock['Composite Score']:.2f}/10")
    print(f"\n{'CATEGORY BREAKDOWN:'}")
    print(f"{'='*30}")
    
    # Show each metric and its score
    metrics = [
        ('Revenue Growth', 'Rev Growth', 'Score Rev Growth'),
        ('EPS Growth', 'EPS Growth', 'Score EPS Growth'),
        ('Market Cap (lower=better)', 'Market Cap', 'Score Market Cap'),
        ('Free Cash Flow', 'FCF', 'Score FCF'),
        ('Return on Equity', 'ROE', 'Score ROE'),
        ('Debt/Equity (lower=better)', 'D/E', 'Score D/E'),
        ('P/E Ratio (lower=better)', 'P/E', 'Score P/E'),
        ('PEG Ratio (lower=better)', 'PEG', 'Score PEG'),
        ('ROCE', 'ROCE', 'Score ROCE'),
        ('Gross Margin', 'Gross Margin', 'Score Gross Margin'),
        ('Founder Led', 'Founder Led', 'Score Founder Led'),
        ('Pre Revenue', 'Pre Revenue', 'Score Pre Revenue')
    ]
    
    for display_name, raw_col, score_col in metrics:
        raw_value = stock[raw_col]
        score = stock[score_col]
        
        # Format the raw value appropriately
        if raw_col == 'Market Cap':
            formatted_value = f"${raw_value/1e9:.1f}B"
        elif raw_col == 'FCF':
            formatted_value = f"${raw_value/1e9:.1f}B" if raw_value != 0 else "$0"
        elif raw_col in ['Rev Growth', 'EPS Growth', 'ROE', 'ROCE', 'Gross Margin']:
            formatted_value = f"{raw_value*100:.1f}%" if raw_value != 0 else "0%"
        elif raw_col in ['Founder Led', 'Pre Revenue']:
            formatted_value = "Yes" if raw_value == 1 else "No"
        else:
            formatted_value = f"{raw_value:.2f}"
        
        print(f"{display_name:<25}: {formatted_value:<15} Score: {score:.1f}/10")

# Main screen: Top 10 overall by composite score
print("Top 10 Stocks Overall by Composite Score:")
top_10 = df.sort_values('Composite Score', ascending=False).head(10)
print(top_10[['Ticker', 'Sector', 'Price', 'Composite Score']])

# Grouped lists: Top 5 per sector by composite score
print("\nTop Stocks by Sector (Top 5 per Sector by Composite Score):")
grouped = df.groupby('Sector')
for sector, group in grouped:
    if sector == 'Unknown':
        continue
    top_in_sector = group.sort_values('Composite Score', ascending=False).head(5)
    print(f"\nSector: {sector}")
    print(top_in_sector[['Ticker', 'Price', 'Composite Score']])

# Interactive filtering and stock analysis
while True:
    print("\nOptions:")
    print("1. Enter a filter criterion (e.g., 'Rev Growth > 0.1' or 'Sector == \"Technology\"')")
    print("2. Enter a stock ticker to see detailed analysis (e.g., 'ABNB')")
    print("3. Type 'exit' to quit")
    print("\nEnter your choice:")
    user_input = input().strip()
    
    if user_input.lower() == 'exit':
        break
    
    # Check if it's a stock ticker (simple heuristic: 1-5 uppercase letters)
    if len(user_input) <= 5 and user_input.replace('.', '').isalpha():
        show_stock_details(user_input)
    else:
        # Treat as filter
        try:
            filtered_df = df.query(user_input)
            print("Filtered Results:")
            print(filtered_df[['Ticker', 'Sector', 'Price', 'Composite Score']])
        except Exception as e:
            print(f"Invalid filter: {e}")
