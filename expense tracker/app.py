import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import logging
from fpdf import FPDF
import google.generativeai as genai
import plotly.graph_objects as go
import sqlite3 
import hashlib 

logging.basicConfig(level=logging.INFO)

# Define database file
DB_FILE = "expenses.db"

#  Gemini API Configuration 
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('models/gemini-pro-latest') 
    logging.info("Gemini API configured successfully.")
except Exception as e:
    logging.error(f"Error configuring Gemini API: {e}")
    st.error("Could not configure Gemini API. Please check your .streamlit/secrets.toml file.")
    gemini_model = None


# Application Constants 
MASTER_CATEGORIES = [
    'Food', 'Transport', 'Entertainment', 'Utilities', 'Shopping', 'Rent', 
    'Health', 'Education', 'Subscriptions', 'Household', 'Investment', 
    'Personal Care', 'Gifts & Donations', 'Travel', 'Miscellaneous', 'Other'
]
DF_COLUMNS = ['Date', 'Category', 'Sub-Category', 'Amount', 'Description']
DB_COLUMNS = ['Date', 'Category', 'Sub-Category', 'Amount', 'Description', 'username']


# Database Setup
def init_db():
    """Creates the expenses and users tables if they don't exist."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        
        # Create expenses table with foreign key to users
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                Date TEXT,
                Category TEXT,
                "Sub-Category" TEXT,
                Amount REAL,
                Description TEXT,
                username TEXT,
                FOREIGN KEY (username) REFERENCES users (username)
            )
        """)
        
        # Create users table for authentication
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT
            )
        """)
        conn.commit()

# User Authentication Functions 
def hash_password(password):
    """Returns a SHA-256 hash of the password."""
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(hashed_password, provided_password):
    """Verifies a provided password against a stored hash."""
    return hashed_password == hash_password(provided_password)

def create_user(username, password):
    """Creates a new user in the database."""
    password_hash = hash_password(password)
    with sqlite3.connect(DB_FILE) as conn:
        try:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            # This error occurs if the username is already taken
            return False

def get_user(username):
    """Retrieves a user's hashed password from the database."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        return result  # Will be (password_hash,) or None

# Core Expense Functions (Per-User)
def load_data_from_db(username):
    """Loads a specific user's expenses from the SQLite database."""
    with sqlite3.connect(DB_FILE) as conn:
        # Select only expenses matching the logged-in user
        query = "SELECT Date, Category, \"Sub-Category\", Amount, Description FROM expenses WHERE username = ?"
        df = pd.read_sql_query(query, conn, params=(username,), parse_dates=['Date'])
        
        # Ensure all columns exist even if DB is empty for the user
        for col in DF_COLUMNS:
            if col not in df.columns:
                df[col] = pd.Series(dtype='object')
        df = df[DF_COLUMNS] # Re-order to be safe
        return df

def add_expense(date, category, sub_category, amount, description, username):
    """Inserts a new expense record for a specific user."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO expenses (Date, Category, "Sub-Category", Amount, Description, username)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (date, category, sub_category, amount, description, username))
        conn.commit()
    # Reload data into session state after adding
    st.session_state.expenses = load_data_from_db(username)

def load_expenses(username):
    """Loads a CSV file and replaces the current user's DB contents with it."""
    uploaded_file = st.file_uploader("Choose a CSV file to replace data", type=['csv'])
    if uploaded_file is not None:
        try:
            new_data = pd.read_csv(uploaded_file)
            
            # Ensure standard columns exist in the CSV
            for col in DF_COLUMNS:
                if col not in new_data.columns:
                    new_data[col] = "" 
                    if col == 'Amount': new_data[col] = 0.0
                    if col == 'Date': new_data[col] = pd.Timestamp.now().date()
                    if col == 'Sub-Category': new_data[col] = "N/A"
            
            # Add the current user's username to all rows
            new_data['username'] = username
            
            # Ensure column order matches DB
            new_data = new_data[DB_COLUMNS] 
            
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()
                # 1. Delete all OLD data for this user
                cursor.execute("DELETE FROM expenses WHERE username = ?", (username,))
                
                # 2. Insert all NEW data from the CSV
                new_data.to_sql('expenses', conn, if_exists='append', index=False)
            
            st.success("Data loaded and saved to database successfully!")
            # Reload data into session state
            st.session_state.expenses = load_data_from_db(username)
            st.rerun()
        except Exception as e:
            st.error(f"Error loading file: {e}")

def clear_expenses(username):
    """Deletes all records for a specific user from the expenses table."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM expenses WHERE username = ?", (username,))
        conn.commit()
    st.success("All your expenses cleared from database!")
    # Reload data into session state
    st.session_state.expenses = load_data_from_db(username)

init_db() # Ensure database and tables exist on startup

# Data Cleaning
def clean_data():
    """
    Cleans the expense data loaded in st.session_state.
    Converts types, fills NaNs, and removes duplicates.
    """
    if 'expenses' not in st.session_state or st.session_state.expenses.empty:
        st.session_state.expenses = pd.DataFrame(columns=DF_COLUMNS)
        return st.session_state.expenses

    df = st.session_state.expenses.copy()

    if 'Sub-Category' not in df.columns:
        df['Sub-Category'] = "N/A"
    for col in DF_COLUMNS:
       if col not in df.columns:
               df[col] = 0.0 if col == 'Amount' else "N/A"

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df = df.dropna(subset=['Date', 'Amount'])

    # Cap dates at today
    today = pd.to_datetime('today').normalize()
    future_dates_mask = df['Date'] > today
    if future_dates_mask.any():
        df.loc[future_dates_mask, 'Date'] = today

    # Standardize text columns
    text_cols = ['Category', 'Sub-Category', 'Description']
    for col in text_cols:
        df[col] = df[col].fillna("N/A").astype(str).str.strip()
    
    df['Category'] = df['Category'].str.title()
    df['Sub-Category'] = df['Sub-Category'].str.title()
    
    df['Category'] = df['Category'].replace(["", "N/A"], "Other")
    df['Sub-Category'] = df['Sub-Category'].replace("", "N/A")
    df['Description'] = df['Description'].replace("", "N/A")
    
    df['Amount'] = df['Amount'].fillna(0)
    
    # Remove duplicates based on key fields
    subset_cols = ['Date', 'Amount', 'Description']
    df = df.drop_duplicates(subset=subset_cols, keep='last')

    df = df[DF_COLUMNS]
    st.session_state.expenses = df 
    return st.session_state.expenses


# Visualization Functions
def get_expense_graph():
    """Creates a Matplotlib bar chart for PDF reporting."""
    df = clean_data() 
    if not df.empty and df['Amount'].sum() > 0:
        fig, ax = plt.subplots()
        category_totals = df.groupby('Category')['Amount'].sum().reset_index()
        
        sns.barplot(data=category_totals, x='Category', y='Amount', ax=ax, errorbar=None)
        
        plt.xticks(rotation=45, ha='right')
        plt.title('Total Expenses by Category')
        plt.ylabel('Total Amount (₹)')
        plt.tight_layout()
        return fig
    return None

def get_category_pie_chart(df):
    """Creates an interactive Plotly Pie Chart by Category."""
    if df.empty or df['Amount'].sum() == 0:
        return None
    category_totals = df.groupby('Category')['Amount'].sum().reset_index()
    
    pulls = [0] * len(category_totals)
    if not category_totals.empty:
        max_index = category_totals['Amount'].idxmax()
        pulls[max_index] = 0.05 # Explode the largest slice
        
    fig = go.Figure(data=[go.Pie(
        labels=category_totals['Category'], 
        values=category_totals['Amount'], 
        pull=pulls,
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Amount: ₹%{value:,.2f}<br>Percent: %{percent:.1%}<extra></extra>'
    )])
    fig.update_layout(
        title_text='Expense Distribution by Category',
        legend_title_text='Categories',
        uniformtext_minsize=12, 
        uniformtext_mode='hide'
    )
    return fig

def get_subcategory_bar_chart(df):
    """Creates an interactive Plotly Bar Chart for top 20 Sub-Categories."""
    if df.empty or df['Amount'].sum() == 0:
        return None
    
    sub_cat_totals = df.groupby('Sub-Category')['Amount'].sum().nlargest(20).reset_index()
    sub_cat_totals = sub_cat_totals.sort_values(by='Amount', ascending=False)
    
    fig = go.Figure(data=[go.Bar(
        x=sub_cat_totals['Sub-Category'],
        y=sub_cat_totals['Amount'],
        text=sub_cat_totals['Amount'].apply(lambda x: f'₹{x:,.2f}'),
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Amount: ₹%{y:,.2f}<extra></extra>'
    )])
    fig.update_layout(
        title_text='Top 20 Spending by Sub-Category',
        xaxis_title='Sub-Category',
        yaxis_title='Total Amount (₹)',
        xaxis_tickangle=-45
    )
    return fig

# categorize_expense_rules
def categorize_expense_rules(description):
    """Classifies expense based on keywords in the description."""
    desc_lower = description.lower()
    
    rule_map = {
        'Food': {
            'Groceries': ['grocery', 'supermarket', 'market', 'vegetables', 'fruit', 'milk', 'eggs', 'bread', 'blinkit', 'zepto', 'bigbasket', 'dunzo', 'grofers'],
            'Restaurant': ['restaurant', 'dining out', 'eats', 'diner', 'brunch', 'lunch', 'dinner', 'mcdonalds', 'kfc', 'burger king', 'dominos', 'pizza hut'],
            'Takeaway/Delivery': ['takeaway', 'takeout', 'delivery', 'swiggy', 'zomato', 'pizza', 'order in', 'foodpanda'],
            'Coffee/Cafe': ['coffee', 'starbucks', 'cafe coffee day', 'ccd', 'cafe', 'tea house', 'barista'],
        },
        'Transport': {
            'Fuel': ['fuel', 'petrol', 'diesel', 'gas', 'cng', 'filling station', 'shell', 'hpcl', 'ioc', 'bharat petroleum'],
            'Taxi/Ride-share': ['taxi', 'cab', 'uber', 'ola', 'lyft', 'ride', 'auto', 'rickshaw', 'rapido'],
            'Public Transport': ['bus', 'metro', 'train', 'subway', 'public transport'],
            'Vehicle Maintenance': ['car wash', 'mechanic', 'car repair', 'car service', 'parking', 'toll', 'fastag'],
            'Flights/Travel': ['flight', 'airline', 'airfare', 'indigo', 'air india', 'vistara', 'spicejet', 'makemytrip', 'goibibo'],
        },
        'Travel': {
            'Hotel': ['hotel', 'motel', 'inn', 'marriott', 'hyatt', 'taj', 'oyo', 'airbnb', 'booking.com'],
            'Travel Tickets': ['train ticket', 'bus ticket', 'redbus', 'irctc'],
            'Vacation': ['vacation', 'holiday', 'trip'],
        },
        'Utilities': {
            'Electricity': ['electricity', 'power bill', 'electric bill', 'tata power', 'bses'],
            'Water': ['water bill', 'water tax', 'jal board'],
            'Gas (Cooking)': ['gas bill', 'cylinder', 'lpg', 'cooking gas', 'indane', 'hp gas'],
            'Internet': ['internet', 'wifi', 'wi-fi', 'broadband', 'airtel fiber', 'jiofiber', 'act', 'hathway'],
            'Phone': ['phone bill', 'mobile bill', 'recharge', 'postpaid', 'prepaid', 'airtel', 'jio', 'vi', 'vodafone idea'],
        },
        'Household': {
            'Cleaning Supplies': ['cleaner', 'cleaning', 'detergent', 'soap', 'housekeeping', 'lizol', 'harpic'],
            'Home Repairs': ['home repair', 'plumber', 'electrician', 'carpenter', 'maintenance', 'urban company'],
            'Furniture/Decor': ['decor', 'furniture', 'ikea', 'home center', 'furnishing', 'pepperfry'],
            'Rent': ['rent', 'lease', 'housing'],
        },
        'Shopping': {
            'Clothing': ['clothes', 'shirt', 'pants', 'shoes', 'dress', 'apparel', 'jeans', 'zara', 'h&m', 'myntra', 'ajio', 'lifestyle'],
            'Electronics': ['electronics', 'gadget', 'croma', 'reliance digital', 'earphones', 'charger', 'laptop', 'mobile'],
            'Online Market': ['amazon', 'flipkart', 'ebay', 'snapdeal'],
            'General': ['shopping', 'mall', 'store'],
        },
        'Personal Care': {
            'Cosmetics/Skincare': ['pharmacy', 'cosmetics', 'makeup', 'skincare', 'sephora', 'nykaa', 'body shop'],
            'Salon/Barber': ['salon', 'barber', 'haircut', 'spa', 'manicure', 'pedicure'],
            'Supplies': ['shampoo', 'toothpaste', 'soap', 'personal hygiene'],
        },
        'Health': {
            'Doctor/Consultation': ['doctor', 'clinic', 'consultation', 'hospital', 'physio', 'practo'],
            'Medicine/Pharmacy': ['pharmacy', 'medicine', 'apollo', 'medplus', 'netmeds', '1mg', 'pharmeasy'],
            'Insurance': ['health insurance', 'premium', 'lic', 'policybazaar'],
        },
        'Entertainment': {
            'Movies/Cinema': ['movie', 'cinema', 'pvr', 'inox', 'bookmyshow', 'ticketnew'],
            'Events/Concerts': ['concert', 'show', 'ticket', 'event', 'play'],
            'Hobbies': ['books', 'hobby', 'craft', 'stationary', 'bookstore', 'crossword'],
            'Games': ['game', 'steam', 'video game', 'playstation', 'xbox', 'nintendo'],
            'OTT': ['netflix', 'spotify', 'hotstar', 'prime video', 'disney+', 'youtube premium', 'sonyliv', 'zee5'],
        },
        'Subscriptions': {
            'Software/SaaS': ['saas', 'subscription', 'adobe', 'icloud', 'google one', 'dropbox', 'microsoft 365'],
            'Gym/Fitness': ['gym', 'fitness', 'cult fit', 'gym membership', 'anytime fitness'],
            'News/Media': ['newspaper', 'magazine', 'times of india', 'the hindu'],
        },
        'Education': {
            'Tuition/Fees': ['tuition', 'school fees', 'college fees', 'course fee'],
            'Supplies/Books': ['stationary', 'textbooks', 'notebook', 'books'],
            'Online Courses': ['course', 'udemy', 'coursera', 'edx', 'byjus'],
        },
        'Investment': {
            'Stocks': ['stocks', 'shares', 'equity', 'zerodha', 'groww', 'upstox', 'nse', 'bse'],
            'Mutual Funds': ['mutual fund', 'sip', 'etf', 'nav'],
            'Other': ['investment', 'crypto', 'bitcoin', 'wazirx', 'coindcx'],
        },
        'Gifts & Donations': {
            'Gift': ['gift', 'present', 'birthday gift'],
            'Donation': ['donation', 'charity', 'giveindia', 'unicef'],
        },
        'Miscellaneous': {
            'Payment Gateway': ['razorpay', 'payu', 'ccavenue', 'instamojo'],
            'General Payment': ['upi', 'paytm', 'google pay', 'gpay', 'phonepe', 'bhim', 'apple pay'],
            'Bank/Card': ['card payment', 'debit card', 'credit card', 'bank charge', 'atm withdrawal', 'neft', 'imps', 'rtgs'],
            'Other': ['misc', 'miscellaneous', 'other'],
        }
    }

    # Define the order of checking to catch specific categories first
    category_precedence = [
        'Food', 'Transport', 'Travel', 'Household', 'Shopping', 'Personal Care', 
        'Health', 'Entertainment', 'Subscriptions', 'Education', 'Investment',
        'Gifts & Donations', 'Utilities', 'Rent', 'Miscellaneous'
    ]

    for category in category_precedence:
        if category in rule_map:
            for sub_category, keywords in rule_map[category].items():
                for keyword in keywords:
                    if keyword in desc_lower:
                        return category, sub_category
                        
    return "Other", "N/A"

def get_gemini_suggestion():
    """Generates AI-powered financial suggestions based on user's data."""
    if gemini_model is None:
        return "Gemini API is not configured. Please check your API key."

    df = clean_data() 
    if df.empty:
        return "Add some expenses first to get suggestions!"
        
    if df['Amount'].sum() == 0:
        return "Your total spending is ₹0. Add some expenses with valid amounts to get suggestions."

    # We only send the user's own data for analysis
    data_string = df.to_string(index=False)
    
    prompt = f"""
    You are a friendly and helpful financial advisor. Analyze the following personal expense data (which includes Category and Sub-Category) and provide smart, actionable suggestions.
    Here is the expense data:
    {data_string}
    Based on this data, please provide:
    1.  A brief, encouraging summary of the user's spending habits.
    2.  Identify the top 1-2 spending **Categories** and their total amounts.
    3.  Identify the top 1-2 spending **Sub-Categories** to pinpoint specific habits.
    4.  Provide 3-5 concrete, actionable tips for reducing spending or improving financial habits, focusing on the highest spending areas.
    5.  Keep the tone positive and supportive. Use markdown for formatting (like bullet points).
    """

    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error calling Gemini API: {e}")
        return f"Sorry, I couldn't generate suggestions. Error: {e}"

def get_forecast_graph():
    """Generates a 30-day forecast using a 7-day moving average."""
    df = clean_data() 
    
    if df.empty:
        st.warning("Need at least 1 day of expense data to generate a forecast.")
        return None
        
    daily_expenses = df.groupby('Date')['Amount'].sum().reset_index()
    daily_expenses.rename(columns={'Date': 'ds', 'Amount': 'y'}, inplace=True)
    
    if len(daily_expenses) < 2:
        st.warning("Need expenses on at least 2 different days to generate a forecast.")
        return None

    try:
        # Calculate 7-day moving average
        daily_expenses['MA_7'] = daily_expenses['y'].rolling(window=7, min_periods=1).mean()
        last_known_ma = daily_expenses['MA_7'].iloc[-1]
        
        # Create future dates for forecast
        last_date = daily_expenses['ds'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
        
        future_df = pd.DataFrame({
            'ds': future_dates,
            'forecast': last_known_ma # Simple forecast based on last moving avg
        })

        fig = go.Figure()

        # Actual data
        fig.add_trace(go.Scatter(
            x=daily_expenses['ds'], 
            y=daily_expenses['y'], 
            mode='markers', 
            name='Actual Daily Spending',
            hovertemplate='Date: %{x|%Y-%m-%d}<br>Amount: ₹%{y:,.2f}<extra></extra>'
        ))

        # Trend line
        fig.add_trace(go.Scatter(
            x=daily_expenses['ds'], 
            y=daily_expenses['MA_7'], 
            mode='lines', 
            name='7-Day Avg. Trend',
            line=dict(color='orange'),
            hovertemplate='Date: %{x|%Y-%m-%d}<br>7-Day Avg: ₹%{y:,.2f}<extra></extra>'
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=future_df['ds'], 
            y=future_df['forecast'], 
            mode='lines', 
            name='Forecast (Projected Avg.)',
            line=dict(color='red', dash='dash'),
            hovertemplate='Date: %{x|%Y-%m-%d}<br>Forecast: ₹%{y:,.2f}<extra></extra>'
        ))

        fig.update_layout(
            title="30-Day Spending Forecast (based on 7-Day Moving Average)",
            xaxis_title="Date",
            yaxis_title="Predicted Spending (₹)",
            hovermode="x unified"
        )
        return fig
        
    except Exception as e:
        st.error(f"An error occurred during forecasting: {e}")
        logging.error(f"Moving Average forecasting error: {e}")
        return None

# PDF Report Generation
class PDF(FPDF):
    """Custom PDF class to define header and footer."""
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Personal Finance Report', 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf_report():
    """Generates a PDF report of the user's expenses and suggestions."""
    df = clean_data() 
    if df.empty:
        st.error("Cannot generate report: No expenses found.")
        return None

    pdf = PDF()
    pdf.add_page()
    
    # Add AI Suggestion
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Smart Suggestion', 0, 1, 'L')
    if gemini_model:
        suggestion_text = get_gemini_suggestion()
    else:
        suggestion_text = "Gemini API not configured. Suggestions unavailable."
    # Handle potential encoding issues for PDF
    suggestion_text = suggestion_text.encode('latin-1', 'replace').decode('latin-1')
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 5, suggestion_text)
    pdf.ln(5)

    # Add Expense Chart
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Expense Visualization (By Category)', 0, 1, 'L')
    fig = get_expense_graph() 
    if fig:
        # Save matplotlib fig to a temporary file
        fig_filename = "temp_report_chart.png"
        fig.savefig(fig_filename, format='png', bbox_inches='tight')
        pdf.image(fig_filename, w=190)
        os.remove(fig_filename) # Clean up temp file
    else:
        pdf.set_font('Arial', '', 11)
        pdf.cell(0, 10, "No data to visualize.", 0, 1, 'L')
    pdf.ln(5)

    # Add Detailed Summary Table
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Detailed Expense Summary', 0, 1, 'L')
    
    summary_totals = df.groupby(['Category', 'Sub-Category'])['Amount'].sum().reset_index()

    # Table Header
    pdf.set_font('Arial', 'B', 10)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(70, 8, 'Category', 1, 0, 'C', 1)
    pdf.cell(70, 8, 'Sub-Category', 1, 0, 'C', 1)
    pdf.cell(50, 8, 'Total Amount (Rs)', 1, 1, 'C', 1)
    
    # Table Rows
    pdf.set_font('Arial', '', 10)
    for _, row in summary_totals.iterrows():
        cat = str(row['Category']).encode('latin-1', 'replace').decode('latin-1')
        sub_cat = str(row['Sub-Category']).encode('latin-1', 'replace').decode('latin-1')
        
        pdf.cell(70, 8, cat, 1, 0, 'L')
        pdf.cell(70, 8, sub_cat, 1, 0, 'L')
        pdf.cell(50, 8, f"{row['Amount']:.2f}", 1, 1, 'R')

    # Table Footer (Total)
    overall_total = summary_totals['Amount'].sum()
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(140, 8, 'OVERALL TOTAL', 1, 0, 'C', 1)
    pdf.cell(50, 8, f"{overall_total:.2f}", 1, 1, 'R', 1)

    return pdf.output(dest='S').encode('latin-1')


# Streamlit App UI 
st.set_page_config(layout="wide")
st.title('🤖 Personal Expense Tracker with Spend Analysis')

# Authentication & App Flow 
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

def check_login():
    """Displays a login/signup UI and stops app execution if not authenticated."""
    if not st.session_state.logged_in:
        
        # IMPROVED LOGIN/SIGNUP UI WITH TABS
        st.info("Please Log In or Sign Up to continue.")
        login_tab, signup_tab = st.tabs(["Login", "Sign Up"])
        
        with login_tab:
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")

            if st.button("Login"):
                user_data = get_user(username) # Fetches (password_hash,)
                if user_data and check_password(user_data[0], password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    # Load user's data on successful login
                    st.session_state.expenses = load_data_from_db(username)
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        with signup_tab:
            new_username = st.text_input("New Username", key="signup_user")
            new_password = st.text_input("New Password", type="password", key="signup_pass")

            if st.button("Sign Up"):
                if not new_username or not new_password:
                    st.warning("Please enter both a username and password")
                elif create_user(new_username, new_password):
                    st.success("User created successfully! Please log in.")
                else:
                    st.error("Username already exists. Please choose another one.")
        
        # Stop execution to prevent the rest of the app from running
        st.stop()

# Run the login check at the start of every app run
check_login()

# Main App (Only runs if logged in)

# Sidebar
with st.sidebar:
    
    # User logout
    st.header(f"Welcome, {st.session_state.username}!")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        # Clear all session state items on logout
        keys_to_del = ['expenses', 'staged_expense', 'staged_category', 'staged_sub_category']
        for key in keys_to_del:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    st.divider()

    # Add Expense Form 
    st.header('Add Expense')
    
    if 'staged_expense' not in st.session_state:
        # Step 1: Input data
        date = st.date_input('Date', value=pd.Timestamp.now().date())
        amount = st.number_input('Amount', min_value=0.0, format="%.2f")
        description = st.text_input('Description (required for categorization)')

        if st.button('Categorize Expense'):
            if amount > 0 and description:
                category, sub_category = categorize_expense_rules(description)
                
                # Stage the expense for confirmation
                st.session_state.staged_expense = {
                    "date": date,
                    "amount": amount,
                    "description": description
                }
                st.session_state.staged_category = category
                st.session_state.staged_sub_category = sub_category
                st.rerun()
            elif not description:
                st.warning("Please enter a description to categorize.")
            else:
                st.warning("Please enter an amount greater than 0.")
    else:
        # Step 2: Confirm or Edit Category
        st.subheader("Confirm or Edit Category")
        
        staged = st.session_state.staged_expense
        st.write(f"**Description:** {staged['description']}")
        st.write(f"**Amount:** ₹{staged['amount']:.2f}")
        
        try:
            default_index = MASTER_CATEGORIES.index(st.session_state.staged_category)
        except ValueError:
            default_index = MASTER_CATEGORIES.index("Other") # Default to 'Other' if rule fails

        user_category = st.selectbox(
            "Category",
            options=MASTER_CATEGORIES,
            index=default_index
        )
        
        user_sub_category = st.text_input(
            "Sub-Category",
            value=st.session_state.staged_sub_category
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Confirm and Add", type="primary"):
                # Pass username to add_expense
                add_expense(
                    staged['date'], 
                    user_category, 
                    user_sub_category.title(),
                    staged['amount'], 
                    staged['description'],
                    st.session_state.username  # Pass the logged-in user
                )
                st.success(f'Expense added! Category: "{user_category}"')
                # Clear the staged expense
                del st.session_state.staged_expense
                del st.session_state.staged_category
                del st.session_state.staged_sub_category
                st.rerun()

        with col2:
            if st.button("Cancel"):
                # Clear the staged expense
                del st.session_state.staged_expense
                del st.session_state.staged_category
                del st.session_state.staged_sub_category
                st.rerun()

    st.header('File Operations')
    
    # Pass username to clear_expenses
    if st.button("Clear All My Expenses"):
        clear_expenses(st.session_state.username)
        if 'staged_expense' in st.session_state:
            del st.session_state.staged_expense
            del st.session_state.staged_category
            del st.session_state.staged_sub_category
        st.rerun()
    
    # Pass username to load_expenses
    load_expenses(st.session_state.username)

# Main page layout
col1, col2 = st.columns([0.6, 0.4])

# Load and clean data for the logged-in user
cleaned_df = clean_data()


with col1:
    st.header('Recent Expenses')
    st.dataframe(cleaned_df.sort_values(by='Date', ascending=False), height=300)

    st.header('Expense Visualization')
    
    tab1, tab2 = st.tabs(["Overall Overview (Pie)", "Detailed Breakdown (Bar)"])

    with tab1:
        st.subheader("Expense Distribution by Category")
        pie_fig = get_category_pie_chart(cleaned_df)
        if pie_fig:
            st.plotly_chart(pie_fig, use_container_width=True)
        else:
            st.warning("No expenses with valid amounts to visualize!")
    
    with tab2:
        st.subheader("Top Spending by Sub-Category")
        sub_bar_fig = get_subcategory_bar_chart(cleaned_df)
        if sub_bar_fig:
            st.plotly_chart(sub_bar_fig, use_container_width=True)
        else:
            st.warning("No expenses with valid amounts to visualize!")

    if not cleaned_df.empty:
        st.subheader("Spending Totals")
        
        overall_total = cleaned_df['Amount'].sum()
        st.metric(label="Total Spending", value=f"₹{overall_total:.2f}")

        st.write("**Totals by Category:**")
        category_totals = cleaned_df.groupby('Category')['Amount'].sum().reset_index()
        st.table(category_totals)
        
        st.write("**Totals by Sub-Category:**")
        sub_category_totals = cleaned_df.groupby(['Category', 'Sub-Category'])['Amount'].sum().reset_index()
        st.table(sub_category_totals.sort_values(by='Amount', ascending=False))


with col2:
    st.header('Gemini Smart Suggestions')
    if st.button('Get AI Suggestions'):
        with st.spinner("🤖 Asking Gemini for advice..."):
            suggestion = get_gemini_suggestion()
            st.markdown(suggestion)

    st.header('Future Trend Prediction')
    if st.button('Generate 30-Day Forecast'):
        with st.spinner("🔮 Running statistical forecast..."):
            forecast_fig = get_forecast_graph()
            if forecast_fig:
                st.plotly_chart(forecast_fig, use_container_width=True)

    st.header('Download Report')
    st.write("Generate a full PDF report with suggestions, summary, and visualization.")
    
    if st.button("Generate PDF Report"):
        pdf_data = create_pdf_report()
        if pdf_data:
            st.download_button(
                label="Click to Download PDF",
                data=pdf_data,
                file_name="expense_report.pdf",
                mime="application/pdf"
            )