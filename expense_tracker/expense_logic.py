import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File name to store expenses
CSV_FILE = "expenses.csv"

# Standard columns for the expense DataFrame
COLUMNS = ["Date", "Category", "Amount", "Description"]

# Set Seaborn style for plots (white background + grid)
sns.set(style="whitegrid")  


def clean_data(df):
    """Clean the expense data:
    - Standardize column names
    - Convert 'Amount' to numeric
    - Convert 'Date' to datetime
    - Fill missing values
    """
    # Standardize column names to lowercase
    df.columns = [col.strip().capitalize() for col in df.columns]

    # Ensure all required columns exist
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # Convert Amount column to numeric (remove currency symbols)
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Fill missing Description or Category with "Unknown"
    df["Description"] = df["Description"].fillna("No description")
    df["Category"] = df["Category"].fillna("Uncategorized")

    return df[COLUMNS]


def load_expenses():
    """Load expenses from CSV. If file doesn't exist, return empty DataFrame."""
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            df = clean_data(df)  # Clean data before returning
            return df
        except Exception as e:
            print(f"[Error loading CSV]: {e}")
            return pd.DataFrame(columns=COLUMNS)
    else:
        return pd.DataFrame(columns=COLUMNS)


def save_expenses(df):
    """Save the DataFrame to the CSV file."""
    df = clean_data(df)  # Ensure data is clean before saving
    df.to_csv(CSV_FILE, index=False)


def clear_expenses():
    """Clear all expenses and delete CSV file if it exists."""
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)
    return pd.DataFrame(columns=COLUMNS)


def add_expense(df, date, category, amount, description):
    """Add a new expense row to the DataFrame and save to CSV."""
    new_row = pd.DataFrame([[date, category, amount, description]], columns=COLUMNS)
    df = pd.concat([df, new_row], ignore_index=True)
    df = clean_data(df)  # Clean after adding new expense
    save_expenses(df)
    return df


def get_category_totals(df):
    """Return a DataFrame with total expenses for each category."""
    if df.empty:
        return pd.DataFrame(columns=["Category", "Amount"])
    return df.groupby("Category")["Amount"].sum().reset_index()


def get_overall_total(df):
    """Return the total amount spent across all expenses."""
    if df.empty:
        return 0.0
    return df["Amount"].sum()


def visualize_expenses(df):
    """Generate a bar chart figure showing expenses by category."""
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    category_data = df.groupby("Category")["Amount"].sum().reset_index()

    sns.barplot(x="Category", y="Amount", data=category_data, ax=ax, palette="viridis")
    ax.set_title("Expenses by Category")
    ax.set_ylabel("Amount (₹)")
    ax.set_xlabel("Category")

    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def chatbot_suggestion(df):
    """Generate a smart suggestion text based on spending habits."""
    if df.empty:
        return "Add some expenses first to get suggestions!"

    total = df["Amount"].sum()
    category_totals = df.groupby("Category")["Amount"].sum()
    top_category = category_totals.idxmax()
    top_amount = category_totals.max()

    suggestion = f"You have spent a total of ₹{total:.2f}.\n"
    suggestion += f"Your highest spending is on {top_category} (₹{top_amount:.2f}).\n"

    if top_category.lower() == "food":
        suggestion += "Try cooking at home more often to save money."
    elif top_category.lower() == "transport":
        suggestion += "Consider using public transport or carpooling."
    elif top_category.lower() == "entertainment":
        suggestion += "Cut down on unnecessary subscriptions or outings."
    elif top_category.lower() == "utilities":
        suggestion += "Look for energy-saving methods to reduce bills."
    else:
        suggestion += "Keep tracking your miscellaneous expenses."

    return suggestion
