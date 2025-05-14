import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Set page title and configuration
st.set_page_config(page_title="Verra Carbon Credits Dashboard", layout="wide", initial_sidebar_state="expanded")

# Configure sidebar
st.sidebar.title("🌿 Dashboard Controls")

# Initialize session state for selection persistence
if 'selected_buyer_key' not in st.session_state:
    st.session_state['selected_buyer_key'] = None
if 'selected_year' not in st.session_state:
    st.session_state['selected_year'] = datetime.now().year

# Load CSV into DuckDB as a temporary view
csv_path = "Verra Data.csv"

@st.cache_resource
def get_connection():
    return duckdb.connect()

con = get_connection()
con.execute(f"CREATE OR REPLACE TEMP VIEW verra AS SELECT * FROM read_csv_auto('{csv_path}', ignore_errors=True)")

# Get all available years in the data
query_years = '''
SELECT DISTINCT
    EXTRACT(YEAR FROM "Retirement/Cancellation Date") AS year
FROM verra
WHERE "Retirement/Cancellation Date" IS NOT NULL
ORDER BY year DESC
'''
years_df = con.execute(query_years).df()
available_years = years_df['year'].dropna().astype(int).tolist()

# Default to most recent year if no years or selection
if not available_years:
    available_years = [datetime.now().year]
if st.session_state['selected_year'] not in available_years:
    st.session_state['selected_year'] = available_years[0]

# Add year selector to sidebar
selected_year = st.sidebar.selectbox(
    "Select Year", 
    options=available_years,
    index=available_years.index(st.session_state['selected_year'])
)
st.session_state['selected_year'] = selected_year

# Threshold slider
credit_threshold = st.sidebar.slider("Minimum Credits Threshold", 
                                    min_value=1000, 
                                    max_value=50000, 
                                    value=5000, 
                                    step=1000,
                                    help="Show beneficiaries with at least this many credits")

# Get raw beneficiary data for the selected year
query_raw = f'''
SELECT
    "Retirement Beneficiary" AS beneficiary,
    SUM(REPLACE("Quantity Issued", ',', '')::DOUBLE) AS total_volume,
    COUNT(DISTINCT "ID") AS project_count,
    COUNT(*) AS transaction_count,
    MIN("Retirement/Cancellation Date") AS first_retirement,
    MAX("Retirement/Cancellation Date") AS last_retirement,
    AVG(REPLACE("Quantity Issued", ',', '')::DOUBLE) AS avg_transaction_size
FROM verra
WHERE "Retirement Beneficiary" IS NOT NULL
  AND EXTRACT(YEAR FROM "Retirement/Cancellation Date") = {selected_year}
GROUP BY "Retirement Beneficiary"
HAVING total_volume > {credit_threshold}
ORDER BY total_volume DESC
'''
raw_buyers_df = con.execute(query_raw).df()

# Check if we have data for this year
if raw_buyers_df.empty:
    st.warning(f"No data found for year {selected_year} with threshold {credit_threshold}")
    st.stop()

# Normalize beneficiary names in Python for more control
raw_buyers_df['beneficiary_key'] = (
    raw_buyers_df['beneficiary']
    .str.lower()                      # Convert to lowercase
    .str.strip()                      # Remove leading/trailing whitespace 
    .str.replace(r'\s+', ' ', regex=True)  # Normalize internal whitespace
)

# Group by normalized names to combine variations of the same company
buyers_df = raw_buyers_df.groupby('beneficiary_key').agg({
    'beneficiary': 'first',  # Keep one representative name
    'total_volume': 'sum',
    'project_count': 'sum',
    'transaction_count': 'sum',
    'first_retirement': 'min',
    'last_retirement': 'max',
    'avg_transaction_size': 'mean'
}).reset_index()

# Sort by total volume
buyers_df = buyers_df.sort_values('total_volume', ascending=False)

# Create mapping for display and lookup
beneficiary_display_to_key = dict(zip(buyers_df['beneficiary'], buyers_df['beneficiary_key']))
beneficiary_key_to_display = dict(zip(buyers_df['beneficiary_key'], buyers_df['beneficiary']))

# Create the actual UI
st.title(f"🌿 Verra Carbon Credits Dashboard ({selected_year})")

# Summary metrics for the selected year
total_credits = buyers_df['total_volume'].sum()
total_companies = len(buyers_df)
total_projects = buyers_df['project_count'].sum()
avg_credits_per_company = total_credits / total_companies if total_companies > 0 else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Credits Retired", f"{total_credits:,.0f}")
with col2:
    st.metric("Companies (>{:,} credits)".format(credit_threshold), f"{total_companies:,}")
with col3:
    st.metric("Total Projects", f"{total_projects:,}")
with col4:
    st.metric("Avg Credits per Company", f"{avg_credits_per_company:,.0f}")

# Display top buyers summary with enhanced metrics
st.subheader(f"Top Retirement Beneficiaries in {selected_year} (>{credit_threshold:,} credits)")
display_df = buyers_df[['beneficiary', 'total_volume', 'project_count', 'transaction_count', 'avg_transaction_size']].copy()
display_df.columns = ['Retirement Beneficiary', 'Total Volume', 'Project Count', 'Transaction Count', 'Avg Transaction Size']
display_df['Avg Transaction Size'] = display_df['Avg Transaction Size'].round(0).astype(int)
display_df['Market Share'] = (display_df['Total Volume'] / total_credits * 100).round(2).astype(str) + '%'

st.dataframe(display_df, use_container_width=True)

# Top 10 visualization
top_10_df = buyers_df.head(10).copy()
top_10_df['beneficiary'] = top_10_df['beneficiary'].str.slice(0, 20)  # Truncate long names

fig = px.bar(
    top_10_df,
    x='beneficiary',
    y='total_volume',
    title=f'Top 10 Carbon Credit Beneficiaries in {selected_year}',
    labels={'beneficiary': 'Beneficiary', 'total_volume': 'Total Credits'},
    color='total_volume',
    color_continuous_scale='viridis'
)
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

# Beneficiary selector for details
beneficiary_options = buyers_df['beneficiary'].tolist()

# Handle initial state and selection
if beneficiary_options:
    # Use the first option as default if nothing is selected yet
    if st.session_state['selected_buyer_key'] not in beneficiary_key_to_display:
        st.session_state['selected_buyer_key'] = buyers_df['beneficiary_key'].iloc[0]
    
    # Find the display name for the current key
    default_display = beneficiary_key_to_display[st.session_state['selected_buyer_key']]
    
    # Create the selectbox
    selected_display = st.selectbox(
        "Select a beneficiary to explore details",
        options=beneficiary_options,
        index=beneficiary_options.index(default_display)
    )
    
    # Update the session state with the normalized key
    selected_key = beneficiary_display_to_key[selected_display]
    st.session_state['selected_buyer_key'] = selected_key
    
    # Query for yearly stats for the selected company
    query_yearly = f"""
    SELECT
        EXTRACT(YEAR FROM "Retirement/Cancellation Date") AS year,
        SUM(REPLACE("Quantity Issued", ',', '')::DOUBLE) AS total_volume,
        COUNT(DISTINCT "ID") AS project_count,
        COUNT(*) AS transaction_count,
        AVG(REPLACE("Quantity Issued", ',', '')::DOUBLE) AS avg_transaction_size
    FROM verra
    WHERE LOWER(TRIM(REGEXP_REPLACE("Retirement Beneficiary", '\\s+', ' '))) = '{selected_key}'
      AND "Retirement/Cancellation Date" IS NOT NULL
    GROUP BY year
    ORDER BY year
    """
    yearly_stats = con.execute(query_yearly).df()
    
    # Query for details using the normalized key
    query_detail = f"""
    SELECT
        EXTRACT(YEAR FROM "Retirement/Cancellation Date") AS year,
        EXTRACT(MONTH FROM "Retirement/Cancellation Date") AS month,
        "Retirement/Cancellation Date" AS retirement_date,
        "Name" AS project_name,
        "ID" AS project_id,
        "Country/Area" AS country,
        "Vintage Start" AS vintage_start,
        "Vintage End" AS vintage_end,
        REPLACE("Quantity Issued", ',', '')::DOUBLE AS volume,
        "Retirement Reason" AS reason,
        "Retirement Details" AS details,
        "Retirement Beneficiary" AS original_beneficiary
    FROM verra
    WHERE LOWER(TRIM(REGEXP_REPLACE("Retirement Beneficiary", '\\s+', ' '))) = '{selected_key}'
      AND "Retirement/Cancellation Date" IS NOT NULL
    ORDER BY "Retirement/Cancellation Date" DESC, volume DESC
    """
    details_df = con.execute(query_detail).df()
    
    if details_df.empty:
        st.warning(f"No detailed data found for {selected_display}")
    else:
        # Company Profile Section
        st.header(f"Company Profile: {selected_display}")
        
        # Year-over-year analysis
        if not yearly_stats.empty:
            st.subheader("Year-over-Year Analysis")
            
            # Convert yearly stats to nicer display
            yearly_display = yearly_stats.copy()
            yearly_display.columns = ['Year', 'Total Volume', 'Project Count', 'Transaction Count', 'Avg Transaction Size']
            yearly_display['Avg Transaction Size'] = yearly_display['Avg Transaction Size'].round(0).astype(int)
            
            # Volume trend chart
            fig_trend = px.line(
                yearly_stats, 
                x='year', 
                y='total_volume',
                markers=True,
                title=f'Carbon Credits Volume Trend for {selected_display}',
                labels={'year': 'Year', 'total_volume': 'Total Credits'}
            )
            fig_trend.update_layout(xaxis_tickangle=0)
            
            # Yearly stats table
            col1, col2 = st.columns([2, 1])
            with col1:
                st.plotly_chart(fig_trend, use_container_width=True)
            with col2:
                st.dataframe(yearly_display, use_container_width=True, hide_index=True)
                
                # Calculate growth metrics
                if len(yearly_stats) > 1:
                    last_year = yearly_stats['year'].max()
                    previous_year = yearly_stats[yearly_stats['year'] < last_year]['year'].max()
                    
                    if not pd.isna(previous_year):
                        current_vol = yearly_stats[yearly_stats['year'] == last_year]['total_volume'].values[0]
                        prev_vol = yearly_stats[yearly_stats['year'] == previous_year]['total_volume'].values[0]
                        
                        yoy_change = ((current_vol - prev_vol) / prev_vol) * 100
                        yoy_label = f"{yoy_change:+.1f}%" 
                        
                        st.metric(
                            f"YoY Change ({previous_year} to {last_year})",
                            value=yoy_label,
                            delta=f"{current_vol - prev_vol:,.0f} credits"
                        )
        
        # Current year detailed metrics for selected company
        this_year_data = details_df[details_df['year'] == selected_year] if not details_df.empty else pd.DataFrame()
        
        if not this_year_data.empty:
            st.subheader(f"Detailed Metrics for {selected_year}")
            
            # Calculate detailed metrics for the selected year
            total_vol = this_year_data['volume'].sum()
            transactions = len(this_year_data)
            projects = this_year_data['project_id'].nunique()
            countries = this_year_data['country'].nunique()
            first_date = this_year_data['retirement_date'].min()
            last_date = this_year_data['retirement_date'].max()
            
            # Detailed metrics for the year
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Credits", f"{total_vol:,.0f}")
            with col2:
                st.metric("Projects Used", f"{projects}")
            with col3:
                st.metric("Countries", f"{countries}")
            with col4:
                st.metric("Transactions", f"{transactions}")
                
            # Project distribution pie chart
            project_dist = this_year_data.groupby('project_id').agg({
                'volume': 'sum',
                'project_name': 'first'
            }).reset_index()
            # Convert project_id to string before concatenation to avoid type error
            project_dist['project_label'] = project_dist['project_id'].astype(str) + ": " + project_dist['project_name'].str.slice(0, 30)
            
            # Analysis tabs for different perspectives
            tab1, tab2, tab3 = st.tabs(["Projects", "Timeline", "Countries"])
            
            with tab1:
                fig_pie = px.pie(
                    project_dist, 
                    values='volume', 
                    names='project_label',
                    title=f'Project Distribution for {selected_display} in {selected_year}'
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with tab2:
                # Group by month for timeline
                monthly_data = this_year_data.groupby(['year', 'month']).agg({
                    'volume': 'sum'
                }).reset_index()
                monthly_data['date'] = monthly_data.apply(lambda x: f"{int(x['year'])}-{int(x['month']):02d}", axis=1)
                
                fig_monthly = px.bar(
                    monthly_data,
                    x='date',
                    y='volume',
                    title=f'Monthly Retirement Timeline for {selected_display} in {selected_year}',
                    labels={'date': 'Month', 'volume': 'Credits Retired'}
                )
                st.plotly_chart(fig_monthly, use_container_width=True)
            
            with tab3:
                # Country distribution
                country_dist = this_year_data.groupby('country').agg({
                    'volume': 'sum'
                }).reset_index()
                
                fig_country = px.bar(
                    country_dist.sort_values('volume', ascending=False),
                    x='country',
                    y='volume',
                    title=f'Carbon Credits by Country for {selected_display} in {selected_year}',
                    labels={'country': 'Country', 'volume': 'Total Credits'},
                    color='volume',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_country, use_container_width=True)
        
        # Display the detailed transaction table
        st.subheader("Transaction Details")
        
        # Filter by selected year if needed
        if st.checkbox("Show only selected year data", value=True):
            filtered_df = details_df[details_df['year'] == selected_year].copy()
        else:
            filtered_df = details_df.copy()
        
        # Reorder and format columns for display
        if not filtered_df.empty:
            display_cols = [
                'retirement_date', 'volume', 'project_id', 'project_name', 
                'country', 'vintage_start', 'vintage_end', 'reason', 'details'
            ]
            
            renamed_cols = {
                'retirement_date': 'Retirement Date',
                'volume': 'Volume',
                'project_id': 'Project ID',
                'project_name': 'Project Name',
                'country': 'Country',
                'vintage_start': 'Vintage Start',
                'vintage_end': 'Vintage End',
                'reason': 'Reason',
                'details': 'Details'
            }
            
            detail_display = filtered_df[display_cols].rename(columns=renamed_cols)
            st.dataframe(detail_display, use_container_width=True)
            
            # Show original beneficiary names for verification
            with st.expander("Show original beneficiary names in data"):
                variations = filtered_df['original_beneficiary'].unique()
                st.write(f"Found {len(variations)} variations of the name '{selected_display}':")
                st.write(variations)
        else:
            st.info(f"No transactions found for {selected_display} in the selected filters")