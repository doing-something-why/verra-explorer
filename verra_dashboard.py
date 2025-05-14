import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from datetime import datetime

# Set page title and configuration
st.set_page_config(page_title="Verra Carbon Credits Dashboard", layout="wide", initial_sidebar_state="expanded")

# Configure sidebar
st.sidebar.title("🌿 Dashboard Controls")

# Debug toggle
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

# Check if the CSV file exists
csv_path = "Verra Data.csv"
file_exists = os.path.exists(csv_path)

# If file doesn't exist, provide an uploader
if not file_exists:
    st.warning("The Verra Data.csv file is not found in the repository due to size limitations (>100MB).")
    
    uploaded_file = st.file_uploader("Please upload the Verra Data.csv file to use the dashboard", type="csv")
    
    if uploaded_file is None:
        st.info("""
        ### How to get the data file:
        1. Download the Verra Registry data from [Verra Registry](https://registry.verra.org/app/search/VCS)
        2. Save it as 'Verra Data.csv'
        3. Upload it using the file uploader above
        
        Until you upload the data file, the dashboard will display sample data.
        """)
        
        # Create sample data for demonstration
        st.subheader("Sample Dashboard (Upload data to see actual results)")
        sample_image_url = "https://placehold.co/800x400?text=Sample+Verra+Dashboard+Visualization"
        st.image(sample_image_url, caption="Sample dashboard visualization")
        st.stop()
    else:
        # Use the uploaded file
        df = pd.read_csv(uploaded_file)
        if debug_mode:
            st.sidebar.write("DataFrame loaded from uploaded file")
            st.sidebar.write(f"Columns: {df.columns.tolist()}")
else:
    # Use the existing file
    try:
        df = pd.read_csv(csv_path)
        if debug_mode:
            st.sidebar.write("DataFrame loaded from local file")
            st.sidebar.write(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        st.stop()

# Initialize session state for selection persistence
if 'selected_buyer_key' not in st.session_state:
    st.session_state['selected_buyer_key'] = None
if 'selected_year' not in st.session_state:
    st.session_state['selected_year'] = datetime.now().year

# Process the data directly with pandas (avoiding DuckDB issues)
try:
    # Process data to get available years
    if 'Retirement/Cancellation Date' in df.columns:
        # Convert date column to datetime
        df['Retirement/Cancellation Date'] = pd.to_datetime(df['Retirement/Cancellation Date'], errors='coerce')
        
        # Extract year
        df['year'] = df['Retirement/Cancellation Date'].dt.year
        
        # Get available years
        available_years = sorted(df['year'].dropna().unique().tolist(), reverse=True)
    else:
        available_years = [datetime.now().year]
        st.warning("'Retirement/Cancellation Date' column not found in data")
        if debug_mode:
            st.sidebar.write(f"Available columns: {df.columns.tolist()}")
    
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
    
    # Process the data for the selected year
    if 'Quantity Issued' in df.columns and 'Retirement Beneficiary' in df.columns:
        # Clean and convert quantity column
        df['quantity_clean'] = df['Quantity Issued'].str.replace(',', '').astype(float)
        
        # Filter for selected year
        year_df = df[df['year'] == selected_year]
        
        # Group by beneficiary
        beneficiaries = year_df.groupby('Retirement Beneficiary').agg({
            'quantity_clean': 'sum',
            'ID': 'nunique',
            'Retirement/Cancellation Date': ['count', 'min', 'max'],
        }).reset_index()
        
        # Rename columns for clarity
        beneficiaries.columns = ['beneficiary', 'total_volume', 'project_count', 'transaction_count', 'first_retirement', 'last_retirement']
        
        # Calculate average transaction size
        beneficiaries['avg_transaction_size'] = beneficiaries['total_volume'] / beneficiaries['transaction_count']
        
        # Filter by threshold
        beneficiaries = beneficiaries[beneficiaries['total_volume'] > credit_threshold]
        
        # Sort by volume
        beneficiaries = beneficiaries.sort_values('total_volume', ascending=False)
        
        # Normalize beneficiary names
        beneficiaries['beneficiary_key'] = beneficiaries['beneficiary'].str.lower().str.strip()
        
        # Check if we have data
        if beneficiaries.empty:
            st.warning(f"No data found for year {selected_year} with threshold {credit_threshold}")
            st.stop()
        
        # Create the actual UI
        st.title(f"🌿 Verra Carbon Credits Dashboard ({selected_year})")
        
        # Summary metrics for the selected year
        total_credits = beneficiaries['total_volume'].sum()
        total_companies = len(beneficiaries)
        total_projects = beneficiaries['project_count'].sum()
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
        
        display_df = beneficiaries[['beneficiary', 'total_volume', 'project_count', 'transaction_count', 'avg_transaction_size']].copy()
        display_df.columns = ['Retirement Beneficiary', 'Total Volume', 'Project Count', 'Transaction Count', 'Avg Transaction Size']
        display_df['Avg Transaction Size'] = display_df['Avg Transaction Size'].round(0).astype(int)
        display_df['Market Share'] = (display_df['Total Volume'] / total_credits * 100).round(2).astype(str) + '%'
        
        st.dataframe(display_df, use_container_width=True)
        
        # Top 10 visualization
        top_10_df = beneficiaries.head(10).copy()
        top_10_df['beneficiary'] = top_10_df['beneficiary'].astype(str).str.slice(0, 20)  # Truncate long names
        
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
        beneficiary_options = beneficiaries['beneficiary'].tolist()
        
        # Create mapping for display and lookup
        beneficiary_display_to_key = dict(zip(beneficiaries['beneficiary'], beneficiaries['beneficiary_key']))
        beneficiary_key_to_display = dict(zip(beneficiaries['beneficiary_key'], beneficiaries['beneficiary']))
        
        # Handle initial state and selection
        if beneficiary_options:
            # Use the first option as default if nothing is selected yet
            if st.session_state['selected_buyer_key'] not in beneficiary_key_to_display:
                st.session_state['selected_buyer_key'] = beneficiaries['beneficiary_key'].iloc[0]
            
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
            
            # Get company data
            # Filter data for the selected company (case insensitive)
            company_data = df[df['Retirement Beneficiary'].str.lower().str.strip() == selected_key]
            
            # Get yearly stats
            yearly_stats = company_data.groupby('year').agg({
                'quantity_clean': ['sum', 'mean'],
                'ID': 'nunique',
                'Retirement/Cancellation Date': 'count'
            }).reset_index()
            
            yearly_stats.columns = ['year', 'total_volume', 'avg_transaction_size', 'project_count', 'transaction_count']
            yearly_stats = yearly_stats.sort_values('year')
            
            # Company Profile Section
            st.header(f"Company Profile: {selected_display}")
            
            # Year-over-year analysis
            if not yearly_stats.empty:
                st.subheader("Year-over-Year Analysis")
                
                # Convert yearly stats to nicer display
                yearly_display = yearly_stats.copy()
                yearly_display.columns = ['Year', 'Total Volume', 'Avg Transaction Size', 'Project Count', 'Transaction Count']
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
                    st.dataframe(yearly_display, use_container_width=True)
                    
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
            this_year_data = company_data[company_data['year'] == selected_year]
            
            if not this_year_data.empty:
                st.subheader(f"Detailed Metrics for {selected_year}")
                
                # Calculate detailed metrics for the selected year
                total_vol = this_year_data['quantity_clean'].sum()
                transactions = len(this_year_data)
                projects = this_year_data['ID'].nunique()
                countries = this_year_data['Country/Area'].nunique()
                
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
                project_dist = this_year_data.groupby('ID').agg({
                    'quantity_clean': 'sum',
                    'Name': 'first'
                }).reset_index()
                
                # Convert project_id to string before concatenation to avoid type error
                project_dist['project_label'] = project_dist['ID'].astype(str) + ": " + project_dist['Name'].str.slice(0, 30)
                
                # Analysis tabs for different perspectives
                tab1, tab2, tab3 = st.tabs(["Projects", "Timeline", "Countries"])
                
                with tab1:
                    fig_pie = px.pie(
                        project_dist, 
                        values='quantity_clean', 
                        names='project_label',
                        title=f'Project Distribution for {selected_display} in {selected_year}'
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with tab2:
                    # Group by month for timeline
                    this_year_data['month'] = this_year_data['Retirement/Cancellation Date'].dt.month
                    monthly_data = this_year_data.groupby(['year', 'month']).agg({
                        'quantity_clean': 'sum'
                    }).reset_index()
                    
                    monthly_data['date'] = monthly_data.apply(lambda x: f"{int(x['year'])}-{int(x['month']):02d}", axis=1)
                    
                    fig_monthly = px.bar(
                        monthly_data,
                        x='date',
                        y='quantity_clean',
                        title=f'Monthly Retirement Timeline for {selected_display} in {selected_year}',
                        labels={'date': 'Month', 'quantity_clean': 'Credits Retired'}
                    )
                    st.plotly_chart(fig_monthly, use_container_width=True)
                
                with tab3:
                    # Country distribution
                    country_dist = this_year_data.groupby('Country/Area').agg({
                        'quantity_clean': 'sum'
                    }).reset_index()
                    
                    fig_country = px.bar(
                        country_dist.sort_values('quantity_clean', ascending=False),
                        x='Country/Area',
                        y='quantity_clean',
                        title=f'Carbon Credits by Country for {selected_display} in {selected_year}',
                        labels={'Country/Area': 'Country', 'quantity_clean': 'Total Credits'},
                        color='quantity_clean',
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig_country, use_container_width=True)
                
                # Display the detailed transaction table
                st.subheader("Transaction Details")
                
                # Filter by selected year if needed
                if st.checkbox("Show only selected year data", value=True):
                    filtered_df = company_data[company_data['year'] == selected_year].copy()
                else:
                    filtered_df = company_data.copy()
                
                # Reorder and format columns for display
                if not filtered_df.empty:
                    display_cols = [
                        'Retirement/Cancellation Date', 'quantity_clean', 'ID', 'Name', 
                        'Country/Area', 'Vintage Start', 'Vintage End', 'Retirement Reason', 'Retirement Details'
                    ]
                    
                    renamed_cols = {
                        'Retirement/Cancellation Date': 'Retirement Date',
                        'quantity_clean': 'Volume',
                        'ID': 'Project ID',
                        'Name': 'Project Name',
                        'Country/Area': 'Country',
                        'Vintage Start': 'Vintage Start',
                        'Vintage End': 'Vintage End',
                        'Retirement Reason': 'Reason',
                        'Retirement Details': 'Details'
                    }
                    
                    detail_display = filtered_df[display_cols].rename(columns=renamed_cols)
                    st.dataframe(detail_display, use_container_width=True)
                    
                    # Show original beneficiary names for verification
                    with st.expander("Show original beneficiary names in data"):
                        variations = filtered_df['Retirement Beneficiary'].unique()
                        st.write(f"Found {len(variations)} variations of the name '{selected_display}':")
                        st.write(variations)
                else:
                    st.info(f"No transactions found for {selected_display} in the selected filters")
    else:
        st.error("Required columns not found in the data. Please ensure your CSV has 'Quantity Issued' and 'Retirement Beneficiary' columns.")
        if debug_mode:
            st.write("Available columns:", df.columns.tolist())

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    if debug_mode:
        st.exception(e)
    st.info("If you're seeing this error, please make sure you've uploaded the correct Verra Data CSV file and that it contains the expected columns.")