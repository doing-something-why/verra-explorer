import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from datetime import datetime

# Set page title and configuration
st.set_page_config(page_title="Carbon Credits Dashboard", layout="wide", initial_sidebar_state="expanded")

# Configure sidebar
st.sidebar.title("🌿 Carbon Credits Dashboard")

# Debug toggle
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

# Function to load and preprocess Verra data
def load_verra_data(file_path=None, uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif file_path is not None and os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        return None
    
    # Add registry column
    df['registry'] = 'Verra'
    
    # Standardize column names for integration
    df.rename(columns={
        'Retirement Beneficiary': 'beneficiary_name',
        'Quantity Issued': 'credit_quantity',
        'Retirement/Cancellation Date': 'retirement_date',
        'ID': 'project_id',
        'Name': 'project_name',
        'Country/Area': 'country',
        'Vintage Start': 'vintage_start',
        'Vintage End': 'vintage_end',
        'Retirement Reason': 'retirement_reason',
        'Retirement Details': 'retirement_details'
    }, inplace=True)
    
    # Convert date columns
    if 'retirement_date' in df.columns:
        df['retirement_date'] = pd.to_datetime(df['retirement_date'], errors='coerce')
    
    # Clean and convert quantity
    if 'credit_quantity' in df.columns:
        df['credit_quantity'] = df['credit_quantity'].astype(str).str.replace(',', '').astype(float)
    
    # Extract year
    if 'retirement_date' in df.columns:
        df['year'] = df['retirement_date'].dt.year
    
    return df

# Function to load and preprocess Gold Standard data
def load_gs_data(file_path=None, uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif file_path is not None and os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        return None
    
    # Add registry column
    df['registry'] = 'Gold Standard'
    
    # Standardize column names for integration
    df.rename(columns={
        'retirement_using_entity.name': 'beneficiary_name',
        'number_of_credits': 'credit_quantity',
        'certified_date': 'retirement_date',  # Using certified_date as proxy for retirement
        'project.id': 'project_id',
        'project.name': 'project_name',
        'project.country': 'country',
        'project.type': 'project_type',
        'project.labels[0]': 'project_label',
        'vintage': 'vintage_year'
    }, inplace=True)
    
    # Convert date columns
    if 'retirement_date' in df.columns:
        df['retirement_date'] = pd.to_datetime(df['retirement_date'], errors='coerce')
    
    # Extract year
    if 'retirement_date' in df.columns:
        df['year'] = df['retirement_date'].dt.year
    
    return df

# Initialize session state for selection persistence
if 'selected_buyer_key' not in st.session_state:
    st.session_state['selected_buyer_key'] = None
if 'selected_year' not in st.session_state:
    st.session_state['selected_year'] = datetime.now().year
if 'selected_registry' not in st.session_state:
    st.session_state['selected_registry'] = 'All'

# Check for Verra data
verra_path = "Verra Data.csv"
verra_exists = os.path.exists(verra_path)

# Check for Gold Standard data
gs_path = "Gold Standard Data.csv"
gs_exists = os.path.exists(gs_path)

# Determine data availability for UI flow
registry_options = []
data_loaded = False
combined_df = pd.DataFrame()

# Data loading section
with st.sidebar.expander("📊 Data Sources", expanded=True):
    st.write("Select data sources to include:")
    
    # Verra data section
    verra_enabled = st.checkbox("Verra Registry", value=verra_exists)
    if verra_enabled:
        if verra_exists:
            st.success("Verra data file found locally")
            verra_df = load_verra_data(file_path=verra_path)
            registry_options.append('Verra')
            if debug_mode and verra_df is not None:
                st.write(f"Verra rows: {len(verra_df)}")
        else:
            verra_upload = st.file_uploader("Upload Verra Data CSV", type="csv")
            if verra_upload is not None:
                verra_df = load_verra_data(uploaded_file=verra_upload)
                registry_options.append('Verra')
                if debug_mode and verra_df is not None:
                    st.write(f"Verra rows: {len(verra_df)}")
            else:
                st.warning("Verra data file not found. Please upload.")
                verra_df = None
    else:
        verra_df = None
    
    # Gold Standard data section
    gs_enabled = st.checkbox("Gold Standard Registry", value=gs_exists)
    if gs_enabled:
        if gs_exists:
            st.success("Gold Standard data file found locally")
            gs_df = load_gs_data(file_path=gs_path)
            registry_options.append('Gold Standard')
            if debug_mode and gs_df is not None:
                st.write(f"Gold Standard rows: {len(gs_df)}")
        else:
            gs_upload = st.file_uploader("Upload Gold Standard Data CSV", type="csv")
            if gs_upload is not None:
                gs_df = load_gs_data(uploaded_file=gs_upload)
                registry_options.append('Gold Standard')
                if debug_mode and gs_df is not None:
                    st.write(f"Gold Standard rows: {len(gs_df)}")
            else:
                st.warning("Gold Standard data file not found. Please upload.")
                gs_df = None
    else:
        gs_df = None

# Combine dataframes if available
dfs_to_combine = []
if verra_df is not None and len(verra_df) > 0:
    dfs_to_combine.append(verra_df)
if gs_df is not None and len(gs_df) > 0:
    dfs_to_combine.append(gs_df)

if len(dfs_to_combine) > 0:
    combined_df = pd.concat(dfs_to_combine, ignore_index=True)
    data_loaded = True
    # Add 'All' option to registry filter
    registry_options = ['All'] + registry_options
else:
    st.error("No data available. Please enable and upload at least one data source.")
    st.stop()

# Sidebar controls
st.sidebar.subheader("🔍 Filters")

# Add registry filter
selected_registry = st.sidebar.selectbox(
    "Registry", 
    options=registry_options,
    index=registry_options.index(st.session_state['selected_registry']) if st.session_state['selected_registry'] in registry_options else 0
)
st.session_state['selected_registry'] = selected_registry

# Filter data by selected registry
if selected_registry != 'All':
    filtered_df = combined_df[combined_df['registry'] == selected_registry]
else:
    filtered_df = combined_df

# Get available years
available_years = sorted(filtered_df['year'].dropna().unique().astype(int).tolist(), reverse=True)
if not available_years:
    available_years = [datetime.now().year]

# Default to most recent year if no years or selection
if st.session_state['selected_year'] not in available_years:
    st.session_state['selected_year'] = available_years[0]

# Add year selector to sidebar
selected_year = st.sidebar.selectbox(
    "Year", 
    options=available_years,
    index=available_years.index(st.session_state['selected_year'])
)
st.session_state['selected_year'] = selected_year

# Threshold slider
credit_threshold = st.sidebar.slider("Minimum Credits Threshold", 
                                  min_value=0, 
                                  max_value=50000, 
                                  value=5000, 
                                  step=1000,
                                  help="Show beneficiaries with at least this many credits")

# Process the data
try:
    # Create two different dataframes:
    # 1. For all beneficiaries across all years (for dropdown)
    # 2. For beneficiaries in the selected year (for display)
    
    # 1. Process all beneficiaries across all years
    all_beneficiaries = filtered_df.groupby('beneficiary_name').agg({
        'credit_quantity': 'sum',
        'registry': 'first'  # Keep track of which registry
    }).reset_index()
    all_beneficiaries = all_beneficiaries[all_beneficiaries['credit_quantity'] > credit_threshold]
    all_beneficiaries['beneficiary_key'] = all_beneficiaries['beneficiary_name'].str.lower().str.strip()
    all_beneficiaries = all_beneficiaries.sort_values('credit_quantity', ascending=False)
    
    # 2. Filter for selected year
    year_df = filtered_df[filtered_df['year'] == selected_year]
    
    # Group by beneficiary for the selected year
    beneficiaries = year_df.groupby(['beneficiary_name', 'registry']).agg({
        'credit_quantity': 'sum',
        'project_id': 'nunique',
        'retirement_date': ['count', 'min', 'max'],
    }).reset_index()
    
    # Rename columns for clarity
    beneficiaries.columns = ['beneficiary_name', 'registry', 'total_volume', 'project_count', 'transaction_count', 'first_retirement', 'last_retirement']
    
    # Calculate average transaction size
    beneficiaries['avg_transaction_size'] = beneficiaries['total_volume'] / beneficiaries['transaction_count']
    
    # Filter by threshold
    beneficiaries = beneficiaries[beneficiaries['total_volume'] > credit_threshold]
    
    # Sort by volume
    beneficiaries = beneficiaries.sort_values('total_volume', ascending=False)
    
    # Normalize beneficiary names
    beneficiaries['beneficiary_key'] = beneficiaries['beneficiary_name'].str.lower().str.strip()
    
    # Create the actual UI
    registry_title = f"({selected_registry})" if selected_registry != 'All' else "(All Registries)"
    st.title(f"🌿 Carbon Credits Dashboard {registry_title}")
    
    # Summary metrics for the selected year
    if not beneficiaries.empty:
        total_credits = beneficiaries['total_volume'].sum()
        total_companies = len(beneficiaries['beneficiary_name'].unique())
        total_projects = beneficiaries['project_count'].sum()
        avg_credits_per_company = total_credits / total_companies if total_companies > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Credits Retired", f"{total_credits:,.0f}")
        with col2:
            st.metric("Companies", f"{total_companies:,}")
        with col3:
            st.metric("Projects", f"{total_projects:,}")
        with col4:
            st.metric("Avg Credits per Company", f"{avg_credits_per_company:,.0f}")
        
        # Display top buyers summary with enhanced metrics
        st.subheader(f"Top Retirement Beneficiaries in {selected_year} (>{credit_threshold:,} credits)")
        
        display_df = beneficiaries[['beneficiary_name', 'registry', 'total_volume', 'project_count', 'transaction_count', 'avg_transaction_size']].copy()
        display_df.columns = ['Retirement Beneficiary', 'Registry', 'Total Volume', 'Project Count', 'Transaction Count', 'Avg Transaction Size']
        display_df['Avg Transaction Size'] = display_df['Avg Transaction Size'].round(0).astype(int)
        display_df['Market Share'] = (display_df['Total Volume'] / total_credits * 100).round(2).astype(str) + '%'
        
        st.dataframe(display_df, use_container_width=True)
        
        # Top 10 visualization with registry color coding
        top_10_df = beneficiaries.head(10).copy()
        top_10_df['beneficiary_short'] = top_10_df['beneficiary_name'].astype(str).str.slice(0, 20)  # Truncate long names
        
        fig = px.bar(
            top_10_df,
            x='beneficiary_short',
            y='total_volume',
            color='registry',  # Color by registry
            title=f'Top 10 Carbon Credit Beneficiaries in {selected_year}',
            labels={'beneficiary_short': 'Beneficiary', 'total_volume': 'Total Credits', 'registry': 'Registry'},
            color_discrete_map={'Verra': '#1f77b4', 'Gold Standard': '#ff7f0e'}  # Custom colors for registries
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Registry comparison if both registries are present
        if 'Verra' in beneficiaries['registry'].values and 'Gold Standard' in beneficiaries['registry'].values:
            st.subheader(f"Registry Comparison in {selected_year}")
            
            registry_summary = beneficiaries.groupby('registry').agg({
                'total_volume': 'sum',
                'beneficiary_name': 'nunique',
                'project_count': 'sum'
            }).reset_index()
            
            registry_summary.columns = ['Registry', 'Total Credits', 'Unique Beneficiaries', 'Total Projects']
            
            # Display summary table
            st.dataframe(registry_summary, use_container_width=True)
            
            # Create comparison pie chart
            fig_registry = px.pie(
                registry_summary,
                values='Total Credits',
                names='Registry',
                title='Credit Volume by Registry',
                color='Registry',
                color_discrete_map={'Verra': '#1f77b4', 'Gold Standard': '#ff7f0e'}
            )
            st.plotly_chart(fig_registry, use_container_width=True)
    else:
        st.info(f"No companies found in {selected_year} with more than {credit_threshold:,} credits. Try selecting a different year or lowering the threshold.")
        
        # Display total beneficiaries across all years
        st.subheader(f"All Beneficiaries (>{credit_threshold:,} credits across all years)")
        all_display_df = all_beneficiaries[['beneficiary_name', 'registry', 'credit_quantity']].head(20).copy()
        all_display_df.columns = ['Retirement Beneficiary', 'Registry', 'Total Volume']
        st.dataframe(all_display_df, use_container_width=True)
    
    # Beneficiary selector for details - using ALL beneficiaries across all years
    beneficiary_options = all_beneficiaries['beneficiary_name'].tolist()
    
    # Create mapping for display and lookup
    beneficiary_display_to_key = dict(zip(all_beneficiaries['beneficiary_name'], all_beneficiaries['beneficiary_key']))
    beneficiary_key_to_display = dict(zip(all_beneficiaries['beneficiary_key'], all_beneficiaries['beneficiary_name']))
    
    # Handle initial state and selection
    if beneficiary_options:
        # Use the first option as default if nothing is selected yet
        if st.session_state['selected_buyer_key'] not in beneficiary_key_to_display.values():
            st.session_state['selected_buyer_key'] = all_beneficiaries['beneficiary_key'].iloc[0]
        
        # Find the display name for the current key
        default_display = next((k for k, v in beneficiary_display_to_key.items() if v == st.session_state['selected_buyer_key']), beneficiary_options[0])
        
        # Create the selectbox - use all companies, not just those in the current year
        selected_display = st.selectbox(
            "Select a beneficiary to explore details (from all years)",
            options=beneficiary_options,
            index=beneficiary_options.index(default_display) if default_display in beneficiary_options else 0
        )
        
        # Update the session state with the normalized key
        selected_key = beneficiary_display_to_key[selected_display]
        st.session_state['selected_buyer_key'] = selected_key
        
        # Get company data - case insensitive match
        company_data = filtered_df[filtered_df['beneficiary_name'].str.lower().str.strip() == selected_key]
        
        # Get yearly stats
        yearly_stats = company_data.groupby(['year', 'registry']).agg({
            'credit_quantity': ['sum', 'mean'],
            'project_id': 'nunique',
            'retirement_date': 'count'
        }).reset_index()
        
        yearly_stats.columns = ['year', 'registry', 'total_volume', 'avg_transaction_size', 'project_count', 'transaction_count']
        yearly_stats = yearly_stats.sort_values(['year', 'registry'])
        
        # Company Profile Section
        st.header(f"Company Profile: {selected_display}")
        
        # Registry distribution for this company
        registry_dist = company_data.groupby('registry').agg({
            'credit_quantity': 'sum'
        }).reset_index()
        
        if len(registry_dist) > 1:  # Only show if multiple registries
            st.subheader("Registry Distribution")
            fig_reg = px.pie(
                registry_dist,
                values='credit_quantity',
                names='registry',
                title=f'Credits by Registry for {selected_display}',
                color='registry',
                color_discrete_map={'Verra': '#1f77b4', 'Gold Standard': '#ff7f0e'}
            )
            st.plotly_chart(fig_reg, use_container_width=True)
        
        # Year-over-year analysis
        if not yearly_stats.empty:
            st.subheader("Year-over-Year Analysis")
            
            # Convert yearly stats to nicer display
            yearly_display = yearly_stats.copy()
            yearly_display.columns = ['Year', 'Registry', 'Total Volume', 'Avg Transaction Size', 'Project Count', 'Transaction Count']
            yearly_display['Avg Transaction Size'] = yearly_display['Avg Transaction Size'].round(0).astype(int)
            
            # Volume trend chart with registry as color
            fig_trend = px.line(
                yearly_stats, 
                x='year', 
                y='total_volume',
                color='registry',
                markers=True,
                title=f'Carbon Credits Volume Trend for {selected_display}',
                labels={'year': 'Year', 'total_volume': 'Total Credits', 'registry': 'Registry'},
                color_discrete_map={'Verra': '#1f77b4', 'Gold Standard': '#ff7f0e'}
            )
            fig_trend.update_layout(xaxis_tickangle=0)
            
            # Yearly stats table
            col1, col2 = st.columns([2, 1])
            with col1:
                st.plotly_chart(fig_trend, use_container_width=True)
            with col2:
                st.dataframe(yearly_display, use_container_width=True)
                
                # Calculate growth metrics if multiple years
                if len(yearly_stats['year'].unique()) > 1:
                    last_year = yearly_stats['year'].max()
                    previous_year = yearly_stats[yearly_stats['year'] < last_year]['year'].max()
                    
                    if not pd.isna(previous_year):
                        current_vol = yearly_stats[yearly_stats['year'] == last_year]['total_volume'].sum()
                        prev_vol = yearly_stats[yearly_stats['year'] == previous_year]['total_volume'].sum()
                        
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
            total_vol = this_year_data['credit_quantity'].sum()
            transactions = len(this_year_data)
            projects = this_year_data['project_id'].nunique()
            countries = this_year_data['country'].nunique() if 'country' in this_year_data.columns else 0
            
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
            project_dist = this_year_data.groupby(['project_id', 'registry']).agg({
                'credit_quantity': 'sum',
                'project_name': 'first'
            }).reset_index()
            
            # Convert project_id to string before concatenation to avoid type error
            project_dist['project_label'] = project_dist['project_id'].astype(str) + ": " + project_dist['project_name'].astype(str).str.slice(0, 30)
            
            # Analysis tabs for different perspectives
            tab1, tab2, tab3 = st.tabs(["Projects", "Timeline", "Countries"])
            
            with tab1:
                fig_pie = px.pie(
                    project_dist, 
                    values='credit_quantity', 
                    names='project_label',
                    title=f'Project Distribution for {selected_display} in {selected_year}',
                    color='registry',  # Color by registry
                    color_discrete_map={'Verra': '#1f77b4', 'Gold Standard': '#ff7f0e'}
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with tab2:
                # Group by month for timeline
                this_year_data['month'] = this_year_data['retirement_date'].dt.month
                monthly_data = this_year_data.groupby(['year', 'month', 'registry']).agg({
                    'credit_quantity': 'sum'
                }).reset_index()
                
                monthly_data['date'] = monthly_data.apply(lambda x: f"{int(x['year'])}-{int(x['month']):02d}", axis=1)
                
                fig_monthly = px.bar(
                    monthly_data,
                    x='date',
                    y='credit_quantity',
                    color='registry',  # Color by registry
                    title=f'Monthly Retirement Timeline for {selected_display} in {selected_year}',
                    labels={'date': 'Month', 'credit_quantity': 'Credits Retired', 'registry': 'Registry'},
                    color_discrete_map={'Verra': '#1f77b4', 'Gold Standard': '#ff7f0e'}
                )
                st.plotly_chart(fig_monthly, use_container_width=True)
            
            with tab3:
                if 'country' in this_year_data.columns:
                    # Country distribution
                    country_dist = this_year_data.groupby(['country', 'registry']).agg({
                        'credit_quantity': 'sum'
                    }).reset_index()
                    
                    fig_country = px.bar(
                        country_dist.sort_values('credit_quantity', ascending=False),
                        x='country',
                        y='credit_quantity',
                        color='registry',  # Color by registry
                        title=f'Carbon Credits by Country for {selected_display} in {selected_year}',
                        labels={'country': 'Country', 'credit_quantity': 'Total Credits', 'registry': 'Registry'},
                        color_discrete_map={'Verra': '#1f77b4', 'Gold Standard': '#ff7f0e'}
                    )
                    st.plotly_chart(fig_country, use_container_width=True)
                else:
                    st.info("Country data not available for this selection")
            
            # Display the detailed transaction table
            st.subheader("Transaction Details")
            
            # Filter by selected year if needed
            if st.checkbox("Show only selected year data", value=True):
                filtered_transactions = company_data[company_data['year'] == selected_year].copy()
            else:
                filtered_transactions = company_data.copy()
            
            # Reorder and format columns for display
            if not filtered_transactions.empty:
                # Determine which columns to display based on what's available
                available_cols = filtered_transactions.columns
                display_cols = ['retirement_date', 'credit_quantity', 'project_id', 'project_name', 'registry']
                
                # Add optional columns if available
                if 'country' in available_cols:
                    display_cols.append('country')
                if 'project_type' in available_cols:
                    display_cols.append('project_type')
                if 'project_label' in available_cols:
                    display_cols.append('project_label')
                if 'vintage_year' in available_cols:
                    display_cols.append('vintage_year')
                if 'vintage_start' in available_cols:
                    display_cols.append('vintage_start')
                if 'vintage_end' in available_cols:
                    display_cols.append('vintage_end')
                if 'retirement_reason' in available_cols:
                    display_cols.append('retirement_reason')
                if 'retirement_details' in available_cols:
                    display_cols.append('retirement_details')
                
                # Create a list of columns that actually exist in the dataframe
                existing_cols = [col for col in display_cols if col in available_cols]
                
                # Renamed columns map
                renamed_cols = {
                    'retirement_date': 'Retirement Date',
                    'credit_quantity': 'Volume',
                    'project_id': 'Project ID',
                    'project_name': 'Project Name',
                    'registry': 'Registry',
                    'country': 'Country',
                    'project_type': 'Project Type',
                    'project_label': 'Project Label',
                    'vintage_year': 'Vintage Year',
                    'vintage_start': 'Vintage Start',
                    'vintage_end': 'Vintage End',
                    'retirement_reason': 'Reason',
                    'retirement_details': 'Details'
                }
                
                detail_display = filtered_transactions[existing_cols].rename(columns=renamed_cols)
                st.dataframe(detail_display, use_container_width=True)
                
                # Show original beneficiary names for verification
                with st.expander("Show original beneficiary names in data"):
                    variations = filtered_transactions['beneficiary_name'].unique()
                    st.write(f"Found {len(variations)} variations of the name '{selected_display}':")
                    st.write(variations)
            else:
                st.info(f"No transactions found for {selected_display} in the selected filters")
                
        else:
            st.info(f"No data found for {selected_display} in {selected_year}")
            
            # Show data from other years
            other_years = company_data['year'].unique()
            if len(other_years) > 0:
                st.write(f"Data is available for the following years: {', '.join(map(str, sorted(other_years)))}")
                st.write("Uncheck 'Show only selected year data' above to see transactions from all years.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    if debug_mode:
        st.exception(e)
    st.info("If you're seeing this error, please check your data format and try again.")