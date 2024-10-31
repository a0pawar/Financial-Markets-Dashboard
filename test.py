import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Equity Markets Performance")


# Helper functions
def data_loader_bloomberg(filename, sheet_name):
    """
    Loads and processes Bloomberg data from an Excel file.
    """
    if sheet_name == 2:
        data = pd.read_excel(filename, sheet_name=sheet_name, skiprows=2).iloc[4:, :]
    else:
        data = pd.read_excel(filename, sheet_name=sheet_name, skiprows=2).iloc[5:, :]
    data.index = pd.to_datetime(data['Unnamed: 0'], format="%d-%m-%Y")
    data = data.drop(['Unnamed: 0'], axis=1)
    return data


@st.cache_data
def load_data():
    """
    Loads data from Excel file.
    """
    sectoral_data = data_loader_bloomberg(filename='Consolidated_Fin_market Data.xlsx', sheet_name=0)
    benchmark_data = data_loader_bloomberg(filename='Consolidated_Fin_market Data.xlsx', sheet_name=1)
    global_data = data_loader_bloomberg(filename='Consolidated_Fin_market Data.xlsx', sheet_name=2)
    valuation_data = data_loader_bloomberg(filename='Consolidated_Fin_market Data.xlsx', sheet_name=3)
    flows_data = data_loader_bloomberg(filename='Consolidated_Fin_market Data.xlsx', sheet_name=4)
    macro_data = data_loader_bloomberg(filename='Consolidated_Fin_market Data.xlsx', sheet_name=5)
    impact_data = data_loader_bloomberg(filename='Consolidated_Fin_market Data.xlsx', sheet_name=7) 
    return sectoral_data, benchmark_data, global_data, valuation_data, flows_data, macro_data, impact_data

def calculate_returns(data):
    """
    Calculate returns over the entire period of the data.
    """
    return ((data.iloc[-1] - data.iloc[0]) / data.iloc[0]) * 100

def normalize(df):
    """
    Generated normalized values.
    """
    return 100 * df / df.iloc[0, :]

def replace_repeated_tail_values(df, look_back=7):
    """
    Replace repeated values at the end of each column in a dataframe with zeros,
    looking back only at the last 'look_back' number of values.
    
    Args:
    df (pd.DataFrame): Input dataframe with DatetimeIndex
    look_back (int): Number of values to look back (default is 7)
    
    Returns:
    pd.DataFrame: Dataframe with repeated tail values replaced by zeros
    """
    # Ensure we have a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    def replace_repeated(series):
        values = series.values
        for i in range(-1, -look_back-1, -1):
            if values[i] == values[i-1]:
                values[i] = 0
            else:
                break
        return pd.Series(values, index=series.index)

    return df.apply(replace_repeated)
  
def change(series, current_index, previous_index, is_percentage=True):
    """
    Calculate the change between two values in a series.
    
    Args:
    series (pd.Series): The input time series data
    current_index (int): Index for the current value (default is -1, the last value)
    previous_index (int): Index for the previous value (default is -13, for year-over-year change in monthly data)
    is_percentage (bool): If True, calculate percentage change; if False, calculate absolute change
    
    Returns:
    float: The calculated change, rounded to 2 decimal places
    """
    current_value = series.iloc[current_index]
    previous_value = series.iloc[previous_index]
    
    if is_percentage:
        change = (current_value / previous_value - 1) * 100
    else:
        change = current_value - previous_value
    
    return round(change, 2)
    
def custom_metric(label, current_value, previous_value, show_delta=True, delta_color="gray"):
    delta = float(current_value) - float(previous_value)
    delta_html = f"""
    <p class="delta" style="color: {delta_color};">
        {'+' if delta > 0 else ''}{delta:.2f}
    </p>
    """ if show_delta else ""

    html = f"""
    <div class="custom-metric">
        <p class="label">{label}</p>
        <p class="current-value">{current_value}</p>
        <p class="previous-value">Previous: {previous_value}</p>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def calculate_impact(data, start_date, end_date):
    """
    Calculate the change in values between two dates for all columns in the dataframe.
    
    Args:
    data (pd.DataFrame): The input dataframe
    start_date (str or datetime): The start date
    end_date (str or datetime): The end date
    
    Returns:
    pd.DataFrame: The calculated changes
    """
    # Convert input dates to pandas Timestamp objects
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    

    # Check if dates are in the index
    if start_date not in data.index:
        st.error(f"Start date {start_date} not found in data. Available dates: {data.index[0]} to {data.index[-1]}")
        return None
    if end_date not in data.index:
        st.error(f"End date {end_date} not found in data. Available dates: {data.index[0]} to {data.index[-1]}")
        return None
    
    start_value = data.loc[start_date]
    end_value = data.loc[end_date]
    
    # Define lists for different change types
    equity_markets = ['Sensex', 'MidCap', 'SmallCap', 'NiftyBank', 'India VIX', 'S&P500', 'Nikkei225', 'Hang Seng', 'EuroStoxx50']
    bond_markets = ['10Y Government Bond', '3M Tbill', '3M CD', '5Y AAA Corporate Bond', '5Y BBB Corporate Bond', '5Y Government Bond', 'US 10Yr Government Bond', '1Y OIS rate']
    commodities = ['Brent 1M Future']
    
    changes = pd.DataFrame(index=data.columns)
    
    for col in data.columns:
        if col in equity_markets or col in commodities or col in ['INR Spot', 'Dollar Index']:
            changes.loc[col, 'Change'] = np.round(((end_value[col] - start_value[col]) / start_value[col] * 100), 2)
            changes.loc[col, 'Unit'] = '%'
        elif col in bond_markets:
            changes.loc[col, 'Change'] = np.round(((end_value[col] - start_value[col]) * 100), 2)  # Convert to basis points
            changes.loc[col, 'Unit'] = 'bps'
    
    # Calculate credit spread and default spread
    credit_spread_start = start_value['5Y AAA Corporate Bond'] - start_value['5Y Government Bond']
    credit_spread_end = end_value['5Y AAA Corporate Bond'] - end_value['5Y Government Bond']
    credit_spread_change = (credit_spread_end - credit_spread_start) * 100  # Convert to basis points
    
    default_spread_start = start_value['5Y BBB Corporate Bond'] - start_value['5Y AAA Corporate Bond']
    default_spread_end = end_value['5Y BBB Corporate Bond'] - end_value['5Y AAA Corporate Bond']
    default_spread_change = (default_spread_end - default_spread_start) * 100  # Convert to basis points
    
    cd_tbill_spread_start = start_value['3M CD'] - start_value['3M Tbill']
    cd_tbill_spread_end = end_value['3M CD'] - end_value['3M Tbill']
    cd_tbill_spread_change = (cd_tbill_spread_end - cd_tbill_spread_start) * 100  # Convert to basis points
    
    # Add credit spread and default spread to the changes dataframe
    changes.loc['Credit Spread (AAA - Govt)', 'Change'] = np.round(credit_spread_change,2)
    changes.loc['Credit Spread (AAA - Govt)', 'Unit'] = 'bps'
    changes.loc['Default Spread (BBB - AAA)', 'Change'] = np.round(default_spread_change,2)
    changes.loc['Default Spread (BBB - AAA)', 'Unit'] = 'bps'
    changes.loc['CD-Tbill Spread', 'Change'] = np.round(cd_tbill_spread_change,2)
    changes.loc['CD-Tbill Spread', 'Unit'] = 'bps'
    
    return changes

# CSS for the custom metric
st.markdown("""
<style>
.custom-metric {
    border: 1px solid #4b5563;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
    background-color: #1f2937;
}
.custom-metric .label {
    font-size: 1em;
    color: #9ca3af;
    margin: 0;
}
.custom-metric .current-value {
    font-size: 1.5em;
    font-weight: bold;
    color: #f3f4f6;
    margin: 5px 0;
}
.custom-metric .previous-value {
    font-size: 1em;
    color: #9ca3af;
    margin: 0;
}
.custom-metric .delta {
    font-size: 0.9em;
    font-weight: bold;
    margin: 5px 0 0 0;
}
</style>
""", unsafe_allow_html=True)

# Main app
st.markdown("<h1 style='text-align: center; font-size: 36px;'>Markets Dashboard</h1>", unsafe_allow_html=True)
st.divider()

# Load data
sectoral_data, benchmark_data, global_data, valuation_data, flows_data, macro_data, impact_data = load_data()



## Macro-Section
st.markdown("<h3 style='text-align: center; font-size: 24px;'>Macro Snapshot</h3>", unsafe_allow_html=True)


col1, col2, col3, col4 = st.columns(4)
with col1:
    custom_metric(f"IIP growth(YoY %) for {macro_data['IIP Index'].index[-2].strftime('%b-%Y')}", 
            f"{change(macro_data['IIP Index'], -2,-14)}", 
            f"{change(macro_data['IIP Index'], -3,-15)}", 
            show_delta=False)
with col2:    
    custom_metric(f"CPI Inflation(YoY %) for {macro_data['CPI Combined Index'].index[-1].strftime('%b-%Y')}", 
            f"{change(macro_data['CPI Combined Index'], -1,-13)}", 
            f"{change(macro_data['CPI Combined Index'], -2,-14)}", 
            show_delta=False)
with col3:    
    custom_metric(f"Policy Repo Rate(%) for {macro_data['Policy Repo Rate'].index[-1].strftime('%b-%Y')}", 
            f"{macro_data['Policy Repo Rate'][-1]}", f"{macro_data['Policy Repo Rate'][-2]}", show_delta=False)
with col4:
    custom_metric(f"HSBC PMI Composite(%) for {macro_data['HSBC India PMI Composite'].index[-1].strftime('%b-%Y')}", 
            f"{macro_data['HSBC India PMI Composite'][-1]}", f"{macro_data['HSBC India PMI Composite'][-2]}", show_delta=False)

tab1, tab2 = st.tabs(['Equity Markets', 'Market Impact Analysis'])
with tab1:
    st.markdown("<h3 style='text-align: center; font-size: 24px;'>Equity Markets</h3>", unsafe_allow_html=True)

    # Date range selector
    col1, col2 = st.columns(2)
    default_end_date = benchmark_data.index.max()
    default_start_date = default_end_date - pd.DateOffset(months=3)

    with col1:
        start_date = st.date_input("Start Date", value=default_start_date, min_value=benchmark_data.index.min(), max_value=benchmark_data.index.max())

    with col2:
        end_date = st.date_input("End Date", value=default_end_date, min_value=benchmark_data.index.min(), max_value=benchmark_data.index.max())

    # Filter data and calculate returns
    mask = (benchmark_data.index >= pd.Timestamp(start_date)) & (benchmark_data.index <= pd.Timestamp(end_date))
    benchmark_data_filtered = benchmark_data.loc[mask].iloc[:-1,:]
    sectoral_data_filtered = sectoral_data.loc[mask].iloc[:-1,:]
    global_data_filtered = global_data.loc[(global_data.index >= pd.Timestamp(start_date)) & (global_data.index <= pd.Timestamp(end_date))].iloc[:-1,:]
    valuation_data_filtered = valuation_data.loc[(valuation_data.index >= pd.Timestamp(start_date)) & (valuation_data.index <= pd.Timestamp(end_date))].iloc[:-1,:]

    benchmark_returns = calculate_returns(benchmark_data_filtered)
    sectoral_returns = calculate_returns(sectoral_data_filtered)
    global_returns = calculate_returns(global_data_filtered)


    def create_line_plot(data, title):
        # Check if data is a Series or a DataFrame
        data = data.reset_index(names='index')  # Reset index for DataFrame with name 'index'
        x_col = 'index'
        y_col = data.columns
        # Create line plot using Plotly Express
        fig = px.line(
            data_frame=data,
            x=x_col, y=y_col,
            labels={'index': 'Date', 'value': 'Index'},
            title=title
        )
        # Update layout settings
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="auto", y=-0.5, xanchor="auto", x=0.5),
            legend_title=None,
            yaxis=dict(showgrid=False),
            font=dict(size=12)
        )

        return fig

    # Create column chart with Plotly Express
    def create_bar_plot(data, title):
        sorted_data = data.sort_values(ascending=False)
        max_value = 2*sorted_data.max()
        min_value = 2*sorted_data.min() if sorted_data.min() < 0 else 0

        fig = px.bar(
            data_frame=sorted_data.reset_index(),
            x=sorted_data.index, y=sorted_data.values,
            text=[f'{x:.2f}%' for x in sorted_data.values],
            labels={'x': 'Indices', 'y': 'Returns (%)'},
            title=title
        )
        fig.update_traces(textposition='outside', textfont=dict(size=10))
        fig.update_layout(height=400, font=dict(size=12), 
                        margin=dict(l=0, r=0, t=30, b=0), 
                        yaxis=dict(range=[min_value, max_value], showgrid=False), 
                        xaxis=dict(tickangle=-90),
                        legend_title=None)
        return fig

    # Display plots in a compact layout
    #st.markdown("<h2 style='text-align: center; font-size: 20px;'>Domestic Equity Indices</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(create_line_plot(normalize(benchmark_data_filtered), "Benchmark Indices"), use_container_width=True)
    with col2:
        st.plotly_chart(create_bar_plot(benchmark_returns, f"Benchmark Returns"), use_container_width=True)
    with col3:
        st.plotly_chart(create_bar_plot(sectoral_returns, f"Sectoral Returns"), use_container_width=True)


    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<h3 style='text-align: justified; font-size: 16px;'>Valuation: Price-to-Earnings Ratio</h3>", unsafe_allow_html=True)
        
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            st.metric("Sensex", f"{np.round(valuation_data_filtered['Sensex'][-1],2)}")
        with subcol2:
            st.metric("Sensex (Long-term Average)", f"{np.round(valuation_data_filtered['Sensex'].mean(),2)}")
        
        subcol3, subcol4 = st.columns(2)
        with subcol3:
            st.metric("MSCI-EM", f"{np.round(valuation_data_filtered['MSCI Emerging Markets'][-1],2)}")
        with subcol4:
            st.metric("MSCI-World", f"{np.round(valuation_data_filtered['MSCI World'][-1],2)}")

    with col2:
        st.markdown("<h3 style='text-align: center; font-size: 16px;'>Net Monthly Flows to Indian Equity Markets</h3>", unsafe_allow_html=True)

        flows_data = replace_repeated_tail_values(flows_data)
        flows_monthly = flows_data.resample('M').sum().round(2)
        flows_monthly = flows_monthly.reset_index(names='Date')
        flows_monthly = flows_monthly.iloc[-36:,:]

        # Convert columns to numeric, excluding the 'Date' column
        for col in flows_monthly.columns[1:]:
            flows_monthly[col] = pd.to_numeric(flows_monthly[col], errors='coerce')

        # Create bar plot for flows data
        fig_flows = px.bar(
            flows_monthly, 
            x='Date', 
            y=flows_monthly.columns[1:], 
            barmode='group',
            labels={col: col for col in flows_monthly.columns[1:]},
        )

        fig_flows.update_layout(
            height=300,
            font=dict(size=12), 
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis_title="₹ crore",
            legend_title=None,
            yaxis=dict(showgrid=False)
        )

        # Display the chart
        st.plotly_chart(fig_flows, use_container_width=True)

    with col3:
        st.plotly_chart(create_bar_plot(global_returns, f"Global Returns"), use_container_width=True)
        
    st.divider()
    
    # Data Display and Download section
    st.markdown("<h2 style='text-align: left; font-size: 18px;'>Data Download</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        data_option = st.selectbox("Select data to display and download",
                                ("Benchmark Indices", "Sectoral Indices", "Global Indices", "Valuation Data", "Flows Data", "Macro data"))
        
        # Add a checkbox to control dataframe display
        show_dataframe = st.checkbox("Show Data Table", value=False)

        if data_option == "Benchmark Indices":
            display_data = benchmark_data_filtered
        elif data_option == "Sectoral Indices":
            display_data = sectoral_data_filtered
        elif data_option == "Global Indices":
            display_data = global_data_filtered
        elif data_option == "Valuation Data":
            display_data = valuation_data_filtered
        elif data_option == "Flows Data":
            display_data = flows_data
        elif data_option == 'Macro data':
            display_data = macro_data.iloc[:,:3]
        
        display_data.index.names = ['Date']
        try:
            display_data.index = display_data.index.strftime('%Y-%m-%d')
        except:
            pass

        # Only display the dataframe if the checkbox is checked
        if show_dataframe:
            st.dataframe(display_data.round(2), height=300)


with tab2:
# Add this section after your existing sections in the main app
    st.markdown("<h3 style='text-align: center; font-size: 24px;'>Market Impact Analysis</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; font-size: 20px;'>This section can help analyze the impact of a global event/news/announcement on select domestic as well as global market segments.</h3>", unsafe_allow_html=True)
    
    # Date selection for market impact analysis
    col1, col2 = st.columns(2)
    with col1:
        impact_start_date = st.date_input("Select start date for impact analysis",
                                        value=None,
                                        min_value=impact_data.index.min(),
                                        max_value=impact_data.index.max())
    with col2:
        impact_end_date = st.date_input("Select end date for impact analysis",
                                        value=None,
                                        min_value=impact_data.index.min(),
                                        max_value=impact_data.index.max())

    # Define the categories and their corresponding indicators
    categories = {
        "Domestic Equity": ["Sensex", "MidCap", "SmallCap", "NiftyBank"],
        "Global Equity": ["S&P500", "Nikkei225", "Hang Seng", "EuroStoxx50"],
        "Money and Bond Markets": ["10Y Government Bond", "3M Tbill", "3M CD", "1Y OIS rate"],
        "Commodity and FX": ["INR Spot", "Dollar Index", "Brent 1M Future"],
        "Stress Indicators": ["India VIX", "Credit Spread (AAA - Govt)", "Default Spread (BBB - AAA)", "CD-Tbill Spread"]
    }

    def display_category(df, category, indicators):
        st.markdown(f"<h3 style='text-align: left; font-size: 18px;'>{category}</h3>", unsafe_allow_html=True)
        category_df = df[df['Indicator'].isin(indicators)]
        st.dataframe(category_df, use_container_width=True, hide_index=True)

    # Only generate the table if both dates are selected
    if impact_start_date is not None and impact_end_date is not None:
        # Ensure end_date is not before start_date
        if impact_end_date < impact_start_date:
            st.error("End date must be after start date.")
        else:
            # Calculate changes for all indicators
            changes = calculate_impact(impact_data, impact_start_date, impact_end_date)
            # Create a dataframe for display
            display_df = pd.DataFrame({
                'Indicator': changes.index,
                'Change': changes['Change'].apply(lambda x: f"{x:+.2f}"),
                'Unit': changes['Unit']
            })
            
            # Display the categorized table
            col1, col2, col3 = st.columns(3)
            with col1:
                for category in ["Domestic Equity", "Global Equity"]:
                    display_category(display_df, category, categories[category])
            with col2:
                for category in ["Money and Bond Markets"]:
                    display_category(display_df, category, categories[category])
            with col3:
                for category in ["Commodity and FX", "Stress Indicators"]:
                    display_category(display_df, category, categories[category])
            
            st.markdown(f"<h3 style='text-align: left; font-size: 24px;'>Combined Data for Download</h3>", unsafe_allow_html=True)
            st.dataframe(display_df)
    else:
        st.info("Please select both start and end dates to view the market impact.")
    
    st.divider()
    
    



