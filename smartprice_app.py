import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import datetime
import io
import base64
from datetime import timedelta
import random
import uuid

# Set page configuration
st.set_page_config(
    page_title="SmartPrice - Dynamic Travel Pricing Engine",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for persistence
if 'pricing_history' not in st.session_state:
    st.session_state.pricing_history = []
if 'current_optimization' not in st.session_state:
    st.session_state.current_optimization = None
if 'revenue_impact' not in st.session_state:
    st.session_state.revenue_impact = 0
if 'total_optimizations' not in st.session_state:
    st.session_state.total_optimizations = 0
if 'successful_optimizations' not in st.session_state:
    st.session_state.successful_optimizations = 0

# Generate sample data
@st.cache_data
def generate_sample_data(n_samples=1000):
    np.random.seed(42)
    
    # Date range for the past year
    today = datetime.datetime.now()
    date_range = [today - datetime.timedelta(days=x) for x in range(365)]
    date_range.reverse()
    
    # Destinations
    destinations = [
        "Goa", "Mumbai", "Delhi", "Jaipur", "Agra", "Varanasi", 
        "Kolkata", "Chennai", "Bangalore", "Hyderabad", "Rishikesh",
        "Darjeeling", "Shimla", "Manali", "Udaipur", "Kerala"
    ]
    
    # Create dataframe
    data = {
        'date': np.random.choice(date_range, n_samples),
        'destination': np.random.choice(destinations, n_samples),
        'product_type': np.random.choice(['Hotel', 'Flight', 'Package', 'Activity'], n_samples),
        'seasonality': np.random.randint(1, 11, n_samples),
        'competitor_price': np.random.randint(5000, 50001, n_samples),
        'demand_level': np.random.randint(1, 11, n_samples),
        'booking_leadtime': np.random.randint(1, 366, n_samples),
        'original_price': np.random.randint(4000, 55000, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Add more derived features
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_holiday'] = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    df['market_condition'] = np.random.randint(1, 11, n_samples)
    df['historical_performance'] = np.random.randint(1, 11, n_samples)
    
    # Calculate optimal price with some randomness but maintain correlation with factors
    df['optimal_price'] = (
        df['original_price'] * 
        (1 + (df['seasonality'] - 5) * 0.02) * 
        (1 + (df['demand_level'] - 5) * 0.03) * 
        (1 + (df['is_weekend'] * 0.05)) *
        (1 + (df['is_holiday'] * 0.1)) *
        (1 + (df['market_condition'] - 5) * 0.01) *
        (0.9 + np.random.rand(n_samples) * 0.2)  # Add randomness
    ).astype(int)
    
    # Calculate revenue impact
    df['bookings'] = np.random.randint(1, 50, n_samples)
    df['revenue_original'] = df['original_price'] * df['bookings']
    
    # Price elasticity: if optimal_price < original_price, more bookings
    df['bookings_after_optimization'] = df.apply(
        lambda row: row['bookings'] * (1 + 0.2 * (row['original_price'] - row['optimal_price'])/row['original_price']) 
        if row['optimal_price'] != row['original_price'] else row['bookings'], 
        axis=1
    )
    df['bookings_after_optimization'] = df['bookings_after_optimization'].apply(lambda x: max(0, x))
    df['revenue_optimized'] = df['optimal_price'] * df['bookings_after_optimization']
    df['revenue_impact'] = df['revenue_optimized'] - df['revenue_original']
    
    # Generate competitor data
    competitors = ['CompetitorA', 'CompetitorB', 'CompetitorC', 'CompetitorD']
    for comp in competitors:
        df[f'{comp}_price'] = df['original_price'] * (0.85 + np.random.rand(n_samples) * 0.3)
    
    # Add confidence score
    df['confidence_score'] = np.random.randint(70, 100, n_samples)
    
    return df

# Generate seasonal trends data
@st.cache_data
def generate_seasonal_trends():
    months = list(range(1, 13))
    destinations = ["Goa", "Mumbai", "Delhi", "Jaipur", "Shimla", "Kerala"]
    
    data = []
    for dest in destinations:
        # Create a unique seasonal pattern for each destination
        if dest == "Goa":  # Peak in winter
            demand = [6, 5, 4, 3, 2, 1, 1, 2, 3, 5, 7, 9]
        elif dest == "Shimla":  # Peak in summer
            demand = [1, 2, 3, 6, 8, 9, 7, 7, 5, 4, 2, 1]
        elif dest == "Kerala":  # Two seasons
            demand = [8, 7, 5, 3, 2, 1, 1, 2, 3, 6, 8, 9]
        elif dest == "Mumbai":  # Avoid monsoon
            demand = [7, 8, 9, 8, 7, 3, 2, 2, 4, 7, 8, 7]
        elif dest == "Delhi":  # Spring and fall
            demand = [4, 5, 7, 8, 6, 3, 2, 2, 6, 8, 7, 5]
        else:  # Random pattern
            demand = [random.randint(3, 9) for _ in range(12)]
            
        for i, month in enumerate(months):
            data.append({
                'month': month,
                'month_name': datetime.date(2023, month, 1).strftime('%b'),
                'destination': dest,
                'demand_index': demand[i],
                'price_multiplier': 1 + (demand[i] - 5) * 0.05
            })
    
    return pd.DataFrame(data)

# Generate market share data
@st.cache_data
def generate_market_share_data():
    companies = ['SmartPrice', 'CompetitorA', 'CompetitorB', 'CompetitorC', 'Others']
    current_share = [23, 32, 18, 15, 12]
    optimized_share = [29, 29, 17, 14, 11]
    
    return pd.DataFrame({
        'company': companies,
        'current_share': current_share,
        'optimized_share': optimized_share
    })

# Train mock ML model
@st.cache_resource
def train_price_model(df):
    # Features for the model
    features = [
        'seasonality', 'competitor_price', 'demand_level', 'booking_leadtime',
        'is_weekend', 'is_holiday', 'market_condition', 'historical_performance', 'month'
    ]
    
    X = df[features]
    y = df['optimal_price']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, features

# Function to predict price
def predict_price(model, scaler, features, input_data):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df[features])
    predicted_price = model.predict(input_scaled)[0]
    return int(predicted_price)

# Function to calculate confidence score
def calculate_confidence(input_data):
    # Mock confidence calculation
    confidence = 85 + random.randint(-10, 10)
    return min(99, max(70, confidence))

# Function to export data to CSV
def export_to_csv(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="smartprice_export.csv">Download CSV File</a>'
    return href

# Load sample data and train model
sample_data = generate_sample_data()
seasonal_trends = generate_seasonal_trends()
market_share_data = generate_market_share_data()
model, scaler, model_features = train_price_model(sample_data)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-bottom: 0.5rem;
    }
    .metric-container {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #4B5563;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-delta {
        font-size: 1rem;
        font-weight: 500;
    }
    .positive-delta {
        color: #10B981;
    }
    .negative-delta {
        color: #EF4444;
    }
    .neutral-delta {
        color: #6B7280;
    }
    # Find this part in the CSS section (around line 225-245)
    # Replace the existing .info-box CSS with this improved version:

    .info-box {
        background-color: #DBEAFE;  /* Lighter blue background */
        border-left: 5px solid #3B82F6;
        border: 1px solid #BFDBFE;  /* Add border all around */
        padding: 1.2rem;
        border-radius: 5px;
        margin-bottom: 1.5rem;
        margin-top: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        font-weight: 500;  /* Make text slightly bolder */
        color: #1E40AF;  /* Darker blue text */
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #DBEAFE;
        border-bottom: 2px solid #2563EB;
        color: #1E3A8A;
        font-weight: bold;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/null/airplane-take-off.png", width=80)
    st.markdown("<div class='main-title'>SmartPrice</div>", unsafe_allow_html=True)
    st.markdown("### Dynamic Travel Pricing Engine")
    
    st.divider()
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["Dashboard", "Price Optimization", "Market Analysis", "Factors Dashboard", "Analytics & Reporting", "Simulation Mode"]
    )
    
    st.divider()
    
    # Quick stats
    st.markdown("### Quick Stats")
    st.metric("Optimizations Today", st.session_state.total_optimizations)
    st.metric("Success Rate", f"{int((st.session_state.successful_optimizations / max(1, st.session_state.total_optimizations)) * 100)}%")
    st.metric("Revenue Impact", f"₹{st.session_state.revenue_impact:,.0f}")
    
    st.divider()
    
    # System status
    st.markdown("### System Status")
    system_status = {
        "ML Model": "Online ✅",
        "Data Pipeline": "Online ✅",
        "API Integration": "Online ✅",
        "Last Update": datetime.datetime.now().strftime("%d-%b-%Y %H:%M")
    }
    
    for key, value in system_status.items():
        st.text(f"{key}: {value}")

# Main content
if page == "Dashboard":
    st.markdown("<div class='main-title'>Dashboard Overview</div>", unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-title'>Active Optimizations</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{len(sample_data['destination'].unique())}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-delta positive-delta'>+5% from last week</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-title'>Avg. Price Optimization</div>", unsafe_allow_html=True)
        avg_opt = ((sample_data['optimal_price'] - sample_data['original_price']) / sample_data['original_price']).mean() * 100
        st.markdown(f"<div class='metric-value'>{avg_opt:.1f}%</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-delta positive-delta'>+2.3% from last month</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-title'>Total Revenue Impact</div>", unsafe_allow_html=True)
        total_impact = sample_data['revenue_impact'].sum()
        st.markdown(f"<div class='metric-value'>₹{total_impact:,.0f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-delta positive-delta'>+12.7% YTD</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-title'>Model Accuracy</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-value'>87%</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-delta positive-delta'>+1.5% improvement</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Second row with charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='sub-title'>Revenue Impact Over Time</div>", unsafe_allow_html=True)
        
        # Group by date and sum revenue impact
        revenue_by_date = sample_data.groupby(sample_data['date'].dt.date)['revenue_impact'].sum().reset_index()
        revenue_by_date = revenue_by_date.sort_values('date')
        
        # Calculate cumulative sum
        revenue_by_date['cumulative_impact'] = revenue_by_date['revenue_impact'].cumsum()
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add daily revenue impact as bars
        fig.add_trace(
            go.Bar(
                x=revenue_by_date['date'],
                y=revenue_by_date['revenue_impact'],
                name="Daily Impact",
                marker_color='#93C5FD'
            ),
            secondary_y=False,
        )
        
        # Add cumulative impact as line
        fig.add_trace(
            go.Scatter(
                x=revenue_by_date['date'],
                y=revenue_by_date['cumulative_impact'],
                name="Cumulative Impact",
                line=dict(color='#2563EB', width=3)
            ),
            secondary_y=True,
        )
        
        # Set x-axis title
        fig.update_xaxes(title_text="Date")
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Daily Revenue Impact (₹)", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative Impact (₹)", secondary_y=True)
        
        fig.update_layout(
            height=400,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<div class='sub-title'>Optimization by Destination</div>", unsafe_allow_html=True)
        
        # Group by destination
        dest_impact = sample_data.groupby('destination').agg({
            'revenue_impact': 'sum',
            'optimal_price': 'mean',
            'original_price': 'mean'
        }).reset_index()
        
        # Calculate percentage change
        dest_impact['price_change_pct'] = ((dest_impact['optimal_price'] - dest_impact['original_price']) / 
                                          dest_impact['original_price'] * 100)
        
        # Sort by revenue impact
        dest_impact = dest_impact.sort_values('revenue_impact', ascending=False).head(10)
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add revenue impact as bars
        fig.add_trace(
            go.Bar(
                x=dest_impact['destination'],
                y=dest_impact['revenue_impact'],
                name="Revenue Impact",
                marker_color='#93C5FD'
            ),
            secondary_y=False,
        )
        
        # Add price change percentage as markers
        fig.add_trace(
            go.Scatter(
                x=dest_impact['destination'],
                y=dest_impact['price_change_pct'],
                name="Price Change %",
                mode='markers+lines',
                marker=dict(size=10, color='#DC2626', symbol='diamond'),
                line=dict(color='#DC2626', width=2, dash='dot')
            ),
            secondary_y=True,
        )
        
        # Set x-axis title
        fig.update_xaxes(title_text="Destination")
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Revenue Impact (₹)", secondary_y=False)
        fig.update_yaxes(title_text="Price Change (%)", secondary_y=True)
        
        fig.update_layout(
            height=400,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Third row
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("<div class='sub-title'>Optimization Status by Product Type</div>", unsafe_allow_html=True)
        
        # Group by product type
        product_stats = sample_data.groupby('product_type').agg({
            'optimal_price': 'mean',
            'original_price': 'mean',
            'revenue_impact': 'sum',
            'confidence_score': 'mean'
        }).reset_index()
        
        # Calculate price difference
        product_stats['price_diff'] = product_stats['optimal_price'] - product_stats['original_price']
        product_stats['price_diff_pct'] = (product_stats['price_diff'] / product_stats['original_price'] * 100).round(1)
        
        # Create a colorful table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Product Type', 'Avg. Original Price', 'Avg. Optimal Price', 'Price Change', 'Revenue Impact', 'Confidence'],
                fill_color='#BFDBFE',
                align='left',
                font=dict(color='#1E3A8A', size=12)
            ),
            cells=dict(
                values=[
                    product_stats['product_type'],
                    ['₹{:,.0f}'.format(x) for x in product_stats['original_price']],
                    ['₹{:,.0f}'.format(x) for x in product_stats['optimal_price']],
                    ['{:+.1f}%'.format(x) for x in product_stats['price_diff_pct']],
                    ['₹{:,.0f}'.format(x) for x in product_stats['revenue_impact']],
                    ['{:.0f}%'.format(x) for x in product_stats['confidence_score']]
                ],
                fill_color=[
                    ['#F3F4F6'] * len(product_stats),
                    ['#F3F4F6'] * len(product_stats),
                    ['#F3F4F6'] * len(product_stats),
                    [['#FEE2E2' if x < 0 else '#DCFCE7' for x in product_stats['price_diff_pct']]],
                    ['#F3F4F6'] * len(product_stats),
                    [['#FEE2E2' if x < 80 else '#DCFCE7' for x in product_stats['confidence_score']]]
                ],
                align='left',
                font=dict(color='#374151', size=11)
            ))
        ])
        
        fig.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<div class='sub-title'>Top Price Increases</div>", unsafe_allow_html=True)
        
        # Filter for price increases
        price_increases = sample_data[sample_data['optimal_price'] > sample_data['original_price']].copy()
        price_increases['price_diff_pct'] = ((price_increases['optimal_price'] - price_increases['original_price']) / 
                                            price_increases['original_price'] * 100)
        
        # Group by destination and product type
        top_increases = price_increases.groupby(['destination', 'product_type']).agg({
            'price_diff_pct': 'mean',
            'revenue_impact': 'sum'
        }).reset_index()
        
        # Sort and take top 5
        top_increases = top_increases.sort_values('price_diff_pct', ascending=False).head(5)
        
        # Create figure
        fig = px.bar(
            top_increases,
            y=top_increases['destination'] + ' - ' + top_increases['product_type'],
            x='price_diff_pct',
            orientation='h',
            color='price_diff_pct',
            color_continuous_scale='Greens',
            labels={'price_diff_pct': 'Price Increase (%)', 'y': ''},
            text=top_increases['price_diff_pct'].apply(lambda x: f'+{x:.1f}%')
        )
        
        fig.update_traces(textposition='outside')
        
        fig.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False,
            xaxis=dict(title='Price Increase (%)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("<div class='sub-title'>Top Price Decreases</div>", unsafe_allow_html=True)
        
        # Filter for price decreases
        price_decreases = sample_data[sample_data['optimal_price'] < sample_data['original_price']].copy()
        price_decreases['price_diff_pct'] = ((price_decreases['optimal_price'] - price_decreases['original_price']) / 
                                           price_decreases['original_price'] * 100)
        
        # Group by destination and product type
        top_decreases = price_decreases.groupby(['destination', 'product_type']).agg({
            'price_diff_pct': 'mean',
            'revenue_impact': 'sum'
        }).reset_index()
        
        # Sort and take top 5
        top_decreases = top_decreases.sort_values('price_diff_pct', ascending=True).head(5)
        
        # Create figure
        fig = px.bar(
            top_decreases,
            y=top_decreases['destination'] + ' - ' + top_decreases['product_type'],
            x='price_diff_pct',
            orientation='h',
            color='price_diff_pct',
            color_continuous_scale='Reds_r',
            labels={'price_diff_pct': 'Price Decrease (%)', 'y': ''},
            text=top_decreases['price_diff_pct'].apply(lambda x: f'{x:.1f}%')
        )
        
        fig.update_traces(textposition='outside')
        
        fig.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False,
            xaxis=dict(title='Price Decrease (%)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Latest optimizations
    st.markdown("<div class='sub-title'>Latest Pricing Decisions</div>", unsafe_allow_html=True)
    
    if not st.session_state.pricing_history:
        st.info("No pricing decisions have been made yet. Use the Price Optimization tool to generate new pricing decisions.")
    else:
        # Show the latest 5 pricing decisions
        recent_decisions = st.session_state.pricing_history[-5:]
        recent_decisions.reverse()  # Show most recent first
        
        for decision in recent_decisions:
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                st.markdown(f"**{decision['destination']} - {decision['product_type']}**")
                st.text(f"Date: {decision['date']}")
            
            with col2:
                st.markdown("Original Price: ₹{:,.0f}".format(decision['original_price']))
                st.markdown("Optimal Price: ₹{:,.0f}".format(decision['optimal_price']))
            
            with col3:
                price_diff = decision['optimal_price'] - decision['original_price']
                price_diff_pct = (price_diff / decision['original_price']) * 100
                
                if price_diff > 0:
                    st.markdown(f"<span style='color:#10B981'>+₹{price_diff:,.0f} (+{price_diff_pct:.1f}%)</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='color:#EF4444'>-₹{abs(price_diff):,.0f} ({price_diff_pct:.1f}%)</span>", unsafe_allow_html=True)
                
                st.markdown(f"Confidence: {decision['confidence_score']}%")
            
            with col4:
                st.markdown(f"<span style='color:#10B981'>+₹{decision['revenue_impact']:,.0f}</span>", unsafe_allow_html=True)
            
            st.divider()

elif page == "Price Optimization":
    st.markdown("<div class='main-title'>Price Optimization Engine</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    Use this tool to calculate the optimal price for your travel products based on market conditions and demand forecasts.
    Enter the product details below and adjust the relevant factors to see real-time price recommendations.
    </div>
    """, unsafe_allow_html=True)
    
    # Create the form for price optimization
    with st.form("price_optimization_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            destination = st.selectbox(
                "Destination",
                options=sorted(sample_data['destination'].unique()),
                index=0
            )
            
            product_type = st.selectbox(
                "Product Type",
                options=sorted(sample_data['product_type'].unique()),
                index=0
            )
        
        with col2:
            travel_date = st.date_input(
                "Travel Date",
                value=datetime.datetime.now() + timedelta(days=30),
                min_value=datetime.datetime.now().date(),
                max_value=datetime.datetime.now().date() + timedelta(days=365)
            )
            
            booking_leadtime = (travel_date - datetime.datetime.now().date()).days
            
            original_price = st.number_input(
                "Original Price (₹)",
                min_value=1000,
                max_value=100000,
                value=15000,
                step=500
            )
        
        with col3:
            expected_bookings = st.number_input(
                "Expected Bookings",
                min_value=1,
                max_value=1000,
                value=20,
                step=1
            )
            
            st.markdown("<div style='height: 75px;'></div>", unsafe_allow_html=True)
            
            optimize_button = st.form_submit_button("Calculate Optimal Price")
    
    # Factors section
    st.markdown("<div class='sub-title'>Pricing Factors</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        seasonality = st.slider(
            "Seasonality (Demand Level)",
            min_value=1,
            max_value=10,
            value=5,
            help="Higher values indicate peak season with higher demand"
        )
        
        demand_level = st.slider(
            "Current Demand Level",
            min_value=1,
            max_value=10,
            value=5,
            help="Current booking velocity and demand for this destination"
        )
        
        market_condition = st.slider(
            "Market Conditions",
            min_value=1,
            max_value=10,
            value=5,
            help="General market conditions (1: Poor, 10: Excellent)"
        )
        
        historical_performance = st.slider(
            "Historical Performance",
            min_value=1,
            max_value=10,
            value=5,
            help="How well this product has performed historically"
        )
    
    with col2:
        # Calculate approximate competitor prices based on original price
        avg_competitor_price = original_price * (0.9 + 0.2 * random.random())
        
        competitor_price = st.slider(
            "Average Competitor Price (₹)",
            min_value=5000,
            max_value=50000,
            value=int(avg_competitor_price),
            step=500,
            help="Average price offered by competitors for similar products"
        )
        
        is_weekend = st.checkbox(
            "Weekend Travel",
            value=False,
            help="Check if the travel date falls on a weekend"
        )
        
        is_holiday = st.checkbox(
            "Holiday/Festival Period",
            value=False,
            help="Check if the travel date coincides with holidays or festivals"
        )
    
    # If the form is submitted
    if optimize_button:
        # Prepare input for prediction
        month = travel_date.month
        
        input_data = {
            'seasonality': seasonality,
            'competitor_price': competitor_price,
            'demand_level': demand_level,
            'booking_leadtime': booking_leadtime,
            'is_weekend': 1 if is_weekend else 0,
            'is_holiday': 1 if is_holiday else 0,
            'market_condition': market_condition,
            'historical_performance': historical_performance,
            'month': month
        }
        
        # Predict optimal price
        optimal_price = predict_price(model, scaler, model_features, input_data)
        
        # Calculate confidence score
        confidence_score = calculate_confidence(input_data)
        
        # Calculate price difference
        price_diff = optimal_price - original_price
        price_diff_pct = (price_diff / original_price) * 100
        
        # Calculate revenue impact
        elasticity_factor = 0.2  # Price elasticity factor
        bookings_after_optimization = max(0, expected_bookings * (1 + elasticity_factor * (-price_diff_pct/100)))
        
        revenue_original = original_price * expected_bookings
        revenue_optimized = optimal_price * bookings_after_optimization
        revenue_impact = revenue_optimized - revenue_original
        
        # Store the optimization result
        optimization_result = {
            'destination': destination,
            'product_type': product_type,
            'date': travel_date.strftime("%d %b %Y"),
            'original_price': original_price,
            'optimal_price': optimal_price,
            'confidence_score': confidence_score,
            'price_diff': price_diff,
            'price_diff_pct': price_diff_pct,
            'expected_bookings': expected_bookings,
            'bookings_after_optimization': bookings_after_optimization,
            'revenue_original': revenue_original,
            'revenue_optimized': revenue_optimized,
            'revenue_impact': revenue_impact
        }
        
        st.session_state.current_optimization = optimization_result
        
        # Update session state
        st.session_state.total_optimizations += 1
        if abs(price_diff_pct) > 2:  # Consider it successful if price diff is significant
            st.session_state.successful_optimizations += 1
        st.session_state.revenue_impact += revenue_impact
        
        # Show the results
        st.markdown("<div class='sub-title'>Optimization Results</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.markdown("### Price Recommendation")
            
            metric_color = "positive-delta" if price_diff > 0 else "negative-delta"
            price_sign = "+" if price_diff > 0 else ""
            
            st.markdown(f"""
            <div style='background-color: #F3F4F6; padding: 1rem; border-radius: 10px;'>
                <div style='font-size: 1.1rem; color: #4B5563;'>Original Price</div>
                <div style='font-size: 1.8rem; font-weight: 700; color: #1E3A8A;'>₹{original_price:,.0f}</div>
                <div style='height: 10px;'></div>
                <div style='font-size: 1.1rem; color: #4B5563;'>Optimal Price</div>
                <div style='font-size: 1.8rem; font-weight: 700; color: #1E3A8A;'>₹{optimal_price:,.0f}</div>
                <div style='font-size: 1.1rem; font-weight: 500;' class='{metric_color}'>
                    {price_sign}₹{abs(price_diff):,.0f} ({price_sign}{price_diff_pct:.1f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Revenue Impact")
            
            metric_color = "positive-delta" if revenue_impact > 0 else "negative-delta"
            revenue_sign = "+" if revenue_impact > 0 else ""
            
            st.markdown(f"""
            <div style='background-color: #F3F4F6; padding: 1rem; border-radius: 10px;'>
                <div style='font-size: 1.1rem; color: #4B5563;'>Current Revenue</div>
                <div style='font-size: 1.8rem; font-weight: 700; color: #1E3A8A;'>₹{revenue_original:,.0f}</div>
                <div style='height: 10px;'></div>
                <div style='font-size: 1.1rem; color: #4B5563;'>Projected Revenue</div>
                <div style='font-size: 1.8rem; font-weight: 700; color: #1E3A8A;'>₹{revenue_optimized:,.0f}</div>
                <div style='font-size: 1.1rem; font-weight: 500;' class='{metric_color}'>
                    {revenue_sign}₹{abs(revenue_impact):,.0f} ({revenue_sign}{(revenue_impact/revenue_original*100):.1f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("### Confidence")
            
            # Determine color based on confidence score
            if confidence_score >= 90:
                gauge_color = "#10B981"  # Green
            elif confidence_score >= 75:
                gauge_color = "#FBBF24"  # Yellow
            else:
                gauge_color = "#EF4444"  # Red
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Confidence Score"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': gauge_color},
                    'steps': [
                        {'range': [0, 75], 'color': "#FECACA"},
                        {'range': [75, 90], 'color': "#FEF3C7"},
                        {'range': [90, 100], 'color': "#D1FAE5"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            
            fig.update_layout(
                height=200,
                margin=dict(l=10, r=10, t=30, b=10),
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Explanation and recommendations
        st.markdown("<div class='sub-title'>Analysis & Recommendations</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Generate analysis text based on the factors
            analysis_text = f"For {destination} {product_type} on {travel_date.strftime('%d %b %Y')}, our analysis shows "
            
            if price_diff > 0:
                analysis_text += f"the current price of ₹{original_price:,.0f} is **below optimal**. "
                analysis_text += f"The market can support a higher price of ₹{optimal_price:,.0f} "
                analysis_text += f"based on the following factors:\n\n"
            else:
                analysis_text += f"the current price of ₹{original_price:,.0f} is **above optimal**. "
                analysis_text += f"We recommend reducing the price to ₹{optimal_price:,.0f} "
                analysis_text += f"based on the following factors:\n\n"
            
            # Add factor-specific analysis
            factors_analysis = []
            
            if seasonality > 6:
                factors_analysis.append(f"- **High seasonality** (rated {seasonality}/10) indicates strong demand during this period")
            elif seasonality < 4:
                factors_analysis.append(f"- **Low seasonality** (rated {seasonality}/10) suggests lower natural demand")
            
            if demand_level > 6:
                factors_analysis.append(f"- **Strong current demand** (rated {demand_level}/10) shows high market interest")
            elif demand_level < 4:
                factors_analysis.append(f"- **Weak current demand** (rated {demand_level}/10) indicates soft booking patterns")
            
            if competitor_price > original_price * 1.1:
                factors_analysis.append(f"- **Competitor prices** (₹{competitor_price:,.0f}) are significantly higher than yours")
            elif competitor_price < original_price * 0.9:
                factors_analysis.append(f"- **Competitor prices** (₹{competitor_price:,.0f}) are significantly lower than yours")
            
            if booking_leadtime > 90:
                factors_analysis.append(f"- **Long booking window** ({booking_leadtime} days) allows for premium pricing")
            elif booking_leadtime < 30:
                factors_analysis.append(f"- **Short booking window** ({booking_leadtime} days) may require more competitive pricing")
            
            if is_weekend:
                factors_analysis.append("- **Weekend travel** typically commands premium pricing")
            
            if is_holiday:
                factors_analysis.append("- **Holiday/festival period** creates additional demand pressure")
            
            # Add recommendations
            if price_diff > 0:
                recommendations = [
                    f"- Increase price to ₹{optimal_price:,.0f} to maximize revenue",
                    f"- Consider targeted upselling to capture additional value",
                    f"- Monitor booking pace after price change and adjust if necessary"
                ]
            else:
                recommendations = [
                    f"- Reduce price to ₹{optimal_price:,.0f} to stimulate demand",
                    f"- Consider adding value-adds instead of deep discounting",
                    f"- Target marketing to price-sensitive segments"
                ]
            
            st.markdown(analysis_text + "\n".join(factors_analysis))
            
            st.markdown("### Recommendations")
            st.markdown("\n".join(recommendations))
        
        with col2:
            # Price comparison with competitors
            st.markdown("### Competitive Positioning")
            
            # Generate some competitor prices
            competitor_data = {
                'Provider': ['Your Price', 'Optimal Price', 'CompetitorA', 'CompetitorB', 'CompetitorC', 'Market Average'],
                'Price': [
                    original_price,
                    optimal_price,
                    int(original_price * (0.9 + 0.3 * random.random())),
                    int(original_price * (0.85 + 0.4 * random.random())),
                    int(original_price * (0.8 + 0.5 * random.random())),
                    int(competitor_price)
                ]
            }
            
            competitor_df = pd.DataFrame(competitor_data)
            competitor_df = competitor_df.sort_values('Price')
            
            # Create the bar chart
            fig = px.bar(
                competitor_df,
                x='Price',
                y='Provider',
                orientation='h',
                color='Provider',
                color_discrete_map={
                    'Your Price': '#93C5FD',
                    'Optimal Price': '#2563EB',
                    'Market Average': '#9CA3AF'
                },
                labels={'Price': 'Price (₹)', 'Provider': ''},
                title="Price Comparison"
            )
            
            fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=30, b=10),
                showlegend=False
            )
            
            # Add price labels on bars
            fig.update_traces(
                texttemplate='₹%{x:,.0f}',
                textposition='outside',
                hovertemplate='%{y}: ₹%{x:,.0f}'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Apply This Price", type="primary"):
                # Add to pricing history
                st.session_state.pricing_history.append(optimization_result)
                st.success("Price recommendation applied successfully!")
        
        with col2:
            if st.button("Modify Factors"):
                st.info("Adjust the factors above and recalculate.")
        
        with col3:
            if st.button("Save for Later"):
                # Add to pricing history with a flag
                optimization_result['status'] = 'saved'
                st.session_state.pricing_history.append(optimization_result)
                st.info("Price recommendation saved for later review.")
    
    # If no optimization has been performed yet
    if not optimize_button and 'current_optimization' not in st.session_state:
        st.info("Enter your product details and pricing factors above, then click 'Calculate Optimal Price' to get recommendations.")

elif page == "Market Analysis":
    st.markdown("<div class='main-title'>Market Analysis</div>", unsafe_allow_html=True)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Competitor Price Comparison", "Seasonal Trend Analysis", 
        "Demand Forecasting", "Market Share Impact"
    ])
    
    with tab1:
        st.markdown("<div class='sub-title'>Competitor Price Comparison</div>", unsafe_allow_html=True)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            dest_filter = st.selectbox(
                "Destination",
                options=['All'] + sorted(sample_data['destination'].unique().tolist()),
                key="comp_dest_filter"
            )
        
        with col2:
            product_filter = st.selectbox(
                "Product Type",
                options=['All'] + sorted(sample_data['product_type'].unique().tolist()),
                key="comp_product_filter"
            )
        
        with col3:
            date_range = st.date_input(
                "Date Range",
                value=(datetime.datetime.now().date() - timedelta(days=30), datetime.datetime.now().date()),
                key="comp_date_range"
            )
        
        # Filter data
        filtered_data = sample_data.copy()
        
        if dest_filter != 'All':
            filtered_data = filtered_data[filtered_data['destination'] == dest_filter]
        
        if product_filter != 'All':
            filtered_data = filtered_data[filtered_data['product_type'] == product_filter]
        
        if len(date_range) == 2:
            filtered_data = filtered_data[
                (filtered_data['date'].dt.date >= date_range[0]) & 
                (filtered_data['date'].dt.date <= date_range[1])
            ]
        
        # Competitor price comparison
        st.markdown("### Price Comparison by Competitor")
        
        # Get competitors from the dataset
        competitors = ['Original Price', 'Optimal Price'] + [col for col in filtered_data.columns if 'Competitor' in col and 'price' in col]
        
        # Prepare data for comparison
        comp_data = filtered_data.groupby(['destination', 'product_type']).agg({
            'original_price': 'mean',
            'optimal_price': 'mean',
            'CompetitorA_price': 'mean',
            'CompetitorB_price': 'mean',
            'CompetitorC_price': 'mean',
            'CompetitorD_price': 'mean'
        }).reset_index()
        
        # Select top destinations by price
        top_destinations = comp_data.sort_values('original_price', ascending=False).head(10)
        
        # Reshape data for plotting
        plot_data = []
        for _, row in top_destinations.iterrows():
            for comp in ['original_price', 'optimal_price', 'CompetitorA_price', 'CompetitorB_price', 'CompetitorC_price', 'CompetitorD_price']:
                plot_data.append({
                    'Destination': f"{row['destination']} - {row['product_type']}",
                    'Provider': comp.replace('_price', '').replace('original', 'Your Price').replace('optimal', 'Optimal Price'),
                    'Price': row[comp]
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create grouped bar chart
        fig = px.bar(
            plot_df,
            x='Destination',
            y='Price',
            color='Provider',
            barmode='group',
            title="Average Price by Destination and Provider",
            color_discrete_map={
                'Your Price': '#93C5FD',
                'Optimal Price': '#2563EB'
            }
        )
        
        fig.update_layout(
            height=500,
            xaxis_title="",
            yaxis_title="Price (₹)",
            legend_title="Provider",
            xaxis={'categoryorder':'total descending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Price difference analysis
        st.markdown("### Price Difference Analysis")
        
        # Calculate average price difference percentage
        comp_data['vs_CompetitorA'] = ((comp_data['original_price'] - comp_data['CompetitorA_price']) / comp_data['CompetitorA_price'] * 100).round(1)
        comp_data['vs_CompetitorB'] = ((comp_data['original_price'] - comp_data['CompetitorB_price']) / comp_data['CompetitorB_price'] * 100).round(1)
        comp_data['vs_CompetitorC'] = ((comp_data['original_price'] - comp_data['CompetitorC_price']) / comp_data['CompetitorC_price'] * 100).round(1)
        comp_data['vs_CompetitorD'] = ((comp_data['original_price'] - comp_data['CompetitorD_price']) / comp_data['CompetitorD_price'] * 100).round(1)
        comp_data['vs_Optimal'] = ((comp_data['original_price'] - comp_data['optimal_price']) / comp_data['optimal_price'] * 100).round(1)
        
        # Create price difference table
        diff_columns = ['destination', 'product_type', 'vs_CompetitorA', 'vs_CompetitorB', 'vs_CompetitorC', 'vs_CompetitorD', 'vs_Optimal']
        diff_table = comp_data[diff_columns].sort_values('vs_Optimal', ascending=False)
        
        # Create table visualization
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Destination', 'Product Type', 'vs CompetitorA', 'vs CompetitorB', 'vs CompetitorC', 'vs CompetitorD', 'vs Optimal'],
                fill_color='#BFDBFE',
                align='left',
                font=dict(color='#1E3A8A', size=12)
            ),
            cells=dict(
                values=[
                    diff_table['destination'],
                    diff_table['product_type'],
                    [f"{x:+.1f}%" for x in diff_table['vs_CompetitorA']],
                    [f"{x:+.1f}%" for x in diff_table['vs_CompetitorB']],
                    [f"{x:+.1f}%" for x in diff_table['vs_CompetitorC']],
                    [f"{x:+.1f}%" for x in diff_table['vs_CompetitorD']],
                    [f"{x:+.1f}%" for x in diff_table['vs_Optimal']]
                ],
                fill_color=[
                    ['#F3F4F6'] * len(diff_table),
                    ['#F3F4F6'] * len(diff_table),
                    [['#FEE2E2' if x > 0 else '#DCFCE7' for x in diff_table['vs_CompetitorA']]],
                    [['#FEE2E2' if x > 0 else '#DCFCE7' for x in diff_table['vs_CompetitorB']]],
                    [['#FEE2E2' if x > 0 else '#DCFCE7' for x in diff_table['vs_CompetitorC']]],
                    [['#FEE2E2' if x > 0 else '#DCFCE7' for x in diff_table['vs_CompetitorD']]],
                    [['#FEE2E2' if x > 0 else '#DCFCE7' for x in diff_table['vs_Optimal']]]
                ],
                align='left',
                font=dict(color='#374151', size=11)
            ))
        ])
        
        fig.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Market positioning summary
        col1, col2 = st.columns(2)
        
        with col1:
            # Calculate overall positioning
            avg_diff = pd.DataFrame({
                'Competitor': ['CompetitorA', 'CompetitorB', 'CompetitorC', 'CompetitorD', 'Optimal'],
                'Avg_Diff': [
                    comp_data['vs_CompetitorA'].mean(),
                    comp_data['vs_CompetitorB'].mean(),
                    comp_data['vs_CompetitorC'].mean(),
                    comp_data['vs_CompetitorD'].mean(),
                    comp_data['vs_Optimal'].mean()
                ]
            })
            
            # Create horizontal bar chart
            fig = px.bar(
                avg_diff,
                y='Competitor',
                x='Avg_Diff',
                orientation='h',
                title="Average Price Difference (%)",
                color='Avg_Diff',
                color_continuous_scale='RdBu_r',
                range_color=[-20, 20],
                labels={'Avg_Diff': 'Price Difference (%)', 'Competitor': ''},
                text=avg_diff['Avg_Diff'].apply(lambda x: f"{x:+.1f}%")
            )
            
            fig.update_traces(textposition='outside')
            
            fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=40, b=10),
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Summary text
            st.markdown("### Market Positioning Summary")
            
            # Generate insights based on the data
            overall_vs_market = avg_diff['Avg_Diff'].mean()
            
            if overall_vs_market > 5:
                st.markdown("""
                📈 **Premium Positioning**
                
                Your prices are generally **higher** than the market average. Consider:
                - Emphasizing your unique value proposition
                - Highlighting premium features
                - Ensuring service quality matches the premium price
                """)
            elif overall_vs_market < -5:
                st.markdown("""
                📉 **Value Positioning**
                
                Your prices are generally **lower** than the market average. Consider:
                - Opportunities to increase prices in select markets
                - Adding value-added services
                - Marketing your competitive pricing as a strength
                """)
            else:
                st.markdown("""
                📊 **Market-aligned Positioning**
                
                Your prices are generally **aligned** with the market average. Consider:
                - Selective optimization in specific destinations
                - Differentiating through service or features
                - Strategic promotions for competitive advantage
                """)
            
            # Action recommendations
            st.markdown("#### Recommended Actions:")
            
            # Find most overpriced and underpriced destinations
            overpriced = diff_table.sort_values('vs_Optimal', ascending=False).head(3)
            underpriced = diff_table.sort_values('vs_Optimal', ascending=True).head(3)
            
            if not overpriced.empty:
                st.markdown("**Review pricing for potentially overpriced destinations:**")
                for _, row in overpriced.iterrows():
                    st.markdown(f"- {row['destination']} ({row['product_type']}): {row['vs_Optimal']:+.1f}% vs optimal")
            
            if not underpriced.empty:
                st.markdown("**Consider price increases for underpriced destinations:**")
                for _, row in underpriced.iterrows():
                    st.markdown(f"- {row['destination']} ({row['product_type']}): {row['vs_Optimal']:+.1f}% vs optimal")
    
    with tab2:
        st.markdown("<div class='sub-title'>Seasonal Trend Analysis</div>", unsafe_allow_html=True)
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            seasonal_dest = st.multiselect(
                "Select Destinations",
                options=sorted(seasonal_trends['destination'].unique()),
                default=["Goa", "Shimla", "Kerala"]
            )
        
        with col2:
            chart_type = st.radio(
                "Chart Type",
                options=["Demand Index", "Price Multiplier"],
                horizontal=True
            )
        
        # Filter data
        if not seasonal_dest:
            seasonal_dest = sorted(seasonal_trends['destination'].unique())
        
        filtered_seasonal = seasonal_trends[seasonal_trends['destination'].isin(seasonal_dest)]
        
        # Create line chart
        if chart_type == "Demand Index":
            y_value = 'demand_index'
            title = "Seasonal Demand Index by Month"
            y_title = "Demand Index (1-10)"
        else:
            y_value = 'price_multiplier'
            title = "Seasonal Price Multiplier by Month"
            y_title = "Price Multiplier"
        
        fig = px.line(
            filtered_seasonal,
            x='month_name',
            y=y_value,
            color='destination',
            markers=True,
            title=title,
            category_orders={"month_name": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]}
        )
        
        fig.update_layout(
            height=500,
            xaxis_title="Month",
            yaxis_title=y_title,
            legend_title="Destination",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal insights
        st.markdown("### Seasonal Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Peak season analysis
            st.markdown("#### Peak Seasons by Destination")
            
            # Find peak month for each destination
            peak_seasons = filtered_seasonal.loc[filtered_seasonal.groupby('destination')['demand_index'].idxmax()]
            
            # Create table visualization
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=['Destination', 'Peak Month', 'Demand Index', 'Price Multiplier'],
                    fill_color='#BFDBFE',
                    align='left',
                    font=dict(color='#1E3A8A', size=12)
                ),
                cells=dict(
                    values=[
                        peak_seasons['destination'],
                        peak_seasons['month_name'],
                        peak_seasons['demand_index'],
                        [f"{x:.2f}" for x in peak_seasons['price_multiplier']]
                    ],
                    fill_color='#F3F4F6',
                    align='left',
                    font=dict(color='#374151', size=11)
                ))
            ])
            
            fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Off-season analysis
            st.markdown("#### Off-Peak Seasons by Destination")
            
            # Find off-peak month for each destination
            off_peak_seasons = filtered_seasonal.loc[filtered_seasonal.groupby('destination')['demand_index'].idxmin()]
            
            # Create table visualization
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=['Destination', 'Off-Peak Month', 'Demand Index', 'Price Multiplier'],
                    fill_color='#BFDBFE',
                    align='left',
                    font=dict(color='#1E3A8A', size=12)
                ),
                cells=dict(
                    values=[
                        off_peak_seasons['destination'],
                        off_peak_seasons['month_name'],
                        off_peak_seasons['demand_index'],
                        [f"{x:.2f}" for x in off_peak_seasons['price_multiplier']]
                    ],
                    fill_color='#F3F4F6',
                    align='left',
                    font=dict(color='#374151', size=11)
                ))
            ])
            
            fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal pricing recommendations
        st.markdown("### Seasonal Pricing Recommendations")
        
        # Generate recommendations based on current month
        current_month = datetime.datetime.now().month
        current_month_name = datetime.date(2023, current_month, 1).strftime('%b')
        
        # Filter for current month
        current_month_data = filtered_seasonal[filtered_seasonal['month'] == current_month]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### Current Month: {current_month_name}")
            
            # Sort destinations by current demand
            current_month_data = current_month_data.sort_values('demand_index', ascending=False)
            
            for _, row in current_month_data.iterrows():
                if row['demand_index'] >= 7:
                    emoji = "🔥"
                    recommendation = "Premium pricing recommended"
                elif row['demand_index'] >= 4:
                    emoji = "⚖️"
                    recommendation = "Standard pricing recommended"
                else:
                    emoji = "❄️"
                    recommendation = "Promotional pricing recommended"
                
                st.markdown(f"""
                **{row['destination']}** {emoji}
                - Demand Index: {row['demand_index']}/10
                - Price Multiplier: {row['price_multiplier']:.2f}
                - **Recommendation**: {recommendation}
                """)
        
        with col2:
            # Next 3 months forecast
            next_months = [(current_month + i) % 12 or 12 for i in range(1, 4)]
            next_month_names = [datetime.date(2023, m, 1).strftime('%b') for m in next_months]
            
            st.markdown(f"#### Upcoming Months: {', '.join(next_month_names)}")
            
            # Get destinations with increasing and decreasing demand
            increasing_demand = []
            decreasing_demand = []
            
            for dest in seasonal_dest:
                dest_data = filtered_seasonal[filtered_seasonal['destination'] == dest]
                current_demand = dest_data[dest_data['month'] == current_month]['demand_index'].values[0]
                next_demand = dest_data[dest_data['month'] == next_months[0]]['demand_index'].values[0]
                
                if next_demand > current_demand:
                    increasing_demand.append((dest, next_demand - current_demand))
                else:
                    decreasing_demand.append((dest, current_demand - next_demand))
            
            if increasing_demand:
                st.markdown("**Destinations with Increasing Demand:**")
                for dest, change in sorted(increasing_demand, key=lambda x: x[1], reverse=True):
                    st.markdown(f"- {dest}: +{change:.1f} demand points")
            
            if decreasing_demand:
                st.markdown("**Destinations with Decreasing Demand:**")
                for dest, change in sorted(decreasing_demand, key=lambda x: x[1], reverse=True):
                    st.markdown(f"- {dest}: -{change:.1f} demand points")
            
            # Strategic recommendations
            st.markdown("**Strategic Recommendations:**")
            st.markdown("- Increase prices for destinations with rising demand")
            st.markdown("- Consider promotions for destinations with decreasing demand")
            st.markdown("- Prepare marketing campaigns for upcoming high seasons")
    
    with tab3:
        st.markdown("<div class='sub-title'>Demand Forecasting</div>", unsafe_allow_html=True)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            forecast_dest = st.selectbox(
                "Select Destination",
                options=sorted(sample_data['destination'].unique()),
                key="forecast_dest"
            )
        
        with col2:
            forecast_product = st.selectbox(
                "Select Product Type",
                options=sorted(sample_data['product_type'].unique()),
                key="forecast_product"
            )
        
        with col3:
            forecast_period = st.slider(
                "Forecast Period (months)",
                min_value=1,
                max_value=12,
                value=6,
                key="forecast_period"
            )
        
        # Generate forecast data
        st.markdown("### Demand Forecast")
        
        # Filter for selected destination and product
        dest_data = sample_data[
            (sample_data['destination'] == forecast_dest) & 
            (sample_data['product_type'] == forecast_product)
        ]
        
        # Group by month and calculate average bookings
        monthly_bookings = dest_data.groupby(dest_data['date'].dt.month).agg({
            'bookings': 'mean',
            'bookings_after_optimization': 'mean'
        }).reset_index()
        
        monthly_bookings.columns = ['month', 'current_bookings', 'optimized_bookings']
        monthly_bookings['month_name'] = monthly_bookings['month'].apply(lambda x: datetime.date(2023, x, 1).strftime('%b'))
        
        # Extend for forecast
        current_month = datetime.datetime.now().month
        forecast_months = [(current_month + i - 1) % 12 + 1 for i in range(forecast_period)]
        forecast_month_names = [datetime.date(2023, m, 1).strftime('%b') for m in forecast_months]
        
        # Get seasonal factors from seasonal_trends
        seasonal_factors = seasonal_trends[seasonal_trends['destination'] == forecast_dest][['month', 'demand_index']].copy()
        
        # Prepare forecast data
        forecast_data = []
        
        for i, month in enumerate(forecast_months):
            month_name = forecast_month_names[i]
            
            # Get seasonal factor
            # Replace line 1558 with this error-handling code:
            try:
                seasonal_factor = seasonal_factors[seasonal_factors['month'] == month]['demand_index'].values[0]
            except IndexError:
                # Fallback to a default value if no data is found for this month
                seasonal_factor = 5  # Use a neutral value as fallback
                st.warning(f"No seasonal data found for {forecast_dest} in month {month}. Using default value.")
            
            # Get base bookings if available in historical data
            base_current = monthly_bookings[monthly_bookings['month'] == month]['current_bookings'].values
            base_optimized = monthly_bookings[monthly_bookings['month'] == month]['optimized_bookings'].values
            
            if len(base_current) > 0 and len(base_optimized) > 0:
                base_current = base_current[0]
                base_optimized = base_optimized[0]
            else:
                # Use average if month not in historical data
                base_current = monthly_bookings['current_bookings'].mean()
                base_optimized = monthly_bookings['optimized_bookings'].mean()
            
            # Calculate forecast with some randomness
            forecast_current = max(0, base_current * (0.9 + 0.2 * seasonal_factor / 5 + 0.1 * random.random()))
            forecast_optimized = max(0, base_optimized * (0.9 + 0.2 * seasonal_factor / 5 + 0.1 * random.random()))
            
            # Add growth trend
            forecast_current *= (1 + 0.02 * i)  # 2% monthly growth
            forecast_optimized *= (1 + 0.03 * i)  # 3% monthly growth with optimization
            
            forecast_data.append({
                'month': month,
                'month_name': month_name,
                'current_forecast': int(forecast_current),
                'optimized_forecast': int(forecast_optimized),
                'seasonal_factor': seasonal_factor
            })
        
        forecast_df = pd.DataFrame(forecast_data)
        
        # Create line chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=forecast_df['month_name'],
            y=forecast_df['current_forecast'],
            mode='lines+markers',
            name='Current Pricing',
            line=dict(color='#93C5FD', width=2),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['month_name'],
            y=forecast_df['optimized_forecast'],
            mode='lines+markers',
            name='Optimized Pricing',
            line=dict(color='#2563EB', width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f"Demand Forecast for {forecast_dest} - {forecast_product}",
            height=500,
            xaxis_title="Month",
            yaxis_title="Forecasted Bookings",
            legend_title="Pricing Strategy",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast details
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Forecast Details")
            
            # Calculate total and average forecast
            total_current = forecast_df['current_forecast'].sum()
            total_optimized = forecast_df['optimized_forecast'].sum()
            avg_current = forecast_df['current_forecast'].mean()
            avg_optimized = forecast_df['optimized_forecast'].mean()
            
            # Calculate improvement
            booking_improvement = total_optimized - total_current
            booking_improvement_pct = (booking_improvement / total_current) * 100
            
            st.markdown(f"""
            **Total Forecasted Bookings ({forecast_period} months):**
            - Current Pricing: {total_current:,.0f} bookings
            - Optimized Pricing: {total_optimized:,.0f} bookings
            - Improvement: +{booking_improvement:,.0f} bookings (+{booking_improvement_pct:.1f}%)
            
            **Average Monthly Bookings:**
            - Current Pricing: {avg_current:.1f} bookings
            - Optimized Pricing: {avg_optimized:.1f} bookings
            """)
            
            # Revenue impact
            avg_price = dest_data['original_price'].mean()
            avg_opt_price = dest_data['optimal_price'].mean()
            
            current_revenue = total_current * avg_price
            optimized_revenue = total_optimized * avg_opt_price
            revenue_impact = optimized_revenue - current_revenue
            revenue_impact_pct = (revenue_impact / current_revenue) * 100
            
            st.markdown(f"""
            **Revenue Impact:**
            - Current Strategy: ₹{current_revenue:,.0f}
            - Optimized Strategy: ₹{optimized_revenue:,.0f}
            - Improvement: +₹{revenue_impact:,.0f} (+{revenue_impact_pct:.1f}%)
            """)
        
        with col2:
            st.markdown("### Demand Drivers")
            
            # Create bar chart of seasonal factors
            fig = px.bar(
                forecast_df,
                x='month_name',
                y='seasonal_factor',
                title="Seasonal Demand Factors",
                color='seasonal_factor',
                color_continuous_scale='Blues',
                text=forecast_df['seasonal_factor']
            )
            
            fig.update_traces(textposition='outside')
            
            fig.update_layout(
                height=300,
                xaxis_title="Month",
                yaxis_title="Seasonal Factor (1-10)",
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations based on forecast
            st.markdown("### Recommendations")
            
            peak_month = forecast_df.loc[forecast_df['seasonal_factor'].idxmax()]
            low_month = forecast_df.loc[forecast_df['seasonal_factor'].idxmin()]
            
            st.markdown(f"""
            **Strategic Pricing Recommendations:**
            
            - **Peak Demand ({peak_month['month_name']})**: Implement premium pricing with factor of {peak_month['seasonal_factor']}/10
              - Forecast: {peak_month['optimized_forecast']} bookings with optimized pricing
              
            - **Low Demand ({low_month['month_name']})**: Consider promotional offers with factor of {low_month['seasonal_factor']}/10
              - Forecast: {low_month['optimized_forecast']} bookings with optimized pricing
              
            - **Overall Strategy**: Dynamic pricing adjustment based on seasonal factors can improve booking volume by {booking_improvement_pct:.1f}% and revenue by {revenue_impact_pct:.1f}%
            """)
    
    with tab4:
        st.markdown("<div class='sub-title'>Market Share Impact</div>", unsafe_allow_html=True)
        
        # Market share visualization
        st.markdown("### Market Share Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create pie chart for current market share
            fig = px.pie(
                market_share_data,
                values='current_share',
                names='company',
                title="Current Market Share",
                color='company',
                color_discrete_map={
                    'SmartPrice': '#2563EB',
                    'CompetitorA': '#93C5FD',
                    'CompetitorB': '#BFDBFE',
                    'CompetitorC': '#DBEAFE',
                    'Others': '#F3F4F6'
                }
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            fig.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Create pie chart for optimized market share
            fig = px.pie(
                market_share_data,
                values='optimized_share',
                names='company',
                title="Projected Market Share with SmartPrice Optimization",
                color='company',
                color_discrete_map={
                    'SmartPrice': '#2563EB',
                    'CompetitorA': '#93C5FD',
                    'CompetitorB': '#BFDBFE',
                    'CompetitorC': '#DBEAFE',
                    'Others': '#F3F4F6'
                }
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            fig.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Market share change analysis
        st.markdown("### Market Share Change Analysis")
        
        # Calculate market share changes
        market_share_data['share_change'] = market_share_data['optimized_share'] - market_share_data['current_share']
        market_share_data['share_change_pct'] = (market_share_data['share_change'] / market_share_data['current_share']) * 100
        
        # Create horizontal bar chart for market share changes
        fig = px.bar(
            market_share_data,
            y='company',
            x='share_change',
            orientation='h',
            title="Market Share Change (Percentage Points)",
            color='share_change',
            color_continuous_scale='RdBu',
            text=market_share_data['share_change'].apply(lambda x: f"{x:+.1f} pp")
        )
        
        fig.update_traces(textposition='outside')
        
        fig.update_layout(
            height=400,
            xaxis_title="Change in Market Share (Percentage Points)",
            yaxis_title="",
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Market share simulation
        st.markdown("### Market Share Impact Simulation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            optimization_level = st.slider(
                "Price Optimization Level",
                min_value=0,
                max_value=100,
                value=80,
                help="Percentage of recommended price optimizations implemented"
            )
        
        with col2:
            market_size = st.number_input(
                "Total Market Size (₹ Cr)",
                min_value=100,
                max_value=10000,
                value=2500,
                step=100,
                help="Total market size in crore rupees"
            )
        
        with col3:
            timeframe = st.selectbox(
                "Implementation Timeframe",
                options=["3 months", "6 months", "12 months", "24 months"],
                index=1,
                help="Timeframe for implementing the optimization strategy"
            )
        
        # Calculate impact based on inputs
        smartprice_current = market_share_data[market_share_data['company'] == 'SmartPrice']['current_share'].values[0]
        smartprice_optimized = market_share_data[market_share_data['company'] == 'SmartPrice']['optimized_share'].values[0]
        
        # Adjust based on optimization level
        share_increase = (smartprice_optimized - smartprice_current) * (optimization_level / 100)
        projected_share = smartprice_current + share_increase
        
        # Calculate revenue impact
        current_revenue = (market_size * smartprice_current / 100)
        projected_revenue = (market_size * projected_share / 100)
        revenue_increase = projected_revenue - current_revenue
        
        # Calculate competitor impact
        competitor_impact = []
        
        for _, row in market_share_data[market_share_data['company'] != 'SmartPrice'].iterrows():
            share_change = row['share_change'] * (optimization_level / 100)
            competitor_impact.append({
                'company': row['company'],
                'current_share': row['current_share'],
                'projected_share': row['current_share'] + share_change,
                'share_change': share_change
            })
        
        # Show results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Projected Results")
            
            st.markdown(f"""
            **SmartPrice Market Position:**
            - Current Market Share: {smartprice_current:.1f}%
            - Projected Market Share: {projected_share:.1f}%
            - Increase: +{share_increase:.1f} percentage points
            
            **Revenue Impact:**
            - Current Annual Revenue: ₹{current_revenue:.1f} Cr
            - Projected Annual Revenue: ₹{projected_revenue:.1f} Cr
            - Increase: +₹{revenue_increase:.1f} Cr (+{(revenue_increase/current_revenue*100):.1f}%)
            """)
        
        with col2:
            st.markdown("#### Competitor Impact")
            
            for comp in competitor_impact:
                share_color = "red" if comp['share_change'] < 0 else "green"
                st.markdown(f"""
                **{comp['company']}:**
                - Current Share: {comp['current_share']:.1f}%
                - Projected Share: {comp['projected_share']:.1f}%
                - Change: <span style='color:{share_color}'>{comp['share_change']:+.1f} pp</span>
                """, unsafe_allow_html=True)
        
        # Implementation timeline
        st.markdown("### Implementation Timeline")
        
        # Parse timeframe
        months = int(timeframe.split()[0])
        
        # Generate timeline data
        timeline_data = []
        
        for month in range(months + 1):
            implementation_pct = min(100, (month / months) * 100)
            share_gain = share_increase * (implementation_pct / 100)
            monthly_share = smartprice_current + share_gain
            
            timeline_data.append({
                'month': month,
                'implementation_pct': implementation_pct,
                'market_share': monthly_share
            })
        
        timeline_df = pd.DataFrame(timeline_data)
        
        # Create line chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(
                x=timeline_df['month'],
                y=timeline_df['market_share'],
                name="Market Share",
                line=dict(color='#2563EB', width=3)
            ),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(
                x=timeline_df['month'],
                y=timeline_df['implementation_pct'],
                name="Implementation %",
                line=dict(color='#93C5FD', width=2, dash='dot')
            ),
            secondary_y=True,
        )
        
        fig.update_layout(
            title=f"Market Share Growth Over {timeframe} Implementation",
            height=400,
            xaxis_title="Month",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        
        fig.update_yaxes(title_text="Market Share (%)", secondary_y=False)
        fig.update_yaxes(title_text="Implementation (%)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)

elif page == "Factors Dashboard":
    st.markdown("<div class='main-title'>Pricing Factors Dashboard</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    Adjust the key pricing factors to see how they influence optimal pricing recommendations.
    This interactive tool helps you understand the relative importance of different factors in your pricing strategy.
    </div>
    """, unsafe_allow_html=True)
    
    # Create base settings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        base_destination = st.selectbox(
            "Destination",
            options=sorted(sample_data['destination'].unique()),
            index=0,
            key="factors_destination"
        )
    
    with col2:
        base_product = st.selectbox(
            "Product Type",
            options=sorted(sample_data['product_type'].unique()),
            index=0,
            key="factors_product"
        )
    
    with col3:
        base_price = st.number_input(
            "Base Price (₹)",
            min_value=1000,
            max_value=100000,
            value=15000,
            step=500,
            key="factors_base_price"
        )
    
    # Create interactive sliders for factors
    st.markdown("<div class='sub-title'>Pricing Factors</div>", unsafe_allow_html=True)
    
    factor_cols = st.columns(2)
    
    # Define the factors and their ranges
    factors = {
        "seasonality": {"label": "Seasonality", "min": 1, "max": 10, "default": 5, "step": 1, "col": 0,
                      "help": "Impact of seasonal demand (1: Off-season, 10: Peak season)"},
        "competitor_price": {"label": "Competitor Pricing (₹)", "min": 5000, "max": 50000, "default": 15000, "step": 500, "col": 1,
                           "help": "Average price offered by competitors for similar products"},
        "demand_level": {"label": "Current Demand", "min": 1, "max": 10, "default": 5, "step": 1, "col": 0,
                       "help": "Current booking velocity and demand (1: Very low, 10: Very high)"},
        "booking_leadtime": {"label": "Booking Lead Time (days)", "min": 1, "max": 365, "default": 30, "step": 1, "col": 1,
                           "help": "Number of days before travel date"},
        "is_weekend": {"label": "Weekend Travel", "min": 0, "max": 1, "default": 0, "step": 1, "col": 0,
                     "help": "Whether the travel date falls on a weekend (0: No, 1: Yes)"},
        "is_holiday": {"label": "Holiday/Festival Period", "min": 0, "max": 1, "default": 0, "step": 1, "col": 1,
                     "help": "Whether the travel date coincides with holidays or festivals (0: No, 1: Yes)"},
        "market_condition": {"label": "Market Conditions", "min": 1, "max": 10, "default": 5, "step": 1, "col": 0,
                           "help": "General market conditions (1: Poor, 10: Excellent)"},
        "historical_performance": {"label": "Historical Performance", "min": 1, "max": 10, "default": 5, "step": 1, "col": 1,
                                 "help": "How well this product has performed historically (1: Poor, 10: Excellent)"},
        "month": {"label": "Month (1-12)", "min": 1, "max": 12, "default": datetime.datetime.now().month, "step": 1, "col": 0,
                "help": "Month of travel (1: January, 12: December)"}
    }
    
    # Create sliders and store values
    factor_values = {}
    
    for factor, config in factors.items():
        with factor_cols[config["col"]]:
            if factor in ["is_weekend", "is_holiday"]:
                factor_values[factor] = st.checkbox(
                    config["label"],
                    value=bool(config["default"]),
                    help=config["help"]
                )
                factor_values[factor] = 1 if factor_values[factor] else 0
            else:
                factor_values[factor] = st.slider(
                    config["label"],
                    min_value=config["min"],
                    max_value=config["max"],
                    value=config["default"],
                    step=config["step"],
                    help=config["help"]
                )
    
    # Calculate optimal price based on factors
    optimal_price = predict_price(model, scaler, model_features, factor_values)
    
    # Calculate price difference
    price_diff = optimal_price - base_price
    price_diff_pct = (price_diff / base_price) * 100
    
    # Show results
    st.markdown("<div class='sub-title'>Price Recommendation</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        metric_color = "positive-delta" if price_diff > 0 else "negative-delta"
        price_sign = "+" if price_diff > 0 else ""
        
        st.markdown(f"""
        <div style='background-color: #F3F4F6; padding: 1rem; border-radius: 10px;'>
            <div style='font-size: 1.1rem; color: #4B5563;'>Base Price</div>
            <div style='font-size: 1.8rem; font-weight: 700; color: #1E3A8A;'>₹{base_price:,.0f}</div>
            <div style='height: 10px;'></div>
            <div style='font-size: 1.1rem; color: #4B5563;'>Optimal Price</div>
            <div style='font-size: 1.8rem; font-weight: 700; color: #1E3A8A;'>₹{optimal_price:,.0f}</div>
            <div style='font-size: 1.1rem; font-weight: 500;' class='{metric_color}'>
                {price_sign}₹{abs(price_diff):,.0f} ({price_sign}{price_diff_pct:.1f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Calculate confidence score
        confidence_score = calculate_confidence(factor_values)
        
        # Determine color based on confidence score
        if confidence_score >= 90:
            gauge_color = "#10B981"  # Green
        elif confidence_score >= 75:
            gauge_color = "#FBBF24"  # Yellow
        else:
            gauge_color = "#EF4444"  # Red
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence Score"},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 75], 'color': "#FECACA"},
                    {'range': [75, 90], 'color': "#FEF3C7"},
                    {'range': [90, 100], 'color': "#D1FAE5"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(
            height=200,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Factor importance visualization
    st.markdown("<div class='sub-title'>Factor Importance</div>", unsafe_allow_html=True)
    
    # Use random forest feature importance as proxy
    feature_importance = pd.DataFrame({
        'Factor': model_features,
        'Importance': model.feature_importances_
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Map feature names to readable labels
    feature_importance['Factor'] = feature_importance['Factor'].map({
        'seasonality': 'Seasonality',
        'competitor_price': 'Competitor Price',
        'demand_level': 'Demand Level',
        'booking_leadtime': 'Booking Lead Time',
        'is_weekend': 'Weekend Travel',
        'is_holiday': 'Holiday Period',
        'market_condition': 'Market Conditions',
        'historical_performance': 'Historical Performance',
        'month': 'Month of Travel'
    })
    
    # Create horizontal bar chart
    fig = px.bar(
        feature_importance,
        y='Factor',
        x='Importance',
        orientation='h',
        title="Relative Importance of Pricing Factors",
        color='Importance',
        color_continuous_scale='Blues',
        text=feature_importance['Importance'].apply(lambda x: f"{x:.3f}")
    )
    
    fig.update_traces(textposition='outside')
    
    fig.update_layout(
        height=400,
        xaxis_title="Relative Importance",
        yaxis_title="",
        coloraxis_showscale=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Factor sensitivity analysis
    st.markdown("<div class='sub-title'>Factor Sensitivity Analysis</div>", unsafe_allow_html=True)
    
    # Select factor to analyze
    factor_to_analyze = st.selectbox(
        "Select Factor to Analyze",
        options=[config["label"] for factor, config in factors.items()],
        index=0
    )
    
    # Map label back to factor key
    factor_map = {config["label"]: factor for factor, config in factors.items()}
    factor_key = factor_map[factor_to_analyze]
    
    # Generate sensitivity data
    sensitivity_data = []
    
    # Get factor config
    factor_config = factors[factor_key]
    
    # Create range of values to test
    if factor_key in ["is_weekend", "is_holiday"]:
        test_values = [0, 1]
    else:
        step_size = max(1, (factor_config["max"] - factor_config["min"]) // 10)
        test_values = range(factor_config["min"], factor_config["max"] + 1, step_size)
    
    # Calculate prices for each test value
    for test_value in test_values:
        # Create a copy of current factor values
        test_factors = factor_values.copy()
        
        # Override the factor being tested
        test_factors[factor_key] = test_value
        
        # Calculate optimal price
        test_optimal_price = predict_price(model, scaler, model_features, test_factors)
        
        # Calculate percentage change from base price
        price_change_pct = ((test_optimal_price - base_price) / base_price) * 100
        
        sensitivity_data.append({
            'Factor_Value': test_value,
            'Optimal_Price': test_optimal_price,
            'Price_Change_Pct': price_change_pct
        })
    
    sensitivity_df = pd.DataFrame(sensitivity_data)
    
    # Create line chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=sensitivity_df['Factor_Value'],
            y=sensitivity_df['Optimal_Price'],
            mode='lines+markers',
            name="Optimal Price",
            line=dict(color='#2563EB', width=3),
            marker=dict(size=8)
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=sensitivity_df['Factor_Value'],
            y=sensitivity_df['Price_Change_Pct'],
            mode='lines+markers',
            name="% Change from Base",
            line=dict(color='#EF4444', width=2, dash='dot'),
            marker=dict(size=8)
        ),
        secondary_y=True,
    )
    
    # Add a horizontal line at 0% change
    fig.add_trace(
        go.Scatter(
            x=[sensitivity_df['Factor_Value'].min(), sensitivity_df['Factor_Value'].max()],
            y=[0, 0],
            mode='lines',
            name="No Change",
            line=dict(color='#9CA3AF', width=1, dash='dash')
        ),
        secondary_y=True,
    )
    
    # Mark the current value
    fig.add_trace(
        go.Scatter(
            x=[factor_values[factor_key]],
            y=[optimal_price],
            mode='markers',
            name="Current Setting",
            marker=dict(color='#10B981', size=12, symbol='diamond')
        ),
        secondary_y=False,
    )
    
    fig.update_layout(
        title=f"Price Sensitivity to {factor_to_analyze}",
        height=500,
        xaxis_title=factor_to_analyze,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    fig.update_yaxes(title_text="Optimal Price (₹)", secondary_y=False)
    fig.update_yaxes(title_text="% Change from Base Price", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Factor interaction analysis
    st.markdown("<div class='sub-title'>Factor Interaction Analysis</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        factor1 = st.selectbox(
            "Select First Factor",
            options=[config["label"] for factor, config in factors.items() if factor not in ["is_weekend", "is_holiday"]],
            index=0,
            key="factor1"
        )
    
    with col2:
        factor2 = st.selectbox(
            "Select Second Factor",
            options=[config["label"] for factor, config in factors.items() if factor not in ["is_weekend", "is_holiday"]],
            index=1,
            key="factor2"
        )
    
    # Map labels back to factor keys
    factor1_key = factor_map[factor1]
    factor2_key = factor_map[factor2]
    
    # Get factor configs
    factor1_config = factors[factor1_key]
    factor2_config = factors[factor2_key]
    
    # Create ranges of values to test
    step_size1 = max(1, (factor1_config["max"] - factor1_config["min"]) // 5)
    step_size2 = max(1, (factor2_config["max"] - factor2_config["min"]) // 5)
    
    test_values1 = list(range(factor1_config["min"], factor1_config["max"] + 1, step_size1))
    test_values2 = list(range(factor2_config["min"], factor2_config["max"] + 1, step_size2))
    
    # Generate interaction data
    interaction_data = []
    
    for val1 in test_values1:
        for val2 in test_values2:
            # Create a copy of current factor values
            test_factors = factor_values.copy()
            
            # Override the factors being tested
            test_factors[factor1_key] = val1
            test_factors[factor2_key] = val2
            
            # Calculate optimal price
            test_optimal_price = predict_price(model, scaler, model_features, test_factors)
            
            # Calculate percentage change from base price
            price_change_pct = ((test_optimal_price - base_price) / base_price) * 100
            
            interaction_data.append({
                'Factor1': val1,
                'Factor2': val2,
                'Optimal_Price': test_optimal_price,
                'Price_Change_Pct': price_change_pct
            })
    
    interaction_df = pd.DataFrame(interaction_data)
    
    # Create heatmap
    fig = px.density_heatmap(
        interaction_df,
        x='Factor1',
        y='Factor2',
        z='Price_Change_Pct',
        title=f"Price Change (%) Interaction: {factor1} vs {factor2}",
        labels={'Factor1': factor1, 'Factor2': factor2, 'Price_Change_Pct': 'Price Change (%)'},
        color_continuous_scale='RdBu_r',
        range_color=[-20, 20]
    )
    
    # Mark the current values
    fig.add_trace(
        go.Scatter(
            x=[factor_values[factor1_key]],
            y=[factor_values[factor2_key]],
            mode='markers',
            name="Current Setting",
            marker=dict(color='#10B981', size=12, symbol='star')
        )
    )
    
    fig.update_layout(
        height=500,
        xaxis_title=factor1,
        yaxis_title=factor2
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif page == "Analytics & Reporting":
    st.markdown("<div class='main-title'>Analytics & Reporting</div>", unsafe_allow_html=True)
    
    # Create tabs for different reports
    tab1, tab2, tab3, tab4 = st.tabs([
        "Revenue Optimization", "Pricing Decision History", 
        "Performance Metrics", "Export Reports"
    ])
    
    with tab1:
        st.markdown("<div class='sub-title'>Revenue Optimization Analysis</div>", unsafe_allow_html=True)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rev_dest_filter = st.selectbox(
                "Destination",
                options=['All'] + sorted(sample_data['destination'].unique().tolist()),
                key="rev_dest_filter"
            )
        
        with col2:
            rev_product_filter = st.selectbox(
                "Product Type",
                options=['All'] + sorted(sample_data['product_type'].unique().tolist()),
                key="rev_product_filter"
            )
        
        with col3:
            rev_date_range = st.date_input(
                "Date Range",
                value=(datetime.datetime.now().date() - timedelta(days=90), datetime.datetime.now().date()),
                key="rev_date_range"
            )
        
        # Filter data
        filtered_data = sample_data.copy()
        
        if rev_dest_filter != 'All':
            filtered_data = filtered_data[filtered_data['destination'] == rev_dest_filter]
        
        if rev_product_filter != 'All':
            filtered_data = filtered_data[filtered_data['product_type'] == rev_product_filter]
        
        if len(rev_date_range) == 2:
            filtered_data = filtered_data[
                (filtered_data['date'].dt.date >= rev_date_range[0]) & 
                (filtered_data['date'].dt.date <= rev_date_range[1])
            ]
        
        # Revenue impact over time
        st.markdown("### Revenue Impact Over Time")
        
        # Group by date
        revenue_by_date = filtered_data.groupby(filtered_data['date'].dt.date).agg({
            'revenue_original': 'sum',
            'revenue_optimized': 'sum',
            'revenue_impact': 'sum'
        }).reset_index()
        
        # Calculate cumulative impact
        revenue_by_date['cumulative_impact'] = revenue_by_date['revenue_impact'].cumsum()
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add daily revenue as bars
        fig.add_trace(
            go.Bar(
                x=revenue_by_date['date'],
                y=revenue_by_date['revenue_impact'],
                name="Daily Impact",
                marker_color='#93C5FD'
            ),
            secondary_y=False,
        )
        
        # Add cumulative impact as line
        fig.add_trace(
            go.Scatter(
                x=revenue_by_date['date'],
                y=revenue_by_date['cumulative_impact'],
                name="Cumulative Impact",
                line=dict(color='#2563EB', width=3)
            ),
            secondary_y=True,
        )
        
        # Add original vs optimized revenue lines
        fig.add_trace(
            go.Scatter(
                x=revenue_by_date['date'],
                y=revenue_by_date['revenue_original'],
                name="Original Revenue",
                line=dict(color='#9CA3AF', width=2, dash='dot')
            ),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(
                x=revenue_by_date['date'],
                y=revenue_by_date['revenue_optimized'],
                name="Optimized Revenue",
                line=dict(color='#10B981', width=2, dash='dot')
            ),
            secondary_y=False,
        )
        
        # Set axis titles
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Daily Revenue (₹)", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative Impact (₹)", secondary_y=True)
        
        fig.update_layout(
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Revenue impact by destination and product
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Revenue Impact by Destination")
            
            # Group by destination
            revenue_by_dest = filtered_data.groupby('destination').agg({
                'revenue_original': 'sum',
                'revenue_optimized': 'sum',
                'revenue_impact': 'sum'
            }).reset_index()
            
            # Calculate percentage improvement
            revenue_by_dest['improvement_pct'] = (revenue_by_dest['revenue_impact'] / revenue_by_dest['revenue_original']) * 100
            
            # Sort by revenue impact
            revenue_by_dest = revenue_by_dest.sort_values('revenue_impact', ascending=False)
            
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add revenue impact as bars
            fig.add_trace(
                go.Bar(
                    x=revenue_by_dest['destination'],
                    y=revenue_by_dest['revenue_impact'],
                    name="Revenue Impact",
                    marker_color='#93C5FD'
                ),
                secondary_y=False,
            )
            
            # Add percentage improvement as line
            fig.add_trace(
                go.Scatter(
                    x=revenue_by_dest['destination'],
                    y=revenue_by_dest['improvement_pct'],
                    name="% Improvement",
                    mode='markers+lines',
                    marker=dict(size=8, color='#EF4444'),
                    line=dict(color='#EF4444', width=2)
                ),
                secondary_y=True,
            )
            
            # Set axis titles
            fig.update_xaxes(title_text="Destination")
            fig.update_yaxes(title_text="Revenue Impact (₹)", secondary_y=False)
            fig.update_yaxes(title_text="Improvement (%)", secondary_y=True)
            
            fig.update_layout(
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis={'categoryorder':'total descending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Revenue Impact by Product Type")
            
            # Group by product type
            revenue_by_product = filtered_data.groupby('product_type').agg({
                'revenue_original': 'sum',
                'revenue_optimized': 'sum',
                'revenue_impact': 'sum'
            }).reset_index()
            
            # Calculate percentage improvement
            revenue_by_product['improvement_pct'] = (revenue_by_product['revenue_impact'] / revenue_by_product['revenue_original']) * 100
            
            # Sort by revenue impact
            revenue_by_product = revenue_by_product.sort_values('revenue_impact', ascending=False)
            
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add revenue impact as bars
            fig.add_trace(
                go.Bar(
                    x=revenue_by_product['product_type'],
                    y=revenue_by_product['revenue_impact'],
                    name="Revenue Impact",
                    marker_color='#93C5FD'
                ),
                secondary_y=False,
            )
            
            # Add percentage improvement as line
            fig.add_trace(
                go.Scatter(
                    x=revenue_by_product['product_type'],
                    y=revenue_by_product['improvement_pct'],
                    name="% Improvement",
                    mode='markers+lines',
                    marker=dict(size=8, color='#EF4444'),
                    line=dict(color='#EF4444', width=2)
                ),
                secondary_y=True,
            )
            
            # Set axis titles
            fig.update_xaxes(title_text="Product Type")
            fig.update_yaxes(title_text="Revenue Impact (₹)", secondary_y=False)
            fig.update_yaxes(title_text="Improvement (%)", secondary_y=True)
            
            fig.update_layout(
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis={'categoryorder':'total descending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Revenue optimization summary
        st.markdown("### Revenue Optimization Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Calculate total values
            total_original = filtered_data['revenue_original'].sum()
            total_optimized = filtered_data['revenue_optimized'].sum()
            total_impact = filtered_data['revenue_impact'].sum()
            
            # Calculate percentage improvement
            improvement_pct = (total_impact / total_original) * 100
            
            st.markdown(f"""
            **Revenue Summary:**
            - Original Revenue: ₹{total_original:,.0f}
            - Optimized Revenue: ₹{total_optimized:,.0f}
            - Total Impact: ₹{total_impact:,.0f} (+{improvement_pct:.1f}%)
            """)
        
        with col2:
            # Calculate booking metrics
            total_bookings_original = filtered_data['bookings'].sum()
            total_bookings_optimized = filtered_data['bookings_after_optimization'].sum()
            bookings_change = total_bookings_optimized - total_bookings_original
            bookings_change_pct = (bookings_change / total_bookings_original) * 100
            
            st.markdown(f"""
            **Booking Metrics:**
            - Original Bookings: {total_bookings_original:,.0f}
            - Optimized Bookings: {total_bookings_optimized:,.0f}
            - Change: {bookings_change:+,.0f} ({bookings_change_pct:+.1f}%)
            """)
        
        with col3:
            # Calculate price metrics
            avg_price_original = filtered_data['original_price'].mean()
            avg_price_optimized = filtered_data['optimal_price'].mean()
            price_change = avg_price_optimized - avg_price_original
            price_change_pct = (price_change / avg_price_original) * 100
            
            st.markdown(f"""
            **Average Price Metrics:**
            - Original Price: ₹{avg_price_original:,.0f}
            - Optimized Price: ₹{avg_price_optimized:,.0f}
            - Change: ₹{price_change:+,.0f} ({price_change_pct:+.1f}%)
            """)
    
    with tab2:
        st.markdown("<div class='sub-title'>Pricing Decision History</div>", unsafe_allow_html=True)
        
        if not st.session_state.pricing_history:
            st.info("No pricing decisions have been made yet. Use the Price Optimization tool to generate pricing decisions.")
        else:
            # Convert pricing history to DataFrame
            history_df = pd.DataFrame(st.session_state.pricing_history)
            
            # Sort by most recent first
            history_df = history_df.sort_values('date', ascending=False)
            
            # Add filters
            col1, col2 = st.columns(2)
            
            with col1:
                hist_dest_filter = st.multiselect(
                    "Filter by Destination",
                    options=sorted(history_df['destination'].unique()),
                    default=[]
                )
            
            with col2:
                hist_product_filter = st.multiselect(
                    "Filter by Product Type",
                    options=sorted(history_df['product_type'].unique()),
                    default=[]
                )
            
            # Apply filters
            filtered_history = history_df.copy()
            
            if hist_dest_filter:
                filtered_history = filtered_history[filtered_history['destination'].isin(hist_dest_filter)]
            
            if hist_product_filter:
                filtered_history = filtered_history[filtered_history['product_type'].isin(hist_product_filter)]
            
            # Show history table
            st.markdown("### Pricing Decisions")
            
            # Create table visualization
            decision_table = filtered_history[['date', 'destination', 'product_type', 'original_price', 'optimal_price', 'confidence_score', 'revenue_impact']]
            
            # Calculate price change percentage
            decision_table['price_change_pct'] = ((filtered_history['optimal_price'] - filtered_history['original_price']) / 
                                                filtered_history['original_price'] * 100)
            
            # Format the table
            table_data = {
                'Date': decision_table['date'],
                'Destination': decision_table['destination'],
                'Product': decision_table['product_type'],
                'Original Price': ['₹{:,.0f}'.format(x) for x in decision_table['original_price']],
                'Optimal Price': ['₹{:,.0f}'.format(x) for x in decision_table['optimal_price']],
                'Change': ['{:+.1f}%'.format(x) for x in decision_table['price_change_pct']],
                'Confidence': ['{:.0f}%'.format(x) for x in decision_table['confidence_score']],
                'Revenue Impact': ['₹{:+,.0f}'.format(x) for x in decision_table['revenue_impact']]
            }
            
            # Create color conditions for the change column
            change_colors = []
            for x in decision_table['price_change_pct']:
                if x > 0:
                    change_colors.append('#DCFCE7')  # Green for price increases
                elif x < 0:
                    change_colors.append('#FEE2E2')  # Red for price decreases
                else:
                    change_colors.append('#F3F4F6')  # Grey for no change
            
            # Create the table
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=list(table_data.keys()),
                    fill_color='#BFDBFE',
                    align='left',
                    font=dict(color='#1E3A8A', size=12)
                ),
                cells=dict(
                    values=list(table_data.values()),
                    fill_color=[
                        ['#F3F4F6'] * len(decision_table),  # Date
                        ['#F3F4F6'] * len(decision_table),  # Destination
                        ['#F3F4F6'] * len(decision_table),  # Product
                        ['#F3F4F6'] * len(decision_table),  # Original Price
                        ['#F3F4F6'] * len(decision_table),  # Optimal Price
                        [change_colors],  # Change
                        ['#F3F4F6'] * len(decision_table),  # Confidence
                        ['#F3F4F6'] * len(decision_table)   # Revenue Impact
                    ],
                    align='left',
                    font=dict(color='#374151', size=11)
                ))
            ])
            
            fig.update_layout(
                height=500,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Decision metrics
            st.markdown("### Decision Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_decisions = len(filtered_history)
                increases = len(filtered_history[filtered_history['optimal_price'] > filtered_history['original_price']])
                decreases = len(filtered_history[filtered_history['optimal_price'] < filtered_history['original_price']])
                
                # Create donut chart
                fig = go.Figure(data=[go.Pie(
                    labels=['Price Increases', 'Price Decreases', 'No Change'],
                    values=[increases, decreases, total_decisions - increases - decreases],
                    hole=.5,
                    marker_colors=['#10B981', '#EF4444', '#9CA3AF']
                )])
                
                fig.update_layout(
                    title="Price Decision Types",
                    height=300,
                    margin=dict(l=10, r=10, t=40, b=10),
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Calculate average metrics
                avg_confidence = filtered_history['confidence_score'].mean()
                avg_price_change = ((filtered_history['optimal_price'] - filtered_history['original_price']) / 
                                  filtered_history['original_price'] * 100).mean()
                avg_revenue_impact = filtered_history['revenue_impact'].mean()
                
                st.markdown(f"""
                **Average Decision Metrics:**
                
                - **Number of Decisions:** {total_decisions}
                - **Average Confidence:** {avg_confidence:.1f}%
                - **Average Price Change:** {avg_price_change:+.1f}%
                - **Average Revenue Impact:** ₹{avg_revenue_impact:+,.0f}
                """)
                
                # Decision effectiveness
                total_revenue_impact = filtered_history['revenue_impact'].sum()
                positive_impact = len(filtered_history[filtered_history['revenue_impact'] > 0])
                positive_impact_pct = (positive_impact / max(1, total_decisions)) * 100
                
                st.markdown(f"""
                **Decision Effectiveness:**
                
                - **Total Revenue Impact:** ₹{total_revenue_impact:+,.0f}
                - **Positive Impact Decisions:** {positive_impact} ({positive_impact_pct:.1f}%)
                """)
            
            with col3:
                # Decision timeline
                filtered_history['month'] = pd.to_datetime(filtered_history['date']).dt.strftime('%b')
                decision_by_month = filtered_history.groupby('month').size().reset_index()
                decision_by_month.columns = ['month', 'count']
                
                # Sort months chronologically
                month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                decision_by_month['month_num'] = decision_by_month['month'].apply(lambda x: month_order.index(x))
                decision_by_month = decision_by_month.sort_values('month_num')
                
                # Create bar chart
                fig = px.bar(
                    decision_by_month,
                    x='month',
                    y='count',
                    title="Decisions by Month",
                    color='count',
                    color_continuous_scale='Blues',
                    category_orders={"month": month_order}
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=40, b=10),
                    xaxis_title="",
                    yaxis_title="Number of Decisions",
                    coloraxis_showscale=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("<div class='sub-title'>Performance Metrics</div>", unsafe_allow_html=True)
        
        # KPI summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.markdown("<div class='metric-title'>Optimization Accuracy</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-value'>87%</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-delta positive-delta'>+2.1% from last quarter</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.markdown("<div class='metric-title'>Revenue Improvement</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-value'>12.3%</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-delta positive-delta'>+1.5% from last quarter</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.markdown("<div class='metric-title'>Avg. Confidence Score</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-value'>84%</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-delta positive-delta'>+3.2% from last quarter</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col4:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.markdown("<div class='metric-title'>Successful Optimizations</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-value'>91%</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-delta positive-delta'>+4.5% from last quarter</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Performance trends
        st.markdown("### Performance Trends")
        
        # Generate performance trend data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        current_month = datetime.datetime.now().month
        
        # Select last 12 months
        trend_months = []
        for i in range(12):
            month_index = (current_month - 12 + i) % 12
            trend_months.append(months[month_index])
        
        # Create trend data
        np.random.seed(42)
        accuracy_trend = [85 + i*0.2 + np.random.uniform(-1, 1) for i in range(12)]
        revenue_improvement_trend = [10 + i*0.2 + np.random.uniform(-1, 1) for i in range(12)]
        confidence_trend = [80 + i*0.3 + np.random.uniform(-1, 1) for i in range(12)]
        success_rate_trend = [86 + i*0.4 + np.random.uniform(-1, 1) for i in range(12)]
        
        # Create a DataFrame for the trends
        trend_data = pd.DataFrame({
            'Month': trend_months,
            'Accuracy': accuracy_trend,
            'Revenue Improvement': revenue_improvement_trend,
            'Confidence Score': confidence_trend,
            'Success Rate': success_rate_trend
        })
        
        # Create line chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=trend_data['Month'],
            y=trend_data['Accuracy'],
            mode='lines+markers',
            name='Accuracy (%)',
            line=dict(color='#2563EB', width=2),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=trend_data['Month'],
            y=trend_data['Revenue Improvement'],
            mode='lines+markers',
            name='Revenue Improvement (%)',
            line=dict(color='#10B981', width=2),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=trend_data['Month'],
            y=trend_data['Confidence Score'],
            mode='lines+markers',
            name='Confidence Score (%)',
            line=dict(color='#FBBF24', width=2),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=trend_data['Month'],
            y=trend_data['Success Rate'],
            mode='lines+markers',
            name='Success Rate (%)',
            line=dict(color='#EF4444', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            height=500,
            xaxis_title="Month",
            yaxis_title="Performance (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance by destination and product
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Performance by Destination")
            
            # Generate performance by destination data
            destinations = sorted(sample_data['destination'].unique())[:10]
            
            destination_performance = []
            for dest in destinations:
                destination_performance.append({
                    'Destination': dest,
                    'Accuracy': 80 + np.random.uniform(0, 15),
                    'Revenue Improvement': 8 + np.random.uniform(0, 10),
                    'Success Rate': 85 + np.random.uniform(0, 10)
                })
            
            dest_perf_df = pd.DataFrame(destination_performance)
            
            # Sort by revenue improvement
            dest_perf_df = dest_perf_df.sort_values('Revenue Improvement', ascending=False)
            
            # Create table visualization
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=['Destination', 'Accuracy (%)', 'Revenue Improvement (%)', 'Success Rate (%)'],
                    fill_color='#BFDBFE',
                    align='left',
                    font=dict(color='#1E3A8A', size=12)
                ),
                cells=dict(
                    values=[
                        dest_perf_df['Destination'],
                        [f"{x:.1f}%" for x in dest_perf_df['Accuracy']],
                        [f"{x:.1f}%" for x in dest_perf_df['Revenue Improvement']],
                        [f"{x:.1f}%" for x in dest_perf_df['Success Rate']]
                    ],
                    fill_color='#F3F4F6',
                    align='left',
                    font=dict(color='#374151', size=11)
                ))
            ])
            
            fig.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Performance by Product Type")
            
            # Generate performance by product type data
            product_types = sorted(sample_data['product_type'].unique())
            
            product_performance = []
            for prod in product_types:
                product_performance.append({
                    'Product Type': prod,
                    'Accuracy': 80 + np.random.uniform(0, 15),
                    'Revenue Improvement': 8 + np.random.uniform(0, 10),
                    'Success Rate': 85 + np.random.uniform(0, 10)
                })
            
            prod_perf_df = pd.DataFrame(product_performance)
            
            # Sort by revenue improvement
            prod_perf_df = prod_perf_df.sort_values('Revenue Improvement', ascending=False)
            
            # Create spider chart
            categories = ['Accuracy', 'Revenue Improvement', 'Success Rate']
            
            fig = go.Figure()
            
            for i, product in enumerate(prod_perf_df['Product Type']):
                values = prod_perf_df.loc[prod_perf_df['Product Type'] == product, categories].values.flatten().tolist()
                # Normalize values for radar chart
                normalized_values = [v / max(prod_perf_df[cat]) * 100 for v, cat in zip(values, categories)]
                # Add the first value again to close the loop
                values_plot = normalized_values + [normalized_values[0]]
                
                fig.add_trace(go.Scatterpolar(
                    r=values_plot,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=product
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=True,
                height=400,
                margin=dict(l=30, r=30, t=30, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Model performance metrics
        st.markdown("### Model Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Generate model performance data
            feature_names = [
                'Seasonality', 'Competitor Price', 'Demand Level', 'Booking Lead Time',
                'Weekend Travel', 'Holiday Period', 'Market Conditions', 'Historical Performance', 'Month'
            ]
            
            feature_metrics = []
            for feature in feature_names:
                feature_metrics.append({
                    'Feature': feature,
                    'Importance': np.random.uniform(0.05, 0.2),
                    'Impact on Accuracy': np.random.uniform(70, 95)
                })
            
            feature_df = pd.DataFrame(feature_metrics)
            feature_df = feature_df.sort_values('Importance', ascending=False)
            
            # Create horizontal bar chart
            fig = px.bar(
                feature_df,
                y='Feature',
                x='Importance',
                orientation='h',
                title="Feature Importance",
                color='Importance',
                color_continuous_scale='Blues',
                text=feature_df['Importance'].apply(lambda x: f"{x:.3f}")
            )
            
            fig.update_traces(textposition='outside')
            
            fig.update_layout(
                height=400,
                xaxis_title="Relative Importance",
                yaxis_title="",
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Generate error distribution data
            error_data = np.random.normal(0, 5, 1000)
            
            # Create histogram
            fig = px.histogram(
                error_data,
                title="Pricing Error Distribution",
                labels={'value': 'Pricing Error (%)', 'count': 'Frequency'},
                color_discrete_sequence=['#93C5FD']
            )
            
            # Add a vertical line at zero
            fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="#2563EB")
            
            # Add a normal distribution curve
            x_range = np.linspace(min(error_data), max(error_data), 100)
            y_range = np.exp(-(x_range**2) / (2 * 5**2)) / (5 * np.sqrt(2 * np.pi)) * len(error_data) * (max(error_data) - min(error_data)) / 30
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_range,
                mode='lines',
                line=dict(color='#EF4444', width=2),
                name='Normal Distribution'
            ))
            
            fig.update_layout(
                height=400,
                xaxis_title="Pricing Error (%)",
                yaxis_title="Frequency",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("<div class='sub-title'>Export Reports</div>", unsafe_allow_html=True)
        
        # Select report type
        report_type = st.selectbox(
            "Select Report Type",
            options=[
                "Revenue Optimization Summary",
                "Pricing Decision History",
                "Performance Metrics",
                "Destination Analysis",
                "Product Type Analysis",
                "Seasonal Trends",
                "Competitor Analysis",
                "Custom Report"
            ]
        )
        
        # Report parameters
        col1, col2 = st.columns(2)
        
        with col1:
            export_dest = st.multiselect(
                "Destinations",
                options=['All'] + sorted(sample_data['destination'].unique().tolist()),
                default=['All']
            )
        
        with col2:
            export_product = st.multiselect(
                "Product Types",
                options=['All'] + sorted(sample_data['product_type'].unique().tolist()),
                default=['All']
            )
        
        # Date range selector
        export_date_range = st.date_input(
            "Date Range",
            value=(datetime.datetime.now().date() - timedelta(days=90), datetime.datetime.now().date()),
            key="export_date_range"
        )
        
        # Additional parameters based on report type
        if report_type == "Custom Report":
            st.markdown("### Select Metrics to Include")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                include_revenue = st.checkbox("Revenue Metrics", value=True)
                include_bookings = st.checkbox("Booking Metrics", value=True)
            
            with col2:
                include_pricing = st.checkbox("Pricing Metrics", value=True)
                include_performance = st.checkbox("Performance Metrics", value=True)
            
            with col3:
                include_competitors = st.checkbox("Competitor Analysis", value=False)
                include_seasonal = st.checkbox("Seasonal Analysis", value=False)
        
        # Generate report preview
        st.markdown("### Report Preview")
        
        # Filter data for report
        filtered_data = sample_data.copy()
        
        if 'All' not in export_dest:
            filtered_data = filtered_data[filtered_data['destination'].isin(export_dest)]
        
        if 'All' not in export_product:
            filtered_data = filtered_data[filtered_data['product_type'].isin(export_product)]
        
        if len(export_date_range) == 2:
            filtered_data = filtered_data[
                (filtered_data['date'].dt.date >= export_date_range[0]) & 
                (filtered_data['date'].dt.date <= export_date_range[1])
            ]
        
        # Generate report based on type
        if report_type == "Revenue Optimization Summary":
            # Group by destination and product type
            report_data = filtered_data.groupby(['destination', 'product_type']).agg({
                'original_price': 'mean',
                'optimal_price': 'mean',
                'bookings': 'sum',
                'bookings_after_optimization': 'sum',
                'revenue_original': 'sum',
                'revenue_optimized': 'sum',
                'revenue_impact': 'sum'
            }).reset_index()
            
            # Calculate percentage changes
            report_data['price_change_pct'] = ((report_data['optimal_price'] - report_data['original_price']) / 
                                             report_data['original_price'] * 100)
            report_data['bookings_change_pct'] = ((report_data['bookings_after_optimization'] - report_data['bookings']) / 
                                                report_data['bookings'] * 100)
            report_data['revenue_change_pct'] = ((report_data['revenue_optimized'] - report_data['revenue_original']) / 
                                               report_data['revenue_original'] * 100)
            
            # Sort by revenue impact
            report_data = report_data.sort_values('revenue_impact', ascending=False)
            
            # Create table visualization
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=[
                        'Destination', 'Product Type', 'Avg. Original Price', 'Avg. Optimal Price', 'Price Change', 
                        'Original Bookings', 'Optimized Bookings', 'Booking Change',
                        'Original Revenue', 'Optimized Revenue', 'Revenue Impact'
                    ],
                    fill_color='#BFDBFE',
                    align='left',
                    font=dict(color='#1E3A8A', size=12)
                ),
                cells=dict(
                    values=[
                        report_data['destination'],
                        report_data['product_type'],
                        ['₹{:,.0f}'.format(x) for x in report_data['original_price']],
                        ['₹{:,.0f}'.format(x) for x in report_data['optimal_price']],
                        ['{:+.1f}%'.format(x) for x in report_data['price_change_pct']],
                        ['{:,.0f}'.format(x) for x in report_data['bookings']],
                        ['{:,.0f}'.format(x) for x in report_data['bookings_after_optimization']],
                        ['{:+.1f}%'.format(x) for x in report_data['bookings_change_pct']],
                        ['₹{:,.0f}'.format(x) for x in report_data['revenue_original']],
                        ['₹{:,.0f}'.format(x) for x in report_data['revenue_optimized']],
                        ['₹{:+,.0f}'.format(x) for x in report_data['revenue_impact']]
                    ],
                    fill_color='#F3F4F6',
                    align='left',
                    font=dict(color='#374151', size=11)
                ))
            ])
            
            fig.update_layout(
                height=500,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add summary metrics
            total_original_revenue = report_data['revenue_original'].sum()
            total_optimized_revenue = report_data['revenue_optimized'].sum()
            total_revenue_impact = report_data['revenue_impact'].sum()
            total_revenue_impact_pct = (total_revenue_impact / total_original_revenue) * 100
            
            st.markdown(f"""
            **Report Summary:**
            
            - **Total Original Revenue:** ₹{total_original_revenue:,.0f}
            - **Total Optimized Revenue:** ₹{total_optimized_revenue:,.0f}
            - **Total Revenue Impact:** ₹{total_revenue_impact:+,.0f} ({total_revenue_impact_pct:+.1f}%)
            - **Report Period:** {export_date_range[0]} to {export_date_range[1]}
            - **Generated On:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            - **User:** {st.session_state.get('username', 'Anurag02012004')}
            """)
            
            # Create CSV export
            csv = report_data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            export_filename = f"SmartPrice_Revenue_Optimization_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
            href = f'<a href="data:file/csv;base64,{b64}" download="{export_filename}" class="btn">Download CSV Report</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        elif report_type == "Pricing Decision History":
            if not st.session_state.pricing_history:
                st.info("No pricing decisions have been made yet. Use the Price Optimization tool to generate pricing decisions.")
            else:
                # Convert pricing history to DataFrame
                history_df = pd.DataFrame(st.session_state.pricing_history)
                
                # Sort by most recent first
                history_df = history_df.sort_values('date', ascending=False)
                
                # Create table visualization
                fig = go.Figure(data=[go.Table(
                    header=dict(
                        values=[
                            'Date', 'Destination', 'Product Type', 'Original Price', 'Optimal Price', 
                            'Price Change', 'Confidence Score', 'Revenue Impact'
                        ],
                        fill_color='#BFDBFE',
                        align='left',
                        font=dict(color='#1E3A8A', size=12)
                    ),
                    cells=dict(
                        values=[
                            history_df['date'],
                            history_df['destination'],
                            history_df['product_type'],
                            ['₹{:,.0f}'.format(x) for x in history_df['original_price']],
                            ['₹{:,.0f}'.format(x) for x in history_df['optimal_price']],
                            ['{:+.1f}%'.format((o-p)/p*100) for o, p in zip(history_df['optimal_price'], history_df['original_price'])],
                            ['{:.0f}%'.format(x) for x in history_df['confidence_score']],
                            ['₹{:+,.0f}'.format(x) for x in history_df['revenue_impact']]
                        ],
                        fill_color='#F3F4F6',
                        align='left',
                        font=dict(color='#374151', size=11)
                    ))
                ])
                
                fig.update_layout(
                    height=500,
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create CSV export
                csv = history_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                export_filename = f"SmartPrice_Decision_History_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
                href = f'<a href="data:file/csv;base64,{b64}" download="{export_filename}" class="btn">Download CSV Report</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        elif report_type == "Competitor Analysis":
            # Group by destination and product type
            comp_data = filtered_data.groupby(['destination', 'product_type']).agg({
                'original_price': 'mean',
                'optimal_price': 'mean',
                'CompetitorA_price': 'mean',
                'CompetitorB_price': 'mean',
                'CompetitorC_price': 'mean',
                'CompetitorD_price': 'mean'
            }).reset_index()
            
            # Calculate percentage differences
            comp_data['vs_CompetitorA'] = ((comp_data['original_price'] - comp_data['CompetitorA_price']) / comp_data['CompetitorA_price'] * 100).round(1)
            comp_data['vs_CompetitorB'] = ((comp_data['original_price'] - comp_data['CompetitorB_price']) / comp_data['CompetitorB_price'] * 100).round(1)
            comp_data['vs_CompetitorC'] = ((comp_data['original_price'] - comp_data['CompetitorC_price']) / comp_data['CompetitorC_price'] * 100).round(1)
            comp_data['vs_CompetitorD'] = ((comp_data['original_price'] - comp_data['CompetitorD_price']) / comp_data['CompetitorD_price'] * 100).round(1)
            comp_data['vs_Optimal'] = ((comp_data['original_price'] - comp_data['optimal_price']) / comp_data['optimal_price'] * 100).round(1)
            
            # Sort by destination
            comp_data = comp_data.sort_values(['destination', 'product_type'])
            
            # Create table visualization
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=[
                        'Destination', 'Product Type', 'Your Price', 'Optimal Price', 
                        'CompetitorA', 'CompetitorB', 'CompetitorC', 'CompetitorD',
                        'vs CompA', 'vs CompB', 'vs CompC', 'vs CompD', 'vs Optimal'
                    ],
                    fill_color='#BFDBFE',
                    align='left',
                    font=dict(color='#1E3A8A', size=12)
                ),
                cells=dict(
                    values=[
                        comp_data['destination'],
                        comp_data['product_type'],
                        ['₹{:,.0f}'.format(x) for x in comp_data['original_price']],
                        ['₹{:,.0f}'.format(x) for x in comp_data['optimal_price']],
                        ['₹{:,.0f}'.format(x) for x in comp_data['CompetitorA_price']],
                        ['₹{:,.0f}'.format(x) for x in comp_data['CompetitorB_price']],
                        ['₹{:,.0f}'.format(x) for x in comp_data['CompetitorC_price']],
                        ['₹{:,.0f}'.format(x) for x in comp_data['CompetitorD_price']],
                        ['{:+.1f}%'.format(x) for x in comp_data['vs_CompetitorA']],
                        ['{:+.1f}%'.format(x) for x in comp_data['vs_CompetitorB']],
                        ['{:+.1f}%'.format(x) for x in comp_data['vs_CompetitorC']],
                        ['{:+.1f}%'.format(x) for x in comp_data['vs_CompetitorD']],
                        ['{:+.1f}%'.format(x) for x in comp_data['vs_Optimal']]
                    ],
                    fill_color='#F3F4F6',
                    align='left',
                    font=dict(color='#374151', size=11)
                ))
            ])
            
            fig.update_layout(
                height=500,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create CSV export
            csv = comp_data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            export_filename = f"SmartPrice_Competitor_Analysis_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
            href = f'<a href="data:file/csv;base64,{b64}" download="{export_filename}" class="btn">Download CSV Report</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        else:
            # Generic report preview for other report types
            st.info(f"Preview for {report_type} will be generated here. Configure the parameters above and click the Download button to export the report.")
            
            # Create dummy export button
            export_filename = f"SmartPrice_{report_type.replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
            st.download_button(
                label="Download Report",
                data=filtered_data.to_csv(index=False),
                file_name=export_filename,
                mime="text/csv"
            )

elif page == "Simulation Mode":
    st.markdown("<div class='main-title'>Simulation Mode</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    Use this simulation environment to test different pricing strategies and market conditions.
    Run A/B tests, analyze different scenarios, and calculate potential ROI from pricing decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different simulations
    tab1, tab2, tab3 = st.tabs([
        "A/B Testing", "Scenario Analysis", "ROI Calculator"
    ])
    
    with tab1:
        st.markdown("<div class='sub-title'>A/B Testing Simulation</div>", unsafe_allow_html=True)
        
        # Setup A/B test
        col1, col2 = st.columns(2)
        
        with col1:
            ab_destination = st.selectbox(
                "Destination",
                options=sorted(sample_data['destination'].unique()),
                key="ab_destination"
            )
            
            ab_product = st.selectbox(
                "Product Type",
                options=sorted(sample_data['product_type'].unique()),
                key="ab_product"
            )
            
            ab_base_price = st.number_input(
                "Current Base Price (₹)",
                min_value=1000,
                max_value=100000,
                value=15000,
                step=500,
                key="ab_base_price"
            )
        
        with col2:
            ab_test_duration = st.slider(
                "Test Duration (Days)",
                min_value=7,
                max_value=90,
                value=30,
                step=1,
                key="ab_test_duration"
            )
            
            ab_daily_visitors = st.number_input(
                "Daily Visitors",
                min_value=10,
                max_value=10000,
                value=500,
                step=10,
                key="ab_daily_visitors"
            )
            
            ab_current_conv_rate = st.slider(
                "Current Conversion Rate (%)",
                min_value=0.1,
                max_value=10.0,
                value=2.5,
                step=0.1,
                key="ab_current_conv_rate"
            ) / 100
        
        # Define test variations
        st.markdown("### Define Test Variations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Control (A)**")
            control_price = ab_base_price
            st.markdown(f"Price: ₹{control_price:,.0f}")
            control_traffic = st.slider(
                "Traffic Allocation A (%)",
                min_value=10,
                max_value=90,
                value=50,
                key="control_traffic"
            )
            st.markdown(f"Conversion Rate: {ab_current_conv_rate:.2%}")
        
        with col2:
            st.markdown("**Variation (B)**")
            variation_type = st.radio(
                "Variation Type",
                options=["Price Increase", "Price Decrease"],
                key="variation_type"
            )
            
            variation_pct = st.slider(
                "Price Change (%)",
                min_value=1,
                max_value=30,
                value=10,
                key="variation_pct"
            )
            
            if variation_type == "Price Increase":
                variation_price = control_price * (1 + variation_pct/100)
            else:
                variation_price = control_price * (1 - variation_pct/100)
            
            st.markdown(f"Price: ₹{variation_price:,.0f}")
            variation_traffic = 100 - control_traffic
            st.markdown(f"Traffic Allocation B: {variation_traffic}%")
        
        with col3:
            st.markdown("**Expected Impact**")
            
            # Calculate expected conversion rate change based on price elasticity
            price_elasticity = -1.2  # Negative value, as higher price -> lower conversion
            
            price_change_pct = (variation_price - control_price) / control_price
            conv_rate_change_pct = price_change_pct * price_elasticity
            variation_conv_rate = ab_current_conv_rate * (1 + conv_rate_change_pct)
            
            # Ensure conversion rate is within reasonable bounds
            variation_conv_rate = max(0.001, min(0.2, variation_conv_rate))
            
            st.markdown(f"Est. Conversion Rate B: {variation_conv_rate:.2%}")
            
            # Calculate conversion rate difference
            conv_rate_diff = variation_conv_rate - ab_current_conv_rate
            conv_rate_diff_pct = (conv_rate_diff / ab_current_conv_rate) * 100
            
            if conv_rate_diff > 0:
                st.markdown(f"Conversion Δ: <span style='color:#10B981'>+{conv_rate_diff:.2%} ({conv_rate_diff_pct:+.1f}%)</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"Conversion Δ: <span style='color:#EF4444'>{conv_rate_diff:.2%} ({conv_rate_diff_pct:+.1f}%)</span>", unsafe_allow_html=True)
        
        # Run simulation button
        if st.button("Run A/B Test Simulation", key="run_ab_test"):
            # Create day-by-day simulation
            np.random.seed(42)
            
            simulation_data = []
            
            total_visitors = ab_daily_visitors * ab_test_duration
            control_visitors = int(total_visitors * (control_traffic / 100))
            variation_visitors = total_visitors - control_visitors
            
            # Daily fluctuation in conversion rates
            daily_fluctuation = 0.2  # 20% random fluctuation day to day
            
            for day in range(1, ab_test_duration + 1):
                # Control group daily data
                control_daily_visitors = int(control_visitors / ab_test_duration)
                control_daily_conv_rate = ab_current_conv_rate * (1 + np.random.uniform(-daily_fluctuation, daily_fluctuation))
                control_daily_conversions = int(control_daily_visitors * control_daily_conv_rate)
                control_daily_revenue = control_daily_conversions * control_price
                
                # Variation group daily data
                variation_daily_visitors = int(variation_visitors / ab_test_duration)
                variation_daily_conv_rate = variation_conv_rate * (1 + np.random.uniform(-daily_fluctuation, daily_fluctuation))
                variation_daily_conversions = int(variation_daily_visitors * variation_daily_conv_rate)
                variation_daily_revenue = variation_daily_conversions * variation_price
                
                # Add day's data to simulation
                simulation_data.append({
                    'day': day,
                    'control_visitors': control_daily_visitors,
                    'control_conversions': control_daily_conversions,
                    'control_revenue': control_daily_revenue,
                    'variation_visitors': variation_daily_visitors,
                    'variation_conversions': variation_daily_conversions,
                    'variation_revenue': variation_daily_revenue
                })
            
            # Convert to DataFrame
            sim_df = pd.DataFrame(simulation_data)
            
            # Calculate cumulative metrics
            sim_df['cum_control_visitors'] = sim_df['control_visitors'].cumsum()
            sim_df['cum_control_conversions'] = sim_df['control_conversions'].cumsum()
            sim_df['cum_control_revenue'] = sim_df['control_revenue'].cumsum()
            sim_df['cum_variation_visitors'] = sim_df['variation_visitors'].cumsum()
            sim_df['cum_variation_conversions'] = sim_df['variation_conversions'].cumsum()
            sim_df['cum_variation_revenue'] = sim_df['variation_revenue'].cumsum()
            
            # Calculate conversion rates
            sim_df['control_conv_rate'] = sim_df['cum_control_conversions'] / sim_df['cum_control_visitors']
            sim_df['variation_conv_rate'] = sim_df['cum_variation_conversions'] / sim_df['cum_variation_visitors']
            
            # Calculate relative difference
            sim_df['conv_rate_diff'] = (sim_df['variation_conv_rate'] - sim_df['control_conv_rate']) / sim_df['control_conv_rate'] * 100
            sim_df['revenue_diff'] = (sim_df['cum_variation_revenue'] - sim_df['cum_control_revenue']) / sim_df['cum_control_revenue'] * 100
            
            # Show simulation results
            st.markdown("### A/B Test Simulation Results")
            
            # Conversion trends
            st.markdown("#### Conversion Rate Over Time")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=sim_df['day'],
                y=sim_df['control_conv_rate'],
                mode='lines',
                name='Control (A)',
                line=dict(color='#93C5FD', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=sim_df['day'],
                y=sim_df['variation_conv_rate'],
                mode='lines',
                name='Variation (B)',
                line=dict(color='#2563EB', width=2)
            ))
            
            # Add confidence interval for variation
            upper_bound = sim_df['variation_conv_rate'] * (1 + 0.1 * np.exp(-0.05 * sim_df['day']))
            lower_bound = sim_df['variation_conv_rate'] * (1 - 0.1 * np.exp(-0.05 * sim_df['day']))
            
            fig.add_trace(go.Scatter(
                x=sim_df['day'],
                y=upper_bound,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=sim_df['day'],
                y=lower_bound,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(37, 99, 235, 0.2)',
                showlegend=False
            ))
            
            fig.update_layout(
                height=400,
                xaxis_title="Day",
                yaxis_title="Conversion Rate",
                yaxis_tickformat='.2%',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=30, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Revenue comparison
            st.markdown("#### Cumulative Revenue")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=sim_df['day'],
                y=sim_df['cum_control_revenue'],
                mode='lines',
                name='Control (A)',
                line=dict(color='#93C5FD', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=sim_df['day'],
                y=sim_df['cum_variation_revenue'],
                mode='lines',
                name='Variation (B)',
                line=dict(color='#2563EB', width=2)
            ))
            
            fig.update_layout(
                height=400,
                xaxis_title="Day",
                yaxis_title="Cumulative Revenue (₹)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=30, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Final results
            st.markdown("### Final Test Results")
            
            # Get final day results
            final_results = sim_df.iloc[-1]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Control (A)")
                
                st.markdown(f"""
                - **Price:** ₹{control_price:,.0f}
                - **Visitors:** {final_results['cum_control_visitors']:,.0f}
                - **Conversions:** {final_results['cum_control_conversions']:,.0f}
                - **Conversion Rate:** {final_results['control_conv_rate']:.2%}
                - **Revenue:** ₹{final_results['cum_control_revenue']:,.0f}
                - **Revenue per Visitor:** ₹{final_results['cum_control_revenue']/final_results['cum_control_visitors']:,.2f}
                """)
            
            with col2:
                st.markdown("#### Variation (B)")
                
                st.markdown(f"""
                - **Price:** ₹{variation_price:,.0f}
                - **Visitors:** {final_results['cum_variation_visitors']:,.0f}
                - **Conversions:** {final_results['cum_variation_conversions']:,.0f}
                - **Conversion Rate:** {final_results['variation_conv_rate']:.2%}
                - **Revenue:** ₹{final_results['cum_variation_revenue']:,.0f}
                - **Revenue per Visitor:** ₹{final_results['cum_variation_revenue']/final_results['cum_variation_visitors']:,.2f}
                """)
            
            # Statistical significance
            conv_rate_diff = final_results['variation_conv_rate'] - final_results['control_conv_rate']
            conv_rate_diff_pct = conv_rate_diff / final_results['control_conv_rate'] * 100
            
            revenue_diff = final_results['cum_variation_revenue'] - final_results['cum_control_revenue']
            revenue_diff_pct = revenue_diff / final_results['cum_control_revenue'] * 100
            
            # Calculate confidence level (simplified)
            n1 = final_results['cum_control_visitors']
            n2 = final_results['cum_variation_visitors']
            p1 = final_results['control_conv_rate']
            p2 = final_results['variation_conv_rate']
            
            # Standard error of difference between proportions
            se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
            
            # Z-score
            z = (p2 - p1) / se
            
            # Convert to confidence level (approximate)
            if abs(z) > 2.58:
                confidence = "99%"
            elif abs(z) > 1.96:
                confidence = "95%"
            elif abs(z) > 1.65:
                confidence = "90%"
            else:
                confidence = f"{min(int(abs(z) * 30), 85)}%"
            
            st.markdown("#### Comparison Results")
            
            if revenue_diff > 0:
                result_color = "#10B981"  # Green
                recommendation = f"Implement Variation B (₹{variation_price:,.0f}) for higher revenue"
            else:
                result_color = "#EF4444"  # Red
                recommendation = f"Keep Control A (₹{control_price:,.0f}) for higher revenue"
            
            st.markdown(f"""
            - **Conversion Rate Difference:** <span style='color:{result_color}'>{conv_rate_diff:+.2%} ({conv_rate_diff_pct:+.1f}%)</span>
            - **Revenue Difference:** <span style='color:{result_color}'>₹{revenue_diff:+,.0f} ({revenue_diff_pct:+.1f}%)</span>
            - **Statistical Confidence:** {confidence}
            - **Recommendation:** {recommendation}
            """, unsafe_allow_html=True)
            
            # Projected annual impact
            annual_control = final_results['cum_control_revenue'] * (365 / ab_test_duration)
            annual_variation = final_results['cum_variation_revenue'] * (365 / ab_test_duration)
            annual_diff = annual_variation - annual_control
            
            st.markdown(f"""
            **Projected Annual Impact:** <span style='color:{result_color}'>₹{annual_diff:+,.0f}</span>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<div class='sub-title'>Scenario Analysis</div>", unsafe_allow_html=True)
        
        # Setup scenario analysis
        col1, col2 = st.columns(2)
        
        with col1:
            scenario_destination = st.selectbox(
                "Destination",
                options=sorted(sample_data['destination'].unique()),
                key="scenario_destination"
            )
            
            scenario_product = st.selectbox(
                "Product Type",
                options=sorted(sample_data['product_type'].unique()),
                key="scenario_product"
            )
        
        with col2:
            scenario_base_price = st.number_input(
                "Current Base Price (₹)",
                min_value=1000,
                max_value=100000,
                value=15000,
                step=500,
                key="scenario_base_price"
            )
            
            scenario_timeframe = st.selectbox(
                "Analysis Timeframe",
                options=["3 months", "6 months", "12 months"],
                index=1,
                key="scenario_timeframe"
            )
        
        # Define scenarios
        st.markdown("### Define Market Scenarios")
        
        # Market conditions definitions
        scenario_definitions = {
            "Pessimistic": {
                "desc": "Low demand, high competition, economic downturn",
                "demand_level": 3,
                "competitor_pricing": "aggressive",
                "market_condition": 3,
                "color": "#FEE2E2"  # Light red
            },
            "Realistic": {
                "desc": "Normal demand, stable competition, neutral market",
                "demand_level": 5,
                "competitor_pricing": "stable",
                "market_condition": 5,
                "color": "#DBEAFE"  # Light blue
            },
            "Optimistic": {
                "desc": "High demand, favorable competition, strong market",
                "demand_level": 8,
                "competitor_pricing": "favorable",
                "market_condition": 8,
                "color": "#DCFCE7"  # Light green
            }
        }
        
        # Create price strategies
        price_strategies = {
            "Conservative": {
                "desc": "Maintain current pricing with minimal adjustments",
                "adjustment": 0,
                "color": "#F3F4F6"  # Light gray
            },
            "Moderate": {
                "desc": "Apply SmartPrice recommendations selectively",
                "adjustment": 1,
                "color": "#BFDBFE"  # Medium blue
            },
            "Aggressive": {
                "desc": "Fully implement SmartPrice dynamic optimization",
                "adjustment": 2,
                "color": "#2563EB"  # Dark blue
            }
        }
        
        # Display scenario options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Pessimistic Scenario**")
            st.markdown(f"""<div style='background-color: {scenario_definitions['Pessimistic']['color']}; padding: 10px; border-radius: 5px;'>
            {scenario_definitions['Pessimistic']['desc']}
            </div>""", unsafe_allow_html=True)
            
            include_pessimistic = st.checkbox("Include in analysis", value=True, key="include_pessimistic")
            
            if include_pessimistic:
                pessimistic_probability = st.slider(
                    "Probability (%)",
                    min_value=5,
                    max_value=95,
                    value=25,
                    step=5,
                    key="pessimistic_probability"
                )
            else:
                pessimistic_probability = 0
        
        with col2:
            st.markdown("**Realistic Scenario**")
            st.markdown(f"""<div style='background-color: {scenario_definitions['Realistic']['color']}; padding: 10px; border-radius: 5px;'>
            {scenario_definitions['Realistic']['desc']}
            </div>""", unsafe_allow_html=True)
            
            include_realistic = st.checkbox("Include in analysis", value=True, key="include_realistic")
            
            if include_realistic:
                realistic_probability = st.slider(
                    "Probability (%)",
                    min_value=5,
                    max_value=95,
                    value=50,
                    step=5,
                    key="realistic_probability"
                )
            else:
                realistic_probability = 0
        
        with col3:
            st.markdown("**Optimistic Scenario**")
            st.markdown(f"""<div style='background-color: {scenario_definitions['Optimistic']['color']}; padding: 10px; border-radius: 5px;'>
            {scenario_definitions['Optimistic']['desc']}
            </div>""", unsafe_allow_html=True)
            
            include_optimistic = st.checkbox("Include in analysis", value=True, key="include_optimistic")
            
            if include_optimistic:
                optimistic_probability = st.slider(
                    "Probability (%)",
                    min_value=5,
                    max_value=95,
                    value=25,
                    step=5,
                    key="optimistic_probability"
                )
            else:
                optimistic_probability = 0
        
        # Check if probabilities sum to 100%
        total_probability = pessimistic_probability + realistic_probability + optimistic_probability
        
        if total_probability != 100 and total_probability > 0:
            st.warning(f"Scenario probabilities should sum to 100% (currently {total_probability}%)")
        
        # Pricing strategies
        st.markdown("### Select Pricing Strategies")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Conservative**")
            st.markdown(f"""<div style='background-color: {price_strategies['Conservative']['color']}; padding: 10px; border-radius: 5px;'>
            {price_strategies['Conservative']['desc']}
            </div>""", unsafe_allow_html=True)
            
            include_conservative = st.checkbox("Include in analysis", value=True, key="include_conservative")
        
        with col2:
            st.markdown("**Moderate**")
            st.markdown(f"""<div style='background-color: {price_strategies['Moderate']['color']}; padding: 10px; border-radius: 5px;'>
            {price_strategies['Moderate']['desc']}
            </div>""", unsafe_allow_html=True)
            
            include_moderate = st.checkbox("Include in analysis", value=True, key="include_moderate")
        
        with col3:
            st.markdown("**Aggressive**")
            st.markdown(f"""<div style='background-color: {price_strategies['Aggressive']['color']}; padding: 10px; border-radius: 5px;'>
            {price_strategies['Aggressive']['desc']}
            </div>""", unsafe_allow_html=True)
            
            include_aggressive = st.checkbox("Include in analysis", value=True, key="include_aggressive")
        
        # Run scenario analysis
        if st.button("Run Scenario Analysis", key="run_scenario_analysis"):
            # Convert timeframe to months
            timeframe_months = int(scenario_timeframe.split()[0])
            
            # Base parameters
            monthly_volume = 1000  # Base monthly booking volume
            
            # Generate scenario analysis data
            scenario_results = []
            
            # Get scenarios to analyze
            scenarios = []
            if include_pessimistic:
                scenarios.append(("Pessimistic", pessimistic_probability/100))
            if include_realistic:
                scenarios.append(("Realistic", realistic_probability/100))
            if include_optimistic:
                scenarios.append(("Optimistic", optimistic_probability/100))
            
            # Get strategies to analyze
            strategies = []
            if include_conservative:
                strategies.append("Conservative")
            if include_moderate:
                strategies.append("Moderate")
            if include_aggressive:
                strategies.append("Aggressive")
            
            # Run analysis for each scenario and strategy
            for scenario_name, probability in scenarios:
                scenario = scenario_definitions[scenario_name]
                
                for strategy_name in strategies:
                    strategy = price_strategies[strategy_name]
                    
                    # Calculate optimal price based on scenario and strategy
                    # Pessimistic scenario tends to lower prices, optimistic to raise them
                    if scenario_name == "Pessimistic":
                        base_adjustment = -0.05  # -5% base adjustment
                    elif scenario_name == "Optimistic":
                        base_adjustment = 0.08  # +8% base adjustment
                    else:
                        base_adjustment = 0.02  # +2% base adjustment
                    
                    # Strategy adjustment factor
                    strategy_factor = strategy["adjustment"] * 0.03  # 0%, 3%, or 6%
                    
                    # Calculate price adjustment
                    price_adjustment = base_adjustment + strategy_factor
                    optimal_price = scenario_base_price * (1 + price_adjustment)
                    
                    # Calculate volume impact based on price elasticity and scenario
                    # Different elasticity for different scenarios
                    if scenario_name == "Pessimistic":
                        elasticity = -1.5  # More sensitive to price
                    elif scenario_name == "Optimistic":
                        elasticity = -0.8  # Less sensitive to price
                    else:
                        elasticity = -1.2  # Medium sensitivity
                    
                    volume_impact = price_adjustment * elasticity
                    
                    # Scenario impact on volume
                    scenario_volume_factor = (scenario["demand_level"] / 5) * 0.2  # -40% to +60%
                    
                    # Calculate expected volume
                    expected_volume = monthly_volume * (1 + scenario_volume_factor + volume_impact) * timeframe_months
                    
                    # Calculate revenue
                    expected_revenue = expected_volume * optimal_price
                    
                    # Base case (conservative strategy in realistic scenario)
                    if scenario_name == "Realistic" and strategy_name == "Conservative":
                        base_volume = expected_volume
                        base_revenue = expected_revenue
                    
                    # Add to results
                    scenario_results.append({
                        'Scenario': scenario_name,
                        'Strategy': strategy_name,
                        'Probability': probability,
                        'Price': optimal_price,
                        'Price_Change': price_adjustment * 100,  # as percentage
                        'Volume': expected_volume,
                        'Revenue': expected_revenue,
                        'Scenario_Color': scenario["color"],
                        'Strategy_Color': strategy["color"]
                    })
            
            # Convert to DataFrame
            results_df = pd.DataFrame(scenario_results)
            
            # Calculate base case for comparison
            base_case = results_df[(results_df['Scenario'] == 'Realistic') & (results_df['Strategy'] == 'Conservative')]
            if not base_case.empty:
                base_revenue = base_case['Revenue'].values[0]
                
                # Calculate difference from base case
                results_df['Revenue_vs_Base'] = ((results_df['Revenue'] - base_revenue) / base_revenue) * 100
                results_df['Volume_vs_Base'] = ((results_df['Volume'] - base_case['Volume'].values[0]) / base_case['Volume'].values[0]) * 100
            else:
                # If base case not available, use first entry as reference
                base_revenue = results_df['Revenue'].iloc[0]
                results_df['Revenue_vs_Base'] = ((results_df['Revenue'] - base_revenue) / base_revenue) * 100
                results_df['Volume_vs_Base'] = ((results_df['Volume'] - results_df['Volume'].iloc[0]) / results_df['Volume'].iloc[0]) * 100
            
            # Calculate weighted average for each strategy
            strategy_weighted = results_df.groupby('Strategy').apply(
                lambda x: pd.Series({
                    'Weighted_Revenue': (x['Revenue'] * x['Probability']).sum(),
                    'Weighted_Volume': (x['Volume'] * x['Probability']).sum(),
                    'Min_Revenue': x['Revenue'].min(),
                    'Max_Revenue': x['Revenue'].max(),
                    'Strategy_Color': x['Strategy_Color'].iloc[0]
                })
            )
            
            # Show results
            st.markdown("### Scenario Analysis Results")
            
            # Results table
            st.markdown("#### Results by Scenario and Strategy")
            
            # Create results table
            results_table = results_df[['Scenario', 'Strategy', 'Price', 'Price_Change', 'Volume', 'Revenue', 'Revenue_vs_Base', 'Probability']]
            
            # Format the table
            table_data = {
                'Scenario': results_table['Scenario'],
                'Strategy': results_table['Strategy'],
                'Price': ['₹{:,.0f}'.format(x) for x in results_table['Price']],
                'Price Change': ['{:+.1f}%'.format(x) for x in results_table['Price_Change']],
                'Volume': ['{:,.0f}'.format(x) for x in results_table['Volume']],
                'Revenue': ['₹{:,.0f}'.format(x) for x in results_table['Revenue']],
                'vs Base Case': ['{:+.1f}%'.format(x) for x in results_table['Revenue_vs_Base']],
                'Probability': ['{:.0f}%'.format(x*100) for x in results_table['Probability']]
            }
            
            # Create the table
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=list(table_data.keys()),
                    fill_color='#BFDBFE',
                    align='left',
                    font=dict(color='#1E3A8A', size=12)
                ),
                cells=dict(
                    values=list(table_data.values()),
                    fill_color=[
                        [results_df['Scenario_Color'][i] for i in range(len(results_df))],
                        [results_df['Strategy_Color'][i] for i in range(len(results_df))],
                        ['#F3F4F6'] * len(results_df),
                        ['#F3F4F6'] * len(results_df),
                        ['#F3F4F6'] * len(results_df),
                        ['#F3F4F6'] * len(results_df),
                        [('#DCFCE7' if x > 0 else '#FEE2E2') for x in results_df['Revenue_vs_Base']],
                        ['#F3F4F6'] * len(results_df)
                    ],
                    align='left',
                    font=dict(color='#374151', size=11)
                ))
            ])
            
            fig.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Results visualization
            st.markdown("#### Strategy Comparison")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create grouped bar chart for revenue by scenario and strategy
                revenue_pivot = results_df.pivot(index='Scenario', columns='Strategy', values='Revenue')
                
                fig = go.Figure()
                
                # Add a trace for each strategy
                for strategy in strategies:
                    if strategy in revenue_pivot.columns:
                        fig.add_trace(go.Bar(
                            x=revenue_pivot.index,
                            y=revenue_pivot[strategy],
                            name=strategy,
                            marker_color=price_strategies[strategy]['color']
                        ))
                
                fig.update_layout(
                    height=400,
                    xaxis_title="Scenario",
                    yaxis_title="Revenue (₹)",
                    legend_title="Strategy",
                    barmode='group',
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Expected Value Analysis
                st.markdown("#### Expected Value Analysis")
                
                # Sort strategies by weighted revenue
                strategy_weighted = strategy_weighted.sort_values('Weighted_Revenue', ascending=False)
                
                for idx, (strategy, row) in enumerate(strategy_weighted.iterrows()):
                    revenue_str = '₹{:,.0f}'.format(row['Weighted_Revenue'])
                    volume_str = '{:,.0f}'.format(row['Weighted_Volume'])
                    min_rev_str = '₹{:,.0f}'.format(row['Min_Revenue'])
                    max_rev_str = '₹{:,.0f}'.format(row['Max_Revenue'])
                    
                    highlight = ""
                    if idx == 0:
                        highlight = "background-color: #DCFCE7; font-weight: bold;"
                    
                    st.markdown(f"""
                    <div style='padding: 10px; border-radius: 5px; margin-bottom: 10px; {highlight}'>
                        <h4>{strategy}</h4>
                        <p>Expected Revenue: {revenue_str}<br>
                        Expected Volume: {volume_str}<br>
                        Range: {min_rev_str} - {max_rev_str}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Risk-reward analysis
            st.markdown("### Risk-Reward Analysis")
            
            # Calculate risk metrics
            risk_metrics = []
            
            for strategy in strategies:
                strategy_data = results_df[results_df['Strategy'] == strategy]
                
                weighted_revenue = (strategy_data['Revenue'] * strategy_data['Probability']).sum()
                revenue_std = strategy_data['Revenue'].std()
                min_revenue = strategy_data['Revenue'].min()
                max_revenue = strategy_data['Revenue'].max()
                range_pct = (max_revenue - min_revenue) / weighted_revenue * 100
                
                # Calculate risk score (higher is more risky)
                risk_score = revenue_std / weighted_revenue * 100
                
                # Calculate reward score (weighted improvement vs. base)
                reward_score = (strategy_data['Revenue_vs_Base'] * strategy_data['Probability']).sum()
                
                risk_metrics.append({
                    'Strategy': strategy,
                    'Expected_Revenue': weighted_revenue,
                    'Risk_Score': risk_score,
                    'Reward_Score': reward_score,
                    'Range_Pct': range_pct,
                    'Strategy_Color': price_strategies[strategy]['color']
                })
            
            risk_df = pd.DataFrame(risk_metrics)
            
            # Create risk-reward scatter plot
            fig = px.scatter(
                risk_df,
                x='Risk_Score',
                y='Reward_Score',
                size='Expected_Revenue',
                color='Strategy',
                color_discrete_map={
                    'Conservative': price_strategies['Conservative']['color'],
                    'Moderate': price_strategies['Moderate']['color'],
                    'Aggressive': price_strategies['Aggressive']['color']
                },
                labels={
                    'Risk_Score': 'Risk (Standard Deviation %)',
                    'Reward_Score': 'Expected Return (%)',
                    'Expected_Revenue': 'Expected Revenue'
                },
                title="Risk-Reward Analysis",
                text='Strategy'
            )
            
            fig.update_traces(
                textposition='top center',
                marker=dict(size=20, opacity=0.8, line=dict(width=2, color='DarkSlateGrey')),
            )
            
            # Add efficient frontier line
            fig.add_shape(
                type="line",
                x0=risk_df['Risk_Score'].min() * 0.9,
                y0=risk_df['Reward_Score'].min() * 0.9,
                x1=risk_df['Risk_Score'].max() * 1.1,
                y1=risk_df['Reward_Score'].max() * 1.1,
                line=dict(color="grey", width=1, dash="dot")
            )
            
            fig.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=30, b=20),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("### Recommendations")
            
            # Find best strategy based on risk-adjusted return
            risk_df['Risk_Adjusted_Return'] = risk_df['Reward_Score'] / risk_df['Risk_Score']
            best_strategy = risk_df.loc[risk_df['Risk_Adjusted_Return'].idxmax()]['Strategy']
            
            # Generate recommendation text
            rec_text = f"""
            Based on the scenario analysis, the following recommendations are provided:
            
            1. **Optimal Strategy:** The {best_strategy} pricing strategy offers the best risk-adjusted return.
            
            2. **Expected Results:**
               - Revenue: ₹{risk_df[risk_df['Strategy'] == best_strategy]['Expected_Revenue'].values[0]:,.0f}
               - Expected Return: {risk_df[risk_df['Strategy'] == best_strategy]['Reward_Score'].values[0]:.1f}%
               - Risk Level: {risk_df[risk_df['Strategy'] == best_strategy]['Risk_Score'].values[0]:.1f}%
            
            3. **Implementation Plan:**
            """
            
            if best_strategy == "Conservative":
                rec_text += """
               - Maintain current pricing with minimal adjustments
               - Focus on high-demand periods for selective price increases
               - Monitor competitor pricing closely
               - Re-evaluate strategy quarterly
            """
            elif best_strategy == "Moderate":
                rec_text += """
               - Implement SmartPrice recommendations selectively
               - Apply larger adjustments for peak seasons and high-confidence recommendations
               - Test price changes in smaller segments before full rollout
               - Review performance monthly
            """
            else:  # Aggressive
                rec_text += """
               - Fully implement SmartPrice dynamic optimization
               - Update prices frequently based on real-time data
               - Establish monitoring system for market response
               - Review performance weekly
            """
            
            st.markdown(rec_text)
    
    with tab3:
        st.markdown("<div class='sub-title'>ROI Calculator</div>", unsafe_allow_html=True)
        
        st.markdown("""
        Calculate the Return on Investment (ROI) for implementing SmartPrice optimization across your travel products.
        Adjust the parameters below to estimate the financial impact and payback period.
        """)
        
        # Implementation parameters
        st.markdown("### Implementation Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            roi_products = st.number_input(
                "Number of Travel Products",
                min_value=1,
                max_value=10000,
                value=100,
                step=10,
                help="Total number of travel products to optimize"
            )
            
            roi_annual_revenue = st.number_input(
                "Annual Revenue (₹ lakhs)",
                min_value=1,
                max_value=100000,
                value=1000,
                step=100,
                help="Current annual revenue in lakhs of rupees"
            ) * 100000  # Convert to rupees
            
            roi_implementation_level = st.slider(
                "Implementation Level (%)",
                min_value=10,
                max_value=100,
                value=80,
                step=10,
                help="Percentage of recommendations implemented"
            )
        
        with col2:
            roi_timeframe = st.selectbox(
                "Implementation Timeframe",
                options=["3 months", "6 months", "12 months", "18 months"],
                index=1,
                help="Time required for full implementation"
            )
            
            roi_confidence = st.slider(
                "Confidence Level",
                min_value=70,
                max_value=99,
                value=87,
                step=1,
                help="Confidence in the accuracy of price recommendations"
            )
            
            roi_market_condition = st.select_slider(
                "Market Conditions",
                options=["Difficult", "Challenging", "Neutral", "Favorable", "Excellent"],
                value="Neutral",
                help="Current market conditions affecting price optimization"
            )
        
        # Cost parameters
        st.markdown("### Cost Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            implementation_cost = st.number_input(
                "Implementation Cost (₹ lakhs)",
                min_value=1,
                max_value=1000,
                value=25,
                step=5,
                help="One-time cost to implement SmartPrice"
            ) * 100000  # Convert to rupees
        
        with col2:
            annual_license_fee = st.number_input(
                "Annual License Fee (₹ lakhs)",
                min_value=0,
                max_value=500,
                value=12,
                step=1,
                help="Annual license/maintenance fee"
            ) * 100000  # Convert to rupees
        
        with col3:
            training_cost = st.number_input(
                "Training Cost (₹ lakhs)",
                min_value=0,
                max_value=100,
                value=5,
                step=1,
                help="Cost of staff training"
            ) * 100000  # Convert to rupees
        
        # Calculate ROI
        if st.button("Calculate ROI", key="calculate_roi"):
            # Convert timeframe to months
            timeframe_months = int(roi_timeframe.split()[0])
            
            # Market condition factor
            market_factors = {
                "Difficult": 0.7,
                "Challenging": 0.85,
                "Neutral": 1.0,
                "Favorable": 1.15,
                "Excellent": 1.3
            }
            market_factor = market_factors[roi_market_condition]
            
            # Calculate expected revenue improvement
            # Base improvement based on typical results
            base_improvement_pct = 0.12  # 12% base improvement
            
            # Adjust for implementation level
            adjusted_improvement = base_improvement_pct * (roi_implementation_level / 100)
            
            # Adjust for market conditions
            market_adjusted_improvement = adjusted_improvement * market_factor
            
            # Adjust for confidence level
            confidence_factor = roi_confidence / 87  # Normalize against baseline 87% confidence
            final_improvement_pct = market_adjusted_improvement * confidence_factor
            
            # Calculate the revenue increase
            annual_revenue_increase = roi_annual_revenue * final_improvement_pct
            
            # Calculate implementation ramp-up
            monthly_revenue = roi_annual_revenue / 12
            monthly_improvement = []
            cumulative_improvement = []
            
            # Simple linear ramp-up
            for month in range(1, 25):  # Calculate for 24 months
                if month <= timeframe_months:
                    month_pct = month / timeframe_months
                    month_improvement = monthly_revenue * final_improvement_pct * month_pct
                else:
                    month_improvement = monthly_revenue * final_improvement_pct
                
                monthly_improvement.append(month_improvement)
                
                if month == 1:
                    cumulative_improvement.append(month_improvement)
                else:
                    cumulative_improvement.append(cumulative_improvement[-1] + month_improvement)
            
            # Calculate total costs
            initial_costs = implementation_cost + training_cost
            monthly_costs = annual_license_fee / 12
            
            cumulative_costs = []
            for month in range(1, 25):
                if month == 1:
                    cumulative_costs.append(initial_costs + monthly_costs)
                else:
                    cumulative_costs.append(cumulative_costs[-1] + monthly_costs)
            
            # Calculate net benefit
            net_benefit = []
            for imp, cost in zip(cumulative_improvement, cumulative_costs):
                net_benefit.append(imp - cost)
            
            # Find payback period (when net benefit becomes positive)
            payback_period = None
            for month, benefit in enumerate(net_benefit, 1):
                if benefit >= 0:
                    payback_period = month
                    break
            
            # Calculate ROI for 1 and 2 years
            year1_roi = (cumulative_improvement[11] - cumulative_costs[11]) / cumulative_costs[11] * 100
            year2_roi = (cumulative_improvement[23] - cumulative_costs[23]) / cumulative_costs[23] * 100
            
            # Create ROI chart
            months = list(range(1, 25))
            
            roi_data = pd.DataFrame({
                'Month': months,
                'Cumulative Improvement': cumulative_improvement,
                'Cumulative Costs': cumulative_costs,
                'Net Benefit': net_benefit
            })
            
            st.markdown("### ROI Analysis Results")
            
            # ROI chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=roi_data['Month'],
                y=roi_data['Cumulative Improvement'],
                mode='lines',
                name='Cumulative Revenue Increase',
                line=dict(color='#10B981', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=roi_data['Month'],
                y=roi_data['Cumulative Costs'],
                mode='lines',
                name='Cumulative Costs',
                line=dict(color='#EF4444', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=roi_data['Month'],
                y=roi_data['Net Benefit'],
                mode='lines',
                name='Net Benefit',
                line=dict(color='#2563EB', width=3)
            ))
            
            # Add breakeven point
            if payback_period:
                fig.add_trace(go.Scatter(
                    x=[payback_period],
                    y=[0],
                    mode='markers',
                    name='Breakeven Point',
                    marker=dict(color='#FBBF24', size=12, symbol='star')
                ))
            
            # Add vertical lines for year 1 and 2
            fig.add_vline(x=12, line_width=1, line_dash="dash", line_color="gray")
            fig.add_vline(x=24, line_width=1, line_dash="dash", line_color="gray")
            
            # Add horizontal line at zero
            fig.add_hline(y=0, line_width=1, line_color="black")
            
            fig.update_layout(
                height=500,
                xaxis_title="Month",
                yaxis_title="Amount (₹)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=30, b=20),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # ROI summary
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Financial Impact")
                
                st.markdown(f"""
                **Revenue Improvement:**
                - Expected Improvement: {final_improvement_pct:.1%}
                - Annual Revenue Increase: ₹{annual_revenue_increase/100000:.1f} lakhs
                - 2-Year Revenue Increase: ₹{cumulative_improvement[-1]/100000:.1f} lakhs
                
                **Investment:**
                - Initial Investment: ₹{initial_costs/100000:.1f} lakhs
                - 2-Year Total Cost: ₹{cumulative_costs[-1]/100000:.1f} lakhs
                """)
            
            with col2:
                st.markdown("#### ROI Metrics")
                
                if payback_period:
                    payback_str = f"{payback_period} months"
                else:
                    payback_str = "Beyond 24 months"
                
                st.markdown(f"""
                **Return on Investment:**
                - 1-Year ROI: {year1_roi:.1f}%
                - 2-Year ROI: {year2_roi:.1f}%
                
                **Payback Period:** {payback_str}
                
                **Net Benefit (2 Years):** ₹{net_benefit[-1]/100000:.1f} lakhs
                """)
            
            # Additional ROI metrics
            st.markdown("#### Implementation Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Benefit-Cost Ratio
                bcr = cumulative_improvement[-1] / cumulative_costs[-1]
                
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=bcr,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Benefit-Cost Ratio"},
                    gauge={
                        'axis': {'range': [0, max(5, bcr * 1.2)]},
                        'bar': {'color': "#10B981"},
                        'steps': [
                            {'range': [0, 1], 'color': "#FEE2E2"},
                            {'range': [1, 2], 'color': "#FEF3C7"},
                            {'range': [2, 5], 'color': "#D1FAE5"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 2},
                            'thickness': 0.75,
                            'value': 1
                        }
                    }
                ))
                
                fig.update_layout(
                    height=250,
                    margin=dict(l=10, r=10, t=30, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Implementation efficiency
                efficiency = min(100, 100 * (roi_implementation_level / 100) * (roi_confidence / 100))
                
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=efficiency,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Implementation Efficiency"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#2563EB"},
                        'steps': [
                            {'range': [0, 60], 'color': "#FEE2E2"},
                            {'range': [60, 80], 'color': "#FEF3C7"},
                            {'range': [80, 100], 'color': "#D1FAE5"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 2},
                            'thickness': 0.75,
                            'value': 80
                        }
                    }
                ))
                
                fig.update_layout(
                    height=250,
                    margin=dict(l=10, r=10, t=30, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                # Risk assessment
                risk_factors = {
                    "Difficult": 80,
                    "Challenging": 65,
                    "Neutral": 50,
                    "Favorable": 35,
                    "Excellent": 20
                }
                risk_score = risk_factors[roi_market_condition]
                
                # Adjust based on implementation level and confidence
                impl_risk = (100 - roi_implementation_level) * 0.2  # 0-20 points
                conf_risk = (100 - roi_confidence) * 0.3  # 0-30 points
                
                final_risk = min(100, max(0, risk_score + impl_risk + conf_risk))
                
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=final_risk,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Implementation Risk"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#EF4444"},
                        'steps': [
                            {'range': [0, 40], 'color': "#D1FAE5"},
                            {'range': [40, 70], 'color': "#FEF3C7"},
                            {'range': [70, 100], 'color': "#FEE2E2"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 2},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                
                fig.update_layout(
                    height=250,
                    margin=dict(l=10, r=10, t=30, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("### Recommendations")
            
            # Generate recommendations based on ROI analysis
            if year1_roi > 100:
                roi_assessment = "extremely favorable"
                action = "proceed with full implementation immediately"
            elif year1_roi > 50:
                roi_assessment = "very favorable"
                action = "proceed with implementation as planned"
            elif year1_roi > 20:
                roi_assessment = "favorable"
                action = "proceed with implementation with regular monitoring"
            elif year1_roi > 0:
                roi_assessment = "positive but modest"
                action = "consider a phased implementation approach"
            else:
                roi_assessment = "negative in the first year"
                action = "reevaluate the implementation parameters or consider a limited pilot"
            
            st.markdown(f"""
            Based on the ROI analysis, the investment in SmartPrice shows a **{roi_assessment}** return on investment. 
            We recommend to **{action}**.
            
            **Key observations:**
            
            1. The project is expected to break even in **{payback_str}**.
            
            2. The 2-year ROI of **{year2_roi:.1f}%** indicates a strong long-term value.
            
            3. With a benefit-cost ratio of **{bcr:.2f}**, the project is {'economically viable' if bcr > 1 else 'not economically viable'}.
            
            4. Implementation risk is **{final_risk:.0f}%** ({final_risk < 50 and 'acceptable' or 'significant'}).
            """)
            
            # Implementation plan
            st.markdown("### Implementation Plan")
            
            # Create implementation timeline
            timeline_data = []
            
            # Define implementation phases
            phases = [
                {"name": "Planning & Setup", "duration": max(1, timeframe_months // 4)},
                {"name": "Initial Deployment", "duration": max(1, timeframe_months // 4)},
                {"name": "Full Rollout", "duration": max(1, timeframe_months // 2)},
                {"name": "Optimization", "duration": max(1, timeframe_months // 4)}
            ]
            
            # Adjust if total duration exceeds timeframe
            total_duration = sum(phase["duration"] for phase in phases)
            if total_duration > timeframe_months:
                scale_factor = timeframe_months / total_duration
                for phase in phases:
                    phase["duration"] = max(1, int(phase["duration"] * scale_factor))
            
            # Create timeline data
            current_month = 1
            for phase in phases:
                start_month = current_month
                end_month = current_month + phase["duration"] - 1
                
                timeline_data.append({
                    "Phase": phase["name"],
                    "Start": start_month,
                    "End": end_month,
                    "Duration": phase["duration"]
                })
                
                current_month = end_month + 1
            
            # Create timeline chart
            fig = go.Figure()
            
            colors = ['#BFDBFE', '#93C5FD', '#60A5FA', '#2563EB']
            
            for i, phase in enumerate(timeline_data):
                fig.add_trace(go.Bar(
                    x=[phase["Duration"]],
                    y=[phase["Phase"]],
                    orientation='h',
                    marker=dict(color=colors[i % len(colors)]),
                    base=phase["Start"] - 1,
                    hovertemplate=f"{phase['Phase']}<br>Months {phase['Start']}-{phase['End']}<br>Duration: {phase['Duration']} months"
                ))
            
            fig.update_layout(
                height=200,
                xaxis_title="Month",
                yaxis=dict(
                    title="",
                    autorange="reversed"
                ),
                barmode='stack',
                showlegend=False,
                margin=dict(l=20, r=20, t=10, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Implementation costs distribution
            col1, col2 = st.columns(2)
            
            with col1:
                # Cost breakdown
                cost_breakdown = {
                    "Implementation": implementation_cost,
                    "Training": training_cost,
                    "Annual License (2 years)": annual_license_fee * 2
                }
                
                # Create pie chart
                fig = px.pie(
                    values=list(cost_breakdown.values()),
                    names=list(cost_breakdown.keys()),
                    title="Cost Breakdown",
                    color_discrete_sequence=['#BFDBFE', '#93C5FD', '#60A5FA']
                )
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Monthly cash flow (first 12 months)
                monthly_cash_flow = []
                
                for i in range(12):
                    if i == 0:
                        cost = initial_costs + monthly_costs
                    else:
                        cost = monthly_costs
                    
                    monthly_cash_flow.append({
                        'Month': i + 1,
                        'Revenue Increase': monthly_improvement[i],
                        'Cost': cost,
                        'Net Cash Flow': monthly_improvement[i] - cost
                    })
                
                monthly_cf_df = pd.DataFrame(monthly_cash_flow)
                
                # Create bar chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=monthly_cf_df['Month'],
                    y=monthly_cf_df['Revenue Increase'],
                    name='Revenue Increase',
                    marker_color='#10B981'
                ))
                
                fig.add_trace(go.Bar(
                    x=monthly_cf_df['Month'],
                    y=[-cost for cost in monthly_cf_df['Cost']],
                    name='Cost',
                    marker_color='#EF4444'
                ))
                
                fig.add_trace(go.Scatter(
                    x=monthly_cf_df['Month'],
                    y=monthly_cf_df['Net Cash Flow'],
                    name='Net Cash Flow',
                    mode='lines+markers',
                    line=dict(color='#2563EB', width=3)
                ))
                
                fig.update_layout(
                    height=300,
                    xaxis_title="Month",
                    yaxis_title="Amount (₹)",
                    barmode='relative',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Run the app
if __name__ == "__main__":
    st.markdown(
        f"""
        <div style='text-align: right; color: #9CA3AF; padding: 10px;'>
        Current User: {st.session_state.get('username', 'Anurag02012004')}<br>
        Last Updated: 2025-07-08 01:51:42
        </div>
        """,
        unsafe_allow_html=True
    )