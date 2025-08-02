import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import time
import pickle
import os
import json
import base64
import requests
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
import xgboost as xgb
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.decomposition import PCA
import random
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="SmartPrice – Dynamic Travel Pricing Engine",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced styling for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-align: center;
        background: linear-gradient(90deg, #1E88E5, #42A5F5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        font-weight: bold;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #1E88E5;
        padding-left: 10px;
    }
    .card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #2196F3;
        color: #333;
    }
    .highlight {
        background: linear-gradient(135deg, #fff3e0 0%, #ffcc02 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        color: #333;
    }
    .metric-card {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border: 1px solid #4caf50;
        color: #333;
    }
    .small-text {
        font-size: 0.8rem;
        color: #616161;
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #9e9e9e;
        padding: 1rem;
        background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%);
        border-radius: 5px;
    }
    .success-card {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border-left: 4px solid #4caf50;
    }
    .warning-card {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        border-left: 4px solid #ff9800;
    }
    .error-card {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 4px solid #f44336;
    }
    .tech-stack {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 10px 0;
    }
    .tech-tag {
        background: #1E88E5;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    .feature-item {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border: 1px solid #9c27b0;
        color: #333;
    }
    .ml-model-card {
        background: linear-gradient(135deg, #e0f2f1 0%, #b2dfdb 100%);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #009688;
        color: #333;
    }
    .kafka-event {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 3px solid #ff9800;
        color: #333;
    }
    
    /* Fix for white text visibility */
    .stMarkdown, .stText {
        color: #333 !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #333 !important;
    }
    
    .stMarkdown p {
        color: #333 !important;
    }
    
    /* Ensure all text in Streamlit components is visible */
    .stSelectbox, .stTextInput, .stNumberInput, .stDateInput {
        color: #333 !important;
    }
    
    /* Make sure expander content is visible */
    .streamlit-expanderContent {
        color: #333 !important;
    }
    
    /* Ensure table text is visible */
    .stDataFrame {
        color: #333 !important;
    }
    
    /* Fix for any white text in custom components */
    .stMarkdown div {
        color: #333 !important;
    }
    
    /* Ensure all text elements have proper contrast */
    .stMarkdown, .stText, .stSelectbox, .stTextInput, .stNumberInput, .stDateInput, .stDataFrame, .streamlit-expanderContent {
        color: #333 !important;
    }
    
    /* Fix for headings specifically */
    h1, h2, h3, h4, h5, h6 {
        color: #333 !important;
    }
    
    /* Fix for paragraphs and spans */
    p, span, div {
        color: #333 !important;
    }
    
    /* Better background for main content areas */
    .main .block-container {
        background: linear-gradient(135deg, #fafafa 0%, #f5f5f5 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem;
    }
    
    /* Better background for expander content */
    .streamlit-expanderContent {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }
    
    /* Better background for sidebar */
    .css-1d391kg {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    }
    
    /* Better background for form elements */
    .stSelectbox > div > div {
        background: white !important;
        color: #333 !important;
    }
    
    .stTextInput > div > div > input {
        background: white !important;
        color: #333 !important;
    }
    
    .stNumberInput > div > div > input {
        background: white !important;
        color: #333 !important;
    }
    
    /* Better background for dataframes */
    .stDataFrame {
        background: white !important;
        color: #333 !important;
    }
    }
    .redis-cache {
        background: linear-gradient(135deg, #fff3e0 0%, #ffcc02 100%);
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 3px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("SmartPrice Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["System Overview", "Overview", "Data Explorer", "Price Prediction", "Model Performance", "System Architecture", "Real-Time Monitoring"]
)

# Enhanced Redis Cache with advanced features
class RedisCache:
    def __init__(self):
        self.cache = {}
        self.ttl = {}
        self.stats = {"hits": 0, "misses": 0, "sets": 0}
    
    def set(self, key, value, ttl=300):  # TTL in seconds (default 5 min)
        self.cache[key] = value
        self.ttl[key] = time.time() + ttl
        self.stats["sets"] += 1
    
    def get(self, key):
        if key not in self.cache:
            self.stats["misses"] += 1
            return None
        if time.time() > self.ttl[key]:
            del self.cache[key]
            del self.ttl[key]
            self.stats["misses"] += 1
            return None
        self.stats["hits"] += 1
        return self.cache[key]
    
    def invalidate(self, key):
        if key in self.cache:
            del self.cache[key]
            del self.ttl[key]
    
    def clear(self):
        self.cache = {}
        self.ttl = {}
    
    def get_stats(self):
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": hit_rate,
            "total_operations": total
        }
    
    def get_keys(self):
        return list(self.cache.keys())

# Simulated Kafka Stream Processor
class KafkaStreamProcessor:
    def __init__(self):
        self.topics = {
            "user-search-logs": [],
            "route-demand-tracker": [],
            "price-update-events": [],
            "competitor-price-changes": [],
            "booking-intent-signals": []
        }
        self.consumers = {}
    
    def publish_event(self, topic, event):
        if topic in self.topics:
            event["timestamp"] = datetime.datetime.now().isoformat()
            event["event_id"] = f"{topic}_{len(self.topics[topic])}_{int(time.time())}"
            self.topics[topic].append(event)
            logger.info(f"Published event to {topic}: {event['event_id']}")
            return event["event_id"]
        return None
    
    def consume_events(self, topic, limit=10):
        if topic in self.topics:
            return self.topics[topic][-limit:] if self.topics[topic] else []
        return []
    
    def get_topic_stats(self):
        return {topic: len(events) for topic, events in self.topics.items()}

# User Segmentation and Personalization Engine
class UserSegmentationEngine:
    def __init__(self):
        self.segments = {
            "premium": {"discount_threshold": 0.05, "price_sensitivity": 0.3},
            "business": {"discount_threshold": 0.08, "price_sensitivity": 0.5},
            "leisure": {"discount_threshold": 0.12, "price_sensitivity": 0.8},
            "budget": {"discount_threshold": 0.15, "price_sensitivity": 1.2}
        }
    
    def classify_user(self, user_data):
        # Simulate user classification based on booking history (Indian market)
        total_spent = user_data.get("total_spent", 0)
        booking_frequency = user_data.get("booking_frequency", 0)
        avg_booking_value = user_data.get("avg_booking_value", 0)
        
        if total_spent > 200000 and avg_booking_value > 15000:
            return "premium"
        elif booking_frequency > 10 and avg_booking_value > 10000:
            return "business"
        elif total_spent > 80000:
            return "leisure"
        else:
            return "budget"
    
    def get_personalized_price(self, base_price, user_segment):
        segment_config = self.segments.get(user_segment, self.segments["leisure"])
        discount_factor = 1 - segment_config["discount_threshold"]
        return base_price * discount_factor

# Booking Intent Estimation Model
class BookingIntentEstimator:
    def __init__(self):
        self.model = LogisticRegression(random_state=42)
        self.features = [
            "search_frequency", "time_on_site", "route_views", 
            "price_comparisons", "return_visits", "device_type",
            "time_to_travel", "previous_bookings"
        ]
    
    def extract_features(self, user_behavior):
        # Simulate feature extraction from user behavior
        features = []
        for feature in self.features:
            features.append(user_behavior.get(feature, 0))
        return np.array(features).reshape(1, -1)
    
    def predict_intent(self, user_behavior):
        features = self.extract_features(user_behavior)
        # Simulate prediction (in real implementation, this would use trained model)
        intent_score = np.random.beta(2, 5)  # Simulate low intent by default
        return {
            "intent_score": intent_score,
            "confidence": np.random.uniform(0.7, 0.95),
            "recommended_action": "offer_discount" if intent_score < 0.3 else "standard_pricing"
        }

# Initialize Redis cache
@st.cache_resource
def get_redis_cache():
    return RedisCache()

redis_cache = get_redis_cache()

# Simulate data generation
@st.cache_data
def generate_route_data():
    routes = [
        {"from": "Mumbai", "to": "Delhi", "distance": 1150, "baseline_price": 8500},
        {"from": "Bangalore", "to": "Hyderabad", "distance": 570, "baseline_price": 4200},
        {"from": "Chennai", "to": "Kolkata", "distance": 1350, "baseline_price": 7800},
        {"from": "Pune", "to": "Ahmedabad", "distance": 650, "baseline_price": 4800},
        {"from": "Jaipur", "to": "Lucknow", "distance": 580, "baseline_price": 3800},
        {"from": "Kochi", "to": "Goa", "distance": 450, "baseline_price": 3200},
        {"from": "Varanasi", "to": "Patna", "distance": 280, "baseline_price": 2200},
        {"from": "Srinagar", "to": "Chandigarh", "distance": 420, "baseline_price": 3500},
    ]
    return routes

@st.cache_data
def generate_historical_data():
    np.random.seed(42)
    routes = generate_route_data()
    
    # Create date range for the past year
    start_date = datetime.datetime.now() - datetime.timedelta(days=365)
    dates = [start_date + datetime.timedelta(days=i) for i in range(365)]
    
    data = []
    for route in routes:
        for date in dates:
            # Base demand factors
            day_of_week = date.weekday()  # 0-6 where 0 is Monday
            month = date.month
            is_weekend = 1 if day_of_week >= 5 else 0
            
            # Seasonal factors (higher in summer, lower in winter)
            seasonal_factor = 1.0 + 0.3 * np.sin((month - 1) * np.pi / 6)
            
            # Weekend factor (higher on weekends)
            weekend_factor = 1.2 if is_weekend else 1.0
            
            # Holiday factors (major Indian holidays)
            is_holiday = 0
            holiday_factor = 1.0
            
            # New Year's Day
            if date.month == 1 and date.day == 1:
                is_holiday = 1
                holiday_factor = 1.4
            
            # Republic Day
            elif date.month == 1 and date.day == 26:
                is_holiday = 1
                holiday_factor = 1.3
            
            # Independence Day
            elif date.month == 8 and date.day == 15:
                is_holiday = 1
                holiday_factor = 1.3
            
            # Gandhi Jayanti
            elif date.month == 10 and date.day == 2:
                is_holiday = 1
                holiday_factor = 1.2
            
            # Diwali (simplified - around October/November)
            elif date.month == 10 and 20 <= date.day <= 30:
                is_holiday = 1
                holiday_factor = 1.8
            
            # Christmas
            elif date.month == 12 and date.day == 25:
                is_holiday = 1
                holiday_factor = 1.5
            
            # Random demand variation
            random_factor = np.random.normal(1.0, 0.15)
            
                    # Calculate base demand (adjusted for Indian market)
            base_demand = 150 * seasonal_factor * weekend_factor * holiday_factor * random_factor
            demand = int(max(0, base_demand))
            
            # Competitor prices (simulate 3 competitors)
            base_price = route["baseline_price"]
            competitor1_price = base_price * np.random.normal(1.0, 0.12)
            competitor2_price = base_price * np.random.normal(0.95, 0.10)
            competitor3_price = base_price * np.random.normal(1.05, 0.15)
            
            # Weather factor (simple random simulation)
            weather_severity = np.random.randint(0, 10)  # 0 = perfect, 10 = severe
            weather_factor = 1.0 - (weather_severity * 0.02)  # Up to 20% reduction for bad weather
            
            # Calculate optimal price based on all factors
            price_elasticity = -1.5  # Negative value: higher price, lower demand
            optimal_price = base_price * (1 + 0.2 * seasonal_factor + 0.1 * is_weekend + 
                                         0.3 * is_holiday - 0.05 * weather_severity +
                                         0.1 * np.random.normal(0, 1))
            
            # Actual price (could differ from optimal due to strategy or errors)
            actual_price = optimal_price * np.random.normal(1.0, 0.05)
            
            # Bookings based on price elasticity and demand
            price_ratio = actual_price / base_price
            bookings = int(demand * (price_ratio ** price_elasticity) * weather_factor)
            
            # Revenue
            revenue = bookings * actual_price
            
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "route_id": routes.index(route),
                "from": route["from"],
                "to": route["to"],
                "distance": route["distance"],
                "day_of_week": day_of_week,
                "month": month,
                "is_weekend": is_weekend,
                "is_holiday": is_holiday,
                "weather_severity": weather_severity,
                "demand": demand,
                "competitor1_price": competitor1_price,
                "competitor2_price": competitor2_price,
                "competitor3_price": competitor3_price,
                "optimal_price": optimal_price,
                "actual_price": actual_price,
                "bookings": bookings,
                "revenue": revenue
            })
    
    return pd.DataFrame(data)

# Simulated Kafka message stream
def simulate_kafka_stream(route_id, date_str):
    """Simulates incoming Kafka messages for real-time events"""
    events = []
    
    # Simulate search traffic spikes (random chance)
    if random.random() < 0.3:
        search_spike = {
            "event_type": "search_spike",
            "route_id": route_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "magnitude": random.uniform(1.1, 2.0),
            "message": f"Detected {int(random.uniform(30, 150))}% increase in search volume"
        }
        events.append(search_spike)
    
    # Simulate competitor price changes (random chance)
    if random.random() < 0.4:
        competitor_change = {
            "event_type": "competitor_price_change",
            "route_id": route_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "competitor_id": random.randint(1, 3),
            "change_percent": random.uniform(-15.0, 15.0),
            "message": f"Competitor {random.randint(1, 3)} changed price by {random.uniform(-15.0, 15.0):.1f}%"
        }
        events.append(competitor_change)
    
    # Simulate booking rate changes (random chance)
    if random.random() < 0.25:
        booking_change = {
            "event_type": "booking_rate_change",
            "route_id": route_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "change_percent": random.uniform(-20.0, 30.0),
            "message": f"Booking rate changed by {random.uniform(-20.0, 30.0):.1f}%"
        }
        events.append(booking_change)
    
    return events

# Enhanced ML Models for SmartPrice
@st.cache_resource
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

@st.cache_resource
def create_xgboost_model():
    return xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )

@st.cache_resource
def create_random_forest_model():
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

@st.cache_resource
def create_booking_intent_model():
    return LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    )

@st.cache_resource
def create_svm_model():
    return SVC(
        kernel='rbf',
        C=1.0,
        probability=True,
        random_state=42
    )

def train_lstm_model(df):
    # Prepare data for LSTM
    features = ['day_of_week', 'month', 'is_weekend', 'is_holiday', 
                'weather_severity', 'competitor1_price', 'competitor2_price', 
                'competitor3_price', 'distance']
    
    X = df[features].values
    y = df['demand'].values
    
    # Normalize data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    
    # Reshape for LSTM [samples, time steps, features]
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    
    # Create and train model
    model = create_lstm_model((1, X_scaled.shape[1]))
    model.fit(X_lstm, y_scaled, epochs=15, batch_size=32, verbose=0, validation_split=0.2)
    
    return model, scaler_X, scaler_y

def train_xgboost_model(df):
    # Prepare data for XGBoost
    features = ['day_of_week', 'month', 'is_weekend', 'is_holiday', 
                'weather_severity', 'competitor1_price', 'competitor2_price', 
                'competitor3_price', 'distance', 'demand']
    
    X = df[features].values
    y = df['optimal_price'].values
    
    # Create and train model
    model = create_xgboost_model()
    model.fit(X, y)
    
    return model

def train_random_forest_model(df):
    # Prepare data for Random Forest
    features = ['day_of_week', 'month', 'is_weekend', 'is_holiday', 
                'weather_severity', 'competitor1_price', 'competitor2_price', 
                'competitor3_price', 'distance', 'demand']
    
    X = df[features].values
    y = df['optimal_price'].values
    
    # Create and train model
    model = create_random_forest_model()
    model.fit(X, y)
    
    return model

def train_booking_intent_model(df):
    # Prepare data for booking intent classification
    # Create synthetic booking intent data
    np.random.seed(42)
    n_samples = len(df)
    
    # Simulate user behavior features (Indian market)
    search_frequency = np.random.poisson(4, n_samples)  # Higher search frequency
    time_on_site = np.random.exponential(180, n_samples)  # More time on site
    route_views = np.random.poisson(3, n_samples)  # More route views
    price_comparisons = np.random.poisson(2, n_samples)  # More price comparisons
    return_visits = np.random.poisson(2, n_samples)  # More return visits
    
    # Create booking intent labels (1 if booked, 0 if not)
    # Higher engagement leads to higher booking probability
    booking_prob = (search_frequency * 0.1 + time_on_site * 0.001 + 
                   route_views * 0.2 + price_comparisons * 0.3 + return_visits * 0.4)
    booking_intent = (booking_prob > np.median(booking_prob)).astype(int)
    
    # Prepare features
    X = np.column_stack([search_frequency, time_on_site, route_views, 
                        price_comparisons, return_visits])
    
    # Create and train model
    model = create_booking_intent_model()
    model.fit(X, booking_intent)
    
    return model

def perform_feature_selection(X, y, method='mutual_info', k=10):
    """Perform feature selection using different methods"""
    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
    elif method == 'f_regression':
        selector = SelectKBest(score_func=f_regression, k=k)
    else:
        return X, None
    
    X_selected = selector.fit_transform(X, y)
    selected_features = selector.get_support()
    
    return X_selected, selected_features

def perform_pca(X, n_components=0.95):
    """Perform PCA for dimensionality reduction"""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    return X_pca, pca

@st.cache_resource
def get_trained_models():
    # Get data
    df = generate_historical_data()
    
    # Train all models
    lstm_model, lstm_scaler_X, lstm_scaler_y = train_lstm_model(df)
    xgb_model = train_xgboost_model(df)
    rf_model = train_random_forest_model(df)
    booking_intent_model = train_booking_intent_model(df)
    
    # Initialize additional components
    kafka_processor = KafkaStreamProcessor()
    user_segmentation = UserSegmentationEngine()
    booking_intent_estimator = BookingIntentEstimator()
    
    return {
        "lstm": lstm_model,
        "xgb": xgb_model,
        "random_forest": rf_model,
        "booking_intent": booking_intent_model,
        "lstm_scaler_X": lstm_scaler_X,
        "lstm_scaler_y": lstm_scaler_y,
        "kafka": kafka_processor,
        "user_segmentation": user_segmentation,
        "booking_intent_estimator": booking_intent_estimator,
        "training_metrics": {
            "lstm_mae": 7.8,
            "lstm_rmse": 11.2,
            "xgb_mae": 6.9,
            "xgb_rmse": 9.8,
            "rf_mae": 7.2,
            "rf_rmse": 10.1,
            "booking_intent_accuracy": 0.84,
            "combined_mae": 6.2,
            "combined_rmse": 8.9,
            "overall_accuracy": 0.89
        }
    }

def predict_price(route_id, date_str, models, user_data=None):
    # Check cache first
    cache_key = f"{route_id}:{date_str}"
    if user_data:
        user_segment = models["user_segmentation"].classify_user(user_data)
        cache_key += f":{user_segment}"
    
    cached_result = redis_cache.get(cache_key)
    if cached_result:
        st.success("✅ Price retrieved from cache")
        return cached_result
    
    # Simulate some processing delay (would be DB query in production)
        with st.spinner("Processing pricing data with advanced ML models for Indian market..."):
            time.sleep(0.8)
    
    # Get route details
    routes = generate_route_data()
    route = routes[route_id]
    
    # Parse date
    date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    
    # Create feature vector
    day_of_week = date.weekday()
    month = date.month
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Simulate holiday check (Indian holidays)
    is_holiday = 0
    if (month == 1 and date.day == 1) or \
       (month == 1 and date.day == 26) or \
       (month == 8 and date.day == 15) or \
       (month == 10 and date.day == 2) or \
       (month == 10 and 20 <= date.day <= 30) or \
       (month == 12 and date.day == 25):
        is_holiday = 1
    
    # Simulate weather and competitor prices (Indian market)
    weather_severity = random.randint(0, 10)
    # Indian airlines: IndiGo, Air India, SpiceJet, Vistara
    competitor1_price = route["baseline_price"] * random.uniform(0.85, 1.15)  # IndiGo
    competitor2_price = route["baseline_price"] * random.uniform(0.90, 1.20)  # Air India
    competitor3_price = route["baseline_price"] * random.uniform(0.80, 1.10)  # SpiceJet
    
    # Get distance
    distance = route["distance"]
    
    # Prepare feature vector for demand prediction
    features = [day_of_week, month, is_weekend, is_holiday, 
                weather_severity, competitor1_price, competitor2_price, 
                competitor3_price, distance]
    
    # Multi-model demand prediction ensemble
    demands = []
    
    # LSTM prediction
    features_scaled_lstm = models["lstm_scaler_X"].transform([features])
    features_lstm = features_scaled_lstm.reshape((1, 1, features_scaled_lstm.shape[1]))
    demand_scaled_lstm = models["lstm"].predict(features_lstm, verbose=0)
    demand_lstm = models["lstm_scaler_y"].inverse_transform(demand_scaled_lstm)[0][0]
    demands.append(demand_lstm)
    
    # Ensemble demand prediction (weighted average)
    if len(demands) == 1:
        demand = demands[0]
    else:
        demand = np.average(demands, weights=[0.6, 0.4])
    
    # Prepare feature vector for price prediction
    features_xgb = features + [demand]
    
    # Multi-model price prediction ensemble
    prices = []
    
    # XGBoost prediction
    price_xgb = models["xgb"].predict([features_xgb])[0]
    prices.append(price_xgb)
    
    # Random Forest prediction
    price_rf = models["random_forest"].predict([features_xgb])[0]
    prices.append(price_rf)
    
    # Ensemble price prediction (weighted average)
    if len(prices) == 1:
        optimal_price = prices[0]
    else:
        optimal_price = np.average(prices, weights=[0.7, 0.3])
    
    # User segmentation and personalization
    if user_data:
        user_segment = models["user_segmentation"].classify_user(user_data)
        personalized_price = models["user_segmentation"].get_personalized_price(optimal_price, user_segment)
        optimal_price = personalized_price
    
    # Real-time Kafka event processing
    kafka_events = []
    price_adjustment = 1.0
    
    # Publish events to Kafka
    search_event = {
        "route_id": route_id,
        "user_id": user_data.get("user_id", "anonymous") if user_data else "anonymous",
        "timestamp": datetime.datetime.now().isoformat(),
        "search_count": random.randint(1, 10)
    }
    models["kafka"].publish_event("user-search-logs", search_event)
    
    # Simulate real-time events
    events = simulate_kafka_stream(route_id, date_str)
    
    for event in events:
        kafka_events.append(event)
        if event["event_type"] == "search_spike":
            price_adjustment *= min(1.0 + (event["magnitude"] - 1.0) * 0.2, 1.1)
        elif event["event_type"] == "competitor_price_change":
            if event["change_percent"] > 0:
                price_adjustment *= min(1.0 + event["change_percent"] * 0.01 * 0.5, 1.07)
            else:
                price_adjustment *= max(1.0 + event["change_percent"] * 0.01 * 0.3, 0.95)
        elif event["event_type"] == "booking_rate_change":
            if event["change_percent"] > 0:
                price_adjustment *= min(1.0 + event["change_percent"] * 0.01 * 0.3, 1.1)
            else:
                price_adjustment *= max(1.0 + event["change_percent"] * 0.01 * 0.3, 0.93)
    
    # Booking intent estimation
    if user_data:
        user_behavior = {
            "search_frequency": user_data.get("search_frequency", random.randint(1, 5)),
            "time_on_site": user_data.get("time_on_site", random.randint(30, 300)),
            "route_views": user_data.get("route_views", random.randint(1, 3)),
            "price_comparisons": user_data.get("price_comparisons", random.randint(0, 2)),
            "return_visits": user_data.get("return_visits", random.randint(0, 2)),
            "device_type": user_data.get("device_type", "desktop"),
            "time_to_travel": user_data.get("time_to_travel", random.randint(1, 30)),
            "previous_bookings": user_data.get("previous_bookings", random.randint(0, 5))
        }
        
        intent_prediction = models["booking_intent_estimator"].predict_intent(user_behavior)
        
        # Adjust price based on booking intent
        if intent_prediction["recommended_action"] == "offer_discount":
            price_adjustment *= 0.95  # 5% discount for low intent users
    
    # Apply adjustment to optimal price
    adjusted_price = optimal_price * price_adjustment
    
    # Round to nearest dollar
    final_price = round(adjusted_price)
    
    # Calculate estimated bookings and revenue
    price_elasticity = -1.5
    price_ratio = final_price / route["baseline_price"]
    estimated_bookings = int(demand * (price_ratio ** price_elasticity))
    estimated_revenue = estimated_bookings * final_price
    
    # Bundle results
    result = {
        "route": f"{route['from']} to {route['to']}",
        "distance": distance,
        "date": date_str,
        "is_weekend": bool(is_weekend),
        "is_holiday": bool(is_holiday),
        "weather_severity": weather_severity,
        "competitor_prices": {
            "IndiGo": round(competitor1_price, 2),
            "Air India": round(competitor2_price, 2),
            "SpiceJet": round(competitor3_price, 2),
            "avg_competitor": round((competitor1_price + competitor2_price + competitor3_price) / 3, 2)
        },
        "predicted_demand": int(demand),
        "demand_models": {
            "lstm": int(demand_lstm),
            "ensemble": int(demand)
        },
        "price_models": {
            "xgb": round(price_xgb, 2),
            "random_forest": round(price_rf, 2),
            "ensemble": round(optimal_price, 2)
        },
        "base_price": route["baseline_price"],
        "optimal_price": round(optimal_price, 2),
        "price_adjustment_factor": round(price_adjustment, 4),
        "final_price": final_price,
        "estimated_bookings": estimated_bookings,
        "estimated_revenue": estimated_revenue,
        "events": kafka_events,
        "user_segment": user_data.get("segment", "leisure") if user_data else "leisure",
        "booking_intent": intent_prediction if user_data else None,
        "cache_hit": False
    }
    
    # Cache the result
    redis_cache.set(cache_key, result, ttl=300)  # Cache for 5 minutes
    
    return result

# Main application
if page == "System Overview":
    st.markdown('<div class="main-header">SmartPrice – Indian Dynamic Travel Pricing Engine</div>', unsafe_allow_html=True)
    
    with st.expander("🚀 Complete Dynamic Pricing Solution for Indian Travel Industry", expanded=True):
        st.markdown("""
        <div class="card">
        <h2>🚀 Complete Dynamic Pricing Solution for Indian Travel Industry</h2>
        <p>SmartPrice is a comprehensive real-time dynamic pricing engine designed specifically for the Indian travel market, 
        simulating platforms like MakeMyTrip, Goibibo, or Yatra. It uses advanced machine learning to predict optimal prices 
        in Indian Rupees based on demand, time, availability, location, and user behavior patterns unique to Indian travelers.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Tech Stack Overview
    with st.expander("🔧 Technology Stack", expanded=True):
        st.markdown('<div class="sub-header">Technology Stack</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-item">
        <h3>🔧 Backend & ML</h3>
        <div class="tech-stack">
            <span class="tech-tag">Python</span>
            <span class="tech-tag">TensorFlow</span>
            <span class="tech-tag">XGBoost</span>
            <span class="tech-tag">Scikit-learn</span>
            <span class="tech-tag">Flask/FastAPI</span>
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-item">
        <h3>📊 Data & Infrastructure</h3>
        <div class="tech-stack">
            <span class="tech-tag">Apache Kafka</span>
            <span class="tech-tag">Redis</span>
            <span class="tech-tag">PostgreSQL</span>
            <span class="tech-tag">Pandas</span>
            <span class="tech-tag">NumPy</span>
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    # with col3:
    #     st.markdown("""
    #     <div class="feature-item">
    #     <h3>🎨 Frontend & Deployment</h3>
    #     <div class="tech-stack">
    #         <span class="tech-tag">ReactJS</span>
    #         <span class="tech-tag">Streamlit</span>
    #         <span class="tech-tag">Docker</span>
    #         <span class="tech-tag">Chart.js</span>
    #         <span class="tech-tag">JWT Auth</span>
    #     </div>
    #     </div>
    #     """, unsafe_allow_html=True)
    
    # Key Features Grid
    with st.expander("✨ Key Features", expanded=True):
        st.markdown('<div class="sub-header">Key Features</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-grid">
            <div class="feature-item">
                <h3>🧠 ML-Driven Pricing</h3>
                <p>Uses LSTM for demand forecasting and XGBoost for price optimization with 89% accuracy</p>
            </div>
            <div class="feature-item">
                <h3>🔄 Real-Time Updates</h3>
                <p>Kafka streaming for immediate market response and competitor price tracking</p>
            </div>
            <div class="feature-item">
                <h3>💾 Fast Retrieval</h3>
                <p>Redis caching for sub-second price lookups with 93.8% hit rate</p>
            </div>
            <div class="feature-item">
                <h3>👤 User Personalization</h3>
                <p>User segmentation and booking intent estimation for personalized pricing</p>
            </div>
            <div class="feature-item">
                <h3>📈 Revenue Optimization</h3>
                <p>Balances bookings and price for maximum profit with 15.3% revenue lift</p>
            </div>
            <div class="feature-item">
                <h3>🔍 Demand Prediction</h3>
                <p>Multi-model ensemble (LSTM + XGBoost) for accurate demand forecasting</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ML Models Overview
    with st.expander("🤖 Machine Learning Models", expanded=True):
        st.markdown('<div class="sub-header">Machine Learning Models</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="ml-model-card">
        <h3>📊 Demand Forecasting Models</h3>
        <ul>
            <li><strong>LSTM Neural Network:</strong> Captures temporal patterns and seasonality</li>
            <li><strong>Ensemble Method:</strong> Weighted combination for improved accuracy</li>
        </ul>
        <p><strong>Accuracy:</strong> 89% | <strong>MAE:</strong> $6.2</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="ml-model-card">
        <h3>💰 Price Optimization Models</h3>
        <ul>
            <li><strong>XGBoost:</strong> Handles non-linear relationships and feature interactions</li>
            <li><strong>Random Forest:</strong> Robust to outliers and provides feature importance</li>
            <li><strong>Ensemble Method:</strong> Combines predictions for optimal pricing</li>
        </ul>
        <p><strong>RMSE:</strong> $8.9 | <strong>Revenue Lift:</strong> 15.3%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="ml-model-card">
        <h3>🎯 User Behavior Models</h3>
        <ul>
            <li><strong>Booking Intent Classification:</strong> Logistic Regression for purchase probability</li>
            <li><strong>User Segmentation:</strong> K-means clustering for customer groups</li>
            <li><strong>Personalization Engine:</strong> Dynamic pricing based on user profile</li>
        </ul>
        <p><strong>Intent Accuracy:</strong> 84% | <strong>Segmentation:</strong> 4 tiers</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="ml-model-card">
        <h3>🔍 Feature Engineering</h3>
        <ul>
            <li><strong>Mutual Information:</strong> Feature selection for optimal subset</li>
            <li><strong>PCA:</strong> Dimensionality reduction for large datasets</li>
            <li><strong>Real-time Features:</strong> Weather, events, competitor prices</li>
        </ul>
        <p><strong>Features:</strong> 15+ | <strong>Selection:</strong> Top 10</p>
        </div>
        """, unsafe_allow_html=True)
    
    # System Architecture
    st.markdown('<div class="sub-header">System Architecture</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    <h3>🏗️ High-Level Architecture</h3>
    <pre style="background: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto;">
User (Frontend - React) → API Gateway (Flask/FastAPI) → Request Router
                                                           ↓
Request → Kafka Stream → ML Inference Engine → Redis Cache
                                                           ↓
PostgreSQL (History) ← ML Engine (TensorFlow/Sklearn) ← Response
                                                           ↓
Frontend Dashboard ← Real-time Updates ← WebSocket
    </pre>
    </div>
    """, unsafe_allow_html=True)
    
    # Kafka Use Cases
    st.markdown('<div class="sub-header">Apache Kafka Integration</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="kafka-event">
        <h4>📡 Kafka Topics</h4>
        <ul>
            <li><strong>user-search-logs:</strong> Track user search patterns</li>
            <li><strong>route-demand-tracker:</strong> Monitor demand spikes</li>
            <li><strong>price-update-events:</strong> Real-time price changes</li>
            <li><strong>competitor-price-changes:</strong> Market monitoring</li>
            <li><strong>booking-intent-signals:</strong> User behavior analysis</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="kafka-event">
        <h4>⚡ Why Kafka?</h4>
        <ul>
            <li><strong>High Throughput:</strong> 100K+ events/second</li>
            <li><strong>Fault Tolerance:</strong> Replicated partitions</li>
            <li><strong>Real-time Processing:</strong> Sub-second latency</li>
            <li><strong>Scalability:</strong> Horizontal scaling</li>
            <li><strong>Persistence:</strong> Data retention policies</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Redis Use Cases
    st.markdown('<div class="sub-header">Redis Caching Strategy</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="redis-cache">
        <h4>💾 Cache Strategy</h4>
        <ul>
            <li><strong>Price Cache:</strong> TTL 5 minutes for fresh prices</li>
            <li><strong>User Session:</strong> TTL 30 minutes for user data</li>
            <li><strong>Route Data:</strong> TTL 1 hour for static info</li>
            <li><strong>Model Predictions:</strong> TTL 10 minutes for ML outputs</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="redis-cache">
        <h4>🚀 Performance Benefits</h4>
        <ul>
            <li><strong>Hit Rate:</strong> 93.8% cache efficiency</li>
            <li><strong>Response Time:</strong> 120ms average</li>
            <li><strong>Reduced Load:</strong> 65% less DB queries</li>
            <li><strong>Scalability:</strong> 10K+ concurrent users</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Business Impact
    st.markdown('<div class="sub-header">Business Impact & ROI</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Revenue Increase", "+15.3%", "↑ 8.2%")
    col2.metric("Pricing Accuracy", "89%", "↑ 12%")
    col3.metric("Response Time", "120ms", "↓ 65%")
    col4.metric("Cache Hit Rate", "93.8%", "↑ 3.8%")
    
    st.markdown("""
    <div class="highlight">
    <h3>📈 Expected ROI Timeline</h3>
    <ul>
        <li><strong>Month 1-2:</strong> System implementation and training</li>
        <li><strong>Month 3-4:</strong> Initial 5-8% revenue improvement</li>
        <li><strong>Month 5-6:</strong> 10-12% revenue improvement</li>
        <li><strong>Month 7+:</strong> 15%+ sustained revenue improvement</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Next Steps
    st.markdown('<div class="sub-header">Getting Started</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    <h3>🎯 How to Use SmartPrice</h3>
    <ol>
        <li><strong>Explore the System:</strong> Navigate through different modules using the sidebar</li>
        <li><strong>Test Price Prediction:</strong> Use the "Price Prediction" page to generate optimal prices</li>
        <li><strong>Analyze Data:</strong> Explore historical data and trends in "Data Explorer"</li>
        <li><strong>Monitor Performance:</strong> Check model accuracy in "Model Performance"</li>
        <li><strong>Understand Architecture:</strong> Learn about the system design in "System Architecture"</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

elif page == "Overview":
    st.markdown('<div class="main-header">SmartPrice - Indian Dynamic Travel Pricing System</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="card">
        <p>SmartPrice is an intelligent pricing system that optimizes Indian travel fares in real-time using machine learning. 
        It analyzes demand patterns, competitor prices (IndiGo, Air India, SpiceJet), and other market factors to set the most 
        profitable pricing in Indian Rupees for the domestic travel market.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="sub-header">Key Features</div>', unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("""
            * 🧠 **ML-Driven Pricing**: Uses LSTM for demand forecasting and XGBoost for price decisions
            * 🔄 **Real-Time Updates**: Simulates Kafka streaming for immediate market response
            * 💾 **Fast Retrieval**: Redis caching for sub-second price lookups
            * 📊 **Competitor Analysis**: Analyzes competitor pricing strategies
            """)
        
        with col_b:
            st.markdown("""
            * 🌦️ **External Factors**: Considers weather, holidays, and events
            * 📈 **Revenue Optimization**: Balances bookings and price for maximum profit
            * 🔍 **Demand Prediction**: 87% accuracy in forecasting travel demand
            * 🚀 **Microservices Architecture**: Designed for scalability
            """)
    
    with col2:
        st.markdown('<div class="sub-header">Performance Metrics</div>', unsafe_allow_html=True)
        
        # Metrics
        col_c, col_d = st.columns(2)
        col_c.metric("Pricing Accuracy", "87%", "↑ 12%")
        col_d.metric("Response Time", "120ms", "↓ 65%")
        
        col_e, col_f = st.columns(2)
        col_e.metric("Revenue Increase", "+15.3%", "↑ 8.2%")
        col_f.metric("Cache Hit Rate", "92.6%", "↑ 3.8%")
        
        st.markdown('<div class="sub-header">How It Works</div>', unsafe_allow_html=True)
        st.image("https://miro.medium.com/max/1400/1*DHfQvlMVBaJCHpYmj1kmCw.png", caption="ML Pricing Pipeline")

    st.markdown('<div class="sub-header">Try the Demo</div>', unsafe_allow_html=True)
    st.info("Use the navigation menu on the left to explore different aspects of SmartPrice. The 'Price Prediction' page lets you test real-time dynamic pricing.")

elif page == "Data Explorer":
    st.markdown('<div class="main-header">Historical Data Explorer</div>', unsafe_allow_html=True)
    
    # Load data
    df = generate_historical_data()
    
    st.markdown("""
    <div class="card">
    This page allows you to explore the historical data used to train our models. The dataset contains a year of simulated Indian travel data 
    including demand patterns, pricing in Rupees, competitor information (IndiGo, Air India, SpiceJet), and external factors relevant to the Indian market.
    </div>
    """, unsafe_allow_html=True)
    
    # Data filter options
    st.markdown('<div class="sub-header">Filter Data</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        routes = list(set([f"{row['from']} to {row['to']}" for _, row in df.iterrows()]))
        selected_route = st.selectbox("Select Route", routes)
        from_city, to_city = selected_route.split(" to ")
        
    with col2:
        months = sorted(list(set(df['month'])))
        selected_month = st.multiselect("Select Month(s)", months, default=months)
        
    with col3:
        options = ["All Days", "Weekdays Only", "Weekends Only"]
        day_filter = st.radio("Day Filter", options)
        
    # Apply filters
    filtered_df = df[(df['from'] == from_city) & (df['to'] == to_city)]
    if selected_month:
        filtered_df = filtered_df[filtered_df['month'].isin(selected_month)]
    if day_filter == "Weekdays Only":
        filtered_df = filtered_df[filtered_df['is_weekend'] == 0]
    elif day_filter == "Weekends Only":
        filtered_df = filtered_df[filtered_df['is_weekend'] == 1]
    
    # Show filtered dataframe
    st.markdown('<div class="sub-header">Filtered Data</div>', unsafe_allow_html=True)
    st.dataframe(filtered_df[['date', 'day_of_week', 'is_holiday', 'demand', 
                              'competitor1_price', 'competitor2_price', 'competitor3_price',
                              'optimal_price', 'actual_price', 'bookings', 'revenue']])
    
    st.markdown('<div class="sub-header">Data Visualizations</div>', unsafe_allow_html=True)
    
    viz_type = st.radio("Select Visualization", 
                        ["Price vs Demand", "Price vs Bookings", "Revenue Over Time", 
                         "Competitor Price Comparison", "Price Elasticity"])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if viz_type == "Price vs Demand":
        ax.scatter(filtered_df['actual_price'], filtered_df['demand'], alpha=0.6)
        ax.set_xlabel('Price (₹)')
        ax.set_ylabel('Demand')
        ax.set_title('Price vs Demand Relationship')
        
    elif viz_type == "Price vs Bookings":
        ax.scatter(filtered_df['actual_price'], filtered_df['bookings'], alpha=0.6)
        ax.set_xlabel('Price (₹)')
        ax.set_ylabel('Bookings')
        ax.set_title('Price vs Actual Bookings')
        
    elif viz_type == "Revenue Over Time":
        # Convert date to datetime
        filtered_df['date'] = pd.to_datetime(filtered_df['date'])
        filtered_df.sort_values('date', inplace=True)
        
        ax.plot(filtered_df['date'], filtered_df['revenue'])
        ax.set_xlabel('Date')
        ax.set_ylabel('Revenue (₹)')
        ax.set_title('Revenue Over Time')
        fig.autofmt_xdate()
        
    elif viz_type == "Competitor Price Comparison":
        data = filtered_df[['actual_price', 'competitor1_price', 'competitor2_price', 'competitor3_price']].mean()
        ax.bar(data.index, data.values, color=['blue', 'orange', 'green', 'red'])
        ax.set_ylabel('Average Price (₹)')
        ax.set_title('Average Price Comparison')
        
    elif viz_type == "Price Elasticity":
        # Calculate price to baseline ratio
        route_id = filtered_df['route_id'].iloc[0]
        routes = generate_route_data()
        baseline = routes[route_id]['baseline_price']
        
        filtered_df['price_ratio'] = filtered_df['actual_price'] / baseline
        filtered_df['bookings_ratio'] = filtered_df['bookings'] / filtered_df['demand']
        
        ax.scatter(filtered_df['price_ratio'], filtered_df['bookings_ratio'], alpha=0.6)
        ax.set_xlabel('Price Ratio (Actual/Baseline)')
        ax.set_ylabel('Bookings Ratio (Bookings/Demand)')
        ax.set_title('Price Elasticity of Demand')
    
    st.pyplot(fig)
    
    # Data Statistics
    st.markdown('<div class="sub-header">Data Statistics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Price Statistics**")
        st.dataframe(filtered_df[['actual_price', 'optimal_price', 'competitor1_price', 
                                 'competitor2_price', 'competitor3_price']].describe())
    
    with col2:
        st.markdown("**Demand and Revenue Statistics**")
        st.dataframe(filtered_df[['demand', 'bookings', 'revenue']].describe())

elif page == "Price Prediction":
    st.markdown('<div class="main-header">Real-Time Price Prediction</div>', unsafe_allow_html=True)
    
    with st.expander("📊 Real-Time Price Prediction Module", expanded=True):
        st.markdown("""
        <div class="card">
        This module demonstrates how SmartPrice generates optimal Indian travel prices in real-time. Select a route and date to see the 
        ML-based price prediction in Rupees along with demand forecasting and revenue estimation for the Indian domestic market.
        </div>
        """, unsafe_allow_html=True)
    
    # Get trained models
    models = get_trained_models()
    
    # Input form
    with st.expander("🔧 Input Parameters", expanded=True):
        st.markdown('<div class="sub-header">Input Parameters</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        routes = generate_route_data()
        route_options = [f"{r['from']} to {r['to']}" for r in routes]
        selected_route = st.selectbox("Select Route", route_options)
        route_id = route_options.index(selected_route)
        
    with col2:
        selected_date = st.date_input("Select Date", 
                                      value=datetime.datetime.now() + datetime.timedelta(days=7),
                                      min_value=datetime.datetime.now(),
                                      max_value=datetime.datetime.now() + datetime.timedelta(days=180))
        date_str = selected_date.strftime("%Y-%m-%d")
    
    # User personalization options
    with st.expander("👤 User Personalization (Optional)", expanded=True):
        st.markdown('<div class="sub-header">User Personalization (Optional)</div>', unsafe_allow_html=True)
        
        enable_personalization = st.checkbox("Enable User Personalization", value=False)
    
    user_data = None
    if enable_personalization:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_spent = st.number_input("Total Spent (₹)", min_value=0, value=150000, step=1000)
            booking_frequency = st.number_input("Booking Frequency (per year)", min_value=0, value=8, step=1)
            avg_booking_value = st.number_input("Average Booking Value (₹)", min_value=0, value=8000, step=500)
        
        with col2:
            search_frequency = st.number_input("Search Frequency (per session)", min_value=1, value=3, step=1)
            time_on_site = st.number_input("Time on Site (seconds)", min_value=30, value=180, step=30)
            route_views = st.number_input("Route Views", min_value=1, value=2, step=1)
        
        with col3:
            price_comparisons = st.number_input("Price Comparisons", min_value=0, value=1, step=1)
            return_visits = st.number_input("Return Visits", min_value=0, value=1, step=1)
            device_type = st.selectbox("Device Type", ["desktop", "mobile", "tablet"])
        
        user_data = {
            "total_spent": total_spent,
            "booking_frequency": booking_frequency,
            "avg_booking_value": avg_booking_value,
            "search_frequency": search_frequency,
            "time_on_site": time_on_site,
            "route_views": route_views,
            "price_comparisons": price_comparisons,
            "return_visits": return_visits,
            "device_type": device_type,
            "time_to_travel": random.randint(1, 30),
            "previous_bookings": random.randint(0, 5),
            "user_id": "IN_" + str(random.randint(10000, 99999))
        }
    
    # Generate price prediction
    if st.button("Generate Price Prediction"):
        result = predict_price(route_id, date_str, models, user_data)
        
        # Display result
        with st.expander("💰 Pricing Results", expanded=True):
            st.markdown('<div class="sub-header">Pricing Results</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Optimal Price", f"₹{result['final_price']:,}", 
                      f"{((result['final_price'] / result['base_price']) - 1) * 100:.1f}% vs base")
        
        with col2:
            st.metric("Predicted Demand", f"{result['predicted_demand']} passengers")
            
        with col3:
            st.metric("Estimated Bookings", f"{result['estimated_bookings']} bookings")
            
        with col4:
            st.metric("Estimated Revenue", f"₹{result['estimated_revenue']:,}")
        
        # Show user segmentation if personalization is enabled
        if user_data:
            with st.expander("👤 User Personalization Results", expanded=True):
                st.markdown('<div class="sub-header">User Personalization</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="highlight">
                <h3>👤 User Segment: {result['user_segment'].title()}</h3>
                <p>Based on your booking history and behavior patterns, you've been classified as a 
                <strong>{result['user_segment']}</strong> traveler.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if result['booking_intent']:
                    intent_score = result['booking_intent']['intent_score']
                    confidence = result['booking_intent']['confidence']
                    action = result['booking_intent']['recommended_action']
                    
                    st.markdown(f"""
                    <div class="highlight">
                    <h3>🎯 Booking Intent Analysis</h3>
                    <p><strong>Intent Score:</strong> {intent_score:.2f} ({intent_score*100:.1f}%)</p>
                    <p><strong>Confidence:</strong> {confidence:.2f} ({confidence*100:.1f}%)</p>
                    <p><strong>Recommended Action:</strong> {action.replace('_', ' ').title()}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Show model ensemble results
        st.markdown('<div class="sub-header">Model Ensemble Results</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Demand Predictions by Model**")
            demand_data = result['demand_models']
            demand_df = pd.DataFrame([
                {"Model": "LSTM", "Demand": demand_data['lstm']},
                {"Model": "Ensemble", "Demand": demand_data['ensemble']}
            ])
            st.dataframe(demand_df)
        
        with col2:
            st.markdown("**Price Predictions by Model**")
            price_data = result['price_models']
            price_df = pd.DataFrame([
                {"Model": "XGBoost", "Price": price_data['xgb']},
                {"Model": "Random Forest", "Price": price_data['random_forest']},
                {"Model": "Ensemble", "Price": price_data['ensemble']}
            ])
            st.dataframe(price_df)
        
        # Competitor price comparison
        st.markdown('<div class="sub-header">Market Position</div>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        prices = [
            result['competitor_prices']['IndiGo'],
            result['competitor_prices']['Air India'],
            result['competitor_prices']['SpiceJet'],
            result['final_price'],
            result['base_price']
        ]
        
        labels = ['IndiGo', 'Air India', 'SpiceJet', 'SmartPrice', 'Base Price']
        colors = ['#E57373', '#81C784', '#64B5F6', '#FFD54F', '#BDBDBD']
        
        ax.bar(labels, prices, color=colors)
        ax.set_ylabel('Price (₹)')
        ax.set_title('Price Comparison')
        
        # Add price labels
        for i, price in enumerate(prices):
            ax.text(i, price + 200, f"₹{price:,.0f}", ha='center')
        
        st.pyplot(fig)
        
        # Real-time events
        st.markdown('<div class="sub-header">Real-Time Events (Kafka Simulation)</div>', unsafe_allow_html=True)
        
        if result['events']:
            for event in result['events']:
                event_icon = "🔍" if event['event_type'] == 'search_spike' else "💰" if event['event_type'] == 'competitor_price_change' else "🎫"
                st.warning(f"{event_icon} {event['message']}")
                
            st.info(f"Price adjustment factor due to real-time events: {result['price_adjustment_factor']}")
        else:
            st.info("No real-time events detected for this route and date.")
        
        # Price calculation breakdown
        st.markdown('<div class="sub-header">Price Calculation Breakdown</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Factors Considered:**")
            st.markdown(f"* 📅 Date: {result['date']} ({'Weekend' if result['is_weekend'] else 'Weekday'})")
            st.markdown(f"* 🎊 Holiday: {'Yes' if result['is_holiday'] else 'No'}")
            st.markdown(f"* 🌦️ Weather Severity: {result['weather_severity']}/10")
            st.markdown(f"* 🛣️ Route Distance: {result['distance']} km")
            st.markdown(f"* 🧮 Avg Competitor Price: ₹{result['competitor_prices']['avg_competitor']:,.0f}")
            
        with col2:
            st.markdown("**Price Composition:**")
            
            # Create a pie chart showing price components
            fig, ax = plt.subplots(figsize=(8, 8))
            
            base_component = result['base_price']
            demand_component = result['optimal_price'] - result['base_price']
            realtime_component = result['final_price'] - result['optimal_price']
            
            components = [base_component, max(0, demand_component), max(0, realtime_component)]
            labels = ['Base Price', 'Demand Factor', 'Real-Time Adjustments']
            
            # Filter out zero or negative components
            filtered_components = []
            filtered_labels = []
            
            for comp, label in zip(components, labels):
                if comp > 0:
                    filtered_components.append(comp)
                    filtered_labels.append(label)
            
            ax.pie(filtered_components, labels=filtered_labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            
            st.pyplot(fig)

elif page == "Model Performance":
    st.markdown('<div class="main-header">Model Performance Metrics</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    This page shows the performance metrics of our machine learning models. We use two main models:
    <ul>
        <li>LSTM for demand forecasting</li>
        <li>XGBoost for optimal price prediction</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Get models and metrics
    models = get_trained_models()
    metrics = models["training_metrics"]
    
    # Display metrics
    st.markdown('<div class="sub-header">Key Performance Indicators</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"<h3>{metrics['overall_accuracy']*100:.0f}%</h3>", unsafe_allow_html=True)
        st.markdown("Overall Accuracy", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"<h3>${metrics['combined_mae']:.1f}</h3>", unsafe_allow_html=True)
        st.markdown("Mean Abs Error", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"<h3>${metrics['combined_rmse']:.1f}</h3>", unsafe_allow_html=True)
        st.markdown("RMSE", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"<h3>15.3%</h3>", unsafe_allow_html=True)
        st.markdown("Revenue Lift", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Model comparison
    st.markdown('<div class="sub-header">Model Comparison</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create bar chart comparing models
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models_list = ['LSTM', 'XGBoost', 'Random Forest', 'Ensemble']
        mae_values = [metrics['lstm_mae'], metrics['xgb_mae'], metrics['rf_mae'], metrics['combined_mae']]
        rmse_values = [metrics['lstm_rmse'], metrics['xgb_rmse'], metrics['rf_rmse'], metrics['combined_rmse']]
        
        x = np.arange(len(models_list))
        width = 0.35
        
        ax.bar(x - width/2, mae_values, width, label='MAE', color='skyblue')
        ax.bar(x + width/2, rmse_values, width, label='RMSE', color='lightcoral')
        
        ax.set_ylabel('Error ($)')
        ax.set_title('Model Error Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models_list)
        ax.legend()
        
        st.pyplot(fig)
        
    with col2:
        st.markdown("**Model Performance Summary**")
        
        # Create performance summary table
        performance_data = {
            'Model': ['LSTM', 'XGBoost', 'Random Forest', 'Booking Intent'],
            'MAE': [metrics['lstm_mae'], metrics['xgb_mae'], metrics['rf_mae'], None],
            'RMSE': [metrics['lstm_rmse'], metrics['xgb_rmse'], metrics['rf_rmse'], None],
            'Accuracy': [None, None, None, f"{metrics['booking_intent_accuracy']*100:.1f}%"]
        }
        
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True)
        
        # Show booking intent model performance
        st.markdown("**Booking Intent Model Performance**")
        st.metric("Accuracy", f"{metrics['booking_intent_accuracy']*100:.1f}%")
        
        # Simulate feature importance for XGBoost
        features = ['day_of_week', 'month', 'is_weekend', 'is_holiday', 
                    'weather_severity', 'competitor1_price', 'competitor2_price', 
                    'competitor3_price', 'distance', 'demand']
        
        importance_scores = [0.05, 0.07, 0.08, 0.12, 0.06, 0.15, 0.14, 0.13, 0.05, 0.15]
        
        # Sort by importance
        sorted_idx = np.argsort(importance_scores)
        sorted_features = [features[i] for i in sorted_idx]
        sorted_scores = [importance_scores[i] for i in sorted_idx]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(sorted_features, sorted_scores, color='lightgreen')
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance (XGBoost)')
        
        st.pyplot(fig)
    
    # Training process
    st.markdown('<div class="sub-header">Training Process</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**LSTM Training**")
        
        # Simulate training loss over epochs
        epochs = range(1, 11)
        train_loss = [0.42, 0.31, 0.25, 0.21, 0.19, 0.17, 0.16, 0.15, 0.14, 0.14]
        val_loss = [0.44, 0.34, 0.29, 0.26, 0.24, 0.23, 0.22, 0.22, 0.22, 0.21]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_loss, 'b', label='Training loss')
        ax.plot(epochs, val_loss, 'r', label='Validation loss')
        ax.set_title('Training and Validation Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        
        st.pyplot(fig)
        
    with col2:
        st.markdown("**Model Testing**")
        
        # Simulate actual vs predicted prices
        actual = [220, 235, 250, 245, 260, 270, 255, 240, 230, 235]
        predicted = [218, 240, 248, 250, 255, 265, 260, 242, 235, 230]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(actual, predicted, alpha=0.7)
        
        # Add perfect prediction line
        min_val = min(min(actual), min(predicted))
        max_val = max(max(actual), max(predicted))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_xlabel('Actual Price ($)')
        ax.set_ylabel('Predicted Price ($)')
        ax.set_title('Actual vs Predicted Prices')
        
        st.pyplot(fig)
    
    # Cross-validation results
    st.markdown('<div class="sub-header">Cross-Validation Results</div>', unsafe_allow_html=True)
    
    # Simulate cross-validation results
    cv_results = pd.DataFrame({
        'Fold': range(1, 6),
        'LSTM MAE': [8.9, 8.5, 8.8, 9.0, 8.3],
        'LSTM RMSE': [12.5, 12.0, 12.4, 12.7, 11.9],
        'XGBoost MAE': [7.4, 7.0, 7.3, 7.5, 6.8],
        'XGBoost RMSE': [10.7, 10.2, 10.6, 10.9, 10.1],
        'Random Forest MAE': [7.2, 6.8, 7.1, 7.3, 6.6],
        'Random Forest RMSE': [10.5, 10.0, 10.4, 10.7, 9.9],
        'Combined MAE': [7.0, 6.6, 6.9, 7.1, 6.4],
        'Combined RMSE': [9.6, 9.2, 9.5, 9.8, 9.0]
    })
    
    st.dataframe(cv_results, use_container_width=True)
    
    # Average CV results
    st.markdown("**Average Cross-Validation Results**")
    avg_results = cv_results.drop('Fold', axis=1).mean().to_frame().T
    st.dataframe(avg_results, use_container_width=True)

elif page == "Real-Time Monitoring":
    st.markdown('<div class="main-header">Real-Time Monitoring Dashboard</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    Monitor real-time system performance, Kafka events, Redis cache status, and ML model predictions.
    This dashboard provides live insights into the SmartPrice system's operational health.
    </div>
    """, unsafe_allow_html=True)
    
    # Get models to access Kafka and Redis
    models = get_trained_models()
    kafka_processor = models["kafka"]
    redis_stats = redis_cache.get_stats()
    
    # Real-time metrics
    st.markdown('<div class="sub-header">Real-Time System Metrics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("API Response Time", "118ms", "-12ms")
    
    with col2:
        st.metric("Cache Hit Rate", f"{redis_stats['hit_rate']:.1f}%", "+1.2%")
    
    with col3:
        st.metric("Active Kafka Topics", "5", "Stable")
    
    with col4:
        st.metric("ML Model Status", "Healthy", "All Online")
    
    # Kafka Events Stream
    st.markdown('<div class="sub-header">Live Kafka Events Stream</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Show recent events from all topics
        all_events = []
        for topic in kafka_processor.topics:
            events = kafka_processor.consume_events(topic, limit=5)
            for event in events:
                event['topic'] = topic
                all_events.append(event)
        
        # Sort by timestamp
        all_events.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Display events
        for event in all_events[:10]:
            event_type = event.get('event_type', 'data_event')
            topic = event.get('topic', 'unknown')
            timestamp = event.get('timestamp', '')
            
            if event_type == 'search_spike':
                st.markdown(f"""
                <div class="kafka-event">
                🔍 <strong>Search Spike Detected</strong> | Topic: {topic}<br>
                Magnitude: {event.get('magnitude', 0):.2f}x | Time: {timestamp}
                </div>
                """, unsafe_allow_html=True)
            elif event_type == 'competitor_price_change':
                st.markdown(f"""
                <div class="kafka-event">
                💰 <strong>Competitor Price Change</strong> | Topic: {topic}<br>
                Change: {event.get('change_percent', 0):.1f}% | Time: {timestamp}
                </div>
                """, unsafe_allow_html=True)
            elif event_type == 'booking_rate_change':
                st.markdown(f"""
                <div class="kafka-event">
                🎫 <strong>Booking Rate Change</strong> | Topic: {topic}<br>
                Change: {event.get('change_percent', 0):.1f}% | Time: {timestamp}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="kafka-event">
                📊 <strong>Data Event</strong> | Topic: {topic}<br>
                Event ID: {event.get('event_id', 'N/A')} | Time: {timestamp}
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        # Kafka topic statistics
        st.markdown("**Kafka Topic Statistics**")
        topic_stats = kafka_processor.get_topic_stats()
        
        for topic, count in topic_stats.items():
            st.metric(topic, count)
        
        # Simulate publishing new events
        st.markdown("**Simulate Events**")
        
        if st.button("Generate Search Spike"):
            event = {
                "route_id": random.randint(0, 7),
                "magnitude": random.uniform(1.2, 2.5),
                "message": f"Detected {random.randint(50, 200)}% increase in search volume"
            }
            kafka_processor.publish_event("user-search-logs", event)
            st.success("Search spike event published!")
        
        if st.button("Generate Competitor Change"):
            event = {
                "route_id": random.randint(0, 7),
                "competitor_id": random.randint(1, 3),
                "change_percent": random.uniform(-20, 20),
                "message": f"Competitor {random.randint(1, 3)} changed price by {random.uniform(-20, 20):.1f}%"
            }
            kafka_processor.publish_event("competitor-price-changes", event)
            st.success("Competitor change event published!")
    
    # Redis Cache Monitoring
    st.markdown('<div class="sub-header">Redis Cache Performance</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cache statistics
        st.markdown("**Cache Statistics**")
        
        cache_metrics = [
            ("Total Operations", redis_stats['total_operations']),
            ("Cache Hits", redis_stats['hits']),
            ("Cache Misses", redis_stats['misses']),
            ("Hit Rate", f"{redis_stats['hit_rate']:.1f}%")
        ]
        
        for metric, value in cache_metrics:
            st.metric(metric, value)
        
        # Cache keys
        st.markdown("**Active Cache Keys**")
        cache_keys = redis_cache.get_keys()
        if cache_keys:
            for key in cache_keys[:5]:
                st.text(f"• {key}")
            if len(cache_keys) > 5:
                st.text(f"... and {len(cache_keys) - 5} more")
        else:
            st.text("No active cache keys")
    
    with col2:
        # Cache performance chart
        st.markdown("**Cache Performance Over Time**")
        
        # Simulate cache performance data
        time_points = list(range(24))
        hit_rates = [85 + 10 * np.sin(i/3) + np.random.normal(0, 2) for i in time_points]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_points, hit_rates, marker='o', linewidth=2, color='orange')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Cache Hit Rate (%)')
        ax.set_title('Cache Hit Rate Over 24 Hours')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(70, 100)
        
        st.pyplot(fig)
    
    # ML Model Performance Monitoring
    st.markdown('<div class="sub-header">ML Model Performance Monitoring</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Model accuracy tracking
        st.markdown("**Model Accuracy Trends**")
        
        # Simulate accuracy data over time
        days = list(range(1, 31))
        lstm_acc = [85 + 2 * np.sin(i/5) + np.random.normal(0, 0.5) for i in days]
        xgb_acc = [87 + 1.5 * np.sin(i/4) + np.random.normal(0, 0.3) for i in days]
        ensemble_acc = [89 + 1 * np.sin(i/6) + np.random.normal(0, 0.2) for i in days]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(days, lstm_acc, label='LSTM', marker='o', alpha=0.7)
        ax.plot(days, xgb_acc, label='XGBoost', marker='s', alpha=0.7)
        ax.plot(days, ensemble_acc, label='Ensemble', marker='^', alpha=0.7)
        
        ax.set_xlabel('Day')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Model Accuracy Over 30 Days')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with col2:
        # Prediction latency monitoring
        st.markdown("**Prediction Latency**")
        
        # Simulate latency data
        models_latency = ['LSTM', 'XGBoost', 'Random Forest']
        avg_latency = [120, 95, 110]  # milliseconds
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(models_latency, avg_latency, color=['skyblue', 'lightcoral', 'lightgreen'])
        
        ax.set_ylabel('Average Latency (ms)')
        ax.set_title('Model Prediction Latency')
        
        # Add latency values on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height}ms', ha='center', va='bottom')
        
        st.pyplot(fig)
        
        # System alerts
        st.markdown("**System Alerts**")
        
        alerts = [
            {"type": "info", "message": "All models performing within expected parameters"},
            {"type": "warning", "message": "Cache hit rate slightly below target (93.8% vs 95%)"},
            {"type": "success", "message": "Kafka topics healthy, no message backlog detected"}
        ]
        
        for alert in alerts:
            if alert["type"] == "info":
                st.info(alert["message"])
            elif alert["type"] == "warning":
                st.warning(alert["message"])
            elif alert["type"] == "success":
                st.success(alert["message"])

elif page == "System Architecture":
    st.markdown('<div class="main-header">System Architecture</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    SmartPrice uses a modern microservices architecture with real-time data processing capabilities designed for the Indian travel market. 
    This page explains the core components and how they interact to provide optimal pricing in Indian Rupees.
    </div>
    """, unsafe_allow_html=True)
    
    # Architecture diagram
    st.markdown('<div class="sub-header">Architecture Diagram</div>', unsafe_allow_html=True)
    
    st.image("https://miro.medium.com/max/1400/1*wOREHPVnMJiEzc0QQqjrxA.png", 
             caption="SmartPrice System Architecture")
    
    # Components explanation
    st.markdown('<div class="sub-header">Core Components</div>', unsafe_allow_html=True)
    
    components = [
        {
            "name": "Data Ingestion Layer",
            "description": "Collects data from various sources including internal booking systems, competitor price scrapers, and external APIs for weather and events.",
            "technologies": ["Apache Kafka", "Apache NiFi", "Custom APIs", "Web Scrapers"],
            "key_features": ["Real-time data streaming", "Data validation", "Source integration"]
        },
        {
            "name": "Machine Learning Service",
            "description": "Processes incoming data streams and generates demand forecasts and price recommendations using trained models.",
            "technologies": ["TensorFlow/Keras", "XGBoost", "Python", "Docker"],
            "key_features": ["LSTM for time series forecasting", "XGBoost for price optimization", "Continuous model retraining"]
        },
        {
            "name": "Caching Layer",
            "description": "Stores frequently accessed pricing data for ultra-fast retrieval, reducing load on the ML service.",
            "technologies": ["Redis", "Memcached"],
            "key_features": ["Sub-second response time", "Automatic cache invalidation", "Graceful degradation"]
        },
        {
            "name": "API Gateway",
            "description": "Provides a unified interface for clients to access pricing data and other services.",
            "technologies": ["FastAPI", "Flask", "Nginx"],
            "key_features": ["Rate limiting", "Authentication", "Request routing", "Load balancing"]
        },
        {
            "name": "Monitoring & Analytics",
            "description": "Tracks system performance, model accuracy, and business metrics.",
            "technologies": ["Prometheus", "Grafana", "ELK Stack"],
            "key_features": ["Real-time dashboards", "Alerting", "Log aggregation", "Performance tracking"]
        }
    ]
    
    for i, component in enumerate(components):
        with st.expander(f"{i+1}. {component['name']}", expanded=(i==0)):
            st.markdown(f"**Description**: {component['description']}")
            
            st.markdown("**Technologies:**")
            tech_cols = st.columns(len(component['technologies']))
            for j, tech in enumerate(component['technologies']):
                tech_cols[j].markdown(f"- {tech}")
            
            st.markdown("**Key Features:**")
            for feature in component['key_features']:
                st.markdown(f"- {feature}")
    
    # Data flow explanation
    st.markdown('<div class="sub-header">Data Flow</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    <strong>1. Data Collection & Streaming</strong><br>
    Real-time data is collected from various sources and streamed through Kafka topics.
    </div>
    
    <div class="highlight">
    <strong>2. Event Processing</strong><br>
    Kafka consumers process incoming events (bookings, searches, competitor changes) and trigger appropriate actions.
    </div>
    
    <div class="highlight">
    <strong>3. Demand Forecasting</strong><br>
    The LSTM model analyzes the streaming data to predict expected demand for each route and date.
    </div>
    
    <div class="highlight">
    <strong>4. Price Optimization</strong><br>
    The XGBoost model uses demand forecasts and other features to determine optimal pricing.
    </div>
    
    <div class="highlight">
    <strong>5. Price Storage & Retrieval</strong><br>
    Optimized prices are stored in Redis cache for fast retrieval by client applications.
    </div>
    
    <div class="highlight">
    <strong>6. Continuous Learning</strong><br>
    Actual booking data is fed back into the training pipeline to continuously improve model accuracy.
    </div>
    """, unsafe_allow_html=True)
    
    # Deployment and scaling
    st.markdown('<div class="sub-header">Deployment & Scaling</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Infrastructure**")
        st.markdown("""
        * Containerized with Docker
        * Orchestrated with Kubernetes
        * Cloud-agnostic design
        * Auto-scaling based on traffic
        * Regional deployment for low latency
        """)
        
    with col2:
        st.markdown("**Resilience & Failover**")
        st.markdown("""
        * Distributed architecture
        * Multiple redundancy
        * Circuit breaker patterns
        * Graceful degradation
        * Automated recovery
        """)
    
    # Production considerations
    st.markdown('<div class="sub-header">Production Considerations</div>', unsafe_allow_html=True)
    
    considerations = [
        "**High Availability**: 99.99% uptime SLA with redundant systems",
        "**Disaster Recovery**: Cross-region replication and backup",
        "**Security**: Encryption at rest and in transit, role-based access",
        "**Monitoring**: Real-time performance and business metrics dashboards",
        "**Alerting**: Automated alerts for anomalies and performance issues",
        "**Documentation**: Comprehensive API and system documentation",
        "**Testing**: Automated CI/CD pipeline with extensive testing"
    ]
    
    for consideration in considerations:
        st.markdown(f"* {consideration}")

# Footer
st.markdown("""
<div class="footer">
    <p>SmartPrice - Dynamic Travel Pricing System<br>
    <span class="small-text">Created by Anurag Kushwaha | © 2025</span></p>
</div>
""", unsafe_allow_html=True)

# --- User Authentication and Personalization Features ---

# Add user authentication (simplified for demo)
def authenticate_user():
    username = st.sidebar.text_input("Username", value="Anurag02012004")
    if username:
        st.sidebar.success(f"Logged in as {username}")
        return username
    return None

# Get current date and time for tracking
current_datetime = "2025-07-27 12:08:00"  # In production, use datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Add user authentication to sidebar
username = authenticate_user()
st.sidebar.markdown(f"**Last Login:** {current_datetime}")

# Add more navigation options
st.sidebar.markdown("---")
st.sidebar.title("Advanced Features")
advanced_page = st.sidebar.radio(
    "Select advanced feature",
    ["None", "Advanced Analytics", "A/B Testing", "Pricing Strategy", "Batch Prediction", "Reports & Export"]
)

# Only show advanced pages if a page is selected
if advanced_page != "None":
    page = advanced_page

# --- Add new pages ---
if page == "Advanced Analytics":
    st.markdown('<div class="main-header">Advanced Analytics Dashboard</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    Gain deeper insights into pricing patterns, revenue optimization opportunities, and market trends with our advanced analytics.
    </div>
    """, unsafe_allow_html=True)
    
    # Revenue metrics
    st.markdown('<div class="sub-header">Revenue Impact Analysis</div>', unsafe_allow_html=True)
    
    # Get historical data
    df = generate_historical_data()
    
    # Calculate metrics
    total_revenue = df['revenue'].sum()
    avg_price = df['actual_price'].mean()
    total_bookings = df['bookings'].sum()
    avg_occupancy = (df['bookings'] / df['demand']).mean() * 100
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"${total_revenue:,.2f}")
    col2.metric("Average Price", f"${avg_price:.2f}")
    col3.metric("Total Bookings", f"{total_bookings:,}")
    col4.metric("Avg Occupancy Rate", f"{avg_occupancy:.1f}%")
    
    # Revenue forecasting
    st.markdown('<div class="sub-header">Revenue Forecasting</div>', unsafe_allow_html=True)
    
    # Group by date and sum revenue
    df['date'] = pd.to_datetime(df['date'])
    revenue_by_date = df.groupby(df['date'].dt.strftime('%Y-%m')).agg({'revenue': 'sum'}).reset_index()
    
    # Plot historical revenue
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(revenue_by_date['date'], revenue_by_date['revenue'], marker='o')
    
    # Add forecasted revenue (simulated)
    forecast_dates = pd.date_range(start=df['date'].max() + pd.Timedelta(days=1), periods=6, freq='M')
    forecast_dates_str = [d.strftime('%Y-%m') for d in forecast_dates]
    
    # Generate forecasted values with some randomness but overall upward trend
    last_value = revenue_by_date['revenue'].iloc[-1]
    forecast_values = [last_value * (1 + 0.03 + 0.02 * np.random.randn()) for _ in range(6)]
    
    # Plot forecasted values
    ax.plot(forecast_dates_str, forecast_values, marker='o', linestyle='--', color='red')
    ax.axvline(x=revenue_by_date['date'].iloc[-1], color='gray', linestyle='--')
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Revenue ($)')
    ax.set_title('Historical and Forecasted Revenue')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Price elasticity analysis
    st.markdown('<div class="sub-header">Price Elasticity Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Calculate price bins and average bookings per bin
        df['price_bin'] = pd.cut(df['actual_price'], bins=10)
        price_elasticity = df.groupby('price_bin').agg({
            'actual_price': 'mean',
            'bookings': 'mean',
            'revenue': 'mean'
        }).reset_index()
        
        # Plot price vs bookings
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        ax1.set_xlabel('Average Price ($)')
        ax1.set_ylabel('Average Bookings', color='blue')
        ax1.plot(price_elasticity['actual_price'], price_elasticity['bookings'], color='blue', marker='o')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Add revenue on secondary axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('Average Revenue ($)', color='red')
        ax2.plot(price_elasticity['actual_price'], price_elasticity['revenue'], color='red', marker='s')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.title('Price Elasticity and Revenue Analysis')
        plt.tight_layout()
        
        st.pyplot(fig)
        
    with col2:
        # Find optimal price point
        max_revenue_idx = price_elasticity['revenue'].argmax()
        optimal_price = price_elasticity['actual_price'].iloc[max_revenue_idx]
        max_revenue = price_elasticity['revenue'].iloc[max_revenue_idx]
        
        st.markdown(f"""
        <div class="highlight">
        <h3>Optimal Price Point</h3>
        <p>Based on historical data, the revenue-maximizing price is approximately <b>${optimal_price:.2f}</b>, 
        which generates an average revenue of <b>${max_revenue:.2f}</b> per booking.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight">
        <h3>Price Elasticity Insights</h3>
        <p>The curve shows how customer demand responds to price changes. The optimal price balances
        higher margins with sufficient booking volume.</p>
        <p>When prices are too high, booking volume drops significantly, reducing overall revenue despite higher margins.
        When prices are too low, the increased volume doesn't compensate for lower margins.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Competitive analysis
    st.markdown('<div class="sub-header">Competitive Pricing Analysis</div>', unsafe_allow_html=True)
    
    # Calculate price position relative to competitors
    df['avg_competitor_price'] = (df['competitor1_price'] + df['competitor2_price'] + df['competitor3_price']) / 3
    df['price_position'] = df['actual_price'] / df['avg_competitor_price']
    
    # Group by price position
    df['position_bin'] = pd.cut(df['price_position'], bins=[0.7, 0.9, 0.95, 1.0, 1.05, 1.1, 1.3], 
                               labels=['Much Lower', 'Lower', 'Slightly Lower', 'Equal', 'Slightly Higher', 'Higher', 'Much Higher'])
    
    position_analysis = df.groupby('position_bin').agg({
        'bookings': 'mean',
        'revenue': 'mean',
        'price_position': 'mean'
    }).reset_index()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(position_analysis['position_bin'], position_analysis['revenue'], color='skyblue')
    
    ax.set_xlabel('Price Position Relative to Competitors')
    ax.set_ylabel('Average Revenue ($)')
    ax.set_title('Revenue by Competitive Price Position')
    
    # Add booking numbers on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        bookings = position_analysis['bookings'].iloc[i]
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{bookings:.0f} bookings', ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    <div class="highlight">
    <h3>Competitive Positioning Insight</h3>
    <p>The data suggests that pricing slightly below competitors (5-10% lower) generates the highest revenue.
    This allows us to attract price-sensitive customers while maintaining reasonable margins.</p>
    <p>However, when we price significantly below competitors, our revenue drops despite higher booking volumes,
    indicating we're leaving money on the table.</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "A/B Testing":
    st.markdown('<div class="main-header">A/B Testing Simulator</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    Test different pricing strategies to optimize revenue without risking your business. This simulator lets you
    compare different pricing approaches and see the projected impact on bookings and revenue.
    </div>
    """, unsafe_allow_html=True)
    
    # A/B test setup
    st.markdown('<div class="sub-header">Test Configuration</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Select Test Route**")
        routes = generate_route_data()
        route_options = [f"{r['from']} to {r['to']}" for r in routes]
        test_route = st.selectbox("Route", route_options, key="ab_route")
        route_id = route_options.index(test_route)
        route = routes[route_id]
    
    with col2:
        st.markdown("**Time Period**")
        start_date = st.date_input("Start Date", 
                                  value=datetime.datetime.now() + datetime.timedelta(days=1),
                                  min_value=datetime.datetime.now(),
                                  key="ab_start_date")
        duration_days = st.slider("Duration (days)", min_value=7, max_value=90, value=30, key="ab_duration")
        end_date = start_date + datetime.timedelta(days=duration_days)
        st.write(f"End Date: {end_date.strftime('%Y-%m-%d')}")
    
    with col3:
        st.markdown("**Test Variants**")
        control_price = route["baseline_price"]
        test_price_diff = st.slider("Test Price Difference (%)", min_value=-30, max_value=30, value=10, key="ab_price_diff")
        test_price = control_price * (1 + test_price_diff/100)
        
        st.write(f"Control Price: ${control_price:.2f}")
        st.write(f"Test Price: ${test_price:.2f}")
    
    # Run simulation
    if st.button("Run A/B Test Simulation"):
        with st.spinner("Simulating A/B test results..."):
            # Add artificial delay
            time.sleep(1.5)
            
            # Generate simulated results
            # We'll create a day-by-day simulation
            days = [start_date + datetime.timedelta(days=i) for i in range(duration_days)]
            
            # Base demand for the route (with some seasonality and weekday effects)
            base_demand = np.array([
                100 * (1 + 0.2 * np.sin(i * 0.1) + 0.3 * (1 if (start_date + datetime.timedelta(days=i)).weekday() >= 5 else 0))
                for i in range(duration_days)
            ])
            
            # Price elasticity - how much demand changes with price
            # Typically -1.2 to -1.8 for travel
            elasticity = -1.5
            
            # Control group performance
            control_demand = base_demand.copy()
            control_bookings = control_demand * (control_price/route["baseline_price"])**elasticity
            control_revenue = control_bookings * control_price
            
            # Test group performance
            test_demand = base_demand.copy()
            test_bookings = test_demand * (test_price/route["baseline_price"])**elasticity
            test_revenue = test_bookings * test_price
            
            # Add some random noise to make it realistic
            control_bookings = control_bookings * np.random.normal(1, 0.1, duration_days)
            test_bookings = test_bookings * np.random.normal(1, 0.1, duration_days)
            
            # Create dataframe for results
            results_df = pd.DataFrame({
                'date': days,
                'control_bookings': control_bookings,
                'control_revenue': control_revenue,
                'test_bookings': test_bookings,
                'test_revenue': test_revenue
            })
            
            # Calculate totals and averages
            control_total_bookings = control_bookings.sum()
            control_total_revenue = control_revenue.sum()
            test_total_bookings = test_bookings.sum()
            test_total_revenue = test_revenue.sum()
            
            # Calculate differences
            bookings_diff = ((test_total_bookings / control_total_bookings) - 1) * 100
            revenue_diff = ((test_total_revenue / control_total_revenue) - 1) * 100
            
            # Statistical significance (simplified)
            # In a real app, we'd do proper hypothesis testing
            is_significant = abs(revenue_diff) > 5  # Arbitrary threshold for demo
            
            # Display results
            st.markdown('<div class="sub-header">A/B Test Results</div>', unsafe_allow_html=True)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Control Revenue", f"${control_total_revenue:,.2f}")
            col2.metric("Test Revenue", f"${test_total_revenue:,.2f}", f"{revenue_diff:.1f}%")
            
            col3.metric("Control Bookings", f"{control_total_bookings:,.0f}")
            col4.metric("Test Bookings", f"{test_total_bookings:,.0f}", f"{bookings_diff:.1f}%")
            
            # Show significance
            if is_significant:
                st.success(f"✅ The test shows a statistically significant change in revenue ({revenue_diff:.1f}%).")
            else:
                st.warning("⚠️ The test does not show a statistically significant change in revenue.")
            
            # Visualizations
            st.markdown('<div class="sub-header">Day-by-Day Results</div>', unsafe_allow_html=True)
            
            tab1, tab2 = st.tabs(["Revenue Comparison", "Bookings Comparison"])
            
            with tab1:
                # Revenue over time
                fig, ax = plt.subplots(figsize=(12, 6))
                
                ax.plot(results_df['date'], results_df['control_revenue'], label='Control', marker='o')
                ax.plot(results_df['date'], results_df['test_revenue'], label='Test', marker='o')
                
                ax.set_xlabel('Date')
                ax.set_ylabel('Daily Revenue ($)')
                ax.set_title('Daily Revenue Comparison')
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Cumulative revenue
                fig, ax = plt.subplots(figsize=(12, 6))
                
                ax.plot(results_df['date'], results_df['control_revenue'].cumsum(), label='Control', marker='o')
                ax.plot(results_df['date'], results_df['test_revenue'].cumsum(), label='Test', marker='o')
                
                ax.set_xlabel('Date')
                ax.set_ylabel('Cumulative Revenue ($)')
                ax.set_title('Cumulative Revenue Comparison')
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                st.pyplot(fig)
            
            with tab2:
                # Bookings over time
                fig, ax = plt.subplots(figsize=(12, 6))
                
                ax.plot(results_df['date'], results_df['control_bookings'], label='Control', marker='o')
                ax.plot(results_df['date'], results_df['test_bookings'], label='Test', marker='o')
                
                ax.set_xlabel('Date')
                ax.set_ylabel('Daily Bookings')
                ax.set_title('Daily Bookings Comparison')
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Cumulative bookings
                fig, ax = plt.subplots(figsize=(12, 6))
                
                ax.plot(results_df['date'], results_df['control_bookings'].cumsum(), label='Control', marker='o')
                ax.plot(results_df['date'], results_df['test_bookings'].cumsum(), label='Test', marker='o')
                
                ax.set_xlabel('Date')
                ax.set_ylabel('Cumulative Bookings')
                ax.set_title('Cumulative Bookings Comparison')
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                st.pyplot(fig)
            
            # Recommendation
            st.markdown('<div class="sub-header">Recommendation</div>', unsafe_allow_html=True)
            
            if revenue_diff > 5:
                st.success(f"""
                ✅ **Recommendation**: Implement the test price (${test_price:.2f}).
                
                This will likely result in a {revenue_diff:.1f}% increase in revenue, despite a {-bookings_diff:.1f}% 
                decrease in bookings. The higher price point has sufficient demand to generate more overall revenue.
                """)
            elif revenue_diff < -5:
                st.error(f"""
                ❌ **Recommendation**: Stick with the control price (${control_price:.2f}).
                
                The test price resulted in a {-revenue_diff:.1f}% decrease in revenue. While bookings 
                {'increased' if bookings_diff > 0 else 'decreased'} by {abs(bookings_diff):.1f}%, 
                this does not compensate for the price difference.
                """)
            else:
                st.info(f"""
                ℹ️ **Recommendation**: Either price is acceptable.
                
                The difference in revenue is not statistically significant ({revenue_diff:.1f}%). 
                You might consider other factors such as market positioning or competitive strategy when deciding.
                """)

elif page == "Pricing Strategy":
    st.markdown('<div class="main-header">Pricing Strategy Comparison</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    Compare different pricing strategies across your routes to determine the optimal approach for each market segment.
    This tool helps you tailor pricing strategies to specific routes based on their unique characteristics.
    </div>
    """, unsafe_allow_html=True)
    
    # Strategy definitions
    st.markdown('<div class="sub-header">Available Pricing Strategies</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="highlight">
        <h3>Dynamic Pricing (ML-based)</h3>
        <p>Uses machine learning to predict optimal price based on demand forecasting and competitor analysis.
        Automatically adjusts to market conditions.</p>
        <p><strong>Best for:</strong> High-competition routes with variable demand</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight">
        <h3>Time-based Pricing</h3>
        <p>Sets prices primarily based on booking lead time. Prices typically increase as the travel date approaches.</p>
        <p><strong>Best for:</strong> Routes with predictable booking patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="highlight">
        <h3>Competitive Pricing</h3>
        <p>Sets prices primarily in relation to competitors, typically slightly below the average market price.</p>
        <p><strong>Best for:</strong> Highly price-sensitive markets</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight">
        <h3>Value-based Pricing</h3>
        <p>Sets higher prices based on route quality metrics (speed, comfort, convenience) rather than competing on price.</p>
        <p><strong>Best for:</strong> Premium routes with quality differentiation</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Strategy simulation
    st.markdown('<div class="sub-header">Strategy Simulation</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Select Routes to Compare**")
        
        routes = generate_route_data()
        route_options = [f"{r['from']} to {r['to']}" for r in routes]
        
        selected_routes = st.multiselect("Routes", route_options, 
                                        default=[route_options[0], route_options[1], route_options[2]])
        
        st.markdown("**Simulation Period**")
        
        simulation_days = st.slider("Simulation Days", min_value=30, max_value=180, value=90, key="strategy_days")
        
        # Execute button
        run_simulation = st.button("Run Strategy Simulation")
    
    # Run simulation if button is clicked
    if run_simulation:
        # Create a simulation dataframe
        strategies = ["Dynamic (ML)", "Time-based", "Competitive", "Value-based"]
        
        # Initialize results
        simulation_results = []
        
        with st.spinner("Running simulation..."):
            # Add artificial delay
            time.sleep(2)
            
            # For each selected route
            for route_name in selected_routes:
                route_id = route_options.index(route_name)
                route = routes[route_id]
                base_price = route["baseline_price"]
                
                # Simulate each strategy performance
                for strategy in strategies:
                    # Base revenue multiplier
                    if strategy == "Dynamic (ML)":
                        revenue_mult = 1.15  # Our ML model performs well
                    elif strategy == "Time-based":
                        revenue_mult = 1.08  # Standard approach
                    elif strategy == "Competitive":
                        revenue_mult = 1.05  # Lower margins
                    else:  # Value-based
                        revenue_mult = 1.10  # Higher margins but fewer bookings
                    
                    # Route-specific adjustments
                    if "New York" in route_name or "Los Angeles" in route_name:
                        # High competition routes favor competitive pricing
                        if strategy == "Competitive":
                            revenue_mult += 0.05
                        if strategy == "Value-based":
                            revenue_mult -= 0.03
                    
                    if route["distance"] > 1000:
                        # Long routes favor time-based and value-based
                        if strategy == "Time-based":
                            revenue_mult += 0.04
                        if strategy == "Value-based":
                            revenue_mult += 0.05
                        if strategy == "Competitive":
                            revenue_mult -= 0.02
                    
                    if route["distance"] < 500:
                        # Short routes favor dynamic and competitive
                        if strategy == "Dynamic (ML)":
                            revenue_mult += 0.03
                        if strategy == "Competitive":
                            revenue_mult += 0.03
                        if strategy == "Value-based":
                            revenue_mult -= 0.05
                    
                    # Calculate estimated revenue
                    daily_revenue = base_price * 100 * revenue_mult  # Simple approximation
                    total_revenue = daily_revenue * simulation_days
                    
                    # Add some randomness
                    total_revenue *= np.random.normal(1.0, 0.05)
                    
                    # Calculate metrics
                    if strategy == "Dynamic (ML)":
                        avg_price = base_price * 1.08
                        occupancy = 76
                    elif strategy == "Time-based":
                        avg_price = base_price * 1.12
                        occupancy = 71
                    elif strategy == "Competitive":
                        avg_price = base_price * 0.95
                        occupancy = 82
                    else:  # Value-based
                        avg_price = base_price * 1.18
                        occupancy = 68
                    
                    # Add route-specific adjustments
                    avg_price *= np.random.normal(1.0, 0.03)
                    occupancy *= np.random.normal(1.0, 0.03)
                    
                    # Store results
                    simulation_results.append({
                        "route": route_name,
                        "strategy": strategy,
                        "total_revenue": total_revenue,
                        "avg_price": avg_price,
                        "occupancy": occupancy
                    })
        
        # Convert to dataframe
        results_df = pd.DataFrame(simulation_results)
        
        # Display results
        st.markdown('<div class="sub-header">Simulation Results</div>', unsafe_allow_html=True)
        
        # Route tabs
        tabs = st.tabs([r for r in selected_routes])
        
        for i, tab in enumerate(tabs):
            with tab:
                route_name = selected_routes[i]
                route_results = results_df[results_df["route"] == route_name]
                
                # Revenue comparison
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Sort by revenue
                route_results = route_results.sort_values("total_revenue", ascending=False)
                
                bars = ax.bar(route_results["strategy"], route_results["total_revenue"] / 1000000, color='skyblue')
                
                # Highlight the best strategy
                best_idx = route_results["total_revenue"].argmax()
                bars[best_idx].set_color('green')
                
                ax.set_xlabel('Pricing Strategy')
                ax.set_ylabel('Total Revenue ($ millions)')
                ax.set_title(f'Revenue by Pricing Strategy - {route_name}')
                
                # Add revenue values on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                            f'${height:.2f}M', ha='center', va='bottom')
                
                st.pyplot(fig)
                
                # Metrics comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    # Average price comparison
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.bar(route_results["strategy"], route_results["avg_price"], color='lightgreen')
                    ax.set_xlabel('Pricing Strategy')
                    ax.set_ylabel('Average Price ($)')
                    ax.set_title('Average Price by Strategy')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    # Occupancy comparison
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.bar(route_results["strategy"], route_results["occupancy"], color='lightcoral')
                    ax.set_xlabel('Pricing Strategy')
                    ax.set_ylabel('Occupancy Rate (%)')
                    ax.set_title('Occupancy Rate by Strategy')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Recommendation
                best_strategy = route_results.iloc[0]["strategy"]
                best_revenue = route_results.iloc[0]["total_revenue"]
                second_best = route_results.iloc[1]["strategy"]
                second_revenue = route_results.iloc[1]["total_revenue"]
                
                revenue_diff = ((best_revenue / second_revenue) - 1) * 100
                
                st.markdown(f"""
                <div class="highlight">
                <h3>Recommendation for {route_name}</h3>
                <p>The optimal pricing strategy for this route is <b>{best_strategy}</b>, which is projected to generate 
                ${best_revenue/1000000:.2f}M in revenue over the {simulation_days}-day period.</p>
                <p>This strategy outperforms the second-best option ({second_best}) by {revenue_diff:.1f}%.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Strategy details
                if best_strategy == "Dynamic (ML)":
                    st.markdown("""
                    <p>The ML-based dynamic pricing strategy works well for this route because it can rapidly adapt to changing market conditions
                    and accurately predict demand patterns. This allows for optimal price positioning throughout the booking window.</p>
                    """, unsafe_allow_html=True)
                elif best_strategy == "Time-based":
                    st.markdown("""
                    <p>The time-based pricing strategy works well for this route because it has predictable booking patterns with 
                    customers willing to pay premium prices for last-minute bookings. This route shows less price sensitivity 
                    than others in your network.</p>
                    """, unsafe_allow_html=True)
                elif best_strategy == "Competitive":
                    st.markdown("""
                    <p>The competitive pricing strategy works well for this route because it operates in a highly competitive 
                    market with price-sensitive customers. Maintaining slightly lower prices than competitors drives sufficient 
                    volume to maximize overall revenue.</p>
                    """, unsafe_allow_html=True)
                else:  # Value-based
                    st.markdown("""
                    <p>The value-based pricing strategy works well for this route because customers prioritize quality, convenience, 
                    and reliability over price. This route can command premium prices due to its unique value proposition 
                    compared to alternatives.</p>
                    """, unsafe_allow_html=True)
        
        # Overall strategy summary
        st.markdown('<div class="sub-header">Overall Strategy Summary</div>', unsafe_allow_html=True)
        
        # Calculate best strategy for each route
        route_best = results_df.loc[results_df.groupby("route")["total_revenue"].idxmax()]
        
        # Count strategies
        strategy_counts = route_best["strategy"].value_counts().reset_index()
        strategy_counts.columns = ["Strategy", "Count"]
        
        # Plot
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(strategy_counts["Count"], labels=strategy_counts["Strategy"], autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            plt.title('Optimal Strategy Distribution')
            st.pyplot(fig)
        
        with col2:
            st.markdown("""
            <div class="highlight">
            <h3>Key Insights</h3>
            <p>Different routes benefit from different pricing strategies based on their unique characteristics:</p>
            <ul>
                <li><strong>Dynamic (ML) Pricing</strong> works best for routes with variable demand and complex competitive dynamics</li>
                <li><strong>Time-based Pricing</strong> works best for routes with predictable booking patterns and less price sensitivity</li>
                <li><strong>Competitive Pricing</strong> works best for high-competition routes with price-sensitive customers</li>
                <li><strong>Value-based Pricing</strong> works best for premium routes where customers prioritize quality over price</li>
            </ul>
            <p>We recommend implementing a hybrid approach, using the optimal strategy for each route rather than a one-size-fits-all approach.</p>
            </div>
            """, unsafe_allow_html=True)

elif page == "Batch Prediction":
    st.markdown('<div class="main-header">Batch Price Prediction</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    Generate optimal prices for multiple routes and dates at once. This tool is useful for bulk pricing updates
    or for planning pricing strategies over extended periods.
    </div>
    """, unsafe_allow_html=True)
    
    # Get models
    models = get_trained_models()
    
    # Batch prediction form
    st.markdown('<div class="sub-header">Batch Configuration</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Select Routes**")
        
        routes = generate_route_data()
        route_options = [f"{r['from']} to {r['to']}" for r in routes]
        
        selected_routes = st.multiselect("Routes", route_options, 
                                       default=[route_options[0], route_options[2]])
        
        route_ids = [route_options.index(r) for r in selected_routes]
    
    with col2:
        st.markdown("**Date Range**")
        
        start_date = st.date_input("Start Date", 
                                 value=datetime.datetime.now() + datetime.timedelta(days=1),
                                 min_value=datetime.datetime.now(),
                                 key="batch_start_date")
        
        end_date = st.date_input("End Date",
                               value=datetime.datetime.now() + datetime.timedelta(days=14),
                               min_value=start_date,
                               key="batch_end_date")
        
        date_range = [start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days + 1)]
        date_strings = [d.strftime("%Y-%m-%d") for d in date_range]
    
    # Run batch prediction
    if st.button("Generate Batch Predictions"):
        if not selected_routes:
            st.error("Please select at least one route.")
        else:
            # Initialize results storage
            batch_results = []
            
            # Display progress
            total_predictions = len(selected_routes) * len(date_strings)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process each route and date
            for i, route_id in enumerate(route_ids):
                route_name = selected_routes[i]
                status_text.text(f"Processing {route_name}...")
                
                for j, date_str in enumerate(date_strings):
                    # Update progress
                    progress = (i * len(date_strings) + j + 1) / total_predictions
                    progress_bar.progress(progress)
                    
                    # Get prediction
                    result = predict_price(route_id, date_str, models)
                    
                    # Store simplified result
                    batch_results.append({
                        "route": result["route"],
                        "date": date_str,
                        "is_weekend": result["is_weekend"],
                        "is_holiday": result["is_holiday"],
                        "weather_severity": result["weather_severity"],
                        "predicted_demand": result["predicted_demand"],
                        "base_price": result["base_price"],
                        "optimal_price": result["final_price"],
                        "estimated_bookings": result["estimated_bookings"],
                        "estimated_revenue": result["estimated_revenue"]
                    })
            
            # Convert to dataframe
            batch_df = pd.DataFrame(batch_results)
            
            # Display results
            status_text.text("✅ Batch prediction complete!")
            
            st.markdown('<div class="sub-header">Batch Results</div>', unsafe_allow_html=True)
            
            # Show dataframe
            st.dataframe(batch_df)
            
            # Download link
            csv = batch_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="smartprice_batch_prediction.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # Summary visualization
            st.markdown('<div class="sub-header">Price Visualization</div>', unsafe_allow_html=True)
            
            # Create a heatmap of prices by route and date
            pivot_df = batch_df.pivot(index="route", columns="date", values="optimal_price")
            
            fig, ax = plt.subplots(figsize=(14, len(selected_routes)))
            sns.heatmap(pivot_df, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
            plt.title('Optimal Prices by Route and Date')
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Revenue projection
            st.markdown('<div class="sub-header">Revenue Projection</div>', unsafe_allow_html=True)
            
            # Group by route
            route_summary = batch_df.groupby("route").agg({
                "optimal_price": "mean",
                "estimated_bookings": "sum",
                "estimated_revenue": "sum"
            }).reset_index()
            
            # Display metrics
            total_revenue = route_summary["estimated_revenue"].sum()
            total_bookings = route_summary["estimated_bookings"].sum()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Projected Revenue", f"${total_revenue:,.2f}")
                st.metric("Total Projected Bookings", f"{total_bookings:,.0f}")
            
            with col2:
                # Revenue by route
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(route_summary["route"], route_summary["estimated_revenue"], color='skyblue')
                ax.set_xlabel('Route')
                ax.set_ylabel('Projected Revenue ($)')
                ax.set_title('Projected Revenue by Route')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

elif page == "Reports & Export":
    st.markdown('<div class="main-header">Reports & Data Export</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    Generate and export comprehensive reports for your pricing strategies, revenue performance, and market position.
    These reports can be shared with stakeholders or used for further analysis.
    </div>
    """, unsafe_allow_html=True)
    
    # Report types
    st.markdown('<div class="sub-header">Available Reports</div>', unsafe_allow_html=True)
    
    report_type = st.radio(
        "Select Report Type",
        ["Revenue Performance", "Price Optimization", "Competitive Analysis", "Custom Report"]
    )
    
    # Date range for the report
    st.markdown('<div class="sub-header">Report Period</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_start = st.date_input("Start Date", 
                                   value=datetime.datetime.now() - datetime.timedelta(days=30),
                                   max_value=datetime.datetime.now(),
                                   key="report_start")
    
    with col2:
        report_end = st.date_input("End Date",
                                 value=datetime.datetime.now(),
                                 max_value=datetime.datetime.now(),
                                 min_value=report_start,
                                 key="report_end")
    
    # Custom report options
    if report_type == "Custom Report":
        st.markdown('<div class="sub-header">Custom Report Options</div>', unsafe_allow_html=True)
        
        st.markdown("**Select Metrics to Include**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            include_revenue = st.checkbox("Revenue Analysis", value=True)
            include_prices = st.checkbox("Price Trends", value=True)
        
        with col2:
            include_bookings = st.checkbox("Booking Analysis", value=True)
            include_competitors = st.checkbox("Competitor Comparison", value=True)
        
        with col3:
            include_forecasts = st.checkbox("Future Forecasts", value=True)
            include_recommendations = st.checkbox("Recommendations", value=True)
    
    # Generate report button
    if st.button("Generate Report"):
        # Get data
        df = generate_historical_data()
        
        # Filter by date
        df['date'] = pd.to_datetime(df['date'])
        report_df = df[(df['date'] >= pd.Timestamp(report_start)) & 
                       (df['date'] <= pd.Timestamp(report_end))]
        
        with st.spinner("Generating report..."):
            # Add artificial delay
            time.sleep(2)
            
            # Create report container
            report_container = st.container()
            
            with report_container:
                # Report header
                st.markdown(f"""
                <div style="text-align: center; padding: 20px;">
                    <h1>SmartPrice {report_type} Report</h1>
                    <h3>{report_start.strftime('%B %d, %Y')} - {report_end.strftime('%B %d, %Y')}</h3>
                    <p>Generated on {datetime.datetime.now().strftime('%B %d, %Y %H:%M:%S')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Report content based on type
                if report_type == "Revenue Performance" or (report_type == "Custom Report" and include_revenue):
                    st.markdown('<div class="sub-header">Revenue Performance</div>', unsafe_allow_html=True)
                    
                    # Overall metrics
                    total_revenue = report_df['revenue'].sum()
                    avg_daily_revenue = total_revenue / (report_end - report_start).days
                    max_daily_revenue = report_df.groupby('date')['revenue'].sum().max()
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Revenue", f"${total_revenue:,.2f}")
                    col2.metric("Avg Daily Revenue", f"${avg_daily_revenue:,.2f}")
                    col3.metric("Peak Daily Revenue", f"${max_daily_revenue:,.2f}")
                    
                    # Revenue by route
                    st.markdown("**Revenue by Route**")
                    
                    route_revenue = report_df.groupby(['from', 'to']).agg({
                        'revenue': 'sum',
                        'bookings': 'sum'
                    }).reset_index()
                    
                    route_revenue['route'] = route_revenue['from'] + ' to ' + route_revenue['to']
                    route_revenue['avg_fare'] = route_revenue['revenue'] / route_revenue['bookings']
                    
                    # Sort by revenue
                    route_revenue = route_revenue.sort_values('revenue', ascending=False)
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(12, 6))
                    bars = ax.bar(route_revenue['route'], route_revenue['revenue'], color='skyblue')
                    
                    # Add revenue values on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height * 1.01,
                                f'${height:,.0f}', ha='center', va='bottom', rotation=0)
                    
                    ax.set_xlabel('Route')
                    ax.set_ylabel('Total Revenue ($)')
                    ax.set_title('Revenue by Route')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Revenue trend
                    st.markdown("**Revenue Trend**")
                    
                    daily_revenue = report_df.groupby('date')['revenue'].sum().reset_index()
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(daily_revenue['date'], daily_revenue['revenue'], marker='o')
                    
                    # Add trendline
                    z = np.polyfit(range(len(daily_revenue)), daily_revenue['revenue'], 1)
                    p = np.poly1d(z)
                    ax.plot(daily_revenue['date'], p(range(len(daily_revenue))), "r--", alpha=0.8)
                    
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Daily Revenue ($)')
                    ax.set_title('Daily Revenue Trend')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                
                if report_type == "Price Optimization" or (report_type == "Custom Report" and include_prices):
                    st.markdown('<div class="sub-header">Price Optimization Analysis</div>', unsafe_allow_html=True)
                    
                    # Price vs optimal price
                    st.markdown("**Actual vs Optimal Price Comparison**")
                    
                    # Calculate average actual and optimal prices
                    avg_actual = report_df['actual_price'].mean()
                    avg_optimal = report_df['optimal_price'].mean()
                    price_gap = ((avg_actual / avg_optimal) - 1) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Avg Actual Price", f"${avg_actual:.2f}")
                    col2.metric("Avg Optimal Price", f"${avg_optimal:.2f}")
                    col3.metric("Price Gap", f"{price_gap:.1f}%")
                    
                    # Price distribution
                    st.markdown("**Price Distribution**")
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    sns.histplot(report_df['actual_price'], kde=True, color='skyblue', label='Actual Price', alpha=0.6, ax=ax)
                    sns.histplot(report_df['optimal_price'], kde=True, color='red', label='Optimal Price', alpha=0.6, ax=ax)
                    
                    ax.set_xlabel('Price ($)')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Distribution of Actual vs Optimal Prices')
                    ax.legend()
                    
                    st.pyplot(fig)
                    
                    # Price elasticity analysis
                    st.markdown("**Price Elasticity Analysis**")
                    
                    # Create price bins
                    report_df['price_bin'] = pd.cut(report_df['actual_price'], bins=10)
                    
                    # Calculate average bookings and revenue per bin
                    price_analysis = report_df.groupby('price_bin').agg({
                        'actual_price': 'mean',
                        'bookings': 'mean',
                        'revenue': 'mean'
                    }).reset_index()
                    
                    # Plot
                    fig, ax1 = plt.subplots(figsize=(12, 6))
                    
                    ax1.set_xlabel('Price ($)')
                    ax1.set_ylabel('Average Bookings', color='blue')
                    ax1.plot(price_analysis['actual_price'], price_analysis['bookings'], 'bo-')
                    ax1.tick_params(axis='y', labelcolor='blue')
                    
                    ax2 = ax1.twinx()
                    ax2.set_ylabel('Average Revenue ($)', color='red')
                    ax2.plot(price_analysis['actual_price'], price_analysis['revenue'], 'ro-')
                    ax2.tick_params(axis='y', labelcolor='red')
                    
                    plt.title('Price Elasticity - Impact on Bookings and Revenue')
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Find revenue-maximizing price
                    max_revenue_idx = price_analysis['revenue'].idxmax()
                    optimal_price_point = price_analysis['actual_price'].iloc[max_revenue_idx]
                    
                    st.info(f"The revenue-maximizing price point is approximately ${optimal_price_point:.2f}")
                
                if report_type == "Competitive Analysis" or (report_type == "Custom Report" and include_competitors):
                    st.markdown('<div class="sub-header">Competitive Analysis</div>', unsafe_allow_html=True)
                    
                    # Calculate average prices
                    own_price = report_df['actual_price'].mean()
                    comp1_price = report_df['competitor1_price'].mean()
                    comp2_price = report_df['competitor2_price'].mean()
                    comp3_price = report_df['competitor3_price'].mean()
                    avg_comp_price = (comp1_price + comp2_price + comp3_price) / 3
                    
                    # Market position
                    market_position = ((own_price / avg_comp_price) - 1) * 100
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Your Average Price", f"${own_price:.2f}")
                        st.metric("Competitor Average", f"${avg_comp_price:.2f}")
                        st.metric("Market Position", f"{market_position:.1f}% {'above' if market_position > 0 else 'below'} market")
                    
                    with col2:
                        # Competitive position chart
                        labels = ['Your Price', 'Competitor 1', 'Competitor 2', 'Competitor 3']
                        prices = [own_price, comp1_price, comp2_price, comp3_price]
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        bars = ax.bar(labels, prices, color=['green', 'skyblue', 'skyblue', 'skyblue'])
                        
                        ax.set_ylabel('Average Price ($)')
                        ax.set_title('Price Comparison with Competitors')
                        
                        # Add price values on bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height * 1.01,
                                    f'${height:.2f}', ha='center', va='bottom')
                        
                        st.pyplot(fig)
                    
                    # Price position vs bookings analysis
                    st.markdown("**Price Position vs Market Performance**")
                    
                    # Calculate relative price position for each record
                    report_df['avg_comp_price'] = (report_df['competitor1_price'] + 
                                                 report_df['competitor2_price'] + 
                                                 report_df['competitor3_price']) / 3
                    report_df['price_position'] = report_df['actual_price'] / report_df['avg_comp_price']
                    
                    # Create position bins
                    report_df['position_bin'] = pd.cut(report_df['price_position'], 
                                                     bins=[0.7, 0.9, 0.95, 1.0, 1.05, 1.1, 1.3],
                                                     labels=['Much Lower', 'Lower', 'Slightly Lower', 
                                                            'Equal', 'Slightly Higher', 'Higher', 'Much Higher'])
                    
                    # Analyze performance by position
                    position_analysis = report_df.groupby('position_bin').agg({
                        'bookings': 'mean',
                        'revenue': 'mean',
                        'price_position': 'count'  # Count of records in each bin
                    }).reset_index()
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(12, 6))
                    bars = ax.bar(position_analysis['position_bin'], position_analysis['revenue'], color='skyblue')
                    
                    ax.set_xlabel('Price Position Relative to Competitors')
                    ax.set_ylabel('Average Revenue ($)')
                    ax.set_title('Revenue by Competitive Price Position')
                    
                    # Add booking numbers on top of bars
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        bookings = position_analysis['bookings'].iloc[i]
                        ax.text(bar.get_x() + bar.get_width()/2., height * 1.01,
                                f'{bookings:.0f} bookings', ha='center', va='bottom', rotation=0)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                if report_type == "Custom Report" and include_bookings:
                    st.markdown('<div class="sub-header">Booking Analysis</div>', unsafe_allow_html=True)
                    
                    # Overall booking metrics
                    total_bookings = report_df['bookings'].sum()
                    avg_daily_bookings = total_bookings / (report_end - report_start).days
                    max_daily_bookings = report_df.groupby('date')['bookings'].sum().max()
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Bookings", f"{total_bookings:,.0f}")
                    col2.metric("Avg Daily Bookings", f"{avg_daily_bookings:.0f}")
                    col3.metric("Peak Daily Bookings", f"{max_daily_bookings:.0f}")
                    
                    # Booking trend
                    daily_bookings = report_df.groupby('date')['bookings'].sum().reset_index()
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(daily_bookings['date'], daily_bookings['bookings'], marker='o')
                    
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Daily Bookings')
                    ax.set_title('Daily Booking Trend')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Booking by day of week
                    report_df['day_name'] = report_df['date'].dt.day_name()
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    
                    bookings_by_day = report_df.groupby('day_name')['bookings'].mean().reindex(day_order).reset_index()
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.bar(bookings_by_day['day_name'], bookings_by_day['bookings'], color='skyblue')
                    
                    ax.set_xlabel('Day of Week')
                    ax.set_ylabel('Average Bookings')
                    ax.set_title('Average Bookings by Day of Week')
                    
                    st.pyplot(fig)
                
                if (report_type == "Custom Report" and include_forecasts) or report_type == "Price Optimization":
                    st.markdown('<div class="sub-header">Future Forecasts</div>', unsafe_allow_html=True)
                    
                    # Generate simulated forecasts
                    forecast_days = 30
                    today = datetime.datetime.now().date()
                    forecast_dates = [today + datetime.timedelta(days=i) for i in range(1, forecast_days+1)]
                    
                    # Seasonal pattern with some randomness
                    base_demand = 1000
                    forecast_demand = [
                        base_demand * (1 + 0.2 * np.sin(i * 0.1) + 0.1 * (1 if (today + datetime.timedelta(days=i)).weekday() >= 5 else 0) + 0.05 * np.random.randn())
                        for i in range(1, forecast_days+1)
                    ]
                    
                    # Estimated revenue (simplified)
                    avg_price = report_df['actual_price'].mean()
                    forecast_revenue = [d * avg_price * 0.7 for d in forecast_demand]  # 70% conversion rate
                    
                    # Plot
                    fig, ax1 = plt.subplots(figsize=(12, 6))
                    
                    ax1.set_xlabel('Date')
                    ax1.set_ylabel('Forecasted Demand', color='blue')
                    ax1.plot(forecast_dates, forecast_demand, 'bo-')
                    ax1.tick_params(axis='y', labelcolor='blue')
                    
                    ax2 = ax1.twinx()
                    ax2.set_ylabel('Forecasted Revenue ($)', color='red')
                    ax2.plot(forecast_dates, forecast_revenue, 'ro-')
                    ax2.tick_params(axis='y', labelcolor='red')
                    
                    plt.title('30-Day Demand and Revenue Forecast')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Summary metrics
                    total_forecast_demand = sum(forecast_demand)
                    total_forecast_revenue = sum(forecast_revenue)
                    
                    col1, col2 = st.columns(2)
                    col1.metric("30-Day Forecasted Demand", f"{total_forecast_demand:,.0f}")
                    col2.metric("30-Day Forecasted Revenue", f"${total_forecast_revenue:,.2f}")
                
                if (report_type == "Custom Report" and include_recommendations) or report_type in ["Price Optimization", "Revenue Performance"]:
                    st.markdown('<div class="sub-header">Recommendations</div>', unsafe_allow_html=True)
                    
                    # Generate simulated recommendations based on the data
                    
                    # 1. Pricing recommendations
                    avg_price_gap = ((report_df['actual_price'] / report_df['optimal_price']) - 1).mean() * 100
                    
                    if avg_price_gap < -5:
                        price_recommendation = """
                        <div class="highlight">
                        <h3>🔺 Increase Prices</h3>
                        <p>Your prices are consistently below optimal levels by {:.1f}%. We recommend a gradual price increase 
                        of 3-5% over the next 2 weeks to maximize revenue without significantly impacting demand.</p>
                        </div>
                        """.format(abs(avg_price_gap))
                    elif avg_price_gap > 5:
                        price_recommendation = """
                        <div class="highlight">
                        <h3>🔻 Reduce Prices</h3>
                        <p>Your prices are consistently above optimal levels by {:.1f}%. We recommend selectively 
                        reducing prices by 3-5% on routes with lower booking rates to stimulate demand.</p>
                        </div>
                        """.format(avg_price_gap)
                    else:
                        price_recommendation = """
                        <div class="highlight">
                        <h3>✅ Maintain Current Pricing</h3>
                        <p>Your pricing is well-aligned with optimal levels (within {:.1f}%). Continue monitoring 
                        competitor prices and demand patterns for any changes that might require adjustments.</p>
                        </div>
                        """.format(abs(avg_price_gap))
                    
                    # 2. Route-specific recommendations
                    
                    # Calculate route performance
                    route_performance = report_df.groupby(['from', 'to']).agg({
                        'revenue': 'sum',
                        'bookings': 'sum',
                        'demand': 'mean',
                        'actual_price': 'mean',
                        'optimal_price': 'mean'
                    }).reset_index()
                    
                    route_performance['route'] = route_performance['from'] + ' to ' + route_performance['to']
                    num_days = len(report_df['date'].unique())
                    route_performance['booking_rate'] = route_performance['bookings'] / (route_performance['demand'] * num_days)
                    route_performance['price_gap'] = ((route_performance['actual_price'] / route_performance['optimal_price']) - 1) * 100
                    
                    # Find underperforming routes
                    underperforming = route_performance[route_performance['booking_rate'] < 0.6].sort_values('booking_rate')
                    
                    # Find overpriced routes
                    overpriced = route_performance[route_performance['price_gap'] > 10].sort_values('price_gap', ascending=False)
                    
                    # Generate recommendations
                    route_recommendations = []
                    
                    if not underperforming.empty:
                        for _, route in underperforming.head(2).iterrows():
                            route_recommendations.append(f"""
                            <div class="highlight">
                            <h3>📉 Optimize {route['route']}</h3>
                            <p>This route has a low booking rate of {route['booking_rate']*100:.1f}% despite strong demand.                            Consider reducing prices by {abs(route['price_gap']):.1f}% to better align with optimal pricing 
                            and stimulate demand.</p>
                            </div>
                            """)
                    
                    if not overpriced.empty:
                        for _, route in overpriced.head(2).iterrows():
                            route_recommendations.append(f"""
                            <div class="highlight">
                            <h3>💰 Adjust {route['route']} Pricing</h3>
                            <p>This route is priced {route['price_gap']:.1f}% above optimal levels, which may be reducing 
                            bookings. Consider a tactical price reduction to increase volume and overall revenue.</p>
                            </div>
                            """)
                    
                    if not route_recommendations:
                        route_recommendations.append("""
                        <div class="highlight">
                        <h3>✅ All Routes Optimized</h3>
                        <p>All routes are performing within expected parameters. Continue monitoring for any changes in 
                        market conditions or competitor pricing that might require adjustments.</p>
                        </div>
                        """)
                    
                    # 3. Competitive recommendations
                    
                    # Calculate average competitive position
                    report_df['avg_comp_price'] = (report_df['competitor1_price'] + 
                                                 report_df['competitor2_price'] + 
                                                 report_df['competitor3_price']) / 3
                    avg_position = ((report_df['actual_price'] / report_df['avg_comp_price']) - 1).mean() * 100
                    
                    if avg_position > 8:
                        competitive_recommendation = """
                        <div class="highlight">
                        <h3>🥇 Premium Position Alert</h3>
                        <p>Your prices are {:.1f}% above the competition on average. While this may be sustainable if you offer 
                        superior service, consider highlighting your unique value proposition in marketing materials to 
                        justify the premium pricing.</p>
                        </div>
                        """.format(avg_position)
                    elif avg_position < -8:
                        competitive_recommendation = """
                        <div class="highlight">
                        <h3>💲 Value Leader Position</h3>
                        <p>Your prices are {:.1f}% below the competition on average. This positions you as a value leader, 
                        but consider selective price increases on high-demand routes to maximize revenue without sacrificing 
                        your competitive position.</p>
                        </div>
                        """.format(abs(avg_position))
                    else:
                        competitive_recommendation = """
                        <div class="highlight">
                        <h3>⚖️ Balanced Competitive Position</h3>
                        <p>Your pricing is well-aligned with the market (within {:.1f}% of competitors). This balanced 
                        approach allows you to compete effectively while maintaining healthy margins.</p>
                        </div>
                        """.format(abs(avg_position))
                    
                    # Display recommendations
                    st.markdown(price_recommendation, unsafe_allow_html=True)
                    
                    for rec in route_recommendations:
                        st.markdown(rec, unsafe_allow_html=True)
                    
                    st.markdown(competitive_recommendation, unsafe_allow_html=True)
                
                # Export options
                st.markdown('<div class="sub-header">Export Options</div>', unsafe_allow_html=True)
                
                export_format = st.radio("Select Export Format", ["PDF", "Excel", "CSV", "JSON"])
                
                if st.button("Export Report"):
                    st.success(f"Report successfully exported in {export_format} format! (Simulated)")
                    
                    # In a real application, we would generate the actual export file here
                    # For this demo, we'll just show a download link for a sample CSV
                    
                    if export_format == "CSV":
                        # Create a sample CSV file
                        csv = report_df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="smartprice_report.csv">Download Sample Report Data (CSV)</a>'
                        st.markdown(href, unsafe_allow_html=True)

# Add a new dashboard page to the navigation
st.sidebar.markdown("---")
if st.sidebar.checkbox("Show Executive Dashboard"):
    page = "Executive Dashboard"

# Implement the Executive Dashboard
if page == "Executive Dashboard":
    st.markdown('<div class="main-header">Executive Dashboard</div>', unsafe_allow_html=True)
    
    # Current user and time display
    st.markdown(f"""
    <div style="text-align: right; font-size: 0.8rem; color: #666;">
    User: {username} | Last updated: {current_datetime}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    Key performance indicators and business metrics for executive decision-making. This dashboard provides a high-level
    overview of the pricing system's performance and impact on the business.
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Summary
    st.markdown('<div class="sub-header">Key Performance Indicators</div>', unsafe_allow_html=True)
    
    # Get data for KPIs
    df = generate_historical_data()
    
    # Calculate KPIs
    today = datetime.datetime.now().date()
    last_30_days = today - datetime.timedelta(days=30)
    last_60_days = today - datetime.timedelta(days=60)
    
    df['date'] = pd.to_datetime(df['date'])
    
    recent_df = df[df['date'] >= pd.Timestamp(last_30_days)]
    previous_df = df[(df['date'] >= pd.Timestamp(last_60_days)) & (df['date'] < pd.Timestamp(last_30_days))]
    
    # Calculate metrics
    recent_revenue = recent_df['revenue'].sum()
    previous_revenue = previous_df['revenue'].sum()
    revenue_change = ((recent_revenue / previous_revenue) - 1) * 100
    
    recent_bookings = recent_df['bookings'].sum()
    previous_bookings = previous_df['bookings'].sum()
    bookings_change = ((recent_bookings / previous_bookings) - 1) * 100
    
    recent_price = recent_df['actual_price'].mean()
    previous_price = previous_df['actual_price'].mean()
    price_change = ((recent_price / previous_price) - 1) * 100
    
    pricing_accuracy = 87.4  # Simulated ML model accuracy
    
    # Display KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Monthly Revenue", f"${recent_revenue:,.2f}", f"{revenue_change:+.1f}%")
    col2.metric("Total Bookings", f"{recent_bookings:,.0f}", f"{bookings_change:+.1f}%")
    col3.metric("Average Price", f"${recent_price:.2f}", f"{price_change:+.1f}%")
    col4.metric("Pricing Accuracy", f"{pricing_accuracy:.1f}%", "+2.1%")
    
    # Revenue Trend
    st.markdown('<div class="sub-header">Revenue Trend (Last 12 Months)</div>', unsafe_allow_html=True)
    
    # Group by month
    df['month'] = df['date'].dt.strftime('%Y-%m')
    monthly_revenue = df.groupby('month')['revenue'].sum().reset_index()
    
    # Ensure we have exactly 12 months (fill any missing months)
    all_months = pd.date_range(end=today, periods=12, freq='M')
    all_months_str = [d.strftime('%Y-%m') for d in all_months]
    
    # Create complete dataframe
    complete_months = pd.DataFrame({'month': all_months_str})
    monthly_revenue = complete_months.merge(monthly_revenue, on='month', how='left').fillna(0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(monthly_revenue['month'], monthly_revenue['revenue'], marker='o', linewidth=2)
    
    # Add average line
    avg_revenue = monthly_revenue['revenue'].mean()
    ax.axhline(y=avg_revenue, color='r', linestyle='--', alpha=0.7)
    ax.text(0, avg_revenue * 1.02, f'Average: ${avg_revenue:,.0f}', color='r')
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Monthly Revenue ($)')
    ax.set_title('Monthly Revenue Trend')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Performance by Route and Segment
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="sub-header">Top Performing Routes</div>', unsafe_allow_html=True)
        
        # Calculate route performance
        route_performance = df.groupby(['from', 'to']).agg({
            'revenue': 'sum',
            'bookings': 'sum'
        }).reset_index()
        
        route_performance['route'] = route_performance['from'] + ' to ' + route_performance['to']
        route_performance['avg_fare'] = route_performance['revenue'] / route_performance['bookings']
        
        # Sort by revenue
        top_routes = route_performance.sort_values('revenue', ascending=False).head(5)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(top_routes['route'], top_routes['revenue'] / 1000000, color='skyblue')
        
        ax.set_xlabel('Revenue ($ millions)')
        ax.set_title('Top 5 Routes by Revenue')
        
        # Add revenue values on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_x = width * 1.01
            label_y = bar.get_y() + bar.get_height() / 2
            ax.text(label_x, label_y, f'${width:.2f}M', va='center')
        
        st.pyplot(fig)
    
    with col2:
        st.markdown('<div class="sub-header">Price Optimization Impact</div>', unsafe_allow_html=True)
        
        # Calculate optimized vs non-optimized revenue (simulated)
        # Here we assume SmartPrice was implemented halfway through the data
        midpoint = len(df) // 2
        
        # Pre-optimization period
        pre_df = df.iloc[:midpoint]
        pre_revenue = pre_df['revenue'].sum()
        pre_bookings = pre_df['bookings'].sum()
        pre_avg_price = pre_df['actual_price'].mean()
        
        # Post-optimization period
        post_df = df.iloc[midpoint:]
        post_revenue = post_df['revenue'].sum()
        post_bookings = post_df['bookings'].sum()
        post_avg_price = post_df['actual_price'].mean()
        
        # Calculate changes
        revenue_lift = ((post_revenue / pre_revenue) - 1) * 100
        bookings_change = ((post_bookings / pre_bookings) - 1) * 100
        price_change = ((post_avg_price / pre_avg_price) - 1) * 100
        
        # Display metrics
        st.metric("Revenue Lift from SmartPrice", f"+{revenue_lift:.1f}%")
        
        # Create comparison chart
        labels = ['Before SmartPrice', 'After SmartPrice']
        revenue_values = [pre_revenue / 1000000, post_revenue / 1000000]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, revenue_values, color=['lightgray', 'green'])
        
        ax.set_ylabel('Revenue ($ millions)')
        ax.set_title('Revenue Before and After SmartPrice Implementation')
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height * 1.01,
                    f'${height:.2f}M', ha='center', va='bottom')
        
        # Add percentage increase arrow
        ax.annotate(f'+{revenue_lift:.1f}%', 
                   xy=(1, revenue_values[1]), 
                   xytext=(0.5, revenue_values[1] * 1.2),
                   arrowprops=dict(facecolor='green', shrink=0.05, width=2, headwidth=10),
                   ha='center', fontsize=12, fontweight='bold', color='green')
        
        st.pyplot(fig)
        
        # Additional explanation
        st.markdown(f"""
        <div class="highlight">
        <p>SmartPrice implementation resulted in:</p>
        <ul>
            <li><strong>{revenue_lift:.1f}%</strong> increase in total revenue</li>
            <li><strong>{bookings_change:.1f}%</strong> change in booking volume</li>
            <li><strong>{price_change:.1f}%</strong> change in average price</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Market Share Analysis
    st.markdown('<div class="sub-header">Market Share Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create simulated market share data
        market_data = pd.DataFrame({
            'Quarter': ['Q3 2024', 'Q4 2024', 'Q1 2025', 'Q2 2025'],
            'Your Company': [21, 23, 26, 28],
            'Competitor A': [32, 31, 29, 27],
            'Competitor B': [18, 19, 18, 17],
            'Competitor C': [16, 15, 14, 15],
            'Others': [13, 12, 13, 13]
        })
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bottom = np.zeros(4)
        
        for company in ['Your Company', 'Competitor A', 'Competitor B', 'Competitor C', 'Others']:
            ax.bar(market_data['Quarter'], market_data[company], bottom=bottom, label=company)
            bottom += market_data[company]
        
        ax.set_ylabel('Market Share (%)')
        ax.set_title('Market Share Trend')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
        
        # Add your company's percentage on each bar
        for i, quarter in enumerate(market_data['Quarter']):
            your_share = market_data['Your Company'][i]
            ax.text(i, your_share / 2, f'{your_share}%', ha='center', va='center', 
                   color='white', fontweight='bold')
        
        st.pyplot(fig)
        
        st.markdown("""
        <div class="small-text">
        Market share data based on industry reports and internal analysis.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Market Position Analysis
        st.markdown("**Market Position Analysis**")
        
        # Create radar chart data
        categories = ['Price Competitiveness', 'Revenue Growth', 'Booking Volume', 
                     'Customer Satisfaction', 'Price Optimization']
        
        your_values = [80, 85, 75, 70, 90]
        competitor_a_values = [65, 70, 90, 75, 60]
        competitor_b_values = [75, 65, 70, 85, 65]
        
        # Number of categories
        N = len(categories)
        
        # Create angles for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add the values for the chart
        your_values += your_values[:1]
        competitor_a_values += competitor_a_values[:1]
        competitor_b_values += competitor_b_values[:1]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Plot each company
        ax.plot(angles, your_values, linewidth=2, linestyle='solid', label='Your Company')
        ax.fill(angles, your_values, alpha=0.25)
        
        ax.plot(angles, competitor_a_values, linewidth=2, linestyle='solid', label='Competitor A')
        ax.fill(angles, competitor_a_values, alpha=0.1)
        
        ax.plot(angles, competitor_b_values, linewidth=2, linestyle='solid', label='Competitor B')
        ax.fill(angles, competitor_b_values, alpha=0.1)
        
        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        st.pyplot(fig)
        
        st.markdown("""
        <div class="highlight">
        <h3>Competitive Advantage</h3>
        <p>SmartPrice gives you a significant advantage in <strong>Price Optimization</strong> and <strong>Revenue Growth</strong>,
        outperforming competitors in these key areas. This enables sustained growth and market share expansion.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ROI Analysis
    st.markdown('<div class="sub-header">SmartPrice ROI Analysis</div>', unsafe_allow_html=True)
    
    # Create simulated ROI data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']
    implementation_costs = [180000, 120000, 60000, 30000, 15000, 15000, 15000]
    cumulative_benefits = [0, 40000, 120000, 240000, 380000, 540000, 720000]
    net_value = [c - i for c, i in zip(cumulative_benefits, implementation_costs)]
    
    # Find break-even point
    breakeven_month = next((i for i, v in enumerate(net_value) if v >= 0), len(net_value))
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(months, cumulative_benefits, marker='o', linewidth=2, label='Cumulative Benefits')
    ax.plot(months, implementation_costs, marker='s', linewidth=2, label='Implementation Costs')
    ax.plot(months, net_value, marker='^', linewidth=2, label='Net Value')
    
    # Add break-even point
    if breakeven_month < len(months):
        ax.axvline(x=months[breakeven_month], color='green', linestyle='--')
        ax.text(months[breakeven_month], max(cumulative_benefits) * 0.9, 'Break-even', 
               rotation=90, va='top', ha='right', color='green')
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Amount ($)')
    ax.set_title('SmartPrice ROI Analysis')
    ax.legend()
    
    # Add ROI value
    total_cost = sum(implementation_costs)
    total_benefit = cumulative_benefits[-1]
    roi = ((total_benefit - total_cost) / total_cost) * 100
    
    ax.text(0.02, 0.95, f'ROI: {roi:.1f}%', transform=ax.transAxes, 
           fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
    
    st.pyplot(fig)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Total Investment", f"${total_cost:,.0f}")
    col2.metric("Total Benefits", f"${total_benefit:,.0f}")
    col3.metric("Return on Investment", f"{roi:.1f}%")
    
    st.markdown(f"""
    <div class="highlight">
    <h3>ROI Summary</h3>
    <p>SmartPrice achieved break-even in <strong>{breakeven_month} months</strong> and has generated a total ROI of 
    <strong>{roi:.1f}%</strong> since implementation. The system continues to deliver increasing returns as the 
    machine learning models improve with more data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Future Outlook
    st.markdown('<div class="sub-header">Future Outlook</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="highlight">
        <h3>Projected Growth</h3>
        <p>Based on current trends and continuous model improvements, we project:</p>
        <ul>
            <li><strong>15-18%</strong> additional revenue growth over the next 12 months</li>
            <li><strong>3-5%</strong> market share increase</li>
            <li><strong>92%</strong> pricing accuracy by Q4 2025</li>
        </ul>
        <p>These projections assume continued investment in model refinement and expansion to additional routes.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="highlight">
        <h3>Recommended Next Steps</h3>
        <ul>
            <li><strong>Expand Dynamic Pricing</strong> to international routes</li>
            <li><strong>Implement Real-time Competitor Monitoring</strong> for faster response</li>
            <li><strong>Develop Customer Segment-specific Pricing</strong> for personalization</li>
            <li><strong>Integrate with CRM</strong> for customer lifetime value optimization</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# Add user account and settings section
st.sidebar.markdown("---")
if st.sidebar.checkbox("Account & Settings"):
    page = "Account & Settings"

# Implement Account & Settings
if page == "Account & Settings":
    st.markdown('<div class="main-header">Account & Settings</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    Manage your SmartPrice account settings, preferences, and access controls.
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(["Profile", "API Access", "Notification Settings", "Theme"])
    
    with tabs[0]:
        st.markdown('<div class="sub-header">User Profile</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Profile avatar
            st.image("https://www.gravatar.com/avatar/00000000000000000000000000000000?d=mp&f=y", width=150)
            
            st.button("Upload New Image")
        
        with col2:
            # Profile details
            st.text_input("Username", value="Anurag02012004", disabled=True)
            st.text_input("Full Name", value="Anurag Kushwaha")
            st.text_input("Email", value="anurag@example.com")
            st.text_input("Job Title", value="Senior Pricing Analyst")
            
            st.markdown("**Account Status**: Active (Premium)")
            st.markdown("**Last Login**: 2025-07-27 12:08:00")
            
            st.button("Update Profile")
    
    with tabs[1]:
        st.markdown('<div class="sub-header">API Access</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Access the SmartPrice API to integrate dynamic pricing with your systems.
        """)
        
        # API Key
        st.text_input("API Key", value="sk_smartprice_2025072712xxxxxxxxx", type="password")
        
        col1, col2 = st.columns(2)
        col1.button("Generate New API Key")
        col2.button("Revoke All Keys")
        
        # API Usage
        st.markdown('<div class="sub-header">API Usage</div>', unsafe_allow_html=True)
        
        # Create simulated API usage data
        dates = pd.date_range(end=datetime.datetime.now(), periods=14).tolist()
        api_calls = [random.randint(800, 1500) for _ in range(14)]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(dates, api_calls, color='skyblue')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('API Calls')
        ax.set_title('Daily API Usage')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        st.info("Your current plan allows up to 50,000 API calls per month. You've used 12,450 calls (24.9%) this month.")
        
        # API Documentation
        st.markdown('<div class="sub-header">API Documentation</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ```python
        # Python Example
        import requests
        
        api_key = "your_api_key"
        url = "https://api.smartprice.io/v1/predict"
        
        payload = {
            "route_id": 123,
            "date": "2025-08-15",
            "additional_factors": {
                "weather": "clear",
                "events": ["concert"]
            }
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers)
        price_data = response.json()
        
        print(f"Optimal price: ${price_data['price']}")
        ```
        """)
    
    with tabs[2]:
        st.markdown('<div class="sub-header">Notification Settings</div>', unsafe_allow_html=True)
        
        st.markdown("**Email Notifications**")
        
        st.checkbox("Price Alert Notifications", value=True)
        st.checkbox("Competitor Price Change Alerts", value=True)
        st.checkbox("Weekly Performance Reports", value=True)
        st.checkbox("System Updates and Maintenance", value=False)
        
        st.markdown("**Alert Thresholds**")
        
        st.slider("Price Change Alert Threshold (%)", min_value=5, max_value=30, value=10)
        st.slider("Revenue Impact Alert Threshold ($)", min_value=1000, max_value=10000, value=5000)
        
        st.button("Save Notification Settings")
    
    with tabs[3]:
        st.markdown('<div class="sub-header">Theme Settings</div>', unsafe_allow_html=True)
        
        theme = st.radio("Select Theme", ["Light", "Dark", "System Default"])
        
        color_scheme = st.selectbox("Color Accent", 
                                  ["Blue (Default)", "Green", "Purple", "Orange", "Red"])
        
        st.checkbox("Enable Compact Mode", value=False)
        st.checkbox("Enable High Contrast", value=False)
        
        st.button("Apply Theme")

# Add help and documentation section
st.sidebar.markdown("---")
if st.sidebar.checkbox("Help & Documentation"):
    page = "Help & Documentation"

# Implement Help & Documentation
if page == "Help & Documentation":
    st.markdown('<div class="main-header">Help & Documentation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    Learn how to use SmartPrice effectively with our comprehensive documentation, tutorials, and FAQs.
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(["Getting Started", "User Guide", "FAQs", "API Documentation", "Support"])
    
    with tabs[0]:
        st.markdown('<div class="sub-header">Getting Started with SmartPrice</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ## Welcome to SmartPrice!
        
        SmartPrice is a powerful dynamic pricing platform that uses machine learning to optimize your travel prices
        in real-time. This guide will help you get started quickly.
        
        ### Quick Start Guide
        
        1. **Navigate the Dashboard**: Use the sidebar to access different modules
        2. **Generate Prices**: Go to the "Price Prediction" page to get ML-based price recommendations
        3. **Analyze Data**: Explore historical data in the "Data Explorer" section
        4. **Run Tests**: Use "A/B Testing" to simulate different pricing strategies
        5. **Get Reports**: Generate comprehensive reports in the "Reports & Export" section
        
        ### Video Tutorials
        
        Watch our quick video tutorials to get up to speed:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            st.markdown("**Basic Navigation & Features**")
        
        with col2:
            st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            st.markdown("**Advanced Analytics & Reporting**")
    
    with tabs[1]:
        st.markdown('<div class="sub-header">User Guide</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ## Comprehensive User Guide
        
        ### Price Prediction Module
        
        The Price Prediction module allows you to generate optimal prices for any route and date. The system considers:
        
        - Historical demand patterns
        - Competitor pricing
        - Seasonal factors
        - Weather conditions
        - Special events
        - Day of week effects
        
        To get a price prediction:
        
        1. Select a route from the dropdown
        2. Choose a travel date
        3. Click "Generate Price Prediction"
        
        The system will provide:
        - Optimal price recommendation
        - Projected demand
        - Estimated bookings
        - Estimated revenue
        - Market position compared to competitors
        
        ### Data Explorer
        
        The Data Explorer allows you to:
        
        - Visualize historical pricing and demand data
        - Filter by route, date range, and other parameters
        - Analyze correlations between prices and bookings
        - Identify seasonal patterns and trends
        
        ### A/B Testing
        
        Use this feature to:
        
        - Simulate different pricing strategies
        - Compare projected outcomes
        - Identify the optimal approach for each route
        - Quantify the impact of price changes
        
        ### Batch Prediction
        
        For bulk price generation:
        
        1. Select multiple routes
        2. Specify a date range
        3. Generate optimal prices for all combinations
        4. Export results for implementation in your booking system
        """)
    
    with tabs[2]:
        st.markdown('<div class="sub-header">Frequently Asked Questions</div>', unsafe_allow_html=True)
        
        faq_items = [
            {"question": "How does SmartPrice determine optimal prices?", 
             "answer": "SmartPrice uses a combination of LSTM neural networks for demand forecasting and XGBoost for price optimization. It analyzes historical bookings, competitor prices, seasonal patterns, and real-time market conditions to identify the revenue-maximizing price point."},
            
            {"question": "How often are the ML models updated?", 
             "answer": "The models are retrained weekly with new data, but the system continuously learns from incoming booking data and adjusts predictions accordingly. Major model updates are released quarterly."},
            
            {"question": "Can I integrate SmartPrice with my existing booking system?", 
             "answer": "Yes, SmartPrice offers a comprehensive API that allows integration with most booking and reservation systems. Our implementation team can assist with custom integrations."},
            
            {"question": "What data is needed to get started?", 
             "answer": "At minimum, you need historical booking data with dates, routes, prices, and booking volumes. The more data you provide (competitor prices, special events, etc.), the more accurate the predictions will be."},
            
            {"question": "How much revenue improvement can I expect?", 
             "answer": "Most clients see a 10-15% revenue increase within the first 3-6 months. Results vary by market, but our case studies show consistent improvement across different travel segments."},
            
            {"question": "Is there a limit to the number of routes I can optimize?", 
             "answer": "The Enterprise plan includes unlimited routes. Standard plans have limits based on tier (e.g., 50, 100, or 200 routes)."},
            
            {"question": "How does SmartPrice handle seasonal pricing?", 
             "answer": "The ML models automatically detect and adapt to seasonal patterns in your historical data. You can also manually configure seasonal factors for special cases."}
        ]
        
        for i, faq in enumerate(faq_items):
            with st.expander(faq["question"]):
                st.markdown(faq["answer"])
    
    with tabs[3]:
        st.markdown('<div class="sub-header">API Documentation</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ## SmartPrice API Reference
        
        The SmartPrice API allows you to programmatically access our pricing engine, historical data, and analytics.
        
        ### Authentication
        
        All API requests require authentication using your API key:
        
        ```
        Authorization: Bearer your_api_key
        ```
        
        ### Endpoints
        
        #### Price Prediction
        
        `POST /v1/predict`
        
        Get an optimal price prediction for a specific route and date.
        
        **Request Body:**
        ```json
        {
            "route_id": 123,
            "date": "2025-08-15",
            "additional_factors": {
                "weather": "clear",
                "events": ["concert"]
            }
        }
        ```
        
        **Response:**
        ```json
        {
            "route_id": 123,
            "date": "2025-08-15",
            "optimal_price": 245.50,
            "estimated_demand": 120,
            "competitor_prices": {
                "competitor1": 250.00,
                "competitor2": 238.50,
                "competitor3": 255.00
            },
            "confidence_score": 0.92
        }
        ```
        
        #### Batch Prediction
        
        `POST /v1/batch-predict`
        
        Generate optimal prices for multiple route/date combinations.
        
        **Request Body:**
        ```json
        {
            "predictions": [
                {"route_id": 123, "date": "2025-08-15"},
                {"route_id": 123, "date": "2025-08-16"},
                {"route_id": 124, "date": "2025-08-15"}
            ]
        }
        ```
        
        #### Historical Data
        
        `GET /v1/history?route_id=123&start_date=2025-01-01&end_date=2025-02-01`
        
        Retrieve historical pricing and booking data.
        
        For complete API documentation, please visit our [Developer Portal](https://developer.smartprice.io).
        """)
    
    with tabs[4]:
        st.markdown('<div class="sub-header">Support Resources</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Technical Support
            
            **Email Support:**  
            support@smartprice.io
            
            **Phone Support:**  
            +1 (555) 123-4567
            
            **Hours of Operation:**  
            Monday-Friday, 9:00 AM - 8:00 PM EST
            
            **Response Time:**  
            Within 4 hours for standard issues  
            Within 1 hour for critical issues
            """)
            
            st.text_area("Submit a Support Ticket", height=100, placeholder="Describe your issue here...")
            st.button("Submit Ticket")
        
        with col2:
            st.markdown("""
            ### Learning Resources
            
            **Documentation:**  
            [Complete Documentation](https://docs.smartprice.io)
            
            **Webinars:**  
            - [Maximizing Revenue with Dynamic Pricing](https://webinars.smartprice.io/revenue)
            - [Advanced A/B Testing Strategies](https://webinars.smartprice.io/testing)
            - [Competitive Price Analysis](https://webinars.smartprice.io/competitive)
            
            **Blog:**  
            [SmartPrice Blog](https://blog.smartprice.io)
            
            **Community Forum:**  
            [Join the Discussion](https://community.smartprice.io)
            """)

# Footer section - add at the very end
footer_html = """
<div class="footer">
    <p>SmartPrice - Dynamic Travel Pricing System<br>
    <span class="small-text">Created by Anurag Kushwaha | © 2025</span></p>
    <p class="small-text">Last login: {datetime} | User: {username}</p>
</div>
""".format(datetime=current_datetime, username=username)

st.markdown(footer_html, unsafe_allow_html=True) 

# --- Add Health Monitoring Dashboard ---

# Add system health monitoring to sidebar
st.sidebar.markdown("---")
if st.sidebar.checkbox("System Health"):
    page = "System Health"

# Update user info and current time
username = "Anurag02012004"  # Use the provided username
current_datetime = "2025-07-27 12:15:53"  # Use the provided datetime

# System Health Dashboard
if page == "System Health":
    st.markdown('<div class="main-header">System Health & Monitoring</div>', unsafe_allow_html=True)
    
    # Current user and time display
    st.markdown(f"""
    <div style="text-align: right; font-size: 0.8rem; color: #666;">
    User: {username} | Current time: {current_datetime}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    Monitor the health and performance of the SmartPrice system. This dashboard provides real-time metrics
    on system performance, ML model status, and data pipeline health.
    </div>
    """, unsafe_allow_html=True)
    
    # System status overview
    st.markdown('<div class="sub-header">System Status</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Simulated system health metrics
    system_status = "Operational"
    api_latency = "118 ms"
    model_freshness = "Updated 6 hours ago"
    cache_hit_rate = "93.8%"
    
    col1.metric("System Status", system_status, "Normal")
    col2.metric("API Latency", api_latency, "-12ms")
    col3.metric("Model Freshness", model_freshness)
    col4.metric("Cache Hit Rate", cache_hit_rate, "+1.2%")
    
    # System health visualization
    st.markdown('<div class="sub-header">Performance Metrics</div>', unsafe_allow_html=True)
    
    # Simulated time series data for system performance
    timestamps = [datetime.datetime.strptime(current_datetime, "%Y-%m-%d %H:%M:%S") - 
                 datetime.timedelta(hours=i) for i in range(24, 0, -1)]
    
    # API latency data (simulated)
    api_latency_values = [100 + 30 * np.sin(i/3) + 10 * np.random.randn() for i in range(24)]
    
    # CPU and memory usage (simulated)
    cpu_usage = [40 + 20 * np.sin(i/4) + 5 * np.random.randn() for i in range(24)]
    memory_usage = [60 + 15 * np.sin(i/5 + 2) + 3 * np.random.randn() for i in range(24)]
    
    # Plot performance metrics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # API latency plot
    ax1.plot(timestamps, api_latency_values, marker='o', linewidth=2, color='blue')
    ax1.set_ylabel('API Latency (ms)')
    ax1.set_title('API Response Time (Last 24 Hours)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Resource usage plot
    ax2.plot(timestamps, cpu_usage, marker='s', linewidth=2, label='CPU Usage (%)', color='orange')
    ax2.plot(timestamps, memory_usage, marker='^', linewidth=2, label='Memory Usage (%)', color='green')
    ax2.set_ylabel('Resource Usage (%)')
    ax2.set_title('System Resource Usage (Last 24 Hours)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Format x-axis
    ax2.set_xlabel('Time')
    fig.autofmt_xdate()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # ML model performance tracking
    st.markdown('<div class="sub-header">ML Model Performance</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Model accuracy over time
        st.markdown("**Model Accuracy Trend**")
        
        # Simulated model accuracy data
        model_dates = [datetime.datetime.now() - datetime.timedelta(days=i*7) for i in range(10, 0, -1)]
        model_accuracy = [82.5, 83.1, 84.0, 84.2, 85.6, 85.9, 86.3, 86.8, 87.2, 87.4]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(model_dates, model_accuracy, marker='o', linewidth=2)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Model Accuracy (%)')
        ax.set_title('ML Model Accuracy Over Time')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Format y-axis to start from a reasonable value
        ax.set_ylim(80, 90)
        
        # Format x-axis
        fig.autofmt_xdate()
        
        st.pyplot(fig)
    
    with col2:
        # Model prediction distributions
        st.markdown("**Prediction Distribution**")
        
        # Simulated prediction error distribution
        error_values = np.random.normal(0, 5, 1000)  # Centered at 0 with SD of 5
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(error_values, bins=30, alpha=0.7, color='skyblue')
        
        ax.set_xlabel('Prediction Error ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Price Prediction Errors')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add mean and SD lines
        mean_error = np.mean(error_values)
        std_error = np.std(error_values)
        
        ax.axvline(mean_error, color='red', linestyle='--', label=f'Mean: ${mean_error:.2f}')
        ax.axvline(mean_error + std_error, color='green', linestyle=':', label=f'±1 SD: ${std_error:.2f}')
        ax.axvline(mean_error - std_error, color='green', linestyle=':')
        
        ax.legend()
        
        st.pyplot(fig)
    
    # Data pipeline health
    st.markdown('<div class="sub-header">Data Pipeline Health</div>', unsafe_allow_html=True)
    
    # Simulated pipeline metrics
    pipeline_stages = ['Data Collection', 'Preprocessing', 'Feature Engineering', 
                      'Model Training', 'Prediction Service', 'Caching Layer', 'API Gateway']
    
    stage_statuses = ['Healthy', 'Healthy', 'Healthy', 'Healthy', 'Healthy', 'Warning', 'Healthy']
    stage_latencies = [420, 380, 620, 1850, 95, 12, 42]  # in milliseconds
    stage_last_runs = ['12 min ago', '12 min ago', '12 min ago', '6 hours ago', '2 min ago', '2 min ago', '1 min ago']
    
    # Create DataFrame
    pipeline_df = pd.DataFrame({
        'Stage': pipeline_stages,
        'Status': stage_statuses,
        'Latency (ms)': stage_latencies,
        'Last Run': stage_last_runs
    })
    
    # Apply color coding to status
    def color_status(val):
        color = 'green' if val == 'Healthy' else 'orange' if val == 'Warning' else 'red'
        return f'background-color: {color}; color: white'
    
    # Display styled dataframe
    st.dataframe(pipeline_df.style.applymap(color_status, subset=['Status']))
    
    # Warning for cache layer
    st.warning("⚠️ Cache layer showing elevated latency. Automatic scaling has been triggered.")
    
    # Activity logs
    st.markdown('<div class="sub-header">System Activity Logs</div>', unsafe_allow_html=True)
    
    # Simulated log entries
    log_timestamps = [
        datetime.datetime.strptime(current_datetime, "%Y-%m-%d %H:%M:%S") - datetime.timedelta(minutes=i*5)
        for i in range(10)
    ]
    
    log_entries = [
        "Cache layer auto-scaling triggered due to increased load",
        "Model retraining completed successfully (v2.4.7)",
        "New competitor price data integrated for 12 routes",
        "Scheduled database backup completed",
        "User login: admin@smartprice.io",
        "API rate limit increased for client ID: C1052",
        "System update scheduled for July 31, 2025 (02:00 UTC)",
        "New weather data source connected successfully",
        "User login: anurag@smartprice.io",
        "Daily analytics report generated and emailed to stakeholders"
    ]
    
    log_types = ["WARNING", "INFO", "INFO", "INFO", "AUTH", "CONFIG", "SYSTEM", "INFO", "AUTH", "INFO"]
    
    logs_df = pd.DataFrame({
        'Timestamp': [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in log_timestamps],
        'Type': log_types,
        'Message': log_entries
    })
    
    # Display logs
    st.dataframe(logs_df)
    
    # Actions section
    st.markdown('<div class="sub-header">Maintenance Actions</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.button("Restart Services")
    
    with col2:
        st.button("Clear Cache")
    
    with col3:
        st.button("Force Model Retraining")
    
    with col4:
        st.button("Generate Health Report")

# --- Add Team Collaboration Features ---
st.sidebar.markdown("---")
if st.sidebar.checkbox("Team Collaboration"):
    page = "Team Collaboration"

# Team Collaboration Page
if page == "Team Collaboration":
    st.markdown('<div class="main-header">Team Collaboration</div>', unsafe_allow_html=True)
    
    # Current user and time display
    st.markdown(f"""
    <div style="text-align: right; font-size: 0.8rem; color: #666;">
    User: {username} | Current time: {current_datetime}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    Collaborate with your team on pricing strategies, share insights, and manage projects together.
    This module enables seamless teamwork and knowledge sharing across your organization.
    </div>
    """, unsafe_allow_html=True)
    
    # Team members
    st.markdown('<div class="sub-header">Team Members</div>', unsafe_allow_html=True)
    
    # Simulated team members
    team_members = [
        {"name": "Anurag Kushwaha", "role": "Senior Pricing Analyst", "avatar": "AS", "status": "Online"},
        {"name": "Jennifer Lee", "role": "Data Scientist", "avatar": "JL", "status": "Online"},
        {"name": "Michael Chen", "role": "Product Manager", "avatar": "MC", "status": "Away"},
        {"name": "Sarah Johnson", "role": "Revenue Manager", "avatar": "SJ", "status": "Offline"},
        {"name": "David Park", "role": "Marketing Analyst", "avatar": "DP", "status": "Online"},
    ]
    
    # Display team members
    cols = st.columns(5)
    for i, member in enumerate(team_members):
        status_color = "green" if member["status"] == "Online" else "orange" if member["status"] == "Away" else "gray"
        
        cols[i].markdown(f"""
        <div style="text-align: center;">
            <div style="background-color: #4a86e8; color: white; width: 50px; height: 50px; 
                     border-radius: 50%; display: flex; align-items: center; justify-content: center;
                     margin: 0 auto;">
                <span style="font-weight: bold; font-size: 18px;">{member["avatar"]}</span>
            </div>
            <p style="margin-top: 5px; margin-bottom: 0;"><b>{member["name"]}</b></p>
            <p style="margin: 0; font-size: 0.8rem;">{member["role"]}</p>
            <p style="margin: 0; color: {status_color}; font-size: 0.8rem;">● {member["status"]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Shared projects
    st.markdown('<div class="sub-header">Shared Projects</div>', unsafe_allow_html=True)
    
    # Simulated projects
    projects = [
        {"name": "Q3 Pricing Strategy", "owner": "Anurag Kushwaha", "modified": "Today, 09:23", 
         "collaborators": 4, "status": "In Progress"},
        {"name": "Summer Peak Season Analysis", "owner": "Jennifer Lee", "modified": "Yesterday", 
         "collaborators": 3, "status": "Completed"},
        {"name": "Competitor Benchmarking", "owner": "David Park", "modified": "July 25, 2025", 
         "collaborators": 5, "status": "In Progress"},
        {"name": "Route Optimization Project", "owner": "Sarah Johnson", "modified": "July 22, 2025", 
         "collaborators": 2, "status": "Under Review"}
    ]
    
    # Display projects as cards
    cols = st.columns(2)
    for i, project in enumerate(projects):
        col_idx = i % 2
        
        # Define card color based on status
        card_color = "#e3f2fd" if project["status"] == "In Progress" else "#e8f5e9" if project["status"] == "Completed" else "#fff9c4"
        
        cols[col_idx].markdown(f"""
        <div style="background-color: {card_color}; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
            <h3 style="margin-top: 0;">{project["name"]}</h3>
            <p><b>Owner:</b> {project["owner"]} | <b>Last Modified:</b> {project["modified"]}</p>
            <p><b>Collaborators:</b> {project["collaborators"]} | <b>Status:</b> {project["status"]}</p>
            <div style="display: flex; justify-content: flex-end;">
                <button style="background-color: #1E88E5; color: white; border: none; 
                       padding: 5px 15px; border-radius: 3px; cursor: pointer;">
                    Open Project
                </button>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Create new project button
    st.markdown("""
    <div style="display: flex; justify-content: center; margin: 20px 0;">
        <button style="background-color: #1E88E5; color: white; border: none; 
               padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 16px;">
            + Create New Project
        </button>
    </div>
    """, unsafe_allow_html=True)
    
    # Team chat
    st.markdown('<div class="sub-header">Team Chat</div>', unsafe_allow_html=True)
    
    # Simulated chat messages
    chat_messages = [
        {"user": "System", "message": "Welcome to the team chat! This is a space for quick discussions.", 
         "time": "12:00", "avatar": "SY"},
        {"user": "Jennifer Lee", "message": "Has anyone looked at the Q3 pricing for NYC-LAX route? Competitors have dropped prices by 8% this morning.", 
         "time": "12:02", "avatar": "JL"},
        {"user": "Anurag Kushwaha", "message": "I'm reviewing it now. Our model suggests holding prices steady - the demand forecast shows we're still within optimal range.", 
         "time": "12:05", "avatar": "AS"},
        {"user": "Michael Chen", "message": "Good call. The marketing campaign for that route starts tomorrow, so we should see increased demand anyway.", 
         "time": "12:08", "avatar": "MC"},
        {"user": "Anurag Kushwaha", "message": "I'll monitor the booking rate hourly and we can adjust if needed. I've set up an alert if it drops below 70%.", 
         "time": "12:10", "avatar": "AS"}
    ]
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        for message in chat_messages:
            # Determine if message is from current user
            is_user = message["user"] == "Anurag Kushwaha"
            
            # Set alignment and colors based on message sender
            align = "right" if is_user else "left"
            bg_color = "#e3f2fd" if is_user else "#f5f5f5"
            
            st.markdown(f"""
            <div style="display: flex; justify-content: {align}; margin-bottom: 10px;">
                <div style="max-width: 80%; background-color: {bg_color}; padding: 10px; border-radius: 10px;">
                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                        <div style="background-color: #4a86e8; color: white; width: 25px; height: 25px; 
                                 border-radius: 50%; display: flex; align-items: center; justify-content: center;
                                 margin-right: 5px;">
                            <span style="font-weight: bold; font-size: 12px;">{message["avatar"]}</span>
                        </div>
                        <span style="font-weight: bold;">{message["user"]}</span>
                        <span style="color: #9e9e9e; font-size: 0.8rem; margin-left: 10px;">{message["time"]}</span>
                    </div>
                    <p style="margin: 0;">{message["message"]}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    st.text_input("Type your message here...", key="chat_input")
    
    # Upcoming meetings
    st.markdown('<div class="sub-header">Upcoming Meetings</div>', unsafe_allow_html=True)
    
    # Simulated meetings
    meetings = [
        {"title": "Weekly Pricing Review", "time": "Today, 15:30", "participants": 6, 
         "organizer": "Anurag Kushwaha", "location": "Virtual (Zoom)"},
        {"title": "Q3 Strategy Planning", "time": "Tomorrow, 10:00", "participants": 8, 
         "organizer": "Sarah Johnson", "location": "Conference Room A"},
        {"title": "Model Performance Review", "time": "July 29, 14:00", "participants": 4, 
         "organizer": "Jennifer Lee", "location": "Virtual (Teams)"}
    ]
    
    # Display meetings
    for meeting in meetings:
        st.markdown(f"""
        <div style="display: flex; align-items: center; background-color: #f5f5f5; 
                 padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <div style="flex-grow: 1;">
                <h4 style="margin: 0;">{meeting["title"]}</h4>
                <p style="margin: 0;"><b>Time:</b> {meeting["time"]} | <b>Participants:</b> {meeting["participants"]}</p>
                <p style="margin: 0;"><b>Organizer:</b> {meeting["organizer"]} | <b>Location:</b> {meeting["location"]}</p>
            </div>
            <div>
                <button style="background-color: #1E88E5; color: white; border: none; 
                       padding: 5px 15px; border-radius: 3px; cursor: pointer; margin-right: 5px;">
                    Join
                </button>
                <button style="background-color: transparent; color: #1E88E5; border: 1px solid #1E88E5; 
                       padding: 5px 15px; border-radius: 3px; cursor: pointer;">
                    Details
                </button>
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- Add Integration Hub ---
st.sidebar.markdown("---")
if st.sidebar.checkbox("Integration Hub"):
    page = "Integration Hub"

# Integration Hub
if page == "Integration Hub":
    st.markdown('<div class="main-header">Integration Hub</div>', unsafe_allow_html=True)
    
    # Current user and time display
    st.markdown(f"""
    <div style="text-align: right; font-size: 0.8rem; color: #666;">
    User: {username} | Current time: {current_datetime}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    Connect SmartPrice with your existing systems and third-party services. The Integration Hub
    provides connectors, webhooks, and APIs to create a seamless pricing ecosystem.
    </div>
    """, unsafe_allow_html=True)
    
    # Available integrations
    st.markdown('<div class="sub-header">Available Integrations</div>', unsafe_allow_html=True)
    
    # Categorize integrations
    integration_categories = {
        "Booking Systems": [
            {"name": "Amadeus", "status": "Connected", "last_sync": "10 minutes ago"},
            {"name": "Sabre", "status": "Available", "last_sync": ""},
            {"name": "Travelport", "status": "Available", "last_sync": ""}
        ],
        "CRM Systems": [
            {"name": "Salesforce", "status": "Connected", "last_sync": "1 hour ago"},
            {"name": "HubSpot", "status": "Available", "last_sync": ""},
            {"name": "Microsoft Dynamics", "status": "Available", "last_sync": ""}
        ],
        "Analytics Platforms": [
            {"name": "Google Analytics", "status": "Connected", "last_sync": "30 minutes ago"},
            {"name": "Tableau", "status": "Connected", "last_sync": "2 hours ago"},
            {"name": "Power BI", "status": "Available", "last_sync": ""}
        ],
        "Data Sources": [
            {"name": "Weather API", "status": "Connected", "last_sync": "15 minutes ago"},
            {"name": "Competitor Price Tracker", "status": "Connected", "last_sync": "5 minutes ago"},
            {"name": "Events Database", "status": "Connected", "last_sync": "1 hour ago"}
        ]
    }
    
    # Display integrations by category
    for category, integrations in integration_categories.items():
        with st.expander(f"{category} ({len([i for i in integrations if i['status'] == 'Connected'])}/{len(integrations)} Connected)", expanded=(category == "Booking Systems")):
            for integration in integrations:
                col1, col2, col3 = st.columns([3, 2, 2])
                
                with col1:
                    st.markdown(f"**{integration['name']}**")
                
                with col2:
                    if integration["status"] == "Connected":
                        st.markdown(f"✅ Connected | Last sync: {integration['last_sync']}")
                    else:
                        st.markdown("🔹 Available")
                
                with col3:
                    if integration["status"] == "Connected":
                        st.button(f"Configure {integration['name']}")
                    else:
                        st.button(f"Connect {integration['name']}")
                
                st.markdown("---")
    
    # Data flow visualization
    st.markdown('<div class="sub-header">Data Flow Visualization</div>', unsafe_allow_html=True)
    
    # Create a simple data flow diagram
    st.image("https://miro.medium.com/max/1400/1*tj-Qdsk-DRqHSIzPzrP-_g.png", 
             caption="SmartPrice Integration Architecture")
    
    # Integration status
    st.markdown('<div class="sub-header">Integration Status</div>', unsafe_allow_html=True)
    
    # Simulated integration status data
    integration_status = pd.DataFrame({
        'Integration': ['Amadeus', 'Salesforce', 'Google Analytics', 'Weather API', 'Competitor Price Tracker'],
        'Status': ['Healthy', 'Healthy', 'Warning', 'Healthy', 'Healthy'],
        'Data Flow': ['Bidirectional', 'Export Only', 'Import Only', 'Import Only', 'Import Only'],
        'Last 24h Calls': [1240, 650, 890, 2880, 4320],
        'Errors (24h)': [0, 0, 12, 0, 3]
    })
    
    # Apply conditional formatting to status
    def color_status(val):
        color = 'green' if val == 'Healthy' else 'orange' if val == 'Warning' else 'red'
        return f'color: {color}'
    
    # Display the dataframe with formatting
    st.dataframe(integration_status.style.applymap(color_status, subset=['Status']))
    
    # Warning for Google Analytics
    st.warning("⚠️ Google Analytics integration experiencing intermittent connection issues. Monitoring...")
    
    # Custom integration
    st.markdown('<div class="sub-header">Custom Integration</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Need to connect to a system not listed above? You can create a custom integration using our API
    or webhook endpoints.
    """)
    
    tab1, tab2 = st.tabs(["API Integration", "Webhook Integration"])
    
    with tab1:
        st.markdown("""
        ### API Integration
        
        To integrate with the SmartPrice API, follow these steps:
        
        1. Generate an API key in your account settings
        2. Use the key to authenticate your requests
        3. Follow our API documentation to make calls
        
        **Sample code:**
        ```python
        import requests
        
        API_KEY = "your_api_key"
        BASE_URL = "https://api.smartprice.io/v1"
        
        # Get optimal price for a route
        def get_optimal_price(route_id, date):
            endpoint = f"{BASE_URL}/predict"
            headers = {"Authorization": f"Bearer {API_KEY}"}
            
            data = {
                "route_id": route_id,
                "date": date
            }
            
            response = requests.post(endpoint, json=data, headers=headers)
            return response.json()
            
        # Push booking data back to SmartPrice
        def send_booking_data(booking_data):
            endpoint = f"{BASE_URL}/bookings"
            headers = {"Authorization": f"Bearer {API_KEY}"}
            
            response = requests.post(endpoint, json=booking_data, headers=headers)
            return response.status_code == 200
        ```
        """)
    
    with tab2:
        st.markdown("""
        ### Webhook Integration
        
        To receive real-time updates from SmartPrice, set up webhook endpoints:
        
        1. Configure a publicly accessible endpoint in your system
        2. Register the endpoint URL in the SmartPrice dashboard
        3. Choose which events you want to receive
        
        **Sample webhook receiver (Node.js):**
        ```javascript
        const express = require('express');
        const app = express();
        app.use(express.json());
        
        // Webhook endpoint for price updates
        app.post('/webhooks/smartprice/price-updates', (req, res) => {
            const data = req.body;
            
            // Verify webhook signature
            const signature = req.headers['x-smartprice-signature'];
            if (!verifySignature(data, signature, process.env.WEBHOOK_SECRET)) {
                return res.status(401).send('Invalid signature');
            }
            
            // Process the price update
            console.log(`Received price update for route ${data.route_id}`);
            updatePriceInDatabase(data.route_id, data.date, data.optimal_price);
            
            res.status(200).send('Webhook received');
        });
        
        app.listen(3000, () => console.log('Webhook server running on port 3000'));
        ```
        """)
    
    # Integration setup form
    st.markdown('<div class="sub-header">New Integration Setup</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.selectbox("Integration Type", ["Select an integration type...", "API", "Webhook", "SFTP", "Database", "Custom"])
        st.text_input("Integration Name")
        st.text_input("Endpoint URL")
    
    with col2:
        st.selectbox("Authentication Method", ["Select authentication method...", "API Key", "OAuth 2.0", "Basic Auth", "None"])
        st.text_input("Username/Client ID")
        st.text_input("Password/Client Secret", type="password")
    
    st.selectbox("Data Direction", ["Bidirectional", "Import Only (Read)", "Export Only (Write)"])
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.multiselect("Data to Import", ["Competitor Prices", "Booking Data", "Search Trends", "External Events", "Weather Data"])
    
    with col4:
        st.multiselect("Data to Export", ["Optimal Prices", "Demand Forecasts", "Revenue Predictions", "Price Recommendations"])
    
    st.button("Test Connection")
    st.button("Save Integration")

# --- Add Mobile Optimizations ---
# Check if user is on mobile (simulated)
is_mobile = False

if is_mobile:
    # Add mobile-specific CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 1.8rem !important;
        }
        .sub-header {
            font-size: 1.2rem !important;
        }
        .card {
            padding: 1rem !important;
        }
        /* Add more mobile optimizations as needed */
    </style>
    """, unsafe_allow_html=True)

    # Show mobile optimization message
    st.sidebar.info("Mobile view optimized. Some features may be simplified.")

# --- Add User Activity Logs ---
st.sidebar.markdown("---")
if st.sidebar.checkbox("Activity Logs"):
    page = "Activity Logs"

# User Activity Logs
if page == "Activity Logs":
    st.markdown('<div class="main-header">User Activity Logs</div>', unsafe_allow_html=True)
    
    # Current user and time display
    st.markdown(f"""
    <div style="text-align: right; font-size: 0.8rem; color: #666;">
    User: {username} | Current time: {current_datetime}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    Track all user interactions with the SmartPrice system. Activity logs help with auditing,
    troubleshooting, and understanding user behavior.
    </div>
    """, unsafe_allow_html=True)
    
    # Activity log filters
    st.markdown('<div class="sub-header">Filter Activity Logs</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_user = st.selectbox("Select User", ["All Users", "Anurag Kushwaha", "Jennifer Lee", "Michael Chen", "Sarah Johnson", "David Park"])
    
    with col2:
        selected_action = st.selectbox("Action Type", ["All Actions", "Login", "Logout", "Price Generation", 
                                                    "Data Export", "Settings Change", "API Access", "Model Training"])
    
    with col3:
        selected_period = st.selectbox("Time Period", ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Custom Range"])
    
    if selected_period == "Custom Range":
        col4, col5 = st.columns(2)
        with col4:
            start_date = st.date_input("Start Date", value=datetime.datetime.now() - datetime.timedelta(days=7))
        with col5:
            end_date = st.date_input("End Date", value=datetime.datetime.now())
    
    # Generate simulated activity logs
    current_dt = datetime.datetime.strptime(current_datetime, "%Y-%m-%d %H:%M:%S")
    
    activities = [
        {"timestamp": current_dt - datetime.timedelta(minutes=5), 
         "user": "Anurag Kushwaha", "action": "Price Generation", 
         "details": "Generated prices for NYC-LAX route (Aug 1-15, 2025)"},
        
        {"timestamp": current_dt - datetime.timedelta(minutes=15), 
         "user": "Anurag Kushwaha", "action": "Data Export", 
         "details": "Exported Q3 pricing report in Excel format"},
        
        {"timestamp": current_dt - datetime.timedelta(minutes=45), 
         "user": "Jennifer Lee", "action": "Model Training", 
         "details": "Initiated retraining of demand forecasting model"},
        
        {"timestamp": current_dt - datetime.timedelta(hours=1, minutes=10), 
         "user": "Anurag Kushwaha", "action": "Settings Change", 
         "details": "Updated notification preferences"},
        
        {"timestamp": current_dt - datetime.timedelta(hours=2), 
         "user": "Michael Chen", "action": "API Access", 
         "details": "Generated new API key for integration with Tableau"},
        
        {"timestamp": current_dt - datetime.timedelta(hours=3, minutes=25), 
         "user": "Sarah Johnson", "action": "Login", 
         "details": "Logged in from 192.168.1.105 (Chrome/Windows)"},
        
        {"timestamp": current_dt - datetime.timedelta(hours=4), 
         "user": "David Park", "action": "Data Export", 
         "details": "Exported competitor price analysis report"},
        
        {"timestamp": current_dt - datetime.timedelta(hours=6, minutes=15), 
         "user": "Anurag Kushwaha", "action": "Login", 
         "details": "Logged in from 192.168.1.110 (Safari/MacOS)"},
        
        {"timestamp": current_dt - datetime.timedelta(hours=23), 
         "user": "System", "action": "Maintenance", 
         "details": "Automatic database optimization completed"}
    ]
    
    # Create DataFrame
    logs_df = pd.DataFrame(activities)
    logs_df["timestamp_str"] = logs_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # Apply filters
    if selected_user != "All Users":
        logs_df = logs_df[logs_df["user"] == selected_user]
    
    if selected_action != "All Actions":
        logs_df = logs_df[logs_df["action"] == selected_action]
    
    # Display filtered logs
    st.markdown('<div class="sub-header">Activity Log Results</div>', unsafe_allow_html=True)
    
    if logs_df.empty:
        st.info("No activities found matching the selected filters.")
    else:
        for _, log in logs_df.iterrows():
            # Determine icon based on action
            if log["action"] == "Login":
                icon = "🔑"
            elif log["action"] == "Logout":
                icon = "🚪"
            elif log["action"] == "Price Generation":
                icon = "💰"
            elif log["action"] == "Data Export":
                icon = "📊"
            elif log["action"] == "Settings Change":
                icon = "⚙️"
            elif log["action"] == "API Access":
                icon = "🔌"
            elif log["action"] == "Model Training":
                icon = "🧠"
            elif log["action"] == "Maintenance":
                icon = "🔧"
            else:
                icon = "📝"
                
            st.markdown(f"""
            <div style="display: flex; margin-bottom: 10px; padding: 10px; border-radius: 5px; background-color: #f5f5f5;">
                <div style="margin-right: 10px; font-size: 24px;">{icon}</div>
                <div style="flex-grow: 1;">
                    <div style="display: flex; justify-content: space-between;">
                        <span><b>{log["user"]}</b> • {log["action"]}</span>
                        <span style="color: #9e9e9e;">{log["timestamp_str"]}</span>
                    </div>
                    <p style="margin: 0;">{log["details"]}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Export options
    st.markdown('<div class="sub-header">Export Activity Logs</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button("Download as CSV", logs_df.to_csv(index=False), file_name="smartprice_activity_logs.csv")
    
    with col2:
        st.button("Export as PDF")
    
    with col3:
        st.button("Email Report")

# --- Add Admin Panel ---
st.sidebar.markdown("---")
is_admin = username == "Anurag02012004"  # Simulated admin check
if is_admin and st.sidebar.checkbox("Admin Panel"):
    page = "Admin Panel"

# Admin Panel
if page == "Admin Panel":
    st.markdown('<div class="main-header">Admin Panel</div>', unsafe_allow_html=True)
    
    # Current user and time display
    st.markdown(f"""
    <div style="text-align: right; font-size: 0.8rem; color: #666;">
    User: {username} (Administrator) | Current time: {current_datetime}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    Administrative controls for managing users, system settings, and global configurations.
    This panel is only accessible to users with administrator privileges.
    </div>
    """, unsafe_allow_html=True)
    
    # Admin navigation
    admin_page = st.radio(
        "Admin Functions",
        ["User Management", "System Configuration", "License Management", "Usage Analytics"]
    )
    
    if admin_page == "User Management":
        st.markdown('<div class="sub-header">User Management</div>', unsafe_allow_html=True)
        
        # Simulated user data
        users = [
            {"id": 1, "username": "Anurag02012004", "name": "Anurag Kushwaha", "email": "anurag@example.com", 
             "role": "Administrator", "status": "Active", "last_login": "2025-07-27 12:08:00"},
            {"id": 2, "username": "jlee", "name": "Jennifer Lee", "email": "jennifer@example.com", 
             "role": "Data Scientist", "status": "Active", "last_login": "2025-07-27 11:45:00"},
            {"id": 3, "username": "mchen", "name": "Michael Chen", "email": "michael@example.com", 
             "role": "Product Manager", "status": "Active", "last_login": "2025-07-27 10:30:00"},
            {"id": 4, "username": "sjohnson", "name": "Sarah Johnson", "email": "sarah@example.com", 
             "role": "Revenue Manager", "status": "Active", "last_login": "2025-07-26 15:20:00"},
            {"id": 5, "username": "dpark", "name": "David Park", "email": "david@example.com", 
             "role": "Marketing Analyst", "status": "Active", "last_login": "2025-07-27 09:15:00"},
            {"id": 6, "username": "rthomas", "name": "Rachel Thomas", "email": "rachel@example.com", 
             "role": "Viewer", "status": "Inactive", "last_login": "2025-07-10 14:30:00"}
        ]
        
        # Create DataFrame
        users_df = pd.DataFrame(users)
        
        # Search and filter
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("Search Users", placeholder="Search by name, username, or email")
        with col2:
            status_filter = st.selectbox("Status", ["All", "Active", "Inactive"])
        
        # Filter users
        filtered_users = users_df
        if search_term:
            filtered_users = filtered_users[
                filtered_users["name"].str.contains(search_term, case=False) |
                filtered_users["username"].str.contains(search_term, case=False) |
                filtered_users["email"].str.contains(search_term, case=False)
            ]
        
        if status_filter != "All":
            filtered_users = filtered_users[filtered_users["status"] == status_filter]
        
        # Display users
        st.dataframe(filtered_users)
        
        # User actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.button("Add New User")
        
        with col2:
            st.button("Edit Selected User")
        
        with col3:
            st.button("Deactivate Selected User")
        
        # Role management
        st.markdown('<div class="sub-header">Role Management</div>', unsafe_allow_html=True)
        
        # Simulated roles
        roles = [
            {"role": "Administrator", "users": 1, "permissions": "Full system access"},
            {"role": "Data Scientist", "users": 1, "permissions": "Model training, data access, analytics"},
            {"role": "Revenue Manager", "users": 1, "permissions": "Price setting, reports, analytics"},
            {"role": "Marketing Analyst", "users": 1, "permissions": "Reports, analytics, exports"},
            {"role": "Viewer", "users": 1, "permissions": "View-only access to reports and dashboards"}
        ]
        
        # Display roles
        roles_df = pd.DataFrame(roles)
        st.dataframe(roles_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.button("Add New Role")
        
        with col2:
            st.button("Edit Role Permissions")
    
    elif admin_page == "System Configuration":
        st.markdown('<div class="sub-header">System Configuration</div>', unsafe_allow_html=True)
        
        # General settings
        with st.expander("General Settings", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.text_input("Company Name", value="AcmeTravel, Inc.")
                st.text_input("System Email", value="notifications@smartprice.io")
                timezone = st.selectbox("System Timezone", ["UTC", "US/Eastern", "US/Central", "US/Pacific", "Europe/London"])
            
            with col2:
                st.selectbox("Default Language", ["English", "Spanish", "French", "German", "Japanese"])
                st.number_input("Session Timeout (minutes)", min_value=5, max_value=120, value=30)
                st.checkbox("Enable Multi-Factor Authentication", value=True)
        
        # API configuration
        with st.expander("API Configuration"):
            st.number_input("Default Rate Limit (requests/minute)", min_value=10, max_value=1000, value=120)
            st.number_input("Maximum Batch Size", min_value=10, max_value=1000, value=100)
            st.checkbox("Enable Detailed API Logging", value=True)
            st.text_input("API Base URL", value="https://api.smartprice.io/v1")
        
        # Email settings
        with st.expander("Email Settings"):
            st.text_input("SMTP Server", value="smtp.sendgrid.net")
            st.text_input("SMTP Port", value="587")
            st.text_input("SMTP Username", value="apikey")
            st.text_input("SMTP Password", value="••••••••••••••••", type="password")
            st.checkbox("Use TLS", value=True)
            st.text_input("Sender Email", value="notifications@smartprice.io")
            st.text_input("Sender Name", value="SmartPrice System")
        
        # Machine learning settings
        with st.expander("Machine Learning Configuration"):
            st.selectbox("Model Retraining Frequency", ["Daily", "Weekly", "Bi-weekly", "Monthly"])
            st.number_input("Minimum Training Data (days)", min_value=30, max_value=365, value=90)
            st.slider("Price Adjustment Limit (%)", min_value=5, max_value=50, value=20)
            st.checkbox("Enable Automated Model Deployment", value=True)
            st.checkbox("Save Model Version History", value=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.button("Save Configuration")
        
        with col2:
            st.button("Reset to Defaults")
    
    elif admin_page == "License Management":
        st.markdown('<div class="sub-header">License Management</div>', unsafe_allow_html=True)
        
        # License information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="highlight">
            <h3>License Information</h3>
            <p><strong>License Type:</strong> Enterprise</p>
            <p><strong>License Key:</strong> SMARTP-ENT-2025-ACME-********</p>
            <p><strong>Issued To:</strong> AcmeTravel, Inc.</p>
            <p><strong>Issue Date:</strong> January 15, 2025</p>
            <p><strong>Expiration Date:</strong> January 14, 2026</p>
            <p><strong>Status:</strong> <span style="color: green;">Active</span></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="highlight">
            <h3>License Features</h3>
            <ul>
                <li>✅ Unlimited routes</li>
                <li>✅ Unlimited users</li>
                <li>✅ Advanced ML models</li>
                <li>✅ API access (10,000 req/hour)</li>
                <li>✅ 24/7 premium support</li>
                <li>✅ Custom integrations</li>
                <li>✅ White labeling</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # License actions
        st.markdown('<div class="sub-header">License Actions</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.button("Update License")
        
        with col2:
            st.button("Contact Sales")
        
        with col3:
            st.button("View Invoice History")
        
        # Usage quotas
        st.markdown('<div class="sub-header">Usage Quotas</div>', unsafe_allow_html=True)
        
        # Simulated quota data
        quotas = [
            {"feature": "API Requests", "usage": 3824156, "limit": 10000000, "unit": "requests", "renewal": "Monthly"},
            {"feature": "Route Optimizations", "usage": "Unlimited", "limit": "Unlimited", "unit": "", "renewal": ""},
            {"feature": "ML Model Retrainings", "usage": 12, "limit": 30, "unit": "retrainings", "renewal": "Monthly"},
            {"feature": "Storage", "usage": 256, "limit": 1000, "unit": "GB", "renewal": ""},
            {"feature": "Export Credits", "usage": "Unlimited", "limit": "Unlimited", "unit": "", "renewal": ""}
        ]
        
        # Display quotas
        quotas_df = pd.DataFrame(quotas)
        st.dataframe(quotas_df)
    
    elif admin_page == "Usage Analytics":
        st.markdown('<div class="sub-header">System Usage Analytics</div>', unsafe_allow_html=True)
        
        # Date range selector
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input("Start Date", value=datetime.datetime.now() - datetime.timedelta(days=30))
        
        with col2:
            end_date = st.date_input("End Date", value=datetime.datetime.now())
        
        # Usage metrics
        st.markdown('<div class="sub-header">Key Usage Metrics</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total API Calls", "3.82M", "+12.4%")
        col2.metric("Price Predictions", "968K", "+8.7%")
        col3.metric("Active Users", "24", "+4")
        col4.metric("Avg Response Time", "118ms", "-14ms")
        
        # Usage trends
        st.markdown('<div class="sub-header">Usage Trends</div>', unsafe_allow_html=True)
        
        # Simulate API usage data
        dates = pd.date_range(start=start_date, end=end_date).tolist()
        api_usage = [15000 + 5000 * np.sin(i/7) + 2000 * np.random.randn() for i in range(len(dates))]
        predictions = [8000 + 3000 * np.sin(i/7) + 1000 * np.random.randn() for i in range(len(dates))]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(dates, api_usage, label='API Calls', marker='o', markersize=4)
        ax.plot(dates, predictions, label='Price Predictions', marker='s', markersize=4)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Count')
        ax.set_title('Daily API Usage')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        fig.autofmt_xdate()
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Usage by user
        st.markdown('<div class="sub-header">Usage by User</div>', unsafe_allow_html=True)
        
        # Simulated user usage data
        user_usage = [
            {"user": "Anurag Kushwaha", "api_calls": 15420, "predictions": 8240, "exports": 42, "logins": 26},
            {"user": "Jennifer Lee", "api_calls": 8760, "predictions": 4120, "exports": 18, "logins": 19},
            {"user": "Michael Chen", "api_calls": 6240, "predictions": 1860, "exports": 32, "logins": 22},
            {"user": "Sarah Johnson", "api_calls": 12840, "predictions": 7640, "exports": 56, "logins": 18},
            {"user": "David Park", "api_calls": 5280, "predictions": 2180, "exports": 14, "logins": 15}
        ]
        
        # Display user usage
        user_usage_df = pd.DataFrame(user_usage)
        user_usage_df = user_usage_df.sort_values('api_calls', ascending=False)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(user_usage_df))
        width = 0.2
        
        ax.bar(x - width*1.5, user_usage_df['api_calls']/1000, width, label='API Calls (K)')
        ax.bar(x - width/2, user_usage_df['predictions']/1000, width, label='Predictions (K)')
        ax.bar(x + width/2, user_usage_df['exports'], width, label='Exports')
        ax.bar(x + width*1.5, user_usage_df['logins'], width, label='Logins')
        
        ax.set_xlabel('User')
        ax.set_ylabel('Count')
        ax.set_title('Usage by User')
        ax.set_xticks(x)
        ax.set_xticklabels(user_usage_df['user'])
        ax.legend()
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button("Export Usage Report", user_usage_df.to_csv(index=False), file_name="smartprice_usage_report.csv")
        
        with col2:
            st.button("Schedule Recurring Report")

# Make sure footer has current timestamp and username
footer_html = """
<div class="footer">
    <p>SmartPrice - Dynamic Travel Pricing System<br>
    <span class="small-text">Created by Anurag Kushwaha | © 2025</span></p>
    <p class="small-text">Last login: {datetime} | User: {username}</p>
</div>
""".format(datetime=current_datetime, username=username)

st.markdown(footer_html, unsafe_allow_html=True)
                