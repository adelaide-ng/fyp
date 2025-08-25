import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import base64
import shap
import urllib.parse
from plotly.subplots import make_subplots
from pathlib import Path
from PIL import Image
from streamlit.components.v1 import html
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# API Key for Google Maps
api_key = st.secrets.get("GOOGLE_MAPS_API_KEY", "")

# Page configuration
st.set_page_config(
    page_title="Air Quality Prediction System",
    page_icon="⛅",
    layout="wide",
    initial_sidebar_state="collapsed",
)

defaults = {
    "current_tab": "Predict AQI",
    "aqi_value": None,
    "prediction_data": None,
    "prediction_history": []
}
for k, v in defaults.items():
        st.session_state.setdefault(k, v)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header */
    .custom-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    
    .custom-header h1 {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .custom-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Navigation tabs */
    .nav-tabs {
        display: flex;
        gap: 10px;
        margin-bottom: 30px;
        background: rgba(102, 126, 234, 0.1);
        border-radius: 15px;
        padding: 10px;
        backdrop-filter: blur(10px);
    }
    
    /* Custom metric styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .aqi-circle {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 20px auto 10px;
        font-size: 3rem;
        font-weight: 700;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .good { background: linear-gradient(45deg, #00e400, #38a169); }
    .moderate { background: linear-gradient(45deg, #ffff00, #dd6b20); }
    .unhealthy-sensitive { background: linear-gradient(45deg, #ff7e00, #e53e3e); }
    .unhealthy { background: linear-gradient(45deg, #ff0000, #805ad5); }
    .very-unhealthy { background: linear-gradient(45deg, #9f7aea, #4a5568); }
    .hazardous { background: linear-gradient(45deg, #7e0023, #1a202c); }
    
    /* Feature cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        height: 100%;
    }
    
    .feature-icon {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(45deg, #667eea, #764ba2);
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 20px;
        font-size: 1.5rem;
    }
    
    /* Health recommendations */
    .health-card {
        background: #f7fafc;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #48bb78;
    }
    
    .recommendation-item {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
        padding: 10px;
        background: white;
        border-radius: 8px;
        border-left: 4px solid #48bb78;
    }
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(45deg, #4299e1, #667eea);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(66, 153, 225, 0.3);
    }
    
    /* Slider customization */
    .stSlider > div > div > div > div {
        background: linear-gradient(to right, #48bb78, #ed8936, #e53e3e);
    }
    
    /* Product cards */
    .product-item {
        background: #f7fafc;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4299e1;
        transition: all 0.3s ease;
    }
    
    .product-item:hover {
        transform: translateX(5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    .product-item img {
        width: 100%;
        height: auto;
        max-height: 200px;
        object-fit: contain;
        border-radius: 5px;
        margin-bottom: 10px;
        display: block;
    }
            
    .amazon-btn {
        background-color: #ff9900;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 5px;
        text-decoration: none;
        font-size: 14px;
        font-weight: bold;
        display: inline-block;
        margin-top: 10px;
        transition: background-color 0.3s ease;
    }
            
    .amazon-btn:hover {
        background-color: #e68900;
        text-decoration: none;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load transformers and model
pt_features = joblib.load('pt_features.pkl')
pt_target = joblib.load('pt_target.pkl')
model = joblib.load('lgb_tuned_model_20250722_092150.pkl')

# Initialize session state
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Predict AQI"

# Header
st.markdown("""
<div class="custom-header">
    <h1>⛅ Air Quality Prediction System</h1>
    <p>Interactive AQI Prediction for Health Guidance and Educational Purposes</p>
</div>
""", unsafe_allow_html=True)

# Navigation tabs
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🎯 Predict AQI", key="tab1", use_container_width=True):
        st.session_state.current_tab = "Predict AQI"
with col2:
    if st.button("📊 Analytics", key="tab2", use_container_width=True):
        st.session_state.current_tab = "Analytics"
with col3:
    if st.button("💡 Learn/Contact", key="tab3", use_container_width=True):
        st.session_state.current_tab = "Learn/Contact"
with col4:
    if st.button("🛒 Products", key="tab4", use_container_width=True):
        st.session_state.current_tab = "Products"
        

# Boxes for precise user input
def precise_slider(label, min_val, max_val, default, step, *, key, help=""):
    """
    Render a slider and a synced number_input side by side.
    """
    base_key   = f"val_{key}"
    slider_key = f"sl_{key}"
    num_key    = f"ni_{key}"

    # Initialize session state with default values if they don't exist
    if base_key not in st.session_state:
        st.session_state[base_key] = default

    # Local callbacks that only update the shared value
    def _from_slider():
        st.session_state[base_key] = st.session_state[slider_key]

    def _from_number():
        st.session_state[base_key] = st.session_state[num_key]

    col_s, col_n = st.columns([5, 1], gap="small")
    
    # Slider (writes to base via on_change)
    col_s.slider(
        label=label,
        min_value=min_val,
        max_value=max_val,
        step=step,
        key=slider_key,
        value=st.session_state[base_key],
        help=help,
        on_change=_from_slider,
    )

    # Formatting
    is_int = float(step).is_integer() and float(min_val).is_integer() and float(max_val).is_integer()
    num_format = "%.0f" if is_int else "%.1f"

    # Number input (writes to base via on_change)
    col_n.number_input(
        label="Enter value",
        min_value=min_val,
        max_value=max_val,
        step=step,
        key=num_key,
        value=st.session_state[base_key],
        format="%.1f",
        label_visibility="hidden",
        on_change=_from_number,
    )

    return float(st.session_state[base_key])


# To display images
def get_base64_image(image_path):
    """Convert local image to base64 string"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.error(f"Image not found: {image_path}")
        return None


# Helper functions
def get_aqi_category(aqi_value):
    """Return AQI category and color based on the range it falls into"""
    if aqi_value <= 50:
        return "Good", "good", "#00e400"
    elif aqi_value <= 100:
        return "Moderate", "moderate", "#ffff00"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups", "unhealthy-sensitive", "#ff7e00"
    elif aqi_value <= 200:
        return "Unhealthy", "unhealthy", "#ff0000"
    elif aqi_value <= 300:
        return "Very Unhealthy", "very-unhealthy", "#9f7aea"
    else:
        return "Hazardous", "hazardous", "#7e0023"

def get_health_recommendations(aqi_value, category):
    # Return health recommendations based on AQI
    recommendations = {
        "Good": [
            "✅ Perfect day for all outdoor activities",
            "🏃 Great for exercising outside",
            "🚶 Ideal for walking and jogging"
        ],
        "Moderate": [
            "✅ Normal outdoor activities are safe",
            "⚠️ Sensitive individuals should limit prolonged outdoor activities",
            "🚶 Great day for walking and light exercise"
        ],
        "Unhealthy for Sensitive Groups": [
            "⚠️ Sensitive groups should limit outdoor activities",
            "😷 Consider wearing a mask if sensitive to air pollution",
            "🏠 Keep windows closed if possible"
        ],
        "Unhealthy": [
            "🚫 Avoid outdoor activities",
            "😷 Wear N95 masks when going outside",
            "🏠 Stay indoors and use air purifiers"
        ],
        "Very Unhealthy": [
            "🚨 Stay indoors at all times",
            "😷 N95 masks are essential if you must go outside",
            "💧 Use HEPA air purifiers indoors"
        ],
        "Hazardous": [
            "🚨 Emergency conditions - stay indoors",
            "😷 Avoid all outdoor activities",
            "🏥 Seek medical attention if experiencing symptoms"
        ]
    }
    return recommendations.get(category, [])

# Define skewed features that were transformed during training
SKEWED_FEATURES = [
    "o3_8hr",
    "pm10",
    "pm2.5",
    "no2",
    "nox",
    "pm2.5_avg",
    "pm10_avg",
    "so2_avg",
    "windspeed"
]

# Define label names for plots
FEATURE_LABELS = {
    "pm2.5": "PM₂.₅ (µg/m³)",
    "pm2.5_avg": "PM₂.₅ 24-hr (µg/m³)",
    "pm10": "PM₁₀ (µg/m³)",
    "pm10_avg": "PM₁₀ 24-hr (µg/m³)",
    "o3": "O₃ 1-hr (ppb)",
    "o3_8hr": "O₃ 8-hr (ppb)",
    "so2": "SO₂ (ppb)",
    "so2_avg": "SO₂ 24-hr (ppb)",
    "no2": "NO₂ (ppb)",
    "nox": "NOₓ (ppb)",
    "co": "CO (ppm)",
    "co_8hr": "CO 8-hr (ppm)",
    "windspeed": "Wind Speed (m/s)",
    "winddirec": "Wind Direction (°)"
}


def predict_aqi(so2, co, o3, o3_8hr, pm10, pm25, no2, nox, co_8hr, pm25_avg, 
                pm10_avg, so2_avg, windspeed, winddirec, record_history: bool = False):
    """
    Build a 1-row DataFrame with all features, then:
      1) Apply pt_features only to the skewed subset,
      2) Pass the full transformed DataFrame to the model,
      3) Inverse-transform the model's output via pt_target back to real AQI.
      4) Store everything for Analytics. If record_history=True, it will append to a simple in-memory history.
    """
    # 1) Assemble inputs (including untransformed features)
    data = {
        "so2":        so2,
        "co":         co,
        "o3":         o3,
        "o3_8hr":     o3_8hr,
        "pm10":       pm10,
        "pm2.5":      pm25,
        "no2":        no2,
        "nox":        nox,
        "co_8hr":     co_8hr,
        "pm2.5_avg":  pm25_avg,
        "pm10_avg":   pm10_avg,
        "so2_avg":    so2_avg,
        "windspeed":  windspeed,
        "winddirec":  winddirec,
    }

    # Create DataFrame
    input_df = pd.DataFrame([data])

    # 2) Copy and transform only the skewed columns
    transformed = input_df.copy()
    # This will only touch the columns in SKEWED_FEATURES that are present in input_df
    to_transform = [c for c in SKEWED_FEATURES if c in input_df.columns]
    transformed[to_transform] = pt_features.transform(input_df[to_transform])

    # 3) Model prediction in the transformed target space
    y_trans = model.predict(transformed).reshape(-1, 1)

    # 4) Invert Yeo–Johnson transform back to original AQI units
    aqi = pt_target.inverse_transform(y_trans).flatten()[0]
    aqi_int = int(round(aqi))

    # Store prediction data in session state for analytics
    st.session_state.prediction_data = {
        'overall_aqi': aqi_int,
        'input_values': data,
        'transformed_input': transformed
    }

    # Maintain a small prediction history
    if record_history:
        st.session_state.setdefault('prediction_history', []).append({
            'ts': datetime.now().isoformat(timespec='seconds'),
            'aqi': aqi_int,
            'inputs': data
        })

    return aqi_int


# SHAP helper function
def compute_shap(row_df: pd.DataFrame):
    """Returns (shap_values_1d, expected_value, feature_names) for a single-row DF"""
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(row_df)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_row = shap_vals[0]

    exp_val = explainer.expected_value
    if isinstance(exp_val, (list, np.ndarray)):
        exp_val = exp_val[0]

    return shap_row, exp_val, list(row_df.columns)


# Clinic Finder using Google Maps API
def clinic_finder_component(api_key: str, height: int = 360):
    """
    Renders a Google Map with a toolbar on top and the results list BELOW the map.
    - Shows nearby doctor/hospital/pharmacy
    - Fetches phone numbers via Places Details (if available)
    - Keeps your same 'height' param for the MAP; results area is extra space below
    """
    # Default center: Kuala Lumpur
    default_lat, default_lng = 3.1390, 101.6869
    # Extra space for results list (scrollable)
    results_height = 360
    total_height = height + results_height + 90  # toolbar + spacing
    html_str = f"""
    <div id="finder" style="width:100%;max-width:1200px;margin:0 auto;">
      <!-- Toolbar -->
      <div id="toolbar" style="display:flex;gap:8px;flex-wrap:wrap;margin:0 0 10px 0">
        <input id="place-input" placeholder="Enter a location (city, postcode)" style="flex:1;min-width:220px;padding:8px;border:1px solid #e2e8f0;border-radius:8px">
        <select id="radius" style="width:130px;padding:8px;border:1px solid #e2e8f0;border-radius:8px">
          <option value="1000">1 km</option>
          <option value="3000" selected>3 km</option>
          <option value="5000">5 km</option>
          <option value="10000">10 km</option>
        </select>
        <button id="btn-search" style="padding:10px 14px;border:none;border-radius:10px;background:linear-gradient(45deg,#4299e1,#667eea);color:#fff;font-weight:600">Search around location</button>
        <button id="btn-near-me" style="padding:10px 14px;border:1px solid #cbd5e0;border-radius:10px;background:#f7fafc">Use my location</button>
      </div>

      <!-- Map -->
      <div id="map" style="width:100%;height:{height}px;border-radius:12px;box-shadow:0 10px 20px rgba(0,0,0,0.05)"></div>

      <!-- Status -->
      <div id="status" style="margin-top:10px;color:#4a5568;font-size:13px"></div>

      <!-- Results BELOW the map -->
      <div id="results" style="margin-top:8px;max-height:{results_height}px;overflow:auto;border:1px solid #edf2f7;border-radius:12px;padding:8px;background:#fff;box-shadow:0 10px 20px rgba(0,0,0,0.03)"></div>

      <p style="color:#718096;font-size:12px;margin-top:8px">
        Tip: If “Use my location” is blocked by your browser, type a nearby landmark or postcode instead.
      </p>
    </div>

    <script>
      // Helpful auth error message if key/restrictions are wrong
      window.gm_authFailure = function() {{
        const el = document.getElementById('map');
        if (el) el.innerHTML = '<div style="padding:12px;color:#e53e3e;font-weight:600">'
          + 'Google Maps auth failed. Likely causes: wrong key, missing billing, or referrer not allowed.'
          + '</div>';
      }};
    </script>

    <script>
      let map, service, geocoder, infowindow, markers=[];
      let lastOrigin = {{lat:{default_lat}, lng:{default_lng}}};

      function setStatus(msg) {{
        const s = document.getElementById('status');
        if (s) s.textContent = msg || '';
      }}

      function initMap(){{
        map = new google.maps.Map(document.getElementById('map'), {{
          center: lastOrigin, zoom: 13, mapTypeControl:false, fullscreenControl:true
        }});
        service = new google.maps.places.PlacesService(map);
        geocoder = new google.maps.Geocoder();
        infowindow = new google.maps.InfoWindow();

        document.getElementById('btn-search').onclick = () => geocodeAndSearch(document.getElementById('place-input').value);
        document.getElementById('btn-near-me').onclick = tryNearMe;

        // Auto-search on load: try geolocation; fallback to default center
        tryNearMe(true);
      }}

      function tryNearMe(silent=false){{
        if (!navigator.geolocation) {{ if (!silent) alert('Geolocation not supported. Please type a location.'); return fallbackSearch(); }}
        navigator.geolocation.getCurrentPosition(
          (pos)=> {{
            lastOrigin = {{lat: pos.coords.latitude, lng: pos.coords.longitude}};
            map.setCenter(lastOrigin); map.setZoom(14);
            searchNearbyMultiple(lastOrigin, getRadius());
          }},
          (err)=> {{
            if (!silent) alert('Location access denied. Typing a location will still work.');
            fallbackSearch();
          }},
          {{enableHighAccuracy:true, timeout:8000}}
        );
      }}

      function fallbackSearch(){{
        map.setCenter(lastOrigin); map.setZoom(13);
        searchNearbyMultiple(lastOrigin, getRadius());
      }}

      function getRadius(){{
        return parseInt(document.getElementById('radius').value, 10) || 3000;
      }}

      function geocodeAndSearch(query){{
        if(!query) {{ alert('Enter a location'); return; }}
        geocoder.geocode({{address: query}}, (results, status)=> {{
          if (status === 'OK' && results[0]) {{
            const loc = results[0].geometry.location;
            lastOrigin = {{lat: loc.lat(), lng: loc.lng()}};
            map.setCenter(lastOrigin); map.setZoom(14);
            searchNearbyMultiple(lastOrigin, getRadius());
          }} else {{
            alert('Location not found. Try a city or postcode.');
          }}
        }});
      }}

      function clearMarkers(){{
        for(const m of markers) m.setMap(null);
        markers = [];
      }}
      function clearResults(){{
        document.getElementById('results').innerHTML = '';
      }}

      // Call Nearby Search once per type and merge results; then call Details for phone numbers.
      function searchNearbyMultiple(center, radius){{
        clearMarkers();
        clearResults();
        setStatus('Searching clinics nearby...');
        const types = ['doctor','hospital','pharmacy'];
        const seen = new Map(); // place_id -> place (from Nearby)
        let pending = types.length;

        types.forEach((t)=>{{
          const request = {{
            location: center,
            radius: radius,
            type: t,
            keyword: 'clinic'
          }};
          service.nearbySearch(request, (results, status, pagination)=> {{
            handleNearbyBatch(results, status, seen);
            if (pagination && pagination.hasNextPage) {{
              pagination.nextPage();
            }} else {{
              if (--pending === 0) {{
                const places = Array.from(seen.values());
                if (places.length === 0) {{
                  setStatus('No clinics found. Try a wider radius.');
                  return;
                }}
                // Fetch details (phone) for up to N results to avoid rate-limit
                const MAX_PLACES_TO_DETAIL = 30;
                const toDetail = places.slice(0, MAX_PLACES_TO_DETAIL);
                fetchDetailsSequential(toDetail, () => {{
                  renderAggregate(places, center);
                }});
              }}
            }}
          }});
        }});
      }}

      function handleNearbyBatch(results, status, seen){{
        if (status !== google.maps.places.PlacesServiceStatus.OK || !results) return;
        for (const p of results) {{
          if (!p.place_id) continue;
          if (!seen.has(p.place_id)) seen.set(p.place_id, p);
        }}
      }}

      // Sequential Details requests (gentle on quota)
      function fetchDetailsSequential(places, done){{
        let i = 0, withPhone = 0;
        function next(){{
          if (i >= places.length) {{ setStatus('Found ' + places.length + ' place(s), ' + withPhone + ' with phone number.'); return done(); }}
          const place = places[i++];
          service.getDetails({{
            placeId: place.place_id,
            fields: ['formatted_phone_number','international_phone_number','website','opening_hours','formatted_address','name','geometry','place_id','rating','vicinity']
          }}, (detail, status)=> {{
            if (status === google.maps.places.PlacesServiceStatus.OK && detail) {{
              place.formatted_phone_number = detail.formatted_phone_number || detail.international_phone_number || null;
              place.website = detail.website || null;
              place.formatted_address = detail.formatted_address || place.vicinity || '';
              if (place.formatted_phone_number) withPhone++;
            }}
            // Small delay to be polite with the API
            setTimeout(next, 120);
          }});
        }}
        next();
      }}

      function renderAggregate(places, origin){{
        // Sort by rating desc
        places.sort((a,b)=> (b.rating||0) - (a.rating||0));

        // Render markers & list
        places.forEach((place, idx)=> {{
          addMarker(place, idx);
        }});
        renderList(places, origin);
        if (places.length > 0 && places[0].geometry && places[0].geometry.location) {{
          map.panTo(places[0].geometry.location);
        }}
      }}

      function addMarker(place, idx){{
        const pos = place.geometry && place.geometry.location;
        if (!pos) return;
        const m = new google.maps.Marker({{map, position: pos, label: String(idx+1)}});
        markers.push(m);
        google.maps.event.addListener(m, 'click', ()=> {{
          const phone = place.formatted_phone_number ? ('<br/>' + place.formatted_phone_number) : '';
          infowindow.setContent('<div style="max-width:240px"><strong>' + (place.name||'Clinic') + '</strong><br/>' + (place.vicinity||place.formatted_address||'') + phone + '<br/>' + (place.rating?('★ '+place.rating):'') + '</div>');
          infowindow.open(map, m);
        }});
      }}

      function renderList(places, origin){{
        const list = document.getElementById('results');
        list.innerHTML = '';
        places.forEach((place, idx)=> {{
          const pos = place.geometry && place.geometry.location;
          const lat = pos ? pos.lat() : null;
          const lng = pos ? pos.lng() : null;
          const originStr = encodeURIComponent(origin.lat + ',' + origin.lng);
          const dest = (lat && lng) ? encodeURIComponent(lat + ',' + lng) : '';
          const directions = dest ? ('https://www.google.com/maps/dir/?api=1&origin=' + originStr + '&destination=' + dest + '&travelmode=driving') : '#';
          const phone = place.formatted_phone_number || '';
          const phoneLink = phone ? ('<a href="tel:' + phone.replace(/\\s|-/g,'') + '">Call</a>') : '';
          const website = place.website ? ('<a href="' + place.website + '" target="_blank">Website</a>') : '';
          const actions = [phoneLink, website, '<a target="_blank" href="' + directions + '">Directions</a>'].filter(Boolean).join(' • ');

          const item = document.createElement('div');
          item.style.padding='10px';
          item.style.border='1px solid #edf2f7';
          item.style.borderRadius='10px';
          item.style.margin='8px 0';
          item.style.background='#fff';
          item.style.cursor='pointer';
          item.innerHTML = `
            <div style="display:flex;gap:10px;align-items:flex-start">
              <div style="min-width:28px;height:28px;border-radius:50%;background:#667eea;color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700">${{idx+1}}</div>
              <div>
                <div style="font-weight:700">${{place.name || 'Clinic'}}</div>
                <div style="color:#4a5568">${{place.vicinity || place.formatted_address || ''}}</div>
                <div style="color:#2d3748;margin:4px 0">${{phone || '—'}}</div>
                <div style="color:#3182ce">${{actions}}</div>
                <div style="color:#718096;font-size:12px">${{place.rating ? ('★ ' + place.rating) : ''}}</div>
              </div>
            </div>`;
          item.onclick = ()=> {{
            // click list -> open marker info
            const marker = markers[idx];
            if (marker) google.maps.event.trigger(marker, 'click');
            // scroll map into view (optional)
            document.getElementById('map').scrollIntoView({{behavior:'smooth', block:'start'}});
          }};
          list.appendChild(item);
        }});
        setStatus('Found ' + places.length + ' place(s).');
      }}

      window.initMap = initMap;
    </script>

    <!-- IMPORTANT: include libraries=places -->
    <script src="https://maps.googleapis.com/maps/api/js?key={api_key}&libraries=places&callback=initMap" async defer></script>
    """
    html(html_str, height=total_height)


# Helper functions for Products
def amazon_search_link(query: str, marketplace: str = "www.amazon.com") -> str:
    q = urllib.parse.quote_plus(query)
    return f"https://{marketplace}/s?k={q}"

def product_card(title: str, description: str, url: str, image_path: str | None = None):
    if image_path and Path(image_path).exists():
        # Get base64 encoded image
        img_base64 = get_base64_image(image_path)
        if img_base64:
            # Detect image format
            image_format = Path(image_path).suffix.lower().replace('.', '')
            if image_format == 'jpg':
                image_format = 'jpeg'
            
            img_html = f'<img src="data:image/{image_format};base64,{img_base64}" alt="{title}">'
        else:
            img_html = ""
    else:
        img_html = ""

    st.markdown(
        f"""
        <div class="product-item">
          {img_html}
          <strong>{title}</strong><br>
          <small>{description}</small><br>
          <a class="amazon-btn" href="{url}" target="_blank" rel="noopener">
            🛒 Search on Amazon
          </a>
        </div>
        """,
        unsafe_allow_html=True
    )


# MAIN content based on selected tab
if st.session_state.current_tab == "Predict AQI":
    # Main prediction interface
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
                <div class="feature-icon">🎛️</div>
                <h2>Environmental Parameters</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Input sliders
        so2         = precise_slider(
                          "SO₂ Concentration (ppb)", 0.0, 1004.0, 10.0, 1.0, 
                          key="so2", help="Sulphur dioxide level"
                      )
        co          = precise_slider(
                          "CO Concentration (ppm)", 0.0, 50.4, 0.5, 0.1, 
                          key="co", help="Carbon monoxide level"
                      )
        o3          = precise_slider(
                          "O₃ 1-hr Concentration (ppb)", 0.0, 604.0, 0.0, 1.0, 
                          key="o3", help="Ozone level over the past hour"
                      )
        o3_8hr      = precise_slider(
                          "O₃ 8-hr Average Concentration (ppb)", 0.0, 200.0, 45.0, 1.0, 
                          key="o3_8hr", help="Mean O₃ over the past 8 hours"
                      )
        pm10        = precise_slider(
                          "PM₁₀ Concentration (µg/m³)", 0.0, 604.0, 50.0, 0.1, 
                          key="pm10", help="Coarse particulate matter"
                      )
        pm25        = precise_slider(
                          "PM₂.₅ Concentration (µg/m³)", 0.0, 500.4, 35.0, 0.1, 
                          key="pm25", help="Fine particulate matter"
                      )
        no2         = precise_slider(
                          "NO₂ Concentration (ppb)", 0.0, 2049.0, 15.0, 1.0, 
                          key="no2", help="Nitrogen dioxide level"
                      )
        nox         = precise_slider(
                          "NOₓ Concentration (ppb)", 0.0, 2049.0, 20.0, 1.0, 
                          key="nox", help="Total NOₓ level"
                      )
        co_8hr      = precise_slider(
                          "CO 8-hr Average Concentration (ppm)", 0.0, 50.4, 1.0, 0.1, 
                          key="co_8hr", help="Mean CO over the past 8 hours"
                      )
        pm25_avg    = precise_slider(
                          "PM₂.₅ 24-hr Average Concentration (µg/m³)", 0.0, 500.4, 35.0, 0.1, 
                          key="pm25_avg", help="Mean PM₂.₅ over the past 24 hours"
                      )
        pm10_avg    = precise_slider(
                          "PM₁₀ 24-hr Average Concentration (µg/m³)", 0.0, 604.0, 50.0, 0.1, 
                          key="pm10_avg", help="Mean PM₁₀ over the past 24 hours"
                      )
        so2_avg     = precise_slider(
                          "SO₂ 24-hr Average Concentration (ppb)", 0.0, 1004.0, 8.0, 1.0, 
                          key="so2_avg", help="Mean SO₂ over the past 24 hours"
                      )
        windspeed   = precise_slider(
                          "Wind Speed (m/s)", 0.0, 30.0, 5.0, 0.1, 
                          key="windspeed", help="Dispersion effect"
                      )
        winddirec   = precise_slider(
                          "Wind Direction (°)", 0.0, 359.0, 10.0, 1.0, 
                          key="winddirec", help="Direction from which wind originates"
                      )
        
        predict_button = st.button("🔮 Predict Air Quality", type="primary", use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
                <div class="feature-icon">📈</div>
                <h2>AQI Prediction Results</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Make prediction
        aqi_value = st.session_state.get("aqi_value")

        if predict_button:
            aqi_value = predict_aqi(so2, co, o3, o3_8hr, pm10, pm25, no2, nox, co_8hr, pm25_avg, 
                                    pm10_avg, so2_avg, windspeed, winddirec, record_history=True)
            st.session_state.aqi_value = aqi_value
        
        if aqi_value is None:
            st.info("Adjust the inputs and click *Predict Air Quality* to generate result.")
            st.stop()
        
        category, css_class, color = get_aqi_category(aqi_value)
        
        # Display AQI circle
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 30px;">
            <div class="aqi-circle {css_class}">
                {aqi_value}
            </div>
            <h2 style="color: {color}; margin-bottom: 10px;">{category}</h2>
            <p style="color: #666;">Current Air Quality Index</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Health recommendations
        st.markdown("""
        <div class="health-card">
            <h3 style="margin-bottom: 15px; color: #2d3748;">🏥 Health Recommendations</h3>
        """, unsafe_allow_html=True)
        
        recommendations = get_health_recommendations(aqi_value, category)
        for rec in recommendations:
            st.markdown(f"""
            <div class="recommendation-item">
                <span>{rec}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

        # AQI Breakpoints
        st.markdown("""
        <div style="margin-top: 1rem;">
            <h4 style="margin-bottom: 0.5rem; color: #2d3748;">AQI Breakpoints</h4>
        </div>
        """, unsafe_allow_html=True)

        img_path = Path("assets/AQI Breakpoints.png")
        if img_path.exists():
            img = np.array(Image.open(img_path))
            fig = go.Figure(go.Image(z=img))
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                dragmode="zoom",
                hovermode=False
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": True,
                    "displaylogo": False,
                    "modeBarButtonsToAdd": ["zoom2d","pan2d","autoScale2d","resetScale2d"]
                }
            )
            st.caption("Tip: Scroll/pinch to zoom • Drag to pan • Double-click to reset")
        else:
            st.info("Place *'AQI Breakpoints.jpg'* in the same folder as this app to show the image.")


elif st.session_state.current_tab == "Analytics":
    st.markdown("""
    <div class="feature-card">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
            <div class="feature-icon">📊</div>
            <h2>Analytics Dashboard</h2>
        </div>
        <p style="color:#4a5568;margin-top:-8px">All visuals below reflect your <strong>latest prediction</strong> only.</p>
    </div>
    """, unsafe_allow_html=True)

    payload = st.session_state.get("prediction_data")
    if not payload:
        st.info("Make a prediction first on *Predict AQI* to see SHAP and the donut chart here.")
        st.stop()
    else:
        latest = payload
        X_row  = latest['transformed_input']      # 1-row DF passed to the model
        inputs = latest['input_values']           # raw values (dict)

        # ----- SHAP values for the single prediction -----
        shap_row, expected_val, feat_names = compute_shap(X_row)

        # Build a tidy DF for plotting
        df_shap = pd.DataFrame({
            "feature": feat_names,
            "display": [FEATURE_LABELS.get(f, f) for f in feat_names],
            "shap": shap_row,
            "abs_shap": np.abs(shap_row),
            "value": [inputs.get(f, X_row.iloc[0][f]) for f in feat_names]
        }).sort_values("abs_shap", ascending=True)

        # Layout: left = SHAP bar, right = donut
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔎 Feature Importance")
            fig_bar = px.bar(
                df_shap,
                x="shap",
                y="display",
                orientation="h",
                hover_data={"value": True, "shap": ":.3f", "display": False, "abs_shap": False, "feature": False},
                title=None
            )
            fig_bar.update_layout(
                height=520,
                xaxis_title="Contribution to predicted AQI",
                yaxis_title="",
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.caption("The graph above shows how much each pollutant affects the AQI prediction.")

        with col2:
            st.subheader("🍩 Influence of each input on AQI")
            # Donut from absolute SHAP values
            df_pie = df_shap.sort_values("abs_shap", ascending=False).copy()

            # Group long tails into 'Other'
            TOP_K = 8
            if len(df_pie) > TOP_K:
                top = df_pie.head(TOP_K)
                other = pd.DataFrame({
                    "display": ["Other"],
                    "abs_shap": [df_pie.iloc[TOP_K:]["abs_shap"].sum()]
                })
                pie_df = pd.concat([top[["display","abs_shap"]], other], ignore_index=True)
            else:
                pie_df = df_pie[["display","abs_shap"]]

            fig_pie = px.pie(
                pie_df,
                names="display",
                values="abs_shap",
                hole=0.55
            )
            fig_pie.update_layout(
                height=520,
                showlegend=True
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            st.caption("*Share is based on |SHAP| (absolute impact) so positives/negatives don’t cancel out.")

        # Global Feature Importance for model (not user-input-driven)
        with st.expander("Global Feature Importance", expanded=False):
            try:
                importances = model.feature_importances_
                try:
                    feature_names = model.booster_.feature_name()
                except AttributeError:
                    feature_names = model.feature_name_

                fi_df = pd.DataFrame({
                    "feature": feature_names,
                    "display": [FEATURE_LABELS.get(f, f) for f in feature_names],
                    "importance": importances
                }).sort_values("importance", ascending=True)

                fig_fi = px.bar(fi_df, x="importance", y="display", orientation="h")
                fig_fi.update_layout(height=420, xaxis_title="Importance", yaxis_title="")
                st.plotly_chart(fig_fi, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not read model feature importances: {e}")


elif st.session_state.current_tab == "Learn/Contact":
    st.markdown("""
    <div class="feature-card">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
            <div class="feature-icon">📚</div>
            <h2>Learn / Contact</h2>
        </div>
        <p style="color:#4a5568;margin-top:-8px">Explore air quality blogs and resources or find nearby clinics/pharmacies if you need care</p>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("🧠 Helpful Resources")
        resources = [
            {"name":"WHO – Ambient Air Pollution", "desc":"Global facts, health impacts, and guidance", "url":"https://www.who.int/health-topics/air-pollution"},
            {"name":"WHO – Air Quality Guidelines", "desc":"Evidence-based guideline values and reports", "url":"https://www.who.int/publications/i/item/9789240034228"},
            {"name":"CDC – Air Quality & Your Health", "desc":"Health tips for sensitive groups and general public", "url":"https://www.cdc.gov/air"},
            {"name":"US EPA – AQI Basics", "desc":"How AQI works and what the colors mean based on US EPA standards", "url":"https://www.airnow.gov/aqi/aqi-basics"},
            {"name":"UNEP – Air Pollution", "desc":"Policy, actions, and clean-air initiatives", "url":"https://www.unep.org/topics/air"},
            {"name":"NIEHS Blog", "desc":"Air pollution, types of pollutants, affected groups", "url":"https://www.niehs.nih.gov/health/topics/agents/air-pollution"},
            {"name":"Clean Air Asia (Knowledge Center)", "desc":"Asia-focused research and best practices", "url":"https://cleanairasia.org/"},
            {"name":"Malaysia MoH – Haze/Health Advice", "desc":"Local guidance during poor air episodes", "url":"https://www.moh.gov.my/"}
        ]
        for r in resources:
            st.markdown(f"""
            <div class="product-item" style="border-left-color:#805ad5">
                <strong><a href="{r['url']}" target="_blank">{r['name']}</a></strong><br>
                <small style="color:#4a5568">{r['desc']}</small>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.subheader("🗺️ Find nearby clinics/pharmacies")
        api_key = st.secrets.get("GOOGLE_MAPS_API_KEY", "")
        if api_key:
            clinic_finder_component(api_key, height=360)
        else:
            st.warning(
                "Add your Google Maps API key to st.secrets to enable the clinic finder.\n\n"
                "In .streamlit/secrets.toml (local) or *Deploy → Secrets* (Streamlit Cloud):\n\n"
                "\nGOOGLE_MAPS_API_KEY = \"YOUR_KEY_HERE\"\n"
            )
            st.info("Temporary fallback shown below (no API key): a basic Google Maps search embed.")
            html(
                '<iframe src="https://www.google.com/maps?q=clinics%20near%20me&output=embed" '
                'style="width:100%;height:560px;border:0;border-radius:12px"></iframe>',
                height=570
            )

elif st.session_state.current_tab == "Products":
    st.markdown("""
    <div class="feature-card">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
            <div class="feature-icon">🛒</div>
            <h2>Recommended Products</h2>
        </div>
        <p style="color:#4a5568;margin-top:-8px">Buying guide for recommended protective face masks and air purifiers</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Product categories
    col1, col2 = st.columns(2)

    MARKETPLACE = "www.amazon.com"
    
    with col1:
        st.subheader("😷 Personal Protection")
        
        masks = [
        {
            "title": "N95 Respirator Masks",
            "image": "assets/n95_respirator.jpg",
            "query": "N95 masks",
            "description": "NIOSH-approved face masks, filters 95% of air particles"
        },
        {
            "title": "KN95 Face Masks",
            "image": "assets/kn95_mask.jpg",
            "query": "KN95 masks",
            "description": "Comfortable fit, multiple layers"
        },
        {
            "title": "P100 Respirator",
            "image": "assets/p100_respirator.jpg",
            "query": "P100 respirator",
            "description": "Maximum protection, reusable"
        },
        ]

        for mask in masks:
            url = amazon_search_link(mask["query"], marketplace=MARKETPLACE)
            product_card(mask["title"], mask["description"], url, mask["image"])
    
    with col2:
        st.subheader("💧 Air Purification")
        
        purifiers = [
        {
            "title": "HEPA Air Purifier",
            "image": "assets/hepa_air_purifier.jpg",
            "query": "HEPA air purifier",
            "description": "Removes 99.97% of particles, quiet operation"
        },
        {
            "title": "Smart Air Monitor",
            "image": "assets/smart_air_monitor.jpg",
            "query": "smart air quality monitor PM2.5",
            "description": "Real-time PM2.5 and AQI monitoring"
        },
        {
            "title": "UV-C Air Sanitizer",
            "image": "assets/uvc_air_sanitizer.jpg",
            "query": "UV-C air sanitizer purifier",
            "description": "Kills bacteria and viruses"
        }
        ]
        
        for purifier in purifiers:
            url = amazon_search_link(purifier["query"], marketplace=MARKETPLACE)
            product_card(purifier["title"], purifier["description"], url, purifier["image"])
    
    # Buying guide
    st.subheader("💡 Buying Guide")
    
    guide_sections = [
        {
            "title": "For High PM2.5 Days (AQI > 100)",
            "items": ["N95 or KN95 masks", "Portable air purifier", "Indoor plants for natural filtration"]
        },
        {
            "title": "For Sensitive Individuals",
            "items": ["Personal air quality monitor", "HEPA air purifier for bedroom", "P100 respirator for outdoor work"]
        },
        {
            "title": "For Families with Children",
            "items": ["Child-sized N95 masks", "Whole-house air purification", "Indoor air quality monitoring"]
        }
    ]
    
    for section in guide_sections:
        with st.expander(f"🎯 {section['title']}"):
            for item in section['items']:
                st.write(f"• {item}")

# Footer
st.markdown("---")
colA, colB = st.columns([1, 9])
with colA:
    st.image("assets/Sustainable_Development_Goal_03GoodHealth.png", width=40)
with colB:
    st.markdown(
        "<div style='color:#666;margin-top:6px'>"
        "<strong>⛅ Air Quality Prediction System</strong><br>"
        "Built with Streamlit • Data-driven insights for healthier living<br>"
        "<span style='font-size:0.9rem'>SDG 3: Good Health and Well-being</span>"
        "</div>",
        unsafe_allow_html=True
    )