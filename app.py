"""
CSW Savings Calculator - Streamlit Web App
MULTI-BUILDING VERSION - Supports Office and Hotel buildings
Uses merged regression_coefficients.csv with both building types
"""

import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Winsert Savings Calculator",
    page_icon="🏢",
    layout="wide"
)

if 'step' not in st.session_state:
    st.session_state.step = 0  # Start at building type selection

# ============================================================================
# DATA: HVAC AND INPUT OPTIONS
# ============================================================================

# Office HVAC Systems
OFFICE_HVAC_SYSTEMS = [
    'Packaged VAV with electric reheat',
    'Packaged VAV with hydronic reheat',
    'Built-up VAV with hydronic reheat',
    'Other'
]

# Hotel HVAC Systems
HOTEL_HVAC_SYSTEMS = [
    'PTAC',
    'PTHP',
    'Fan Coil Unit',
    'Other'
]

HEATING_FUELS = ['Electric', 'Natural Gas', 'None']
COOLING_OPTIONS = ['Yes', 'No']
WINDOW_TYPES = ['Single pane', 'Double pane']
CSW_TYPES = ['Winsert Lite', 'Winsert Plus']
CSW_TYPE_MAPPING = {'Winsert Lite': 'Single', 'Winsert Plus': 'Double'}

# Cooling adjustment polynomial coefficients for Office
COOLING_MULT_COEFFICIENTS_OFFICE = {
    'Mid': {'a': 0.6972151451662, 'b': -0.0001078176371, 'c': 3.60507e-8, 'd': -6.4e-12},
    'Large': {'a': 0.779295373677, 'b': 0.000049630331, 'c': -2.8839e-8, 'd': 1e-12}
}

# ============================================================================
# LOAD DATA FROM CSV FILES
# ============================================================================

@st.cache_data
def load_weather_data():
    """Load weather data from CSV file"""
    try:
        df = pd.read_csv('weather_information.csv')
        df['State'] = df['State'].replace('Aklaska', 'Alaska')
        
        weather_dict = {}
        for _, row in df.iterrows():
            state = row['State']
            city = row['Cities']
            hdd = row['Heating Degree Days (HDD)']
            cdd = row['Cooling Degree Days (CDD)']
            
            if state not in weather_dict:
                weather_dict[state] = {}
            
            weather_dict[state][city] = {'HDD': hdd, 'CDD': cdd}
        
        return weather_dict
    except FileNotFoundError:
        st.error("⚠️ Weather data file not found")
        return {}

@st.cache_data
def load_regression_coefficients():
    """Load merged regression coefficients from CSV (Office + Hotel)"""
    try:
        # CRITICAL: keep_default_na=False prevents pandas from converting 'N/A' string to NaN
        df = pd.read_csv('regression_coefficients.csv', keep_default_na=False, na_values=[''])
        return df
    except FileNotFoundError:
        st.error("⚠️ Regression coefficients file not found")
        return pd.DataFrame()

# Load data
WEATHER_DATA_BY_STATE = load_weather_data()
REGRESSION_COEFFICIENTS = load_regression_coefficients()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_wwr(csw_area, building_area, num_floors):
    """Calculate Window-to-Wall Ratio"""
    if num_floors == 0 or building_area == 0:
        return 0
    floor_area = building_area / num_floors
    wall_area = (floor_area ** 0.5) * 4 * 15 * num_floors
    return csw_area / wall_area if wall_area > 0 else 0

def calculate_cooling_multiplier_office(cdd, building_size):
    """Calculate cooling adjustment multiplier for Office based on CDD"""
    coeffs = COOLING_MULT_COEFFICIENTS_OFFICE[building_size]
    a, b, c, d = coeffs['a'], coeffs['b'], coeffs['c'], coeffs['d']
    multiplier = a + b * cdd + c * (cdd ** 2) + d * (cdd ** 3)
    return max(0.0, min(1.0, multiplier))

def build_lookup_config_office(inputs, hours):
    """Build configuration for finding Office regression row"""
    base = 'Single' if inputs['existing_window'] == 'Single pane' else 'Double'
    csw_type = CSW_TYPE_MAPPING.get(inputs['csw_type'], inputs['csw_type'])
    
    if inputs['building_area'] > 30000 and inputs['hvac_system'] == 'Built-up VAV with hydronic reheat':
        size = 'Large'
    else:
        size = 'Mid'
    
    heating_fuel = inputs['heating_fuel']
    if size == 'Mid':
        hvac_fuel = 'PVAV_Elec' if heating_fuel in ['Electric', 'None'] else 'PVAV_Gas'
    else:
        hvac_fuel = 'VAV'
    
    fuel = 'Electric' if heating_fuel == 'None' else heating_fuel
    
    return {
        'base': base,
        'csw': csw_type,
        'size': size,
        'hvac_fuel': hvac_fuel,
        'fuel': fuel,
        'occupancy': '',
        'hours': hours
    }

def build_lookup_config_hotel(inputs, occupancy_level):
    """Build configuration for finding Hotel CSW regression row"""
    base = 'Single' if inputs['existing_window'] == 'Single pane' else 'Double'
    csw_type = CSW_TYPE_MAPPING.get(inputs['csw_type'], inputs['csw_type'])
    
    hvac_system = inputs['hvac_system']
    size = 'Small' if hvac_system in ['PTAC', 'PTHP'] else 'Large'
    
    hvac_mapping = {'PTAC': 'PTAC', 'PTHP': 'PTHP', 'Fan Coil Unit': '', 'Other': ''}
    hvac_fuel = hvac_mapping.get(hvac_system, '')
    
    heating_fuel = inputs['heating_fuel']
    if hvac_system in ['PTAC', 'PTHP'] or heating_fuel == 'None':
        fuel = 'Electric'
    else:
        fuel = 'Gas' if heating_fuel == 'Natural Gas' else 'Electric'
    
    return {
        'base': base,
        'csw': csw_type,
        'size': size,
        'hvac_fuel': hvac_fuel,
        'fuel': fuel,
        'occupancy': occupancy_level,
        'hours': ''
    }

def build_baseline_config_hotel(inputs, occupancy_level):
    """Build configuration for finding Hotel BASELINE row - uses different logic than CSW lookup"""
    base = 'Single' if inputs['existing_window'] == 'Single pane' else 'Double'
    
    hvac_system = inputs['hvac_system']
    heating_fuel = inputs['heating_fuel']
    
    # Determine size and hvac_fuel for baseline lookup
    # Small hotel (PTAC/PTHP with Electric/None) uses PTAC/PTHP baseline
    # Large hotel uses Gas or Electric baseline based on heating fuel
    if hvac_system == 'PTAC':
        size = 'Small'
        hvac_fuel_baseline = 'PTAC'
    elif hvac_system == 'PTHP':
        size = 'Small'
        hvac_fuel_baseline = 'PTHP'
    else:
        # Large hotel (Fan Coil, Other)
        size = 'Large'
        hvac_fuel_baseline = 'Gas' if heating_fuel == 'Natural Gas' else 'Electric'
    
    return {
        'base': base,
        'size': size,
        'hvac_fuel': hvac_fuel_baseline,
        'occupancy': occupancy_level,
        'hours': ''
    }

def find_regression_row(config, building_type):
    """Find matching regression row for CSW savings"""
    if REGRESSION_COEFFICIENTS.empty:
        return None
    
    if building_type == 'Office':
        # Check occupancy column - it should be empty string or NaN for Office
        mask = (
            (REGRESSION_COEFFICIENTS['base'] == config['base']) &
            (REGRESSION_COEFFICIENTS['csw'] == config['csw']) &
            (REGRESSION_COEFFICIENTS['size'] == config['size']) &
            (REGRESSION_COEFFICIENTS['hvac_fuel'] == config['hvac_fuel']) &
            (REGRESSION_COEFFICIENTS['hours'] == config['hours']) &
            ((REGRESSION_COEFFICIENTS['occupancy'] == '') | (REGRESSION_COEFFICIENTS['occupancy'].isna()))
        )
        if pd.notna(config['fuel']) and config['fuel'] != '':
            mask = mask & (REGRESSION_COEFFICIENTS['fuel'] == config['fuel'])
        
        result = REGRESSION_COEFFICIENTS[mask]
    else:  # Hotel
        mask = (
            (REGRESSION_COEFFICIENTS['base'] == config['base']) &
            (REGRESSION_COEFFICIENTS['csw'] == config['csw']) &
            (REGRESSION_COEFFICIENTS['size'] == config['size']) &
            (REGRESSION_COEFFICIENTS['occupancy'] == config['occupancy']) &
            ((REGRESSION_COEFFICIENTS['hours'] == '') | (REGRESSION_COEFFICIENTS['hours'].isna()))
        )
        if config['hvac_fuel']:
            mask = mask & (REGRESSION_COEFFICIENTS['hvac_fuel'] == config['hvac_fuel'])
        else:
            mask = mask & ((REGRESSION_COEFFICIENTS['hvac_fuel'] == '') | (REGRESSION_COEFFICIENTS['hvac_fuel'].isna()))
        if pd.notna(config['fuel']):
            mask = mask & (REGRESSION_COEFFICIENTS['fuel'] == config['fuel'])
    
        result = REGRESSION_COEFFICIENTS[mask]
    
    return result.iloc[0] if not result.empty else None

def find_baseline_eui_row(config, building_type):
    """Find baseline EUI regression row"""
    if REGRESSION_COEFFICIENTS.empty:
        return None
    
    if building_type == 'Office':
        fuel_type = 'Gas' if config['fuel'] == 'Natural Gas' else 'Electric'
        
        # Office baseline: csw='N/A', occupancy is NaN or empty, hours must match
        mask = (
            (REGRESSION_COEFFICIENTS['base'] == config['base']) &
            (REGRESSION_COEFFICIENTS['csw'] == 'N/A') &
            (REGRESSION_COEFFICIENTS['size'] == config['size']) &
            (REGRESSION_COEFFICIENTS['hvac_fuel'] == fuel_type) &
            (REGRESSION_COEFFICIENTS['hours'] == config['hours']) &
            ((REGRESSION_COEFFICIENTS['occupancy'] == '') | (REGRESSION_COEFFICIENTS['occupancy'].isna()))
        )
        
        result = REGRESSION_COEFFICIENTS[mask]
        
        # If no exact match, try with N/A fuel
        if result.empty:
            mask = (
                (REGRESSION_COEFFICIENTS['base'] == config['base']) &
                (REGRESSION_COEFFICIENTS['csw'] == 'N/A') &
                (REGRESSION_COEFFICIENTS['size'] == config['size']) &
                (REGRESSION_COEFFICIENTS['hvac_fuel'] == fuel_type) &
                (REGRESSION_COEFFICIENTS['fuel'] == 'N/A') &
                (REGRESSION_COEFFICIENTS['hours'] == config['hours']) &
                ((REGRESSION_COEFFICIENTS['occupancy'] == '') | (REGRESSION_COEFFICIENTS['occupancy'].isna()))
            )
            result = REGRESSION_COEFFICIENTS[mask]
    
    else:  # Hotel
        # Hotel baseline lookup uses the hvac_fuel from the baseline config
        mask = (
            (REGRESSION_COEFFICIENTS['base'] == config['base']) &
            (REGRESSION_COEFFICIENTS['csw'] == 'N/A') &
            (REGRESSION_COEFFICIENTS['size'] == config['size']) &
            (REGRESSION_COEFFICIENTS['hvac_fuel'] == config['hvac_fuel']) &
            (REGRESSION_COEFFICIENTS['occupancy'] == config['occupancy']) &
            ((REGRESSION_COEFFICIENTS['hours'] == '') | (REGRESSION_COEFFICIENTS['hours'].isna()))
        )
        
        result = REGRESSION_COEFFICIENTS[mask]
    
    return result.iloc[0] if not result.empty else None

def calculate_from_regression(row, degree_days, is_heating=True):
    """Calculate value using regression formula: value = a + b*DD + c*DD²"""
    if row is None:
        return 0
    
    if is_heating:
        a, b, c = row['heat_a'], row['heat_b'], row['heat_c']
    else:
        a, b, c = row['cool_a'], row['cool_b'], row['cool_c']
    
    return a + b * degree_days + c * (degree_days ** 2)

def interpolate_values(value_param, val_high, val_low, param_high, param_low):
    """Generic interpolation formula"""
    if value_param <= param_low:
        return val_low
    elif value_param >= param_high:
        return val_high
    else:
        return ((value_param - param_low) / (param_high - param_low)) * (val_high - val_low) + val_low

def calculate_savings_office(inputs):
    """Calculate savings for Office buildings"""
    building_area = inputs['building_area']
    csw_area = inputs['csw_area']
    operating_hours = inputs['operating_hours']
    num_floors = inputs['num_floors']
    electric_rate = inputs['electric_rate']
    gas_rate = inputs['gas_rate']
    cooling_installed = inputs['cooling_installed']
    heating_fuel = inputs['heating_fuel']
    hdd = inputs.get('hdd', 0)
    cdd = inputs.get('cdd', 0)
    
    # Determine operating hours brackets
    hours_high = 8760 if operating_hours > 2912 else 2912
    hours_low = 2912 if hours_high == 8760 else 2080
    
    config_high = build_lookup_config_office(inputs, hours_high)
    config_low = build_lookup_config_office(inputs, hours_low)
    
    row_high = find_regression_row(config_high, 'Office')
    row_low = find_regression_row(config_low, 'Office')
    
    if row_high is None or row_low is None:
        st.error(f"⚠️ Could not find Office regression coefficients")
        return None
    
    # Calculate heating savings
    if heating_fuel == 'Natural Gas':
        heating_high = calculate_from_regression(row_high, hdd, is_heating=True)
        heating_low = calculate_from_regression(row_low, hdd, is_heating=True)
        gas_savings_high, gas_savings_low = heating_high, heating_low
        electric_heating_high, electric_heating_low = 0, 0
    else:
        electric_heating_high = calculate_from_regression(row_high, hdd, is_heating=True)
        electric_heating_low = calculate_from_regression(row_low, hdd, is_heating=True)
        gas_savings_high, gas_savings_low = 0, 0
    
    # Calculate cooling savings
    cooling_high = calculate_from_regression(row_high, cdd, is_heating=False)
    cooling_low = calculate_from_regression(row_low, cdd, is_heating=False)
    
    # Interpolate
    if heating_fuel == 'Natural Gas':
        c31 = 0
        c33 = interpolate_values(operating_hours, gas_savings_high, gas_savings_low, hours_high, hours_low)
    else:
        c31 = interpolate_values(operating_hours, electric_heating_high, electric_heating_low, hours_high, hours_low)
        c33 = 0
    
    c32_base = interpolate_values(operating_hours, cooling_high, cooling_low, hours_high, hours_low)
    
    # Apply cooling multiplier
    if cooling_installed == "Yes":
        w24 = 1.0
    else:
        w24 = calculate_cooling_multiplier_office(cdd, config_high['size'])
    
    c32 = c32_base * w24
    
    # Find baseline EUI
    baseline_row_high = find_baseline_eui_row(config_high, 'Office')
    baseline_row_low = find_baseline_eui_row(config_low, 'Office')
    
    if baseline_row_high is None or baseline_row_low is None:
        st.error("⚠️ Could not find Office baseline EUI coefficients")
        return None
    
    baseline_eui_high = calculate_from_regression(baseline_row_high, hdd, is_heating=True)
    baseline_eui_low = calculate_from_regression(baseline_row_low, hdd, is_heating=True)
    baseline_eui = interpolate_values(operating_hours, baseline_eui_high, baseline_eui_low, hours_high, hours_low)
    
    # Calculate final savings
    electric_savings_kwh = (c31 + c32) * csw_area
    gas_savings_therms = c33 * csw_area
    electric_cost_savings = electric_savings_kwh * electric_rate
    gas_cost_savings = gas_savings_therms * gas_rate
    total_cost_savings = electric_cost_savings + gas_cost_savings
    total_savings_kbtu_sf = (electric_savings_kwh * 3.413 + gas_savings_therms * 100) / building_area
    new_eui = baseline_eui - total_savings_kbtu_sf
    percent_eui_savings = (total_savings_kbtu_sf / baseline_eui * 100) if baseline_eui > 0 else 0
    wwr = calculate_wwr(csw_area, building_area, num_floors) if csw_area > 0 and num_floors > 0 else None
    
    return {
        'electric_savings_kwh': electric_savings_kwh,
        'gas_savings_therms': gas_savings_therms,
        'electric_cost_savings': electric_cost_savings,
        'gas_cost_savings': gas_cost_savings,
        'total_cost_savings': total_cost_savings,
        'total_savings_kbtu_sf': total_savings_kbtu_sf,
        'baseline_eui': baseline_eui,
        'new_eui': new_eui,
        'percent_eui_savings': percent_eui_savings,
        'wwr': wwr,
        'hdd': hdd,
        'cdd': cdd,
        'heating_per_sf': c31,
        'cooling_per_sf': c32,
        'gas_per_sf': c33
    }

def calculate_savings_hotel(inputs):
    """Calculate savings for Hotel buildings"""
    building_area = inputs['building_area']
    csw_area = inputs['csw_area']
    occupancy_percent = inputs['occupancy_percent']
    num_floors = inputs['num_floors']
    electric_rate = inputs['electric_rate']
    gas_rate = inputs['gas_rate']
    cooling_installed = inputs['cooling_installed']
    heating_fuel = inputs['heating_fuel']
    hdd = inputs.get('hdd', 0)
    cdd = inputs.get('cdd', 0)
    
    # Hotel uses occupancy interpolation (33% = Low, 100% = High)
    occupancy_high = 100
    occupancy_low = 33
    
    # CSW savings lookup - uses standard config
    config_high = build_lookup_config_hotel(inputs, 'High')
    config_low = build_lookup_config_hotel(inputs, 'Low')
    
    row_high = find_regression_row(config_high, 'Hotel')
    row_low = find_regression_row(config_low, 'Hotel')
    
    if row_high is None or row_low is None:
        st.error(f"⚠️ Could not find Hotel regression coefficients")
        return None
    
    # Calculate heating savings
    if heating_fuel == 'Natural Gas':
        heating_high = calculate_from_regression(row_high, hdd, is_heating=True)
        heating_low = calculate_from_regression(row_low, hdd, is_heating=True)
        gas_savings_high, gas_savings_low = heating_high, heating_low
        electric_heating_high, electric_heating_low = 0, 0
    else:
        electric_heating_high = calculate_from_regression(row_high, hdd, is_heating=True)
        electric_heating_low = calculate_from_regression(row_low, hdd, is_heating=True)
        gas_savings_high, gas_savings_low = 0, 0
    
    # Calculate cooling savings
    cooling_high = calculate_from_regression(row_high, cdd, is_heating=False)
    cooling_low = calculate_from_regression(row_low, cdd, is_heating=False)
    
    # Interpolate based on occupancy
    if heating_fuel == 'Natural Gas':
        c31 = 0
        c33 = interpolate_values(occupancy_percent, gas_savings_high, gas_savings_low, occupancy_high, occupancy_low)
    else:
        c31 = interpolate_values(occupancy_percent, electric_heating_high, electric_heating_low, occupancy_high, occupancy_low)
        c33 = 0
    
    c32_base = interpolate_values(occupancy_percent, cooling_high, cooling_low, occupancy_high, occupancy_low)
    
    # Hotels always apply cooling (no multiplier adjustment needed)
    c32 = c32_base if cooling_installed == "Yes" else 0
    
    # Baseline lookup - uses separate baseline config
    baseline_config_high = build_baseline_config_hotel(inputs, 'High')
    baseline_config_low = build_baseline_config_hotel(inputs, 'Low')
    
    baseline_row_high = find_baseline_eui_row(baseline_config_high, 'Hotel')
    baseline_row_low = find_baseline_eui_row(baseline_config_low, 'Hotel')
    
    if baseline_row_high is None or baseline_row_low is None:
        st.error("⚠️ Could not find Hotel baseline EUI coefficients")
        return None
    
    baseline_eui_high = calculate_from_regression(baseline_row_high, hdd, is_heating=True)
    baseline_eui_low = calculate_from_regression(baseline_row_low, hdd, is_heating=True)
    baseline_eui = interpolate_values(occupancy_percent, baseline_eui_high, baseline_eui_low, occupancy_high, occupancy_low)
    
    # Calculate final savings
    electric_savings_kwh = (c31 + c32) * csw_area
    gas_savings_therms = c33 * csw_area
    electric_cost_savings = electric_savings_kwh * electric_rate
    gas_cost_savings = gas_savings_therms * gas_rate
    total_cost_savings = electric_cost_savings + gas_cost_savings
    total_savings_kbtu_sf = (electric_savings_kwh * 3.413 + gas_savings_therms * 100) / building_area
    new_eui = baseline_eui - total_savings_kbtu_sf
    percent_eui_savings = (total_savings_kbtu_sf / baseline_eui * 100) if baseline_eui > 0 else 0
    wwr = calculate_wwr(csw_area, building_area, num_floors) if csw_area > 0 and num_floors > 0 else None
    
    return {
        'electric_savings_kwh': electric_savings_kwh,
        'gas_savings_therms': gas_savings_therms,
        'electric_cost_savings': electric_cost_savings,
        'gas_cost_savings': gas_cost_savings,
        'total_cost_savings': total_cost_savings,
        'total_savings_kbtu_sf': total_savings_kbtu_sf,
        'baseline_eui': baseline_eui,
        'new_eui': new_eui,
        'percent_eui_savings': percent_eui_savings,
        'wwr': wwr,
        'hdd': hdd,
        'cdd': cdd,
        'heating_per_sf': c31,
        'cooling_per_sf': c32,
        'gas_per_sf': c33
    }

# ============================================================================
# UI
# ============================================================================

# Header
col_logo, col_title = st.columns([1, 6])
with col_logo:
    if os.path.exists('logo.png'):
        st.image('logo.png', width=180)
with col_title:
    st.markdown("<h1 style='margin-bottom: 0;'>Winsert Savings Calculator</h1>", unsafe_allow_html=True)
    building_type_display = st.session_state.get('building_type', 'Select Building Type')
    st.markdown(f"<p style='font-size: 1.2em; color: #666; margin-top: 0;'>{building_type_display}</p>", unsafe_allow_html=True)

st.markdown('---')

# Check if data loaded
if not WEATHER_DATA_BY_STATE:
    st.error("⚠️ Unable to load weather data.")
    st.stop()

if REGRESSION_COEFFICIENTS.empty:
    st.error("⚠️ Unable to load regression coefficients.")
    st.stop()

# Calculate total steps
total_steps = 5
if st.session_state.step > 0:
    progress = st.session_state.step / total_steps
    st.progress(progress)
    st.write(f'Step {st.session_state.step} of {total_steps}')

# STEP 1: Building Type Selection
if st.session_state.step == 0:
    st.header('Step 1: Select Building Type')
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button('🏢 Office Building', use_container_width=True, type='primary'):
            st.session_state.building_type = 'Office'
            st.session_state.step = 1
            st.rerun()
    
    with col2:
        if st.button('🏨 Hotel', use_container_width=True, type='primary'):
            st.session_state.building_type = 'Hotel'
            st.session_state.step = 1
            st.rerun()

# STEP 2: Location
elif st.session_state.step == 1:
    st.header('Step 2: Project Location')
    
    state_options = sorted(WEATHER_DATA_BY_STATE.keys())
    default_state_idx = 0
    if 'state' in st.session_state and st.session_state.state in state_options:
        default_state_idx = state_options.index(st.session_state.state)
    
    state = st.selectbox('Select State', options=state_options, index=default_state_idx, key='state_select')
    st.session_state.state = state
    
    if state:
        city_options = sorted(WEATHER_DATA_BY_STATE[state].keys())
        default_city_idx = 0
        if 'city' in st.session_state and st.session_state.city in city_options:
            default_city_idx = city_options.index(st.session_state.city)
        
        city = st.selectbox('Select City', options=city_options, index=default_city_idx, key='city_select')
        st.session_state.city = city
        
        if city:
            weather = WEATHER_DATA_BY_STATE[state][city]
            st.session_state.hdd = weather['HDD']
            st.session_state.cdd = weather['CDD']
            
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"""<div style='padding: 12px; background-color: #f0f2f6; border-radius: 8px; text-align: center;'>
                    <p style='margin: 0; font-size: 0.9em; color: #666;'>Heating Degree Days</p>
                    <p style='margin: 5px 0 0 0; font-size: 1.4em; font-weight: bold; color: #333;'>{weather['HDD']:,.0f}</p>
                    </div>""",
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f"""<div style='padding: 12px; background-color: #f0f2f6; border-radius: 8px; text-align: center;'>
                    <p style='margin: 0; font-size: 0.9em; color: #666;'>Cooling Degree Days</p>
                    <p style='margin: 5px 0 0 0; font-size: 1.4em; font-weight: bold; color: #333;'>{weather['CDD']:,.0f}</p>
                    </div>""",
                    unsafe_allow_html=True
                )
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_back, col_next = st.columns([1, 1])
    with col_back:
        if st.button('← Back'):
            st.session_state.step = 0
            st.rerun()
    with col_next:
        if st.button('Next →', type='primary'):
            if state and city:
                st.session_state.step = 2
                st.rerun()

# STEP 3: Building Envelope
elif st.session_state.step == 2:
    building_type = st.session_state.get('building_type', 'Office')
    st.header('Step 3: Building Envelope Information')
    col1, col2 = st.columns(2)
    
    with col1:
        # Set building area limits based on building type
        if building_type == 'Hotel':
            min_area, max_area = 15000, 250000
            area_help = "Hotel building area must be between 15,000 and 250,000 square feet"
        else:  # Office
            min_area, max_area = 15000, 500000
            area_help = "Office building area must be between 15,000 and 500,000 square feet"
        
        building_area = st.number_input(
            'Building Area (Sq.Ft.)', 
            min_value=min_area, 
            max_value=max_area, 
            value=min(max(st.session_state.get('building_area', 75000), min_area), max_area), 
            step=1000, 
            key='building_area_input',
            help=area_help
        )
        st.session_state.building_area = building_area
        
        num_floors = st.number_input(
            'Number of Floors', 
            min_value=1, 
            max_value=100, 
            value=st.session_state.get('num_floors', 5), 
            key='num_floors_input',
            help="Number of floors must be between 1 and 100"
        )
        st.session_state.num_floors = num_floors
        
        window_types_list = WINDOW_TYPES
        existing_window_idx = 0
        if 'existing_window' in st.session_state and st.session_state.existing_window in window_types_list:
            existing_window_idx = window_types_list.index(st.session_state.existing_window)
        existing_window = st.selectbox('Type of Existing Window', options=window_types_list, index=existing_window_idx, key='existing_window_select')
        st.session_state.existing_window = existing_window
    
    with col2:
        if existing_window == 'Double pane':
            csw_types_list = ['Winsert Lite']
            csw_type_idx = 0
        else:
            csw_types_list = CSW_TYPES
            csw_type_idx = 0
            if 'csw_type' in st.session_state and st.session_state.csw_type in csw_types_list:
                csw_type_idx = csw_types_list.index(st.session_state.csw_type)
        
        csw_type = st.selectbox('Secondary Window Product', options=csw_types_list, index=csw_type_idx, key='csw_type_select')
        st.session_state.csw_type = csw_type
        
        csw_area = st.number_input('Total Sq. Ft of Secondary Windows Installed', min_value=0, max_value=int(building_area * 0.5), value=min(st.session_state.get('csw_area', 12000), int(building_area * 0.5)), step=100, key='csw_area_input')
        st.session_state.csw_area = csw_area
        
        if csw_area > 0 and building_area > 0 and num_floors > 0:
            wwr = calculate_wwr(csw_area, building_area, num_floors)
            st.markdown(
                f"""<div style='padding: 12px; background-color: #f0f2f6; border-radius: 8px; margin-top: 10px;'>
                <p style='margin: 0; font-size: 0.9em; color: #666;'>Window-to-Wall Ratio</p>
                <p style='margin: 5px 0 0 0; font-size: 1.4em; font-weight: bold; color: #333;'>{wwr:.0%}</p>
                </div>""",
                unsafe_allow_html=True
            )
            
            # WWR validation
            if wwr > 1.0:
                st.error("⚠️ WWR is larger than physically possible. Please update.")
                st.session_state.wwr_error = True
            elif wwr < 0.10 or wwr > 0.50:
                st.warning("⚠️ Warning: window to wall ratio seems out of norm. Please confirm before proceeding.")
                st.session_state.wwr_error = False
            else:
                st.session_state.wwr_error = False
        else:
            st.session_state.wwr_error = False
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_back, col_next = st.columns([1, 1])
    with col_back:
        if st.button('← Back'):
            st.session_state.step = 1
            st.rerun()
    with col_next:
        can_proceed = not st.session_state.get('wwr_error', False)
        if st.button('Next →', type='primary', disabled=not can_proceed):
            st.session_state.step = 3
            st.rerun()

# STEP 4: HVAC & Operations
elif st.session_state.step == 3:
    building_type = st.session_state.get('building_type', 'Office')
    st.header('Step 4: HVAC & Operations')
    col1, col2 = st.columns(2)
    
    with col1:
        electric_rate = st.number_input('Electric Rate ($/kWh)', min_value=0.01, max_value=1.0, value=st.session_state.get('electric_rate', 0.12), step=0.01, format='%.3f', key='electric_rate_input')
        st.session_state.electric_rate = electric_rate
        
        gas_rate = st.number_input('Natural Gas Rate ($/therm)', min_value=0.01, max_value=10.0, value=st.session_state.get('gas_rate', 0.80), step=0.05, format='%.2f', key='gas_rate_input')
        st.session_state.gas_rate = gas_rate
        
        if building_type == 'Office':
            operating_hours = st.number_input('Annual Operating Hours', min_value=1980, max_value=8760, value=st.session_state.get('operating_hours', 8000), step=100, key='operating_hours_input')
            st.session_state.operating_hours = operating_hours
        else:  # Hotel
            occupancy_percent = st.slider('Average Occupancy (%)', min_value=33, max_value=100, value=st.session_state.get('occupancy_percent', 70), step=1, key='occupancy_input', help='Between 33% and 100%')
            st.session_state.occupancy_percent = occupancy_percent
    
    with col2:
        if building_type == 'Office':
            hvac_systems_list = OFFICE_HVAC_SYSTEMS
        else:  # Hotel
            hvac_systems_list = HOTEL_HVAC_SYSTEMS
        
        hvac_idx = 0
        if 'hvac_system' in st.session_state and st.session_state.hvac_system in hvac_systems_list:
            hvac_idx = hvac_systems_list.index(st.session_state.hvac_system)
        hvac_system = st.selectbox('HVAC System Type', options=hvac_systems_list, index=hvac_idx, key='hvac_system_select')
        st.session_state.hvac_system = hvac_system
        
        # Heating fuel logic
        if building_type == 'Office' and hvac_system == 'Packaged VAV with electric reheat':
            heating_fuels_list = ['Electric']
            fuel_idx = 0
        elif building_type == 'Hotel' and hvac_system in ['PTHP', 'PTAC']:
            # PTAC and PTHP systems are packaged terminal units - only electric heating
            heating_fuels_list = ['Electric', 'None']
            fuel_idx = 0
            if 'heating_fuel' in st.session_state and st.session_state.heating_fuel in heating_fuels_list:
                fuel_idx = heating_fuels_list.index(st.session_state.heating_fuel)
        else:
            heating_fuels_list = HEATING_FUELS
            fuel_idx = 0
            if 'heating_fuel' in st.session_state and st.session_state.heating_fuel in heating_fuels_list:
                fuel_idx = heating_fuels_list.index(st.session_state.heating_fuel)
        
        heating_fuel = st.selectbox('Heating Fuel', options=heating_fuels_list, index=fuel_idx, key='heating_fuel_select')
        st.session_state.heating_fuel = heating_fuel
        
        cooling_options_list = COOLING_OPTIONS
        cooling_idx = 0
        if 'cooling_installed' in st.session_state and st.session_state.cooling_installed in cooling_options_list:
            cooling_idx = cooling_options_list.index(st.session_state.cooling_installed)
        cooling_installed = st.selectbox('Cooling Installed?', options=cooling_options_list, index=cooling_idx, key='cooling_installed_select')
        st.session_state.cooling_installed = cooling_installed
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_back, col_next = st.columns([1, 1])
    with col_back:
        if st.button('← Back'):
            st.session_state.step = 2
            st.rerun()
    with col_next:
        if st.button('Calculate Savings →', type='primary'):
            st.session_state.step = 4
            st.rerun()

# STEP 5: Results
elif st.session_state.step == 4:
    building_type = st.session_state.get('building_type', 'Office')
    st.header('💡 Your Energy Savings Results')
    
    inputs = {
        'state': st.session_state.get('state'),
        'city': st.session_state.get('city'),
        'hdd': st.session_state.get('hdd', 0),
        'cdd': st.session_state.get('cdd', 0),
        'building_area': st.session_state.get('building_area', 75000),
        'num_floors': st.session_state.get('num_floors', 5),
        'hvac_system': st.session_state.get('hvac_system'),
        'heating_fuel': st.session_state.get('heating_fuel', 'Electric'),
        'cooling_installed': st.session_state.get('cooling_installed', 'Yes'),
        'existing_window': st.session_state.get('existing_window', 'Single pane'),
        'csw_type': st.session_state.get('csw_type', 'Winsert Lite'),
        'csw_area': st.session_state.get('csw_area', 12000),
        'electric_rate': st.session_state.get('electric_rate', 0.12),
        'gas_rate': st.session_state.get('gas_rate', 0.80)
    }
    
    if building_type == 'Office':
        inputs['operating_hours'] = st.session_state.get('operating_hours', 8000)
        results = calculate_savings_office(inputs)
    else:  # Hotel
        inputs['occupancy_percent'] = st.session_state.get('occupancy_percent', 70)
        results = calculate_savings_hotel(inputs)
    
    if results:
        st.success('✅ Calculation Complete!')
        
        col_chart, col_cost = st.columns([1.3, 1])
        
        with col_chart:
            st.markdown('<h4 style="text-align: center;">Energy Use Intensity (EUI) Reduction</h4>', unsafe_allow_html=True)
            
            baseline_eui = results['baseline_eui']
            savings_eui = results['total_savings_kbtu_sf']
            new_eui = results['new_eui']
            
            fig = go.Figure(go.Waterfall(
                orientation = "v",
                measure = ["absolute", "relative", "total"],
                x = ["Baseline EUI<br>Before Winsert", "Savings with<br>Winsert", "EUI After<br>Winsert"],
                y = [baseline_eui, -savings_eui, new_eui],
                text = [f"{baseline_eui:.1f}", f"−{savings_eui:.1f}", f"{new_eui:.1f}"],
                textposition = ["inside", "outside", "inside"],
                textfont = dict(size=12, color="white"),
                increasing = {"marker":{"color":"#D32F2F", "line":{"color":"#B71C1C", "width":2}}},
                decreasing = {"marker":{"color":"#FF9800", "line":{"color":"#F57C00", "width":2}}},
                totals = {"marker":{"color":"#4CAF50", "line":{"color":"#388E3C", "width":2}}},
                connector = {"line":{"color":"rgb(100, 100, 100)", "width":1}},
                width = [0.5, 0.5, 0.5]
            ))
            
            fig.update_layout(
                height=320,
                showlegend=False,
                yaxis=dict(title='kBtu/SF-yr', title_font=dict(size=11), gridcolor='#E0E0E0', rangemode='tozero'),
                xaxis=dict(title_font=dict(size=11)),
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(t=30, b=80, l=60, r=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(
                f"""<div style='background: linear-gradient(135deg, #2C5F6F 0%, #4A90A4 100%); 
                            padding: 20px; border-radius: 10px; text-align: center;
                            box-shadow: 0 3px 5px rgba(0,0,0,0.1); margin-top: 10px;'>
                    <h2 style='color: white; margin: 0 0 8px 0; font-size: 2.2em; font-weight: bold;'>
                        {results['percent_eui_savings']:.1f}%, {results['total_savings_kbtu_sf']:.1f} kBtu/SF-yr
                    </h2>
                    <p style='color: white; margin: 0; font-size: 0.85em; opacity: 0.95;'>EUI Savings</p>
                </div>""",
                unsafe_allow_html=True
            )
        
        with col_cost:
            st.markdown('<h4 style="text-align: center;">Annual Cost Savings</h4>', unsafe_allow_html=True)
            
            st.markdown(
                f"""<div style='background: linear-gradient(135deg, #2C5F6F 0%, #4A90A4 100%); 
                            padding: 28px; border-radius: 10px; text-align: center;
                            box-shadow: 0 3px 5px rgba(0,0,0,0.1); margin-bottom: 15px; margin-top: 10px;'>
                    <p style='color: white; margin: 0 0 5px 0; font-size: 0.9em; font-weight: 500;'>Total Annual Savings</p>
                    <h1 style='color: white; margin: 0; font-size: 2.5em; font-weight: bold;'>
                        ${results['total_cost_savings']:,.0f}
                    </h1>
                </div>""",
                unsafe_allow_html=True
            )
            
            st.markdown(
                f"""<div style='background: linear-gradient(135deg, #6FA8B8 0%, #8FC1D0 100%); 
                            padding: 20px; border-radius: 8px; margin-bottom: 12px; text-align: center;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.08);'>
                    <p style='margin: 0 0 5px 0; color: #1A4451; font-size: 0.9em; font-weight: 600;'>Electric Savings</p>
                    <p style='font-size: 1.6em; margin: 0; font-weight: bold; color: #1A4451;'>
                        ${results['electric_cost_savings']:,.0f}<span style='font-size: 0.5em;'>/year</span>
                    </p>
                </div>""",
                unsafe_allow_html=True
            )
            
            st.markdown(
                f"""<div style='background: linear-gradient(135deg, #6FA8B8 0%, #8FC1D0 100%); 
                            padding: 20px; border-radius: 8px; text-align: center;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.08);'>
                    <p style='margin: 0 0 5px 0; color: #1A4451; font-size: 0.9em; font-weight: 600;'>Natural Gas Savings</p>
                    <p style='font-size: 1.6em; margin: 0; font-weight: bold; color: #1A4451;'>
                        ${results['gas_cost_savings']:,.0f}<span style='font-size: 0.5em;'>/year</span>
                    </p>
                </div>""",
                unsafe_allow_html=True
            )
        
        st.markdown('---')
        st.markdown('<h4 style="text-align: center;">Energy Savings Breakdown</h4>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""<div style='padding: 15px; background-color: #f0f2f6; border-radius: 8px;'>
                <p style='margin: 0; font-size: 0.9em; color: #666;'>Electric Energy Savings</p>
                <p style='margin: 5px 0 0 0; font-size: 1.5em; font-weight: bold; color: #333;'>{results["electric_savings_kwh"]:,.0f} <span style='font-size: 0.6em;'>kWh/yr</span></p>
                </div>""",
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"""<div style='padding: 15px; background-color: #f0f2f6; border-radius: 8px;'>
                <p style='margin: 0; font-size: 0.9em; color: #666;'>Natural Gas Savings</p>
                <p style='margin: 5px 0 0 0; font-size: 1.5em; font-weight: bold; color: #333;'>{results["gas_savings_therms"]:,.0f} <span style='font-size: 0.6em;'>therms/yr</span></p>
                </div>""",
                unsafe_allow_html=True
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander('🔍 View Detailed Calculations'):
            st.markdown('**Performance Metrics**')
            detail_col1, detail_col2 = st.columns(2)
            with detail_col1:
                st.write(f"• Baseline EUI: {results['baseline_eui']:.1f} kBtu/SF-yr")
                st.write(f"• New EUI: {results['new_eui']:.1f} kBtu/SF-yr")
                st.write(f"• EUI Reduction: {results['percent_eui_savings']:.1f}%")
            with detail_col2:
                st.write(f"• Electric Heating: {results['heating_per_sf']:.4f} kWh/SF-CSW")
                st.write(f"• Cooling & Fans: {results['cooling_per_sf']:.4f} kWh/SF-CSW")
                st.write(f"• Gas Heating: {results['gas_per_sf']:.4f} therms/SF-CSW")
            
            st.markdown('**Project Details**')
            detail_col3, detail_col4 = st.columns(2)
            with detail_col3:
                st.write(f"• Location: {inputs['city']}, {inputs['state']}")
                st.write(f"• Heating Degree Days: {results['hdd']:,.0f}")
                st.write(f"• Cooling Degree Days: {results['cdd']:,.0f}")
            with detail_col4:
                st.write(f"• Building Type: {building_type}")
                st.write(f"• Building Area: {inputs['building_area']:,} SF")
                st.write(f"• Secondary Window Area: {inputs['csw_area']:,} SF")
                if results['wwr']:
                    st.write(f"• Window-to-Wall Ratio: {results['wwr']:.0%}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button('← Start Over', type='secondary'):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.step = 0
        st.rerun()

# Sidebar
with st.sidebar:
    if st.session_state.step == 4:
        building_type = st.session_state.get('building_type', 'Office')
        st.markdown('### 🎛️ Adjust Inputs')
        st.markdown('Modify values to see updated results:')
        st.markdown('---')
        
        st.markdown(f"**📍 Location**")
        st.text(f"{st.session_state.get('city', 'N/A')}, {st.session_state.get('state', 'N/A')}")
        st.markdown('---')
        
        st.markdown('**🏢 Building Envelope**')
        
        # Set building area limits based on building type
        if building_type == 'Hotel':
            min_area, max_area = 15000, 250000
            area_help = "Hotel building area must be between 15,000 and 250,000 square feet"
        else:  # Office
            min_area, max_area = 15000, 500000
            area_help = "Office building area must be between 15,000 and 500,000 square feet"
        
        building_area = st.number_input(
            'Building Area (SF)', 
            min_value=min_area, 
            max_value=max_area, 
            value=min(max(st.session_state.get('building_area', 75000), min_area), max_area), 
            step=1000, 
            key='sidebar_building_area',
            help=area_help
        )
        if building_area != st.session_state.get('building_area'):
            st.session_state.building_area = building_area
            st.rerun()
        
        num_floors = st.number_input(
            'Floors', 
            min_value=1, 
            max_value=100, 
            value=st.session_state.get('num_floors', 5), 
            key='sidebar_num_floors',
            help="Number of floors must be between 1 and 100"
        )
        if num_floors != st.session_state.get('num_floors'):
            st.session_state.num_floors = num_floors
            st.rerun()
        
        existing_window = st.selectbox('Existing Window', options=WINDOW_TYPES, index=WINDOW_TYPES.index(st.session_state.get('existing_window', 'Single pane')), key='sidebar_existing_window')
        if existing_window != st.session_state.get('existing_window'):
            st.session_state.existing_window = existing_window
            st.rerun()
        
        # Product type logic - limit to Winsert Lite if Double pane existing
        if existing_window == 'Double pane':
            csw_types_list = ['Winsert Lite']
            csw_type_idx = 0
        else:
            csw_types_list = CSW_TYPES
            csw_type_idx = 0
            if 'csw_type' in st.session_state and st.session_state.csw_type in csw_types_list:
                csw_type_idx = csw_types_list.index(st.session_state.csw_type)
        
        csw_type = st.selectbox('Product', options=csw_types_list, index=csw_type_idx, key='sidebar_csw_type')
        if csw_type != st.session_state.get('csw_type'):
            st.session_state.csw_type = csw_type
            st.rerun()
        
        csw_area = st.number_input('Secondary Window Area (SF)', min_value=0, max_value=int(building_area * 0.5), value=min(st.session_state.get('csw_area', 12000), int(building_area * 0.5)), step=100, key='sidebar_csw_area')
        if csw_area != st.session_state.get('csw_area'):
            st.session_state.csw_area = csw_area
            st.rerun()
        
        if csw_area > 0 and building_area > 0 and num_floors > 0:
            wwr = calculate_wwr(csw_area, building_area, num_floors)
            st.text(f"WWR: {wwr:.0%}")
        
        st.markdown('---')
        
        st.markdown('**⚙️ HVAC & Utility**')
        if building_type == 'Office':
            operating_hours = st.number_input('Operating Hours/yr', min_value=1980, max_value=8760, value=st.session_state.get('operating_hours', 8000), step=100, key='sidebar_operating_hours')
            if operating_hours != st.session_state.get('operating_hours'):
                st.session_state.operating_hours = operating_hours
                st.rerun()
        else:  # Hotel
            occupancy = st.slider('Occupancy %', min_value=33, max_value=100, value=st.session_state.get('occupancy_percent', 70), step=1, key='sidebar_occupancy')
            if occupancy != st.session_state.get('occupancy_percent'):
                st.session_state.occupancy_percent = occupancy
                st.rerun()
        
        hvac_systems = OFFICE_HVAC_SYSTEMS if building_type == 'Office' else HOTEL_HVAC_SYSTEMS
        hvac_system = st.selectbox('HVAC System', options=hvac_systems, index=hvac_systems.index(st.session_state.get('hvac_system', hvac_systems[0])), key='sidebar_hvac_system')
        if hvac_system != st.session_state.get('hvac_system'):
            st.session_state.hvac_system = hvac_system
            st.rerun()
        
        # Heating fuel logic - apply same rules as main page
        if building_type == 'Office' and hvac_system == 'Packaged VAV with electric reheat':
            heating_fuels_list = ['Electric']
            fuel_idx = 0
        elif building_type == 'Hotel' and hvac_system in ['PTHP', 'PTAC']:
            heating_fuels_list = ['Electric', 'None']
            fuel_idx = 0
            if 'heating_fuel' in st.session_state and st.session_state.heating_fuel in heating_fuels_list:
                fuel_idx = heating_fuels_list.index(st.session_state.heating_fuel)
        else:
            heating_fuels_list = HEATING_FUELS
            fuel_idx = 0
            if 'heating_fuel' in st.session_state and st.session_state.heating_fuel in heating_fuels_list:
                fuel_idx = heating_fuels_list.index(st.session_state.heating_fuel)
        
        heating_fuel = st.selectbox('Heating Fuel', options=heating_fuels_list, index=fuel_idx, key='sidebar_heating_fuel')
        if heating_fuel != st.session_state.get('heating_fuel'):
            st.session_state.heating_fuel = heating_fuel
            st.rerun()
        
        cooling_installed = st.selectbox('Cooling?', options=COOLING_OPTIONS, index=COOLING_OPTIONS.index(st.session_state.get('cooling_installed', 'Yes')), key='sidebar_cooling_installed')
        if cooling_installed != st.session_state.get('cooling_installed'):
            st.session_state.cooling_installed = cooling_installed
            st.rerun()
        
        electric_rate = st.number_input('Electric Rate ($/kWh)', min_value=0.01, max_value=1.0, value=st.session_state.get('electric_rate', 0.12), step=0.01, format='%.3f', key='sidebar_electric_rate')
        if electric_rate != st.session_state.get('electric_rate'):
            st.session_state.electric_rate = electric_rate
            st.rerun()
        
        gas_rate = st.number_input('Gas Rate ($/therm)', min_value=0.01, max_value=10.0, value=st.session_state.get('gas_rate', 0.80), step=0.05, format='%.2f', key='sidebar_gas_rate')
        if gas_rate != st.session_state.get('gas_rate'):
            st.session_state.gas_rate = gas_rate
            st.rerun()
    else:
        st.markdown('### 📝 Summary')
        if st.session_state.step > 0:
            building_type = st.session_state.get('building_type', 'Not selected')
            st.markdown(f"**Building Type:** {building_type}")
        if st.session_state.step > 1:
            st.markdown(f"**Location:** {st.session_state.get('city', 'N/A')}, {st.session_state.get('state', 'N/A')}")
        if st.session_state.step > 2:
            st.markdown(f"**Building:** {st.session_state.get('building_area', 0):,} SF, {st.session_state.get('num_floors', 0)} floors")
            st.markdown(f"**Windows:** {st.session_state.get('existing_window', 'N/A')} → {st.session_state.get('csw_type', 'N/A')}")
            st.markdown(f"**Secondary Window Area:** {st.session_state.get('csw_area', 0):,} SF")
        if st.session_state.step > 3:
            st.markdown(f"**HVAC:** {st.session_state.get('hvac_system', 'N/A')}")
            st.markdown(f"**Heating:** {st.session_state.get('heating_fuel', 'N/A')}")
            building_type = st.session_state.get('building_type', 'Office')
            if building_type == 'Office':
                st.markdown(f"**Operating Hours:** {st.session_state.get('operating_hours', 0):,}/yr")
            else:
                st.markdown(f"**Occupancy:** {st.session_state.get('occupancy_percent', 0)}%")
