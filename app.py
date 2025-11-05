"""
CSW Savings Calculator - Streamlit Web App
MULTI-BUILDING VERSION - Supports Office, Hotel, and School buildings
DUAL PRODUCT COMPARISON - Shows savings for both Winsert Lite and Winsert Plus

Requirements:
- streamlit
- pandas
- plotly
- reportlab (for PDF generation)
- kaleido (for plotly chart export to PDF)

Install with: pip install streamlit pandas plotly reportlab kaleido
"""

import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.io as pio
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Winsert Savings Calculator",
    page_icon="🏢",
    layout="wide"
)

if 'step' not in st.session_state:
    st.session_state.step = 0

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

# School HVAC Systems
SCHOOL_HVAC_SYSTEMS = [
    'Fan Coil Unit',
    'Variable Air Volume',
    'Other'
]

# School Types
SCHOOL_TYPES = ['Primary School', 'Secondary School']
SCHOOL_TYPE_MAPPING = {'Primary School': 'PS', 'Secondary School': 'SS'}

HEATING_FUELS = ['Electric', 'Natural Gas', 'None']
COOLING_OPTIONS = ['Yes', 'No']
WINDOW_TYPES = ['Single pane', 'Double pane']
CSW_TYPE_MAPPING = {'Winsert Lite': 'Single', 'Winsert Plus': 'Double'}

# Cooling adjustment polynomial coefficients for Office
COOLING_MULT_COEFFICIENTS_OFFICE = {
    'Mid': {'a': 0.6972151451662, 'b': -0.0001078176371, 'c': 3.60507e-8, 'd': -6.4e-12},
    'Large': {'a': 0.779295373677, 'b': 0.000049630331, 'c': -2.8839e-8, 'd': 1e-12}
}

# Cooling adjustment polynomial coefficients for Schools (both PS and SS use these)
COOLING_MULT_COEFFICIENTS_SCHOOL = {
    'VAV': {'a': 0.543627257519, 'b': -0.000267199514, 'c': 6.9504e-8, 'd': -7e-12},
    'FCU': {'a': 0.948347618221, 'b': -0.000239269189, 'c': 5.443e-9, 'd': 1e-12}
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
    """Load regression coefficients from CSV (Office + Hotel + School)"""
    try:
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

def calculate_cooling_multiplier_school(cdd, hvac_type):
    """Calculate cooling adjustment multiplier for School based on CDD and HVAC type"""
    hvac_key = 'VAV' if hvac_type == 'Variable Air Volume' else 'FCU'
    coeffs = COOLING_MULT_COEFFICIENTS_SCHOOL[hvac_key]
    a, b, c, d = coeffs['a'], coeffs['b'], coeffs['c'], coeffs['d']
    multiplier = a + b * cdd + c * (cdd ** 2) + d * (cdd ** 3)
    return max(0.0, min(1.0, multiplier))

def apply_zero_floor_to_results(results):
    """Apply zero floor to savings values for display - never show negative savings"""
    if results is None:
        return None
    
    display_results = results.copy()
    
    # Apply floor of zero to all savings metrics
    display_results['electric_savings_kwh'] = max(0, results['electric_savings_kwh'])
    display_results['gas_savings_therms'] = max(0, results['gas_savings_therms'])
    display_results['electric_cost_savings'] = max(0, results['electric_cost_savings'])
    display_results['gas_cost_savings'] = max(0, results['gas_cost_savings'])
    display_results['total_cost_savings'] = max(0, results['total_cost_savings'])
    display_results['total_savings_kbtu_sf'] = max(0, results['total_savings_kbtu_sf'])
    display_results['percent_eui_savings'] = max(0, results['percent_eui_savings'])
    
    # Recalculate new_eui based on floored savings
    display_results['new_eui'] = results['baseline_eui'] - display_results['total_savings_kbtu_sf']
    
    return display_results

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
    """Build configuration for finding Hotel BASELINE row"""
    base = 'Single' if inputs['existing_window'] == 'Single pane' else 'Double'
    
    hvac_system = inputs['hvac_system']
    heating_fuel = inputs['heating_fuel']
    
    if hvac_system == 'PTAC':
        size = 'Small'
        hvac_fuel_baseline = 'PTAC'
    elif hvac_system == 'PTHP':
        size = 'Small'
        hvac_fuel_baseline = 'PTHP'
    else:
        size = 'Large'
        hvac_fuel_baseline = 'Gas' if heating_fuel == 'Natural Gas' else 'Electric'
    
    return {
        'base': base,
        'size': size,
        'hvac_fuel': hvac_fuel_baseline,
        'occupancy': occupancy_level,
        'hours': ''
    }

def build_lookup_config_school(inputs):
    """Build configuration for finding School CSW regression row"""
    base = 'Single' if inputs['existing_window'] == 'Single pane' else 'Double'
    csw_type = CSW_TYPE_MAPPING.get(inputs['csw_type'], inputs['csw_type'])
    
    hvac_system = inputs['hvac_system']
    if hvac_system == 'Fan Coil Unit':
        hvac_fuel = 'FCU'
    elif hvac_system == 'Variable Air Volume':
        hvac_fuel = 'VAV'
    else:
        hvac_fuel = 'VAV'
    
    school_type = SCHOOL_TYPE_MAPPING.get(inputs['school_type'], inputs['school_type'])
    
    heating_fuel = inputs['heating_fuel']
    fuel = 'Electric' if heating_fuel == 'None' else heating_fuel
    
    return {
        'base': base,
        'csw': csw_type,
        'school_type': school_type,
        'hvac_fuel': hvac_fuel,
        'fuel': fuel,
        'occupancy': '',
        'hours': ''
    }

def find_regression_row(config, building_type):
    """Find matching regression row for CSW savings"""
    if REGRESSION_COEFFICIENTS.empty:
        return None
    
    if building_type == 'Office':
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
        
    elif building_type == 'Hotel':
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
        
    elif building_type == 'School':
        mask = (
            (REGRESSION_COEFFICIENTS['base'] == config['base']) &
            (REGRESSION_COEFFICIENTS['csw'] == config['csw']) &
            (REGRESSION_COEFFICIENTS['school_type'] == config['school_type']) &
            (REGRESSION_COEFFICIENTS['hvac_fuel'] == config['hvac_fuel']) &
            (REGRESSION_COEFFICIENTS['fuel'] == config['fuel']) &
            ((REGRESSION_COEFFICIENTS['occupancy'] == '') | (REGRESSION_COEFFICIENTS['occupancy'].isna())) &
            ((REGRESSION_COEFFICIENTS['hours'] == '') | (REGRESSION_COEFFICIENTS['hours'].isna()))
        )
        
        result = REGRESSION_COEFFICIENTS[mask]
    
    return result.iloc[0] if not result.empty else None

def find_baseline_eui_row(config, building_type):
    """Find baseline EUI regression row"""
    if REGRESSION_COEFFICIENTS.empty:
        return None
    
    if building_type == 'Office':
        fuel_type = 'Gas' if config['fuel'] == 'Natural Gas' else 'Electric'
        
        mask = (
            (REGRESSION_COEFFICIENTS['base'] == config['base']) &
            (REGRESSION_COEFFICIENTS['csw'] == 'N/A') &
            (REGRESSION_COEFFICIENTS['size'] == config['size']) &
            (REGRESSION_COEFFICIENTS['hvac_fuel'] == fuel_type) &
            (REGRESSION_COEFFICIENTS['hours'] == config['hours']) &
            ((REGRESSION_COEFFICIENTS['occupancy'] == '') | (REGRESSION_COEFFICIENTS['occupancy'].isna()))
        )
        
        result = REGRESSION_COEFFICIENTS[mask]
        
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
    
    elif building_type == 'Hotel':
        mask = (
            (REGRESSION_COEFFICIENTS['base'] == config['base']) &
            (REGRESSION_COEFFICIENTS['csw'] == 'N/A') &
            (REGRESSION_COEFFICIENTS['size'] == config['size']) &
            (REGRESSION_COEFFICIENTS['hvac_fuel'] == config['hvac_fuel']) &
            (REGRESSION_COEFFICIENTS['occupancy'] == config['occupancy']) &
            ((REGRESSION_COEFFICIENTS['hours'] == '') | (REGRESSION_COEFFICIENTS['hours'].isna()))
        )
        
        result = REGRESSION_COEFFICIENTS[mask]
        
    elif building_type == 'School':
        mask = (
            (REGRESSION_COEFFICIENTS['base'] == config['base']) &
            (REGRESSION_COEFFICIENTS['csw'] == 'N/A') &
            (REGRESSION_COEFFICIENTS['school_type'] == config['school_type']) &
            (REGRESSION_COEFFICIENTS['hvac_fuel'] == config['hvac_fuel']) &
            (REGRESSION_COEFFICIENTS['fuel'] == config['fuel']) &
            ((REGRESSION_COEFFICIENTS['occupancy'] == '') | (REGRESSION_COEFFICIENTS['occupancy'].isna())) &
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
    
    hours_high = 8760 if operating_hours > 2912 else 2912
    hours_low = 2912 if hours_high == 8760 else 2080
    
    config_high = build_lookup_config_office(inputs, hours_high)
    config_low = build_lookup_config_office(inputs, hours_low)
    
    row_high = find_regression_row(config_high, 'Office')
    row_low = find_regression_row(config_low, 'Office')
    
    if row_high is None or row_low is None:
        st.error(f"⚠️ Could not find Office regression coefficients")
        return None
    
    if heating_fuel == 'Natural Gas':
        heating_high = calculate_from_regression(row_high, hdd, is_heating=True)
        heating_low = calculate_from_regression(row_low, hdd, is_heating=True)
        gas_savings_high, gas_savings_low = heating_high, heating_low
        electric_heating_high, electric_heating_low = 0, 0
    else:
        electric_heating_high = calculate_from_regression(row_high, hdd, is_heating=True)
        electric_heating_low = calculate_from_regression(row_low, hdd, is_heating=True)
        gas_savings_high, gas_savings_low = 0, 0
    
    cooling_high = calculate_from_regression(row_high, cdd, is_heating=False)
    cooling_low = calculate_from_regression(row_low, cdd, is_heating=False)
    
    if heating_fuel == 'Natural Gas':
        c31 = 0
        c33 = interpolate_values(operating_hours, gas_savings_high, gas_savings_low, hours_high, hours_low)
    else:
        c31 = interpolate_values(operating_hours, electric_heating_high, electric_heating_low, hours_high, hours_low)
        c33 = 0
    
    c32_base = interpolate_values(operating_hours, cooling_high, cooling_low, hours_high, hours_low)
    
    if cooling_installed == "Yes":
        w24 = 1.0
    else:
        w24 = calculate_cooling_multiplier_office(cdd, config_high['size'])
    
    c32 = c32_base * w24
    
    baseline_row_high = find_baseline_eui_row(config_high, 'Office')
    baseline_row_low = find_baseline_eui_row(config_low, 'Office')
    
    if baseline_row_high is None or baseline_row_low is None:
        st.error("⚠️ Could not find Office baseline EUI coefficients")
        return None
    
    baseline_eui_high = calculate_from_regression(baseline_row_high, hdd, is_heating=True)
    baseline_eui_low = calculate_from_regression(baseline_row_low, hdd, is_heating=True)
    baseline_eui = interpolate_values(operating_hours, baseline_eui_high, baseline_eui_low, hours_high, hours_low)
    
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
    
    occupancy_high = 100
    occupancy_low = 33
    
    config_high = build_lookup_config_hotel(inputs, 'High')
    config_low = build_lookup_config_hotel(inputs, 'Low')
    
    row_high = find_regression_row(config_high, 'Hotel')
    row_low = find_regression_row(config_low, 'Hotel')
    
    if row_high is None or row_low is None:
        st.error(f"⚠️ Could not find Hotel regression coefficients")
        return None
    
    if heating_fuel == 'Natural Gas':
        heating_high = calculate_from_regression(row_high, hdd, is_heating=True)
        heating_low = calculate_from_regression(row_low, hdd, is_heating=True)
        gas_savings_high, gas_savings_low = heating_high, heating_low
        electric_heating_high, electric_heating_low = 0, 0
    else:
        electric_heating_high = calculate_from_regression(row_high, hdd, is_heating=True)
        electric_heating_low = calculate_from_regression(row_low, hdd, is_heating=True)
        gas_savings_high, gas_savings_low = 0, 0
    
    cooling_high = calculate_from_regression(row_high, cdd, is_heating=False)
    cooling_low = calculate_from_regression(row_low, cdd, is_heating=False)
    
    if heating_fuel == 'Natural Gas':
        c31 = 0
        c33 = interpolate_values(occupancy_percent, gas_savings_high, gas_savings_low, occupancy_high, occupancy_low)
    else:
        c31 = interpolate_values(occupancy_percent, electric_heating_high, electric_heating_low, occupancy_high, occupancy_low)
        c33 = 0
    
    c32_base = interpolate_values(occupancy_percent, cooling_high, cooling_low, occupancy_high, occupancy_low)
    c32 = c32_base if cooling_installed == "Yes" else 0
    
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

def calculate_savings_school(inputs):
    """Calculate savings for School buildings"""
    building_area = inputs['building_area']
    csw_area = inputs['csw_area']
    num_floors = inputs['num_floors']
    electric_rate = inputs['electric_rate']
    gas_rate = inputs['gas_rate']
    cooling_installed = inputs['cooling_installed']
    heating_fuel = inputs['heating_fuel']
    hdd = inputs.get('hdd', 0)
    cdd = inputs.get('cdd', 0)
    
    config = build_lookup_config_school(inputs)
    
    row = find_regression_row(config, 'School')
    
    if row is None:
        st.error(f"⚠️ Could not find School regression coefficients")
        return None
    
    if heating_fuel == 'Natural Gas':
        heating_savings = calculate_from_regression(row, hdd, is_heating=True)
        gas_savings = heating_savings
        electric_heating = 0
    else:
        electric_heating = calculate_from_regression(row, hdd, is_heating=True)
        gas_savings = 0
    
    cooling_savings = calculate_from_regression(row, cdd, is_heating=False)
    
    c31 = electric_heating
    if cooling_installed == "Yes":
        c32 = cooling_savings
    else:
        cooling_multiplier = calculate_cooling_multiplier_school(cdd, inputs['hvac_system'])
        c32 = cooling_savings * cooling_multiplier
    c33 = gas_savings
    
    baseline_row = find_baseline_eui_row(config, 'School')
    
    if baseline_row is None:
        st.error("⚠️ Could not find School baseline EUI coefficients")
        return None
    
    baseline_eui = calculate_from_regression(baseline_row, hdd, is_heating=True)
    
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
# PDF GENERATION
# ============================================================================

def generate_pdf_report(inputs, results_lite, results_plus, building_type):
    """Generate a PDF report with Alpen/Winsert info and results for both products"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2C5F6F'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2C5F6F'),
        spaceAfter=12,
        spaceBefore=12
    )
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        leading=14,
        spaceAfter=12
    )
    
    # PAGE 1: ALPEN & WINSERT INFORMATION
    # Add logo if it exists
    if os.path.exists('logo.png'):
        logo = Image('logo.png', width=2*inch, height=0.8*inch)
        logo.hAlign = 'CENTER'
        story.append(logo)
        story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Winsert™ Secondary Glazing System", title_style))
    story.append(Paragraph("Energy Savings Analysis Report", heading_style))
    story.append(Spacer(1, 0.3*inch))
    
    # About Alpen
    story.append(Paragraph("About Alpen High Performance Products", heading_style))
    story.append(Paragraph(
        "For over four decades, Alpen High Performance Products has been leading the way in "
        "climate-responsive design. We engineer custom solutions for durability, energy savings, "
        "and design freedom. Our lightweight triple-pane and quad-pane glass technology delivers "
        "unparalleled energy efficiency while ensuring water and air resistance.",
        body_style
    ))
    
    # About Winsert
    story.append(Paragraph("The Winsert Advantage", heading_style))
    story.append(Paragraph(
        "<b>WinSert Lite</b> utilizes a super-insulated, low profile fiberglass frame combined with "
        "ultra-lightweight thin glass (typically 1.3 mm) laminated to a customized performance film. "
        "AERC-certified installed U-value as low as 0.33.",
        body_style
    ))
    story.append(Paragraph(
        "<b>WinSert Plus</b> combines the same insulated fiberglass frame with a lightweight "
        "high-performance insulated glass unit (IGU) composed of thin glass with low-emissivity "
        "coating, warm edge spacer, and gas fill. AERC-certified installed U-value as low as 0.16.",
        body_style
    ))
    
    # Benefits of Thin Glass Technology
    story.append(Paragraph("Benefits of Thin Glass Technology", heading_style))
    benefits = [
        "97% reduction in air infiltration over existing windows",
        "30-60% improvement in acoustic performance",
        "Low embodied carbon, lightweight construction",
        "Minimal sightlines ideal for historic applications",
        "99%+ UV blockage protects interior finishes",
        "Installation in under 10 minutes without permanent fixtures",
        "15% average whole-building energy savings across all climate zones",
        "Raises interior glass temperature by 20°F"
    ]
    for benefit in benefits:
        story.append(Paragraph(f"• {benefit}", body_style))
    
    # DOE Recognition
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Department of Energy Recognition", heading_style))
    story.append(Paragraph(
        "Alpen was named a semifinalist in the DOE's $2.1 million Building Envelope Prize competition "
        "and received the DOE Retro 30 Award for achieving 32% building envelope improvement and 13.4% "
        "total energy reduction in the Pacific Tower retrofit project - completed in just over a week "
        "with 89% cost savings compared to full window replacement.",
        body_style
    ))
    
    story.append(PageBreak())
    
    # PAGE 2: PROJECT RESULTS - SECTION A: INPUT SUMMARY
    story.append(Paragraph("Your Energy Savings Analysis", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("A. Input Summary", heading_style))
    input_data = [
        ['Parameter', 'Value'],
        ['Building Type', building_type],
        ['Location', f"{inputs['city']}, {inputs['state']}"],
        ['Building Area', f"{inputs['building_area']:,} sq ft"],
        ['Number of Floors', f"{inputs['num_floors']}"],
        ['Existing Window Type', inputs['existing_window']],
        ['Secondary Window Area', f"{inputs['csw_area']:,} sq ft"],
        ['HVAC System', inputs['hvac_system']],
        ['Heating Fuel', inputs['heating_fuel']],
        ['Cooling Installed', inputs['cooling_installed']],
        ['Electric Rate', f"${inputs['electric_rate']:.3f}/kWh"],
        ['Natural Gas Rate', f"${inputs['gas_rate']:.2f}/therm"],
    ]
    
    # Add building-type specific inputs
    if building_type == 'School' and 'school_type' in inputs:
        input_data.insert(2, ['School Type', inputs['school_type']])
    elif building_type == 'Office' and 'operating_hours' in inputs:
        input_data.append(['Operating Hours', f"{inputs['operating_hours']:,} hrs/year"])
    elif building_type == 'Hotel' and 'occupancy_percent' in inputs:
        input_data.append(['Average Occupancy', f"{inputs['occupancy_percent']}%"])
    
    input_table = Table(input_data, colWidths=[2.5*inch, 3.5*inch])
    input_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C5F6F')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(input_table)
    story.append(Spacer(1, 0.3*inch))
    
    # SECTION B: ENERGY SAVINGS COMPARISON
    story.append(Paragraph("B. Product Comparison: Energy & Cost Savings", heading_style))
    
    # Generate Waterfall Chart
    baseline_eui = results_lite['baseline_eui']
    savings_lite = results_lite['total_savings_kbtu_sf']
    eui_lite = results_lite['new_eui']
    savings_plus = results_plus['total_savings_kbtu_sf']
    eui_plus = results_plus['new_eui']
    additional_savings = max(0, savings_plus - savings_lite)
    
    # Determine if we should show both products or just Lite
    show_both = inputs['existing_window'] == 'Single pane'
    
    if show_both:
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "total", "relative", "total"],
            x=["Baseline EUI", "Savings with<br>Winsert Lite", "EUI with<br>Winsert Lite", 
               "Additional Savings<br>Lite → Plus", "Final EUI with<br>Winsert Plus"],
            y=[baseline_eui, -savings_lite, eui_lite, -additional_savings, eui_plus],
            text=[f"{baseline_eui:.1f}", f"−{savings_lite:.1f}", f"{eui_lite:.1f}", 
                  f"−{additional_savings:.1f}", f"{eui_plus:.1f}"],
            textposition=["inside", "outside", "inside", "outside", "inside"],
            textfont=dict(size=12, color="white"),
            increasing={"marker":{"color":"#D32F2F", "line":{"color":"#B71C1C", "width":2}}},
            decreasing={"marker":{"color":"#FF9800", "line":{"color":"#F57C00", "width":2}}},
            totals={"marker":{"color":"#4CAF50", "line":{"color":"#388E3C", "width":2}}},
            connector={"line":{"color":"rgb(100, 100, 100)", "width":1}},
            width=[0.5, 0.5, 0.5, 0.5, 0.5]
        ))
    else:
        # Only show Lite for double pane existing windows
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "total"],
            x=["Baseline EUI", "Savings with<br>Winsert Lite", "EUI with<br>Winsert Lite"],
            y=[baseline_eui, -savings_lite, eui_lite],
            text=[f"{baseline_eui:.1f}", f"−{savings_lite:.1f}", f"{eui_lite:.1f}"],
            textposition=["inside", "outside", "inside"],
            textfont=dict(size=14, color="white"),
            increasing={"marker":{"color":"#D32F2F", "line":{"color":"#B71C1C", "width":2}}},
            decreasing={"marker":{"color":"#FF9800", "line":{"color":"#F57C00", "width":2}}},
            totals={"marker":{"color":"#4CAF50", "line":{"color":"#388E3C", "width":2}}},
            connector={"line":{"color":"rgb(100, 100, 100)", "width":1}},
            width=[0.5, 0.5, 0.5]
        ))
    
    fig.update_layout(
        title="Energy Use Intensity (EUI) Reduction by Product",
        height=400,
        width=600,
        showlegend=False,
        yaxis=dict(title='kBtu/SF-yr', title_font=dict(size=12), gridcolor='#E0E0E0', rangemode='tozero'),
        xaxis=dict(title_font=dict(size=10)),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=60, b=80, l=60, r=20)
    )
    
    # Convert plotly figure to image
    try:
        img_bytes = pio.to_image(fig, format='png', width=600, height=400)
        img_buffer = BytesIO(img_bytes)
        chart_img = Image(img_buffer, width=5.5*inch, height=3.5*inch)
        chart_img.hAlign = 'CENTER'
        story.append(chart_img)
        story.append(Spacer(1, 0.2*inch))
    except Exception as e:
        story.append(Paragraph(f"Note: Chart visualization unavailable. See tables below for detailed data.", body_style))
        story.append(Spacer(1, 0.1*inch))
    
    # Comparison Table
    if show_both:
        comparison_data = [
            ['Metric', 'Winsert Lite', 'Winsert Plus'],
            ['Baseline EUI', f"{baseline_eui:.1f} kBtu/SF-yr", f"{baseline_eui:.1f} kBtu/SF-yr"],
            ['EUI Savings', f"{savings_lite:.1f} kBtu/SF-yr", f"{savings_plus:.1f} kBtu/SF-yr"],
            ['New EUI', f"{eui_lite:.1f} kBtu/SF-yr", f"{eui_plus:.1f} kBtu/SF-yr"],
            ['% EUI Reduction', f"{results_lite['percent_eui_savings']:.1f}%", f"{results_plus['percent_eui_savings']:.1f}%"],
            ['Electric Savings', f"{results_lite['electric_savings_kwh']:,.0f} kWh/yr", f"{results_plus['electric_savings_kwh']:,.0f} kWh/yr"],
            ['Gas Savings', f"{results_lite['gas_savings_therms']:,.0f} therms/yr", f"{results_plus['gas_savings_therms']:,.0f} therms/yr"],
            ['Annual Cost Savings', f"${results_lite['total_cost_savings']:,.2f}/yr", f"${results_plus['total_cost_savings']:,.2f}/yr"],
        ]
    else:
        comparison_data = [
            ['Metric', 'Winsert Lite'],
            ['Baseline EUI', f"{baseline_eui:.1f} kBtu/SF-yr"],
            ['EUI Savings', f"{savings_lite:.1f} kBtu/SF-yr"],
            ['New EUI', f"{eui_lite:.1f} kBtu/SF-yr"],
            ['% EUI Reduction', f"{results_lite['percent_eui_savings']:.1f}%"],
            ['Electric Savings', f"{results_lite['electric_savings_kwh']:,.0f} kWh/yr"],
            ['Gas Savings', f"{results_lite['gas_savings_therms']:,.0f} therms/yr"],
            ['Annual Cost Savings', f"${results_lite['total_cost_savings']:,.2f}/yr"],
        ]
        story.append(Paragraph("Note: Winsert Plus is only available for single pane existing windows.", body_style))
    
    comparison_table = Table(comparison_data, colWidths=[2.5*inch, 1.75*inch, 1.75*inch] if show_both else [3*inch, 3*inch])
    comparison_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C5F6F')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#E8F4F8')),
    ]))
    story.append(comparison_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Climate Data
    story.append(Paragraph("Climate Data", heading_style))
    climate_data = [
        ['Heating Degree Days (HDD)', f"{results_lite['hdd']:,.0f}"],
        ['Cooling Degree Days (CDD)', f"{results_lite['cdd']:,.0f}"],
    ]
    if results_lite['wwr']:
        climate_data.append(['Window-to-Wall Ratio', f"{results_lite['wwr']:.0%}"])
    
    climate_table = Table(climate_data, colWidths=[3*inch, 3*inch])
    climate_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E8F4F8')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(climate_table)
    
    # Footer
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(
        "For more information about Winsert secondary glazing systems, visit "
        "<link href='https://www.thinkalpen.com/winsert' color='blue'>www.thinkalpen.com/winsert</link>",
        body_style
    ))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

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
    if building_type_display == 'School' and 'school_type' in st.session_state:
        building_type_display = f"School - {st.session_state.get('school_type')}"
    st.markdown(f"<p style='font-size: 1.2em; color: #666; margin-top: 0;'>{building_type_display}</p>", unsafe_allow_html=True)

st.markdown('---')

# Check if data loaded
if not WEATHER_DATA_BY_STATE:
    st.error("⚠️ Unable to load weather data.")
    st.stop()

if REGRESSION_COEFFICIENTS.empty:
    st.error("⚠️ Unable to load regression coefficients.")
    st.stop()

# Progress bar logic - only show for steps 1-3
if st.session_state.step >= 1 and st.session_state.step <= 3:
    display_step = int(st.session_state.step)
    progress = display_step / 3
    st.progress(progress)
    st.write(f'Step {display_step} of 3')

# STEP 0: Building Type Selection
if st.session_state.step == 0:
    st.header('Step 1: Select Building Type')
    
    col1, col2, col3 = st.columns(3)
    
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
    
    with col3:
        if st.button('🏫 School', use_container_width=True, type='primary'):
            st.session_state.building_type = 'School'
            st.session_state.step = 0.5
            st.rerun()

# STEP 0.5: School Type Selection (only for schools)
elif st.session_state.step == 0.5:
    st.header('Step 1b: Select School Type')
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button('🏫 Primary School', use_container_width=True, type='primary'):
            st.session_state.school_type = 'Primary School'
            st.session_state.step = 1
            st.rerun()
    
    with col2:
        if st.button('🏫 Secondary School', use_container_width=True, type='primary'):
            st.session_state.school_type = 'Secondary School'
            st.session_state.step = 1
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button('← Back'):
        st.session_state.building_type = None
        st.session_state.step = 0
        st.rerun()

# STEP 1: Location
elif st.session_state.step == 1:
    st.header('Step 1: Project Location')
    
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
            building_type = st.session_state.get('building_type', 'Office')
            if building_type == 'School':
                st.session_state.step = 0.5
            else:
                st.session_state.step = 0
            st.rerun()
    with col_next:
        if st.button('Next →', type='primary'):
            if state and city:
                st.session_state.step = 2
                st.rerun()

# STEP 2: Building Envelope (CSW Type Selection REMOVED)
elif st.session_state.step == 2:
    building_type = st.session_state.get('building_type', 'Office')
    st.header('Step 2: Building Envelope Information')
    col1, col2 = st.columns(2)
    
    with col1:
        if building_type == 'Hotel':
            min_area, max_area = 15000, 250000
            area_help = "Hotel building area must be between 15,000 and 250,000 square feet"
        elif building_type == 'School':
            min_area, max_area = 25000, 350000
            area_help = "School building area must be between 25,000 and 350,000 square feet"
        else:
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
    
    with col2:
        window_types_list = WINDOW_TYPES
        existing_window_idx = 0
        if 'existing_window' in st.session_state and st.session_state.existing_window in window_types_list:
            existing_window_idx = window_types_list.index(st.session_state.existing_window)
        existing_window = st.selectbox('Type of Existing Window', options=window_types_list, index=existing_window_idx, key='existing_window_select')
        st.session_state.existing_window = existing_window
        
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
    
    # Info box about product comparison
    st.info("ℹ️ The calculator will compare savings for both Winsert Lite and Winsert Plus products" + 
            (" (where applicable)" if existing_window == 'Double pane' else "") + ".")
    
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

# STEP 3: HVAC & Operations
elif st.session_state.step == 3:
    building_type = st.session_state.get('building_type', 'Office')
    st.header('Step 3: HVAC & Operations')
    col1, col2 = st.columns(2)
    
    with col1:
        electric_rate = st.number_input('Electric Rate ($/kWh)', min_value=0.01, max_value=1.0, value=st.session_state.get('electric_rate', 0.12), step=0.01, format='%.3f', key='electric_rate_input')
        st.session_state.electric_rate = electric_rate
        
        gas_rate = st.number_input('Natural Gas Rate ($/therm)', min_value=0.01, max_value=10.0, value=st.session_state.get('gas_rate', 0.80), step=0.05, format='%.2f', key='gas_rate_input')
        st.session_state.gas_rate = gas_rate
        
        if building_type == 'Office':
            operating_hours = st.number_input('Annual Operating Hours', min_value=1980, max_value=8760, value=st.session_state.get('operating_hours', 8000), step=100, key='operating_hours_input')
            st.session_state.operating_hours = operating_hours
        elif building_type == 'Hotel':
            occupancy_percent = st.slider('Average Occupancy (%)', min_value=33, max_value=100, value=st.session_state.get('occupancy_percent', 70), step=1, key='occupancy_input', help='Between 33% and 100%')
            st.session_state.occupancy_percent = occupancy_percent
    
    with col2:
        if building_type == 'Office':
            hvac_systems_list = OFFICE_HVAC_SYSTEMS
        elif building_type == 'Hotel':
            hvac_systems_list = HOTEL_HVAC_SYSTEMS
        else:
            hvac_systems_list = SCHOOL_HVAC_SYSTEMS
        
        hvac_idx = 0
        if 'hvac_system' in st.session_state and st.session_state.hvac_system in hvac_systems_list:
            hvac_idx = hvac_systems_list.index(st.session_state.hvac_system)
        hvac_system = st.selectbox('HVAC System Type', options=hvac_systems_list, index=hvac_idx, key='hvac_system_select')
        st.session_state.hvac_system = hvac_system
        
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

# STEP 4: Results (WITH ZERO FLOOR APPLIED TO ALL SAVINGS)
elif st.session_state.step == 4:
    building_type = st.session_state.get('building_type', 'Office')
    st.header('💡 Your Energy Savings Results')
    
    # Base inputs
    inputs_base = {
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
        'csw_area': st.session_state.get('csw_area', 12000),
        'electric_rate': st.session_state.get('electric_rate', 0.12),
        'gas_rate': st.session_state.get('gas_rate', 0.80)
    }
    
    if building_type == 'Office':
        inputs_base['operating_hours'] = st.session_state.get('operating_hours', 8000)
    elif building_type == 'Hotel':
        inputs_base['occupancy_percent'] = st.session_state.get('occupancy_percent', 70)
    else:
        inputs_base['school_type'] = st.session_state.get('school_type', 'Primary School')
    
    # Calculate for Winsert Lite
    inputs_lite = inputs_base.copy()
    inputs_lite['csw_type'] = 'Winsert Lite'
    
    if building_type == 'Office':
        results_lite = calculate_savings_office(inputs_lite)
    elif building_type == 'Hotel':
        results_lite = calculate_savings_hotel(inputs_lite)
    else:
        results_lite = calculate_savings_school(inputs_lite)
    
    # Calculate for Winsert Plus (only if existing window is single pane)
    existing_window = st.session_state.get('existing_window', 'Single pane')
    show_both_products = (existing_window == 'Single pane')
    
    results_plus = None
    if show_both_products:
        inputs_plus = inputs_base.copy()
        inputs_plus['csw_type'] = 'Winsert Plus'
        
        if building_type == 'Office':
            results_plus = calculate_savings_office(inputs_plus)
        elif building_type == 'Hotel':
            results_plus = calculate_savings_hotel(inputs_plus)
        else:
            results_plus = calculate_savings_school(inputs_plus)
    
    if results_lite and (not show_both_products or results_plus):
        # Apply zero floor to all savings for display purposes
        results_lite_display = apply_zero_floor_to_results(results_lite)
        results_plus_display = apply_zero_floor_to_results(results_plus) if results_plus else None
        
        st.success('✅ Calculation Complete!')
        
        # Main waterfall chart showing both products
        st.markdown('<h3 style="text-align: center;">Energy Use Intensity (EUI) Comparison</h3>', unsafe_allow_html=True)
        
        baseline_eui = results_lite_display['baseline_eui']
        savings_lite = results_lite_display['total_savings_kbtu_sf']
        eui_lite = results_lite_display['new_eui']
        
        if show_both_products:
            savings_plus = results_plus_display['total_savings_kbtu_sf']
            eui_plus = results_plus_display['new_eui']
            additional_savings = max(0, savings_plus - savings_lite)  # Also floor additional savings
            
            fig = go.Figure(go.Waterfall(
                orientation="v",
                measure=["absolute", "relative", "total", "relative", "total"],
                x=["Baseline EUI<br>Before Winsert", "Savings with<br>Winsert Lite", "EUI with<br>Winsert Lite", 
                   "Additional Savings<br>Lite → Plus", "Final EUI with<br>Winsert Plus"],
                y=[baseline_eui, -savings_lite, eui_lite, -additional_savings, eui_plus],
                text=[f"{baseline_eui:.1f}", f"−{savings_lite:.1f}", f"{eui_lite:.1f}", 
                      f"−{additional_savings:.1f}", f"{eui_plus:.1f}"],
                textposition=["inside", "outside", "inside", "outside", "inside"],
                textfont=dict(size=12, color="white"),
                increasing={"marker":{"color":"#D32F2F", "line":{"color":"#B71C1C", "width":2}}},
                decreasing={"marker":{"color":"#FF9800", "line":{"color":"#F57C00", "width":2}}},
                totals={"marker":{"color":"#4CAF50", "line":{"color":"#388E3C", "width":2}}},
                connector={"line":{"color":"rgb(100, 100, 100)", "width":1}},
                width=[0.5, 0.5, 0.5, 0.5, 0.5]
            ))
            
            fig.update_layout(
                height=400,
                showlegend=False,
                yaxis=dict(title='kBtu/SF-yr', title_font=dict(size=11), gridcolor='#E0E0E0', rangemode='tozero'),
                xaxis=dict(title_font=dict(size=10)),
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(t=30, b=100, l=60, r=20)
            )
        else:
            # Only show Lite
            fig = go.Figure(go.Waterfall(
                orientation="v",
                measure=["absolute", "relative", "total"],
                x=["Baseline EUI<br>Before Winsert", "Savings with<br>Winsert Lite", "EUI with<br>Winsert Lite"],
                y=[baseline_eui, -savings_lite, eui_lite],
                text=[f"{baseline_eui:.1f}", f"−{savings_lite:.1f}", f"{eui_lite:.1f}"],
                textposition=["inside", "outside", "inside"],
                textfont=dict(size=14, color="white"),
                increasing={"marker":{"color":"#D32F2F", "line":{"color":"#B71C1C", "width":2}}},
                decreasing={"marker":{"color":"#FF9800", "line":{"color":"#F57C00", "width":2}}},
                totals={"marker":{"color":"#4CAF50", "line":{"color":"#388E3C", "width":2}}},
                connector={"line":{"color":"rgb(100, 100, 100)", "width":1}},
                width=[0.5, 0.5, 0.5]
            ))
            
            fig.update_layout(
                height=400,
                showlegend=False,
                yaxis=dict(title='kBtu/SF-yr', title_font=dict(size=11), gridcolor='#E0E0E0', rangemode='tozero'),
                xaxis=dict(title_font=dict(size=11)),
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(t=30, b=80, l=60, r=20)
            )
            
            st.info("ℹ️ Calculator only evaluates Winsert Lite if existing windows are dual pane. Results shown are for Winsert Lite only.")
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('---')
        
        # Product Comparison Section (WITH RED/BLUE COLOR SCHEME AND BREAKDOWN BOXES)
        if show_both_products:
            st.markdown('<h3 style="text-align: center;">Product Comparison</h3>', unsafe_allow_html=True)
            
            col_lite, col_plus = st.columns(2)
            
            with col_lite:
                st.markdown('<h4 style="text-align: center; color: #C62828;">Winsert Lite</h4>', unsafe_allow_html=True)
                
                # EUI Reduction Box (Red)
                st.markdown(
                    f"""<div style='background: linear-gradient(135deg, #C62828 0%, #E53935 100%); 
                                padding: 20px; border-radius: 10px; text-align: center;
                                box-shadow: 0 3px 5px rgba(0,0,0,0.1); margin-bottom: 15px;'>
                        <h2 style='color: white; margin: 0 0 8px 0; font-size: 2em; font-weight: bold;'>
                            {results_lite_display['percent_eui_savings']:.1f}%
                        </h2>
                        <p style='color: white; margin: 0; font-size: 0.85em; opacity: 0.95;'>EUI Reduction</p>
                        <p style='color: white; margin: 5px 0 0 0; font-size: 1.1em;'>{results_lite_display['total_savings_kbtu_sf']:.1f} kBtu/SF-yr</p>
                    </div>""",
                    unsafe_allow_html=True
                )
                
                # Annual Cost Savings Box (Red)
                st.markdown(
                    f"""<div style='background: linear-gradient(135deg, #D32F2F 0%, #EF5350 100%); 
                                padding: 20px; border-radius: 8px; margin-bottom: 8px; text-align: center;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.08);'>
                        <p style='margin: 0 0 5px 0; color: white; font-size: 0.9em; font-weight: 600;'>Annual Cost Savings</p>
                        <p style='font-size: 1.8em; margin: 0; font-weight: bold; color: white;'>
                            ${results_lite_display['total_cost_savings']:,.0f}
                        </p>
                    </div>""",
                    unsafe_allow_html=True
                )
                
                # Electric Cost Savings Sub-box (Light Red)
                st.markdown(
                    f"""<div style='background: linear-gradient(135deg, #FFCDD2 0%, #EF9A9A 100%); 
                                padding: 12px; border-radius: 6px; margin-bottom: 6px; text-align: center;
                                box-shadow: 0 1px 3px rgba(0,0,0,0.06);'>
                        <p style='margin: 0 0 3px 0; color: #B71C1C; font-size: 0.8em; font-weight: 600;'>Electric Cost Savings</p>
                        <p style='font-size: 1.3em; margin: 0; font-weight: bold; color: #B71C1C;'>
                            ${results_lite_display['electric_cost_savings']:,.0f}/yr
                        </p>
                    </div>""",
                    unsafe_allow_html=True
                )
                
                # Gas Cost Savings Sub-box (Light Red)
                st.markdown(
                    f"""<div style='background: linear-gradient(135deg, #FFCDD2 0%, #EF9A9A 100%); 
                                padding: 12px; border-radius: 6px; margin-bottom: 12px; text-align: center;
                                box-shadow: 0 1px 3px rgba(0,0,0,0.06);'>
                        <p style='margin: 0 0 3px 0; color: #B71C1C; font-size: 0.8em; font-weight: 600;'>Natural Gas Cost Savings</p>
                        <p style='font-size: 1.3em; margin: 0; font-weight: bold; color: #B71C1C;'>
                            ${results_lite_display['gas_cost_savings']:,.0f}/yr
                        </p>
                    </div>""",
                    unsafe_allow_html=True
                )
                
                # Electric Energy Savings (Light Red)
                st.markdown(
                    f"""<div style='background: linear-gradient(135deg, #FFCDD2 0%, #EF9A9A 100%); 
                                padding: 12px; border-radius: 6px; margin-bottom: 6px; text-align: center;
                                box-shadow: 0 1px 3px rgba(0,0,0,0.06);'>
                        <p style='margin: 0 0 3px 0; color: #B71C1C; font-size: 0.8em; font-weight: 600;'>Electric Energy Savings</p>
                        <p style='font-size: 1.2em; margin: 0; font-weight: bold; color: #B71C1C;'>
                            {results_lite_display['electric_savings_kwh']:,.0f} kWh/yr
                        </p>
                    </div>""",
                    unsafe_allow_html=True
                )
                
                # Gas Savings (Light Red)
                st.markdown(
                    f"""<div style='background: linear-gradient(135deg, #FFCDD2 0%, #EF9A9A 100%); 
                                padding: 12px; border-radius: 6px; text-align: center;
                                box-shadow: 0 1px 3px rgba(0,0,0,0.06);'>
                        <p style='margin: 0 0 3px 0; color: #B71C1C; font-size: 0.8em; font-weight: 600;'>Natural Gas Savings</p>
                        <p style='font-size: 1.2em; margin: 0; font-weight: bold; color: #B71C1C;'>
                            {results_lite_display['gas_savings_therms']:,.0f} therms/yr
                        </p>
                    </div>""",
                    unsafe_allow_html=True
                )
            
            with col_plus:
                st.markdown('<h4 style="text-align: center; color: #1565C0;">Winsert Plus</h4>', unsafe_allow_html=True)
                
                # EUI Reduction Box (Blue)
                st.markdown(
                    f"""<div style='background: linear-gradient(135deg, #1565C0 0%, #1976D2 100%); 
                                padding: 20px; border-radius: 10px; text-align: center;
                                box-shadow: 0 3px 5px rgba(0,0,0,0.1); margin-bottom: 15px;'>
                        <h2 style='color: white; margin: 0 0 8px 0; font-size: 2em; font-weight: bold;'>
                            {results_plus_display['percent_eui_savings']:.1f}%
                        </h2>
                        <p style='color: white; margin: 0; font-size: 0.85em; opacity: 0.95;'>EUI Reduction</p>
                        <p style='color: white; margin: 5px 0 0 0; font-size: 1.1em;'>{results_plus_display['total_savings_kbtu_sf']:.1f} kBtu/SF-yr</p>
                    </div>""",
                    unsafe_allow_html=True
                )
                
                # Annual Cost Savings Box (Blue)
                st.markdown(
                    f"""<div style='background: linear-gradient(135deg, #1976D2 0%, #42A5F5 100%); 
                                padding: 20px; border-radius: 8px; margin-bottom: 8px; text-align: center;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.08);'>
                        <p style='margin: 0 0 5px 0; color: white; font-size: 0.9em; font-weight: 600;'>Annual Cost Savings</p>
                        <p style='font-size: 1.8em; margin: 0; font-weight: bold; color: white;'>
                            ${results_plus_display['total_cost_savings']:,.0f}
                        </p>
                    </div>""",
                    unsafe_allow_html=True
                )
                
                # Electric Cost Savings Sub-box (Light Blue)
                st.markdown(
                    f"""<div style='background: linear-gradient(135deg, #BBDEFB 0%, #90CAF9 100%); 
                                padding: 12px; border-radius: 6px; margin-bottom: 6px; text-align: center;
                                box-shadow: 0 1px 3px rgba(0,0,0,0.06);'>
                        <p style='margin: 0 0 3px 0; color: #0D47A1; font-size: 0.8em; font-weight: 600;'>Electric Cost Savings</p>
                        <p style='font-size: 1.3em; margin: 0; font-weight: bold; color: #0D47A1;'>
                            ${results_plus_display['electric_cost_savings']:,.0f}/yr
                        </p>
                    </div>""",
                    unsafe_allow_html=True
                )
                
                # Gas Cost Savings Sub-box (Light Blue)
                st.markdown(
                    f"""<div style='background: linear-gradient(135deg, #BBDEFB 0%, #90CAF9 100%); 
                                padding: 12px; border-radius: 6px; margin-bottom: 12px; text-align: center;
                                box-shadow: 0 1px 3px rgba(0,0,0,0.06);'>
                        <p style='margin: 0 0 3px 0; color: #0D47A1; font-size: 0.8em; font-weight: 600;'>Natural Gas Cost Savings</p>
                        <p style='font-size: 1.3em; margin: 0; font-weight: bold; color: #0D47A1;'>
                            ${results_plus_display['gas_cost_savings']:,.0f}/yr
                        </p>
                    </div>""",
                    unsafe_allow_html=True
                )
                
                # Electric Energy Savings (Light Blue)
                st.markdown(
                    f"""<div style='background: linear-gradient(135deg, #BBDEFB 0%, #90CAF9 100%); 
                                padding: 12px; border-radius: 6px; margin-bottom: 6px; text-align: center;
                                box-shadow: 0 1px 3px rgba(0,0,0,0.06);'>
                        <p style='margin: 0 0 3px 0; color: #0D47A1; font-size: 0.8em; font-weight: 600;'>Electric Energy Savings</p>
                        <p style='font-size: 1.2em; margin: 0; font-weight: bold; color: #0D47A1;'>
                            {results_plus_display['electric_savings_kwh']:,.0f} kWh/yr
                        </p>
                    </div>""",
                    unsafe_allow_html=True
                )
                
                # Gas Savings (Light Blue)
                st.markdown(
                    f"""<div style='background: linear-gradient(135deg, #BBDEFB 0%, #90CAF9 100%); 
                                padding: 12px; border-radius: 6px; margin-bottom: 10px; text-align: center;
                                box-shadow: 0 1px 3px rgba(0,0,0,0.06);'>
                        <p style='margin: 0 0 3px 0; color: #0D47A1; font-size: 0.8em; font-weight: 600;'>Natural Gas Savings</p>
                        <p style='font-size: 1.2em; margin: 0; font-weight: bold; color: #0D47A1;'>
                            {results_plus_display['gas_savings_therms']:,.0f} therms/yr
                        </p>
                    </div>""",
                    unsafe_allow_html=True
                )
                
                # Show additional savings
                additional_cost_savings = max(0, results_plus_display['total_cost_savings'] - results_lite_display['total_cost_savings'])
                additional_eui_savings = max(0, results_plus_display['total_savings_kbtu_sf'] - results_lite_display['total_savings_kbtu_sf'])
                st.markdown(
                    f"""<div style='background-color: #E3F2FD; padding: 12px; border-radius: 6px; border-left: 4px solid #1976D2;'>
                        <p style='margin: 0; font-size: 0.85em; color: #0D47A1; font-weight: 600;'>Additional Savings vs Lite:</p>
                        <p style='margin: 5px 0 0 0; font-size: 0.95em; color: #0D47A1;'>
                            +${additional_cost_savings:,.0f}/yr<br>
                            +{additional_eui_savings:.1f} kBtu/SF-yr
                        </p>
                    </div>""",
                    unsafe_allow_html=True
                )
        else:
            # Only Lite results (RED COLOR SCHEME WITH BREAKDOWN)
            st.markdown('<h3 style="text-align: center;">Winsert Lite Performance</h3>', unsafe_allow_html=True)
            
            col_eui, col_cost = st.columns([1, 1])
            
            with col_eui:
                st.markdown(
                    f"""<div style='background: linear-gradient(135deg, #C62828 0%, #E53935 100%); 
                                padding: 28px; border-radius: 10px; text-align: center;
                                box-shadow: 0 3px 5px rgba(0,0,0,0.1);'>
                        <h2 style='color: white; margin: 0 0 8px 0; font-size: 2.2em; font-weight: bold;'>
                            {results_lite_display['percent_eui_savings']:.1f}%
                        </h2>
                        <p style='color: white; margin: 0; font-size: 0.85em; opacity: 0.95;'>EUI Reduction</p>
                        <p style='color: white; margin: 5px 0 0 0; font-size: 1.1em;'>{results_lite_display['total_savings_kbtu_sf']:.1f} kBtu/SF-yr</p>
                    </div>""",
                    unsafe_allow_html=True
                )
            
            with col_cost:
                st.markdown(
                    f"""<div style='background: linear-gradient(135deg, #C62828 0%, #E53935 100%); 
                                padding: 28px; border-radius: 10px; text-align: center;
                                box-shadow: 0 3px 5px rgba(0,0,0,0.1);'>
                        <p style='color: white; margin: 0 0 5px 0; font-size: 0.9em; font-weight: 500;'>Total Annual Savings</p>
                        <h1 style='color: white; margin: 0; font-size: 2.5em; font-weight: bold;'>
                            ${results_lite_display['total_cost_savings']:,.0f}
                        </h1>
                    </div>""",
                    unsafe_allow_html=True
                )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Breakdown boxes (Red theme)
            col1, col2 = st.columns(2)
            with col1:
                # Electric Cost Savings
                st.markdown(
                    f"""<div style='background: linear-gradient(135deg, #FFCDD2 0%, #EF9A9A 100%); 
                                padding: 15px; border-radius: 8px; margin-bottom: 10px; text-align: center;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.06);'>
                        <p style='margin: 0; font-size: 0.9em; color: #B71C1C; font-weight: 600;'>Electric Cost Savings</p>
                        <p style='margin: 5px 0 0 0; font-size: 1.5em; font-weight: bold; color: #B71C1C;'>
                            ${results_lite_display['electric_cost_savings']:,.0f}<span style='font-size: 0.6em;'>/yr</span>
                        </p>
                    </div>""",
                    unsafe_allow_html=True
                )
                # Electric Energy Savings
                st.markdown(
                    f"""<div style='background: linear-gradient(135deg, #FFCDD2 0%, #EF9A9A 100%); 
                                padding: 15px; border-radius: 8px; text-align: center;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.06);'>
                        <p style='margin: 0; font-size: 0.9em; color: #B71C1C; font-weight: 600;'>Electric Energy Savings</p>
                        <p style='margin: 5px 0 0 0; font-size: 1.5em; font-weight: bold; color: #B71C1C;'>
                            {results_lite_display["electric_savings_kwh"]:,.0f} <span style='font-size: 0.6em;'>kWh/yr</span>
                        </p>
                    </div>""",
                    unsafe_allow_html=True
                )
            with col2:
                # Gas Cost Savings
                st.markdown(
                    f"""<div style='background: linear-gradient(135deg, #FFCDD2 0%, #EF9A9A 100%); 
                                padding: 15px; border-radius: 8px; margin-bottom: 10px; text-align: center;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.06);'>
                        <p style='margin: 0; font-size: 0.9em; color: #B71C1C; font-weight: 600;'>Natural Gas Cost Savings</p>
                        <p style='margin: 5px 0 0 0; font-size: 1.5em; font-weight: bold; color: #B71C1C;'>
                            ${results_lite_display['gas_cost_savings']:,.0f}<span style='font-size: 0.6em;'>/yr</span>
                        </p>
                    </div>""",
                    unsafe_allow_html=True
                )
                # Gas Savings
                st.markdown(
                    f"""<div style='background: linear-gradient(135deg, #FFCDD2 0%, #EF9A9A 100%); 
                                padding: 15px; border-radius: 8px; text-align: center;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.06);'>
                        <p style='margin: 0; font-size: 0.9em; color: #B71C1C; font-weight: 600;'>Natural Gas Savings</p>
                        <p style='margin: 5px 0 0 0; font-size: 1.5em; font-weight: bold; color: #B71C1C;'>
                            {results_lite_display["gas_savings_therms"]:,.0f} <span style='font-size: 0.6em;'>therms/yr</span>
                        </p>
                    </div>""",
                    unsafe_allow_html=True
                )
        
        st.markdown('---')
        
        # Detailed calculations expander
        with st.expander('🔍 View Detailed Calculations'):
            if show_both_products:
                st.markdown('### Winsert Lite')
                detail_col1, detail_col2 = st.columns(2)
                with detail_col1:
                    st.write(f"• Baseline EUI: {results_lite_display['baseline_eui']:.1f} kBtu/SF-yr")
                    st.write(f"• New EUI: {results_lite_display['new_eui']:.1f} kBtu/SF-yr")
                    st.write(f"• EUI Reduction: {results_lite_display['percent_eui_savings']:.1f}%")
                with detail_col2:
                    st.write(f"• Electric Heating: {results_lite_display['heating_per_sf']:.4f} kWh/SF-CSW")
                    st.write(f"• Cooling & Fans: {results_lite_display['cooling_per_sf']:.4f} kWh/SF-CSW")
                    st.write(f"• Gas Heating: {results_lite_display['gas_per_sf']:.4f} therms/SF-CSW")
                
                st.markdown('### Winsert Plus')
                detail_col3, detail_col4 = st.columns(2)
                with detail_col3:
                    st.write(f"• Baseline EUI: {results_plus_display['baseline_eui']:.1f} kBtu/SF-yr")
                    st.write(f"• New EUI: {results_plus_display['new_eui']:.1f} kBtu/SF-yr")
                    st.write(f"• EUI Reduction: {results_plus_display['percent_eui_savings']:.1f}%")
                with detail_col4:
                    st.write(f"• Electric Heating: {results_plus_display['heating_per_sf']:.4f} kWh/SF-CSW")
                    st.write(f"• Cooling & Fans: {results_plus_display['cooling_per_sf']:.4f} kWh/SF-CSW")
                    st.write(f"• Gas Heating: {results_plus_display['gas_per_sf']:.4f} therms/SF-CSW")
            else:
                detail_col1, detail_col2 = st.columns(2)
                with detail_col1:
                    st.write(f"• Baseline EUI: {results_lite_display['baseline_eui']:.1f} kBtu/SF-yr")
                    st.write(f"• New EUI: {results_lite_display['new_eui']:.1f} kBtu/SF-yr")
                    st.write(f"• EUI Reduction: {results_lite_display['percent_eui_savings']:.1f}%")
                with detail_col2:
                    st.write(f"• Electric Heating: {results_lite_display['heating_per_sf']:.4f} kWh/SF-CSW")
                    st.write(f"• Cooling & Fans: {results_lite_display['cooling_per_sf']:.4f} kWh/SF-CSW")
                    st.write(f"• Gas Heating: {results_lite_display['gas_per_sf']:.4f} therms/SF-CSW")
            
            st.markdown('### Project Details')
            detail_col5, detail_col6 = st.columns(2)
            with detail_col5:
                st.write(f"• Location: {inputs_base['city']}, {inputs_base['state']}")
                st.write(f"• Heating Degree Days: {results_lite_display['hdd']:,.0f}")
                st.write(f"• Cooling Degree Days: {results_lite_display['cdd']:,.0f}")
            with detail_col6:
                st.write(f"• Building Type: {building_type}")
                if building_type == 'School':
                    st.write(f"• School Type: {inputs_base['school_type']}")
                st.write(f"• Building Area: {inputs_base['building_area']:,} SF")
                st.write(f"• Secondary Window Area: {inputs_base['csw_area']:,} SF")
                if results_lite_display['wwr']:
                    st.write(f"• Window-to-Wall Ratio: {results_lite_display['wwr']:.0%}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Action buttons
    col_restart, col_download = st.columns([1, 1])
    with col_restart:
        if st.button('← Start Over', type='secondary', use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.step = 0
            st.rerun()
    
    with col_download:
        try:
            if show_both_products:
                pdf_buffer = generate_pdf_report(inputs_base, results_lite_display, results_plus_display, building_type)
                filename_suffix = "Comparison"
            else:
                pdf_buffer = generate_pdf_report(inputs_base, results_lite_display, results_lite_display, building_type)
                filename_suffix = "Lite_Only"
            
            st.download_button(
                label='📄 Download PDF Report',
                data=pdf_buffer,
                file_name=f'Winsert_Savings_Report_{building_type}_{inputs_base["city"].replace(" ", "_")}_{filename_suffix}.pdf',
                mime='application/pdf',
                type='primary',
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error generating PDF: {e}")
            if st.button('← Start Over', type='secondary', use_container_width=True, key='backup_restart'):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.session_state.step = 0
                st.rerun()

# Sidebar (UPDATED TO REMOVE CSW TYPE)
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
        
        if building_type == 'Hotel':
            min_area, max_area = 15000, 250000
        elif building_type == 'School':
            min_area, max_area = 25000, 350000
        else:
            min_area, max_area = 15000, 500000
        
        building_area = st.number_input(
            'Building Area (SF)', 
            min_value=min_area, 
            max_value=max_area, 
            value=min(max(st.session_state.get('building_area', 75000), min_area), max_area), 
            step=1000, 
            key='sidebar_building_area'
        )
        if building_area != st.session_state.get('building_area'):
            st.session_state.building_area = building_area
            st.rerun()
        
        num_floors = st.number_input(
            'Floors', 
            min_value=1, 
            max_value=100, 
            value=st.session_state.get('num_floors', 5), 
            key='sidebar_num_floors'
        )
        if num_floors != st.session_state.get('num_floors'):
            st.session_state.num_floors = num_floors
            st.rerun()
        
        existing_window = st.selectbox('Existing Window', options=WINDOW_TYPES, index=WINDOW_TYPES.index(st.session_state.get('existing_window', 'Single pane')), key='sidebar_existing_window')
        if existing_window != st.session_state.get('existing_window'):
            st.session_state.existing_window = existing_window
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
        elif building_type == 'Hotel':
            occupancy = st.slider('Occupancy %', min_value=33, max_value=100, value=st.session_state.get('occupancy_percent', 70), step=1, key='sidebar_occupancy')
            if occupancy != st.session_state.get('occupancy_percent'):
                st.session_state.occupancy_percent = occupancy
                st.rerun()
        elif building_type == 'School':
            st.text(f"School Type: {st.session_state.get('school_type', 'N/A')}")
        
        if building_type == 'Office':
            hvac_systems = OFFICE_HVAC_SYSTEMS
        elif building_type == 'Hotel':
            hvac_systems = HOTEL_HVAC_SYSTEMS
        else:
            hvac_systems = SCHOOL_HVAC_SYSTEMS
            
        hvac_system = st.selectbox('HVAC System', options=hvac_systems, index=hvac_systems.index(st.session_state.get('hvac_system', hvac_systems[0])), key='sidebar_hvac_system')
        if hvac_system != st.session_state.get('hvac_system'):
            st.session_state.hvac_system = hvac_system
            st.rerun()
        
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
            if building_type == 'School' and 'school_type' in st.session_state:
                st.markdown(f"**Building Type:** {building_type} - {st.session_state.get('school_type')}")
            else:
                st.markdown(f"**Building Type:** {building_type}")
        if st.session_state.step > 1:
            st.markdown(f"**Location:** {st.session_state.get('city', 'N/A')}, {st.session_state.get('state', 'N/A')}")
        if st.session_state.step > 2:
            st.markdown(f"**Building:** {st.session_state.get('building_area', 0):,} SF, {st.session_state.get('num_floors', 0)} floors")
            st.markdown(f"**Existing Windows:** {st.session_state.get('existing_window', 'N/A')}")
            st.markdown(f"**Secondary Window Area:** {st.session_state.get('csw_area', 0):,} SF")
        if st.session_state.step > 3:
            building_type = st.session_state.get('building_type', 'Office')
            st.markdown(f"**HVAC:** {st.session_state.get('hvac_system', 'N/A')}")
            st.markdown(f"**Heating:** {st.session_state.get('heating_fuel', 'N/A')}")
            if building_type == 'Office':
                st.markdown(f"**Operating Hours:** {st.session_state.get('operating_hours', 0):,}/yr")
            elif building_type == 'Hotel':
                st.markdown(f"**Occupancy:** {st.session_state.get('occupancy_percent', 0)}%")
            elif building_type == 'School':
                st.markdown(f"**School Type:** {st.session_state.get('school_type', 'N/A')}")
