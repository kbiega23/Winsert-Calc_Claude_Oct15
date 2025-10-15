"""
DEBUG VERSION - This will show us what's wrong with the School lookup
Replace your app.py with this temporarily
"""

import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="Winsert Savings Calculator - DEBUG",
    page_icon="🏢",
    layout="wide"
)

# Load the CSV and show School rows
st.header("DEBUG: CSV School Rows")

try:
    df = pd.read_csv('regression_coefficients.csv', keep_default_na=False, na_values=[''])
    
    st.write("### Total rows in CSV:", len(df))
    st.write("### Column names:", list(df.columns))
    
    # Show school rows
    st.write("### School rows (with school_type column):")
    
    if 'school_type' in df.columns:
        school_rows = df[df['school_type'].notna() & (df['school_type'] != '')]
        st.write(f"Found {len(school_rows)} school rows")
        st.dataframe(school_rows[['row', 'base', 'csw', 'size', 'hvac_fuel', 'fuel', 'occupancy', 'hours', 'school_type', 'heat_a', 'cool_a']])
        
        # Show unique values
        st.write("### Unique values in school rows:")
        st.write("school_type values:", school_rows['school_type'].unique())
        st.write("base values:", school_rows['base'].unique())
        st.write("csw values:", school_rows['csw'].unique())
        st.write("hvac_fuel values:", school_rows['hvac_fuel'].unique())
        st.write("fuel values:", school_rows['fuel'].unique())
        st.write("size values:", school_rows['size'].unique())
        st.write("occupancy values:", school_rows['occupancy'].unique())
        st.write("hours values:", school_rows['hours'].unique())
    else:
        st.error("school_type column NOT FOUND in CSV!")
        st.write("Available columns:", list(df.columns))
        
except Exception as e:
    st.error(f"Error loading CSV: {e}")

st.write("---")
st.write("### Instructions:")
st.write("1. Look at the output above")
st.write("2. Check if 'school_type' column exists")
st.write("3. Check if there are school rows (should show 40 rows)")
st.write("4. Copy the output and send it back to me")
st.write("5. Then replace this debug file with the regular app.py")
