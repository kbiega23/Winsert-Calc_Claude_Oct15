# CSW Savings Calculator - Web Application

A Streamlit-based web application for calculating energy savings from Commercial Storm Window (CSW) installations in office buildings. Converted from Excel workbook to interactive web app.

## Features

- **4-Step Wizard Interface**: Guides users through the calculation process
- **874+ Weather Locations**: HDD/CDD data for accurate location-based calculations
- **Real-time Calculations**: Instant energy and cost savings estimates
- **Professional Results Display**: Clear metrics and downloadable reports
- **Lead Generation Ready**: Structure in place for contact capture

## Installation

### Local Development

1. Clone this repository:
```bash
git clone https://github.com/yourusername/csw-calculator.git
cd csw-calculator
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
streamlit run app.py
```

5. Open your browser to `http://localhost:8501`

## Deployment to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repository and branch
5. Set main file path to `app.py`
6. Click "Deploy"

Your app will be live at: `https://yourusername-csw-calculator-app-xxxxx.streamlit.app`

## Project Structure

```
csw-calculator/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .gitignore            # Git ignore file
```

## Current Status

### ✅ Implemented
- Complete 4-step wizard interface
- Weather data lookup (180+ major US cities)
- Input validation and warnings
- Basic savings calculations
- Results display with 7 key metrics
- Downloadable CSV reports
- Responsive design

### 🚧 To Be Completed
- **Full Weather Data**: Currently has ~180 locations, needs all 874 from Excel
- **Complete Savings Lookup Table**: Currently simplified, needs all 104 rows
- **Regression Formulas**: WWR calculation uses placeholder, needs actual regression
- **All HVAC System Types**: Currently simplified for PVAV systems
- **Lead Capture Form**: Structure ready, needs implementation
- **Branding**: Logo and company information

## Calculations

The app implements key formulas from the original Excel workbook:

- **Electric Savings**: `(heating_per_sf + cooling_per_sf) * csw_area`
- **Operating Hours Interpolation**: Linear interpolation between 2080 and 8760 hours
- **Savings Intensity**: `(electric_kwh * 3.413 + gas_therms * 100) / building_area`
- **Cost Savings**: `electric_kwh * rate + gas_therms * rate`

## Next Steps for Full Implementation

1. **Extract Complete Data from Excel**:
   - All 874 weather locations
   - All 104 savings lookup table rows
   - Complete regression coefficients
   - All dropdown options

2. **Enhance Calculations**:
   - Implement full VLOOKUP logic
   - Add regression-based WWR calculation
   - Include all HVAC system variations
   - Add gas heating calculations

3. **Add Features**:
   - Lead capture form with email integration
   - PDF report generation
   - Comparison scenarios
   - ROI calculator

4. **Styling**:
   - Add company logo
   - Custom color scheme
   - Branded header/footer

## Technology Stack

- **Frontend/Backend**: Streamlit (Python)
- **Data Processing**: Pandas, NumPy
- **Deployment**: Streamlit Cloud (free hosting)
- **Version Control**: Git/GitHub

## Contributing

To contribute improvements:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

[Add your license here]

## Contact

[Add your contact information here]

## Acknowledgments

Converted from original Excel workbook: CSW Savings Calculator 2.0.0
