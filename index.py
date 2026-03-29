import streamlit as st
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="DiaCheck - Fuzzy Logic",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """Create and return the fuzzy control system for diabetes prediction"""
    pregnancies = ctrl.Antecedent(np.arange(0, 21, 1), 'pregnancies')
    glucose = ctrl.Antecedent(np.arange(0, 201, 1), 'glucose')
    blood_pressure = ctrl.Antecedent(np.arange(0, 121, 1), 'blood_pressure')
    skin_thickness = ctrl.Antecedent(np.arange(0, 61, 1), 'skin_thickness')
    insulin = ctrl.Antecedent(np.arange(0, 301, 1), 'insulin')
    bmi = ctrl.Antecedent(np.arange(0, 61, 0.1), 'bmi')
    pedigree = ctrl.Antecedent(np.arange(0, 3.1, 0.1), 'pedigree')
    age = ctrl.Antecedent(np.arange(20, 81, 1), 'age')
    outcome = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'outcome')

    pregnancies['low'] = fuzz.trapmf(pregnancies.universe, [0, 0, 2, 4])
    pregnancies['medium'] = fuzz.trimf(pregnancies.universe, [2, 5, 8])
    pregnancies['high'] = fuzz.trapmf(pregnancies.universe, [6, 10, 20, 20])

    glucose['low'] = fuzz.trapmf(glucose.universe, [0, 0, 50, 70])
    glucose['normal'] = fuzz.trimf(glucose.universe, [50, 85, 100])
    glucose['prediabetes'] = fuzz.trimf(glucose.universe, [100, 112.5, 125])
    glucose['high'] = fuzz.trapmf(glucose.universe, [125, 150, 200, 200])

    blood_pressure['low'] = fuzz.trapmf(blood_pressure.universe, [0, 0, 50, 60])
    blood_pressure['normal'] = fuzz.trimf(blood_pressure.universe, [50, 70, 90])
    blood_pressure['high'] = fuzz.trapmf(blood_pressure.universe, [80, 100, 120, 120])

    skin_thickness['thin'] = fuzz.trapmf(skin_thickness.universe, [0, 0, 15, 20])
    skin_thickness['normal'] = fuzz.trimf(skin_thickness.universe, [15, 25, 35])
    skin_thickness['thick'] = fuzz.trapmf(skin_thickness.universe, [30, 40, 60, 60])

    insulin['low'] = fuzz.trapmf(insulin.universe, [0, 0, 30, 50])
    insulin['normal'] = fuzz.trimf(insulin.universe, [40, 70, 100])
    insulin['high'] = fuzz.trapmf(insulin.universe, [90, 150, 300, 300])

    bmi['underweight'] = fuzz.trapmf(bmi.universe, [0, 0, 15, 18.5])
    bmi['normal'] = fuzz.trimf(bmi.universe, [18.5, 21.7, 24.9])
    bmi['overweight'] = fuzz.trimf(bmi.universe, [24.9, 27.45, 29.9])
    bmi['obese'] = fuzz.trapmf(bmi.universe, [29.9, 35, 60, 60])

    pedigree['low'] = fuzz.trapmf(pedigree.universe, [0, 0, 0.2, 0.4])
    pedigree['medium'] = fuzz.trimf(pedigree.universe, [0.3, 0.6, 0.9])
    pedigree['high'] = fuzz.trapmf(pedigree.universe, [0.8, 1.2, 3, 3])

    age['young'] = fuzz.trapmf(age.universe, [20, 20, 25, 30])
    age['middle_aged'] = fuzz.trimf(age.universe, [25, 40, 55])
    age['old'] = fuzz.trapmf(age.universe, [50, 60, 80, 80])

    outcome['no_diabetes'] = fuzz.trimf(outcome.universe, [0, 0, 0.5])
    outcome['diabetes'] = fuzz.trimf(outcome.universe, [0.5, 1, 1])

    rule1 = ctrl.Rule(glucose['high'], outcome['diabetes'])
    rule2 = ctrl.Rule(glucose['prediabetes'] & bmi['obese'], outcome['diabetes'])
    rule3 = ctrl.Rule(age['old'] & pedigree['high'], outcome['diabetes'])
    rule4 = ctrl.Rule(pregnancies['high'] & glucose['high'], outcome['diabetes'])
    rule5 = ctrl.Rule(insulin['high'] & glucose['high'], outcome['diabetes'])
    rule6 = ctrl.Rule(blood_pressure['high'] & bmi['obese'], outcome['diabetes'])
    rule7 = ctrl.Rule(skin_thickness['thick'] & bmi['obese'], outcome['diabetes'])
    rule8 = ctrl.Rule(glucose['low'], outcome['no_diabetes'])
    rule9 = ctrl.Rule(bmi['normal'] & age['young'], outcome['no_diabetes'])
    rule10 = ctrl.Rule(pregnancies['low'] & glucose['normal'], outcome['no_diabetes'])
    rule11 = ctrl.Rule(pedigree['low'], outcome['no_diabetes'])

    control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11])
    simulation = ctrl.ControlSystemSimulation(control_system)
    
    return simulation

def predict_diabetes(inputs, simulation):
    """Make a prediction using the fuzzy logic model"""
    for key, value in inputs.items():
        simulation.input[key] = value
    
    try:
        simulation.compute()
        predicted_value = simulation.output['outcome']
        predicted_class = 1 if predicted_value > 0.5 else 0
        confidence = predicted_value if predicted_class == 1 else (1 - predicted_value)
        return predicted_class, predicted_value, confidence
    except KeyError:
        return 0, 0.0, 1.0

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the diabetes dataset"""
    data = pd.read_csv("diabetes.csv")
    
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        for outcome in [0, 1]:
            median_val = data[(data[col] != 0) & (data['Outcome'] == outcome)][col].median()
            data.loc[(data[col] == 0) & (data['Outcome'] == outcome), col] = median_val
    
    for col in data.columns[:-1]:
        lower = data[col].quantile(0.05)
        upper = data[col].quantile(0.95)
        data[col] = np.where(data[col] < lower, lower, data[col])
        data[col] = np.where(data[col] > upper, upper, data[col])
    
    return data

def main():
    css = """
    <style>
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stTabs [data-baseweb="tab-list"] { gap: 15px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; background-color: transparent; border-bottom: 3px solid transparent;
        padding-top: 10px; padding-bottom: 10px; font-size: 1.1rem; font-weight: 500; color: #aaaaaa !important;
    }
    div[data-testid="stSidebar"] { padding-top: 1rem; }
    [data-testid="stAppViewContainer"] { background: linear-gradient(-45deg, #0a0a0f, #151522, #000000, #0a0a0f) !important; background-size: 400% 400% !important; animation: gradientBG 15s ease infinite !important; }
    [data-testid="stSidebar"], [data-testid="stHeader"] { background: transparent !important; }
    
    /* Global Color Overrides for Black Theme */
    .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p, span, label { color: #f0f2f6 !important; }
    div[data-testid="stMetricValue"] { color: #8BC34A !important; font-size: 2.5rem; }
    .stTabs [aria-selected="true"] { color: #ffffff !important; border-bottom: 3px solid #8BC34A !important; font-weight: 700 !important;}
    div[class^="st-emotion-cache"] { border-color: #333333 !important; }
    </style>
    """
        
    st.markdown(css, unsafe_allow_html=True)
    
    simulation = load_model()
    
    # Header layout - Removed Toggle
    col_logo, col_title = st.columns([1, 10])
    with col_logo:
        st.markdown("<h1 style='text-align: center; font-size: 3.5rem;'>🩺</h1>", unsafe_allow_html=True)
    with col_title:
        st.title("DiaCheck")
        st.markdown(f"<p style='color: #cccccc !important; font-style: italic; font-size: 1.1rem;'>AI-Powered Interactive Diagnostic Platform</p>", unsafe_allow_html=True)
            
    st.divider()
    
    # ---------------- DYNAMIC SIDEBAR ---------------- #
    with st.sidebar:
        st.header("⚙️ Patient Vitals")
        st.markdown("Metrics updated in **real-time** across the dashboard.")
        
        pregnancies = st.slider("🤰 Pregnancies", 0, 20, 3)
        glucose = st.slider("🩸 Glucose (mg/dL)", 0, 200, 120, format="%d")
        blood_pressure = st.slider("❤️ Blood Press. (mmHg)", 0, 120, 70, format="%d")
        skin_thickness = st.slider("📏 Skin Fold (mm)", 0, 60, 20, format="%d")
        insulin = st.slider("💉 Insulin (μU/ml)", 0, 300, 80, format="%d")
        bmi = st.slider("⚖️ BMI (kg/m²)", 0.0, 60.0, 25.0, format="%.1f")
        pedigree = st.slider("🧬 Pedigree Func", 0.0, 3.0, 0.5, format="%.2f")
        age = st.slider("🎂 Age (years)", 20, 80, 35, format="%d")
        
    inputs = {
        'pregnancies': pregnancies, 'glucose': glucose, 'blood_pressure': blood_pressure,
        'skin_thickness': skin_thickness, 'insulin': insulin, 'bmi': bmi,
        'pedigree': pedigree, 'age': age
    }

    # ---------------- MAIN WINDOW TABS ---------------- #
    tab1, tab2, tab3 = st.tabs(["🔥 Live Risk Feed", "📊 Interactive Datasets", "🧠 Fuzzy Logic Engine"])
    
    with tab1:
        with st.spinner('Synchronizing Inference Engine...'):
            predicted_class, predicted_value, confidence = predict_diabetes(inputs, simulation)
            
            st.markdown("### 📋 Diagnostic Live Results")
            st.markdown(f"Evaluating data for a **{age}-year-old** patient with a BMI of **{bmi}**.")
            
            res_col1, res_col2 = st.columns([1, 1], gap="large")
            
            with res_col1:
                risk_percent = predicted_value * 100
                st.metric(label="Calculated Risk Indicator", value=f"{risk_percent:.1f}%", delta="Critical Risk" if predicted_class == 1 else "Normal Range", delta_color="inverse")
                
                st.write("")
                if predicted_class == 1:
                    st.error("⚠️ **Alert Triggered:** High probability of diabetes detected.")
                    with st.expander("Urgent Actionable Recommendations", expanded=True):
                        st.write("- **Consult** immediately with an endocrinologist.")
                        st.write("- **Monitor** fasting blood glucose levels closely.")
                        st.write("- **Evaluate** stringent lifestyle exercise paradigms.")
                else:
                    st.success("✅ **Screening Clear:** Low probability of diabetes detected.")
                    with st.expander("Preventative Recommendations", expanded=True):
                        st.write("- **Maintain** robust dietary and aerobic practices.")
                        st.write("- **Continue** standard clinical bi-annual evaluations.")
            
            with res_col2:
                # Black Background Plotly Chart
                gauge_text_color = '#ffffff'
                bar_color = "#8BC34A"
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk_percent,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Confidence Heat Index", 'font': {'size': 18, 'color': gauge_text_color}},
                    number = {'font': {'color': gauge_text_color, 'size': 35}, 'suffix': "%"},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': gauge_text_color},
                        'bar': {'color': bar_color, 'thickness': 0.3},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 25], 'color': "rgba(76, 175, 80, 0.4)"},
                            {'range': [25, 50], 'color': "rgba(139, 195, 74, 0.4)"},
                            {'range': [50, 75], 'color': "rgba(255, 152, 0, 0.4)"},
                            {'range': [75, 100], 'color': "rgba(244, 67, 54, 0.4)"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': risk_percent}
                    }))
                
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': gauge_text_color}, margin=dict(l=20, r=20, t=50, b=20), height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            st.info("💡 **Active Diagnostic Flags**")
            risk_factors = []
            
            if glucose > 125: risk_factors.append(("🩸 Elevated Glucose", "Glucose > 125 mg/dL severely increases diabetic probability."))
            if bmi > 30: risk_factors.append(("⚖️ Clinical Obesity", "BMI > 30 is acting as a highly aggressive risk contributor."))
            if age > 50: risk_factors.append(("🎂 Aging Factor", "Age > 50 naturally escalates predictive baseline risk levels."))
            if pedigree > 0.8: risk_factors.append(("🧬 Heavy Genetic History", "A high pedigree function signifies stark genetic linkages."))
            
            if risk_factors:
                for factor, desc in risk_factors:
                    st.write(f"- **{factor}**: {desc}")
            else:
                st.write("No major physiological red flags detected from variables.")
                
    with tab2:
        st.markdown("### 📊 Interactive Dataset Demographics")
        st.caption("Investigate overarching patterns using Plotly interactive matrices.")
        
        data = load_and_preprocess_data()
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Total Records", len(data))
        col_m2.metric("Diabetic Patients", data['Outcome'].sum())
        col_m3.metric("Diabetes Prevalence", f"{data['Outcome'].mean()*100:.1f}%")
        
        with st.expander("🔍 View Searchable Dataset Array"):
            st.dataframe(data, use_container_width=True)
            
        st.markdown("---")
        st.markdown("#### Dimensional Insights")
        viz_option = st.radio("Select Perspective", ["Interactive Feature Distributions", "Interactive Scatter Correlator"], horizontal=True)
        
        template = "plotly_dark"
        
        if viz_option == "Interactive Feature Distributions":
            selected_feature = st.selectbox("Inspect target variable:", data.columns[:-1])
            fig = px.histogram(data, x=selected_feature, color="Outcome", 
                               color_discrete_sequence=['#4CAF50', '#F44336'],
                               barmode="overlay", title=f"Risk Frequency by {selected_feature}")
            fig.update_layout(template=template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_option == "Interactive Scatter Correlator":
            col_x, col_y = st.columns(2)
            var_x = col_x.selectbox("Select X-Axis:", data.columns[:-1], index=1)
            var_y = col_y.selectbox("Select Y-Axis:", data.columns[:-1], index=5)
            
            fig = px.scatter(data, x=var_x, y=var_y, color="Outcome", 
                             color_discrete_sequence=['#4CAF50', '#F44336'],
                             marginal_x="violin", marginal_y="violin",
                             title=f"Scatter Mapping: {var_x} vs. {var_y}")
            fig.update_layout(template=template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
                
    with tab3:
        st.markdown("### 🌐 Behind the AI Diagnostics")
        st.write("""
        **Fuzzy logic** is a robust framework of many-valued logic dealing with reasoning that is approximate rather than fixed and exact. 
        Unlike traditional binary logic (True/False or 1/0), fuzzy variables harbor truth values that range fluently between 0 and 1.
        
        #### Core Capabilities:
        - Utilizes well-grounded medical intuition into hard-coded rules.
        - Captures the natural uncertainty inherent in abstract medical diagnoses.
        - Provides extremely **interpretable readouts** detailing *why* limits triggered.
        
        #### Dominant Factors Evaluated Continuously:
        - High glucose & Prediabetic states.
        - Severe Obesity (BMI triggers).
        - Direct generic genetic linkages (Pedigree markers > 0.8).
        """)

if __name__ == "__main__":
    main()