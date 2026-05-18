
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Employee Short-Term Absenteeism Risk DSS",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple clean CSS: dark background, white text
st.markdown("""
<style>
    .stApp { background-color: #1a1a2e; color: #f0f0f0; }
    section[data-testid="stSidebar"] { background-color: #16213e; }
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: white;
        margin-bottom: 8px;
        padding-bottom: 6px;
        border-bottom: 2px solid white;
    }
    div[data-testid="stMetric"] {
        background-color: #16213e;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        color: #cccccc !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Load model and data


@st.cache_resource
def load_model():
    return joblib.load('absenteeism_model.pkl')


@st.cache_data
def load_data():
    return pd.read_csv('employee_absenteeism_predictions.csv')


model_data = load_model()
model = model_data['model']
sel_features = model_data['selected_features']
df = load_data()

# Sidebar
st.sidebar.markdown("## Absenteeism Risk DSS")
st.sidebar.markdown("Decision Support System for HR Managers")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Employee Risk Table", "Predict Employee Risk"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Use this system to identify employees who may be at risk of repeated "
    "short-term absenteeism and to take early action before the behaviour escalates."
)

# Page 1: Employee Risk Table
if page == "Employee Risk Table":

    st.title("Employee Risk Table")
    st.markdown(
        "Browse all employees and their predicted absenteeism risk level. "
        "Use the filters below to focus on a specific group, then download "
        "the results as a report."
    )
    st.markdown("---")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        dept_filter = st.multiselect(
            "Filter by Department",
            options=sorted(df['Department'].unique()),
            default=sorted(df['Department'].unique())
        )
    with col2:
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            options=['High Risk', 'Low Risk'],
            default=['High Risk', 'Low Risk']
        )
    with col3:
        ot_filter = st.multiselect(
            "Filter by Overtime",
            options=['Yes', 'No'],
            default=['Yes', 'No']
        )

    col4, col5 = st.columns(2)
    with col4:
        prob_range = st.slider(
            "Filter by Risk Score Range",
            0.0, 1.0, (0.0, 1.0), 0.01
        )
    with col5:
        js_filter = st.slider("Filter by Job Satisfaction", 1, 5, (1, 5))

    # Apply filters
    filtered = df[
        (df['Department'].isin(dept_filter)) &
        (df['Risk_Label'].isin(risk_filter)) &
        (df['Overtime'].isin(ot_filter)) &
        (df['Risk_Probability'] >= prob_range[0]) &
        (df['Risk_Probability'] <= prob_range[1]) &
        (df['Job_Satisfaction'] >= js_filter[0]) &
        (df['Job_Satisfaction'] <= js_filter[1])
    ].copy()

    st.markdown(f"**Showing {len(filtered):,} of {len(df):,} employees**")
    st.markdown("---")

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employees Shown", f"{len(filtered):,}")
    c2.metric("High Risk", f"{int((filtered['Predicted_Risk']==1).sum()):,}")
    c3.metric("Low Risk", f"{int((filtered['Predicted_Risk']==0).sum()):,}")
    avg_r = filtered['Risk_Probability'].mean() * 100
    c4.metric("Avg Risk Score",  f"{avg_r:.1f}%")

    st.markdown("---")

    # Colour coding
    def colour_risk(val):
        if val == 'High Risk':
            return 'background-color: darkred; color: white; font-weight: bold'
        return 'background-color: darkgreen; color: white; font-weight: bold'

    def colour_prob(val):
        if val >= 80:
            return 'background-color: darkred; color: white'
        elif val >= 50:
            return 'background-color: darkorange; color: white'
        return 'background-color: darkgreen; color: white'

    def colour_js(val):
        if val <= 2:
            return 'background-color: darkred; color: white'
        elif val == 3:
            return 'background-color: darkorange; color: white'
        return 'background-color: darkgreen; color: white'

    # Build display table
    display_cols = [
        'Employee_ID', 'Age', 'Department', 'Job_Role',
        'Overtime', 'Job_Satisfaction', 'Work_Life_Balance',
        'Absenteeism', 'Monthly_Income', 'Risk_Label', 'Risk_Probability'
    ]
    display_df = filtered[display_cols].copy()

    # Risk score as percentage with 1 decimal place
    display_df['Risk_Probability'] = (
        display_df['Risk_Probability'] * 100).round(1)

    display_df.columns = [
        'ID', 'Age', 'Department', 'Job Role',
        'Overtime', 'Job Satisfaction', 'Work-Life Balance',
        'Absence Days', 'Monthly Income', 'Risk Level', 'Risk Score (%)'
    ]

    styled = (
        display_df.style
        .applymap(colour_risk, subset=['Risk Level'])
        .applymap(colour_prob, subset=['Risk Score (%)'])
        .applymap(colour_js,   subset=['Job Satisfaction'])
    )
    st.dataframe(styled, width='stretch', hide_index=True, height=480)

    # Download button
    csv = filtered[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Results as CSV",
        data=csv,
        file_name="absenteeism_risk_results.csv",
        mime="text/csv"
    )


# Page 2: Predict Employee Risk

elif page == "Predict Employee Risk":

    st.title("Predict Short-Term Absenteeism Risk for an Employee")
    st.markdown(
        "Fill in the employee's details below and click **Get Risk Prediction**. "
        "The system will predict their absenteeism risk level and show "
        "other employees in the dataset who have a similar risk score."
    )
    st.markdown("---")

    # Input form
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Work Situation**")
        overtime = st.selectbox(
            "Does the employee work overtime?", ["No", "Yes"])
        job_sat = st.slider(
            "Job Satisfaction (1 = Very Low, 5 = Very High)", 1, 5, 3)
        job_involve = st.slider(
            "How involved is the employee in their job? (1 = Low, 5 = High)", 1, 5, 3)
        work_life = st.slider(
            "Work-Life Balance (1 = Poor, 4 = Excellent)", 1, 4, 2)
        perf_rating = st.slider(
            "Performance Rating (1 = Low, 5 = High)", 1, 5, 3)
        work_env = st.slider(
            "Work Environment Satisfaction (1 = Low, 5 = High)", 1, 5, 3)

    with col2:
        st.markdown("**Pay and Personal Details**")
        monthly_inc = st.number_input(
            "Monthly Income (RM)", 1000, 25000, 8000, 500)
        hourly_rate = st.number_input(
            "Hourly Rate (Amount paid per hour)", 10, 200, 60, 5)
        avg_hours = st.slider("Average Hours Worked Per Week", 20, 70, 45)
        distance = st.slider("Distance From Home (km)", 1, 60, 15)
        num_companies = st.slider(
            "Number of Previous Companies Worked", 1, 15, 3)

    with col3:
        st.markdown("**Experience and Career**")
        project_count = st.slider(
            "Number of Projects Currently Handling", 1, 20, 5)
        training_hrs = st.slider("Training Hours Last Year", 0, 100, 30)
        years_company = st.slider("Years at This Company", 0, 30, 5)
        age = st.slider("Age", 20, 60, 35)
        years_promo = st.slider("Years Since Last Promotion", 0, 15, 2)

    st.markdown("---")
    predict_btn = st.button(
        "Get Risk Prediction", width='stretch', type="primary"
    )

    if predict_btn:
        # Compute engineered features from inputs
        work_pressure = avg_hours / (project_count + 1)
        experience_ratio = years_company / (age + 1)
        promotion_gap = years_promo / (years_company + 1)

        feat_map = {
            'Overtime_Yes':                  1 if overtime == "Yes" else 0,
            'Job_Satisfaction':              job_sat,
            'Project_Count':                 project_count,
            'Work_Pressure':                 work_pressure,
            'Job_Involvement':               job_involve,
            'Performance_Rating':            perf_rating,
            'Training_Hours_Last_Year':      training_hrs,
            'Hourly_Rate':                   hourly_rate,
            'Monthly_Income':                monthly_inc,
            'Promotion_Gap':                 promotion_gap,
            'Work_Environment_Satisfaction': work_env,
            'Experience_Ratio':              experience_ratio,
            'Average_Hours_Worked_Per_Week': avg_hours,
            'Distance_From_Home':            distance,
            'Number_of_Companies_Worked':    num_companies,
            'Work_Life_Balance':             work_life,
            'Marital_Status_Married':        0,
            'Marital_Status_Single':         0,
            'Attrition_Yes':                 0,
        }

        input_vals = [feat_map.get(f, 0) for f in sel_features]
        input_df = pd.DataFrame([input_vals], columns=sel_features)

        pred_class = model.predict(input_df)[0]
        pred_prob = model.predict_proba(input_df)[0][1]

        st.markdown("---")

        # Result display
        st.markdown(
            '<p class="section-header">Prediction Result</p>',
            unsafe_allow_html=True
        )

        col_r1, col_r2 = st.columns([1, 1])

        with col_r1:
            if pred_class == 1:
                st.error(
                    "**HIGH RISK — Action Recommended**\n\n"
                    "This employee's profile is associated with a high risk of "
                    "repeated short-term absenteeism. An HR check-in is recommended."
                )
            else:
                st.success(
                    "**LOW RISK — No Immediate Action Needed**\n\n"
                    "This employee's profile does not show strong indicators of "
                    "absenteeism risk. Continue routine monitoring."
                )

            # Risk score displayed as percentage with decimals
            risk_pct = pred_prob * 100
            st.markdown(f"### Predicted Risk Score: **{risk_pct:.2f}%**")
            st.markdown(
                "This score represents how likely this employee is to be "
                "at high risk of repeated short-term absenteeism, based on "
                "their HR profile. A score above 50% indicates high risk."
            )

        with col_r2:
            # Risk flags
            st.markdown("**Main factors contributing to this result:**")
            flags = []
            if overtime == "Yes":
                flags.append(
                    "Working overtime: a known driver of absenteeism risk")
            if job_sat <= 2:
                flags.append(
                    "Low job satisfaction: directly linked to higher absence")
            if work_life <= 2:
                flags.append(
                    "Poor work-life balance: associated with burnout and absence")
            if distance > 35:
                flags.append("Long commute: can increase fatigue and absence")
            if project_count > 7:
                flags.append(
                    "High Project Count: can be burnt out from high amount of project")
            if work_pressure > 10:
                flags.append(
                    "High workload: too many hours relative to project count")
            if promotion_gap > 0.5:
                flags.append(
                    "Limited career progression: may reduce motivation")
            if not flags:
                flags.append(
                    "No major risk factors identified from the inputs provided")
            for flag in flags:
                st.markdown(f"- {flag}")

            st.markdown("---")
            if pred_class == 1:
                st.info(
                    "**HR Recommendation:** Schedule a one-to-one meeting to discuss "
                    "the employee's workload, job satisfaction and wellbeing. Consider "
                    "flexible working or workload reduction if overtime or work pressure "
                    "is high."
                )
            else:
                st.info(
                    "**HR Recommendation:** No immediate action required. "
                    "Include in routine annual wellness check-in."
                )

        # Similar employees section
        st.markdown("---")
        st.markdown(
            '<p class="section-header">Employees with a Similar Risk Score</p>',
            unsafe_allow_html=True
        )

        # Tolerance: show all employees whose risk score is within 5% of the prediction
        tolerance = 0.05
        lower = max(0.0, pred_prob - tolerance)
        upper = min(1.0, pred_prob + tolerance)

        similar = df[
            (df['Risk_Probability'] >= lower) &
            (df['Risk_Probability'] <= upper)
        ].copy()
        similar = similar.sort_values('Risk_Probability', ascending=False)
        st.markdown(
            f"**{len(similar):,} employees found with a similar risk score.**")

        # Build similar employees table
        sim_cols = [
            'Employee_ID', 'Age', 'Department', 'Job_Role',
            'Overtime', 'Job_Satisfaction', 'Work_Life_Balance',
            'Absenteeism', 'Risk_Label', 'Risk_Probability'
        ]
        sim_display = similar[sim_cols].copy()

        # Risk score with decimals as percentage
        sim_display['Risk_Probability'] = (
            sim_display['Risk_Probability'] * 100
        ).round(2)

        sim_display.columns = [
            'ID', 'Age', 'Department', 'Job Role',
            'Overtime', 'Job Satisfaction', 'Work-Life Balance',
            'Absence Days', 'Risk Level', 'Risk Score (%)'
        ]

        def colour_risk_sim(val):
            if val == 'High Risk':
                return 'background-color: darkred; color: white; font-weight: bold'
            return 'background-color: darkgreen; color: white; font-weight: bold'

        def colour_score_sim(val):
            if val >= 80:
                return 'background-color: darkred; color: white'
            elif val >= 50:
                return 'background-color: darkorange; color: white'
            return 'background-color: darkgreen; color: white'

        styled_sim = (
            sim_display.style
            .applymap(colour_risk_sim,  subset=['Risk Level'])
            .applymap(colour_score_sim, subset=['Risk Score (%)'])
        )

        st.dataframe(styled_sim, width='stretch', hide_index=True, height=400)

        # Download similar employees
        csv_sim = similar[sim_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Similar Employees as CSV",
            data=csv_sim,
            file_name="similar_risk_employees.csv",
            mime="text/csv"
        )

        st.caption(
            "Employees are shown whose risk score falls within 5 percentage points "
            "of the predicted score. A smaller or larger range can be explored by "
            "adjusting the Risk Score filter on the Employee Risk Table page."
        )
