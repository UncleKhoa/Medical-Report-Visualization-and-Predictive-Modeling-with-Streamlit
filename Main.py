import streamlit as st
import pandas as pd
import os
import math
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
import joblib

# Set the current working directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Read and inject the CSS file content
# css_path = os.path.join(script_dir, "style.css")
# with open(css_path) as f:
#     st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# st.write(script_dir)
# Import the data
data_path = os.path.join(script_dir, "data", "ICU_full.xlsx")
data = pd.read_excel(data_path, engine="openpyxl")
data['PatientID'] = data['PatientID']

# Import the data_diag
data_path = os.path.join(script_dir, "data", "BME10.xlsx")
data_diag = pd.read_excel(data_path, engine="openpyxl", dtype={'PatientID': str})

# Import directory for model, label_encoder, scaler
model_path = os.path.join(script_dir, "model", "decision_tree_model.pkl")
scaler_path = os.path.join(script_dir, "model", "scaler.pkl")
label_encoder_path = os.path.join(script_dir, "model", "label_encoders.pkl")

# Application
st.markdown("""
    <html>
        <h1>Patient Diagnosis</h1>
        <p>Welcome to the patient diagnosis application.</p>
    </html>
""", unsafe_allow_html=True)

# ID Input
Id_Search = st.text_input("Patient ID", placeholder="Example: 0273...")

def remove_suffix(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    Rename columns by removing a specified suffix from the column names.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    suffix (str): The suffix to remove from the column names.

    Returns:
    pd.DataFrame: DataFrame with renamed columns.
    """
    new_columns = {}

    for col in df.columns:
        if col.endswith(suffix):
            new_col = col[:-len(suffix)]
            new_columns[col] = new_col
        else:
            new_columns[col] = col

    df.rename(columns=new_columns, inplace=True)

    return df

# data_diag = remove_suffix(data_diag, '_admission')
features_to_keep_005 = ['NutritionIntake', 'Cortisol', 'ProBNP', 'UrineOutput', 'K', 'OxygenTherapy', 'BMI', 'Pulse', 'eGFR', 'TSH', 'FT3', 'Cl', 'Na', 'FT4', 'CRP', 'Ca', 'Mg', 'Ure', 'PCT', 'Lactat', 'Antibiotics', 'Adrenalin', 'Noradrenalin', 'Creatinin', 'PaCO2', 'TroponinI']

# Function to show information based on the name
def show_Info(id):
    # matching_rows = data[data['PatientID'].str.contains(id, case=False, na=False)]
    matching_rows = data.loc[data['PatientID'].astype(str) == id]    

    matching_rows = matching_rows.fillna("_")

    if not matching_rows.empty:
        # Patient Background
        name_value = matching_rows['FullName'].iloc[0]
        age_value = matching_rows['Age'].iloc[0]
        gender_value = ""
        if (matching_rows['Male'] == 'x').any():
            gender_value = 'male'
        else:
            gender_value = 'female'
            
        bmi_value = matching_rows['Weight'].iloc[0].astype(float) / (math.pow((matching_rows['Height'].iloc[0].astype(float)/100),2))
        bmi_value = f"{bmi_value:.2f}"

        mc_value = ""
        if (matching_rows['MedicalConditions_InfectiousStatus'] == 'Có nhiễm trùng').any():        
            mc_value = matching_rows.loc[
                matching_rows['MedicalConditions_InfectiousStatus'] == 'Có nhiễm trùng',
                'MedicalConditions_InfectiousStatus'
            ].iloc[0] + " - " + matching_rows.loc[
                matching_rows['MedicalConditions_InfectiousStatus'] == 'Có nhiễm trùng',
                'MedicalConditions_InfectiousArea'
            ].iloc[0]
        else:
            mc_value = 'Không nhiễm trùng'    

        sc_value = matching_rows['SurgicalConditions'].iloc[0]
        if sc_value == "_":
            sc_value = '_'
        else:
            sc_value = matching_rows[
                'SurgicalConditions'
            ].iloc[0] + " - " + matching_rows[                
                'SurgicalConditions_OrgansInvolved'
            ].iloc[0]

        umc_values = [matching_rows[col].iloc[0] for col in ['UMC_1', 'UMC_2', 'UMC_3', 'UMC_4', 'UMC_5']]
        umc_values = [value.replace("_","") if "_" in value else value for value in umc_values]
        filtered_umc_values = [str(value).strip() for value in umc_values if pd.notna(value) and str(value).strip() != '']
        umc_value = ' - '.join(filtered_umc_values)

        # Vital Sign
        pulse = str(matching_rows["Pulse_admission"].iloc[0])
        respiratoryrate = str(matching_rows["RespiratoryRate_admission"].iloc[0])
        bodytemp = str(matching_rows["BodyTemperature_admission"].iloc[0])
        bloodpr = str(matching_rows["BloodPressure_admission"].iloc[0])
        spo = str(matching_rows["Spo2_admission"].iloc[0])
        glasgow = str(matching_rows["Glasgow_admission"].iloc[0])

        # Hematology
        wbc = str(matching_rows["WBC_admission"].iloc[0])
        hct = str(matching_rows["HCT_admission"].iloc[0] )
        plt = str(matching_rows["PLT_admission"].iloc[0])

        # ArterialBloodGas
        ph = str(matching_rows["pH_admission"].iloc[0])
        pao2 = str(matching_rows["PaO2_admission"].iloc[0])
        paco2 = str(matching_rows["PaCO2_admission"].iloc[0])
        hco3 = str(matching_rows["HCO3_admission"].iloc[0])

        # CardiacFunction
        troponin = str(matching_rows["TroponinI_admission"].iloc[0])
        ck_mb = str(matching_rows["CK-MB_admission"].iloc[0])
        probnp = str(matching_rows["ProBNP_admission"].iloc[0])

        # LiverFunction
        ast = str(matching_rows["AST_admission"].iloc[0])
        alt = str(matching_rows["ALT_admission"].iloc[0])
        ldh = str(matching_rows["LDH_admission"].iloc[0])
        albumin = str(matching_rows["Albumin_admission"].iloc[0])
        bilirubintp = str(matching_rows["BilirubinTP_admission"].iloc[0])
        bilirubintt = str(matching_rows["BilirubinTT_admission"].iloc[0])
        bilirubingt = str(matching_rows["BilirubinGT_admission"].iloc[0])

            
        # CoagulationFunction
        pt = str(matching_rows["PT_admission"].iloc[0])
        aptt = str(matching_rows["aPTT_admission"].iloc[0])
        ddimer = str(matching_rows["DDimer_admission"].iloc[0])

        # RenalFunction
        ure = str(matching_rows["Ure_admission"].iloc[0])
        creatinin = str(matching_rows["Creatinin_admission"].iloc[0])
        egfr = str(matching_rows["eGFR_admission"].iloc[0])

        # EndocrineFunction
        cortisol = str(matching_rows["Cortisol_admission"].iloc[0])
        tsh = str(matching_rows["TSH_admission"].iloc[0])
        ft3 = str(matching_rows["FT3_admission"].iloc[0])
        ft4 = str(matching_rows["FT4_admission"].iloc[0])

        # Electrolyte
        Na = str(matching_rows["Na_admission"].iloc[0])
        K = str(matching_rows["K_admission"].iloc[0])
        Cl = str(matching_rows["Cl_admission"].iloc[0])
        Ca = str(matching_rows["Ca_admission"].iloc[0])
        Mg = str(matching_rows["Mg_admission"].iloc[0])

        # Otherparameters
        crp = str(matching_rows["CRP_admission"].iloc[0])
        pct = str(matching_rows["PCT_admission"].iloc[0])
        lactat = str(matching_rows["Lactat_admission"].iloc[0])
        glucose = str(matching_rows["Glucose_admission"].iloc[0])

        # OxygenTherapy
        oxygenthe = str(matching_rows["OxygenTherapy_admission"].iloc[0])

        # Antibiotics
        antibiotics_check = matching_rows["Antibiotics"].iloc[0]
        antibiotics = ""
        if(antibiotics_check == "Có"):
            antibiotics = matching_rows["Antibiotics_admission"].iloc[0]
        else:
            antibiotics = "_"

        # VasopressorsandInotropes
        adrenalin = str(matching_rows["Adrenalin_admission"].iloc[0])
        noradrenalin = str(matching_rows["Noradrenalin_admission"].iloc[0])
        dobutamin = str(matching_rows["Dobutamin_admission"].iloc[0])
        dopamin = str(matching_rows["Dopamin_admission"].iloc[0])

        # Nutrition
        nutritionin = str(matching_rows["NutritionIntake_admission"].iloc[0])
        fluidin = str(matching_rows["FluidIntake_admission"].iloc[0])
        urineout = str(matching_rows["UrineOutput_admission"].iloc[0])

        # Patine Name
        # st.dataframe(matching_rows)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <html>
                <h4>Patient ID</h4>
            </html>
            """, unsafe_allow_html=True)
            st.write(id)

        with col3:
            st.markdown("""
            <html>
                <h4>Full Name</h4>
            </html>
            """, unsafe_allow_html=True)
            st.write(name_value)    

        # Patient Background
        st.markdown("""
            <html>
                <h2><span style="color: red;">*</span>Patient Background</h2>
            </html>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
                <html>
                    <h4>Age</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(str(age_value))

            st.markdown("""
                <html>
                    <h4>BMI</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(bmi_value)

        with col3:
            st.markdown("""
                <html>
                    <h4>Gender</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(gender_value)               

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
                <html>
                    <h4>Medical Conditions</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(mc_value)

        with col3:
            st.markdown("""
                <html>
                    <h4>Surgical Conditions</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(sc_value)   

        st.markdown("""
            <html>
                <h4>Underlying Medical Condition</h4>
            </html>
        """, unsafe_allow_html=True)
        st.write(umc_value)

        st.write("_____________________________")

        # Vitalsigns:
        st.markdown("""
            <html>
                <h2><span style="color: red;">*</span>Vital Signs</h2>
            </html>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            # st.dataframe(matching_rows)
            st.markdown("""
                <html>
                    <h4>Pulse</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(pulse)

            st.markdown("""
                <html>
                    <h4>Body Temperature</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(bodytemp) 

            st.markdown("""
                <html>
                    <h4>SpO2</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(spo)

        with col3:
            st.markdown("""
                <html>
                    <h4>Respiratory Rate</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(respiratoryrate)

            st.markdown("""
                <html>
                    <h4>Blood Pressure</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(bloodpr)

            st.markdown("""
                <html>
                    <h4>Glasgow</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(glasgow)

        st.write("_____________________________")

        # Hematology:
        st.markdown("""
            <html>
                <h2><span style="color: red;">*</span>Hematology</h2>
            </html>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            # st.dataframe(matching_rows)
            st.markdown("""
                <html>
                    <h4>WBC</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(wbc)

            st.markdown("""
                <html>
                    <h4>PLT</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(plt)

        with col3:
            st.markdown("""
                <html>
                    <h4>HCT</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(hct)
 
            st.markdown("""
                <html>
                    <h4>HCO3</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(hco3)

        st.write("_____________________________")

        # ArterialBloodGas:
        st.markdown("""
            <html>
                <h2><span style="color: red;">*</span>Arterial Blood Gas</h2>
            </html>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            # st.dataframe(matching_rows)
            st.markdown("""
                <html>
                    <h4>pH</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(ph)

            st.markdown("""
                <html>
                    <h4>PaCO2</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(paco2)

        with col3:
            st.markdown("""
                <html>
                    <h4>PaO2</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(pao2)

        # ArterialBloodGas:
        st.markdown("""
            <html>
                <h2><span style="color: red;">*</span>Cardiac Function</h2>
            </html>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            # st.dataframe(matching_rows)
            st.markdown("""
                <html>
                    <h4>TroponinI</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(troponin)

            st.markdown("""
                <html>
                    <h4>ProBNP</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(str(probnp))

        with col3:
            st.markdown("""
                <html>
                    <h4>CK-MB</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(ck_mb)    

        st.write("_____________________________")

        # LiverFunction:
        st.markdown("""
            <html>
                <h2><span style="color: red;">*</span>Liver Function</h2>
            </html>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            # st.dataframe(matching_rows)
            st.markdown("""
                <html>
                    <h4>AST</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(ast)

            st.markdown("""
                <html>
                    <h4>LDH</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(ldh)

            st.markdown("""
                <html>
                    <h4>BilirubinTP</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(bilirubintp)

            st.markdown("""
                <html>
                    <h4>BilirubinGT</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(bilirubingt)

        with col3:
            st.markdown("""
                <html>
                    <h4>ALT</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(alt)   

            st.markdown("""
                <html>
                    <h4>Albumin</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(albumin)

            st.markdown("""
                <html>
                    <h4>BilirubinTT</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(bilirubintt)

        st.write("_____________________________")

        # CoagulationFunction:
        st.markdown("""
            <html>
                <h2><span style="color: red;">*</span>Coagulation Function</h2>
            </html>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            # st.dataframe(matching_rows)
            st.markdown("""
                <html>
                    <h4>PT</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(pt)

            st.markdown("""
                <html>
                    <h4>DDimer</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(ddimer)

        with col3:
            st.markdown("""
                <html>
                    <h4>aPTT</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(aptt)   

        st.write("_____________________________")

        # RenalFunction:
        st.markdown("""
            <html>
                <h2><span style="color: red;">*</span>Renal Function</h2>
            </html>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            # st.dataframe(matching_rows)
            st.markdown("""
                <html>
                    <h4>Ure</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(ure)

            st.markdown("""
                <html>
                    <h4>eGFR</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(egfr)

        with col3:
            st.markdown("""
                <html>
                    <h4>Creatinin</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(creatinin)   

        st.write("_____________________________")

        # EndocrineFunction:
        st.markdown("""
            <html>
                <h2><span style="color: red;">*</span>Endocrine Function</h2>
            </html>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            # st.dataframe(matching_rows)
            st.markdown("""
                <html>
                    <h4>Cortisol</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(cortisol)

            st.markdown("""
                <html>
                    <h4>FT3</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(ft3)

        with col3:
            st.markdown("""
                <html>
                    <h4>TSH</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(tsh)   

            st.markdown("""
                <html>
                    <h4>FT4</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(ft4)   

        st.write("_____________________________")

        # Electrolyte:
        st.markdown("""
            <html>
                <h2><span style="color: red;">*</span>Electrolyte</h2>
            </html>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            # st.dataframe(matching_rows)
            st.markdown("""
                <html>
                    <h4>Na</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(Na)

            st.markdown("""
                <html>
                    <h4>Cl</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(Cl)

            st.markdown("""
                <html>
                    <h4>Mg</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(Mg)

        with col3:
            st.markdown("""
                <html>
                    <h4>K</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(K)   

            st.markdown("""
                <html>
                    <h4>Ca</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(Ca)   

        st.write("_____________________________")

        # Otherparameters:
        st.markdown("""
            <html>
                <h2><span style="color: red;">*</span>Other parameters</h2>
            </html>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            # st.dataframe(matching_rows)
            st.markdown("""
                <html>
                    <h4>CRP</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(crp)

            st.markdown("""
                <html>
                    <h4>Lactat</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(lactat)

        with col3:
            st.markdown("""
                <html>
                    <h4>PCT</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(pct)   

            st.markdown("""
                <html>
                    <h4>Glucose</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(glucose)   

        st.write("_____________________________")

        # OxygenTherapy & Antibiotics:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <html>
                <h2><span style="color: red;">*</span>OxygenTherapy</h2>
            </html>
            """, unsafe_allow_html=True)

            st.markdown("""
                <html>
                    <h4>OxygenTherapy</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(oxygenthe) 

        with col3:
            st.markdown("""
            <html>
                <h2><span style="color: red;">*</span>Antibiotics</h2>
            </html>
            """, unsafe_allow_html=True)

            st.markdown("""
                <html>
                    <h4>Antibiotics</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(antibiotics) 

        st.write("_____________________________")

        # VasopressorsandInotropes:
        st.markdown("""
            <html>
                <h2><span style="color: red;">*</span>Vasopressors and Inotropes</h2>
            </html>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            # st.dataframe(matching_rows)
            st.markdown("""
                <html>
                    <h4>Adrenalin</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(adrenalin)

            st.markdown("""
                <html>
                    <h4>Dobutamin</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(dobutamin)

        with col3:
            st.markdown("""
                <html>
                    <h4>Noradrenalin</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(noradrenalin)   

            st.markdown("""
                <html>
                    <h4>Dopamin</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(dopamin)    

        st.write("_____________________________")

        # Nutrition:
        st.markdown("""
            <html>
                <h2><span style="color: red;">*</span>Nutrition</h2>
            </html>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            # st.dataframe(matching_rows)
            st.markdown("""
                <html>
                    <h4>Nutrition Intake</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(nutritionin)

            st.markdown("""
                <html>
                    <h4>Urine Output</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(urineout)

        with col3:
            st.markdown("""
                <html>
                    <h4>Fluid Intake</h4>
                </html>
            """, unsafe_allow_html=True)
            st.write(fluidin)   

        # Layout with columns
        col1, col2, col3 = st.columns(3)

        with col3:
            if st.button("Diagnosis", key='diagnosis_button'):
                st.session_state.diagnosis_clicked = True

            if st.session_state.diagnosis_clicked:
                matching_rows_diag = data_diag.loc[data_diag['PatientID'].astype(str) == id]
                matching_rows_diag = remove_suffix(matching_rows_diag,'_admission')
                matching_rows_diag = matching_rows_diag[features_to_keep_005]
                diagnosis(matching_rows_diag)
        

    else:
        st.write(f"No matching patient found for id: {id}")       

def diagnosis(matching_rows):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoders = joblib.load(label_encoder_path)

    # numberical encoder
    numberical_features = matching_rows.select_dtypes(include=['int64', 'float64']).columns
    matching_rows[numberical_features] = scaler.transform(matching_rows[numberical_features])

    # label_encoder
    categorical_cols = matching_rows.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = label_encoders[col]
        matching_rows[col] = matching_rows[col].apply(lambda x: le.transform([x])[0])

    # st.write(matching_rows)

    # Make predictions
    predictions = model.predict(matching_rows)

    if predictions == 0:
        st.write("Survival")
    else:
        st.write("Death")

if 'search_clicked' not in st.session_state:
    st.session_state.search_clicked = False
if 'diagnosis_clicked' not in st.session_state:
    st.session_state.diagnosis_clicked = False

# Button to trigger the search
if st.button("Search", key='search_button'):
    if Id_Search:
        st.session_state.search_clicked = True
        st.session_state.diagnosis_clicked = False  # Reset diagnosis state when a new search is performed

    else:
        st.write("Please write in the patient id")

# Check if search button was clicked and ID was provided
if st.session_state.search_clicked:
    show_Info(Id_Search)
