import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from copy import deepcopy
import joblib
from tensorflow.keras.models import load_model
from copy import deepcopy

st.set_page_config(
    page_title = "Home",
    page_icon = "🏠",
    layout="wide"
)

st.write("AI phân loại bột dược liệu đinh lăng")

df_list = []
df_names = []
col1, col2, col3 = st.columns([3,4,4])

@st.cache_resource
def get_classification_model():
    preprocessing_pipeline = joblib.load("preprocessing_pipeline.pkl")
    classification_model = load_model("models/250225.h5")
    return preprocessing_pipeline, classification_model

preprocessing_pipeline, classification_model = get_classification_model()
        
def plot_analysis_result(X, name_lst, col2, col3):
    def make_large_font(styler):
        styler.set_table_styles([
            {'selector': 'th', 'props': [('font-size', '20px')]},
            {'selector': 'td', 'props': [('font-size', '20px')]}
        ])
        return styler
    preprocesed_X = preprocessing_pipeline.transform(X)
    y_pred = classification_model.predict(preprocesed_X)
    x = ["Root", "Stem-branch", "Mixed"]
    result_table = pd.DataFrame(deepcopy(y_pred), columns=x)
    max_indices = result_table.idxmax(axis=1)
    
    max_indices.index = name_lst
    max_indices = pd.DataFrame(max_indices, columns=["Classification result"])
    max_indices.index.name = "File name"
    with col3:
        # X-axis: column indices

        # Create bar chart
        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        # Add each row as a separate bar series
        for i in range(y_pred.shape[1]):
            fig.add_trace(go.Bar(x=x, y=y_pred[i, :], name=name_lst[i], marker_color=colors[i % len(colors)]))
            # fig.add_trace(go.Bar(x=x, y=row_1, name='Row 1', marker_color='red'))
            # fig.add_trace(go.Bar(x=x, y=row_2, name='Row 2', marker_color='green'))

        # Update layout
        fig.update_layout(
            title="Classification result",
            xaxis_title="Column Index",
            yaxis_title="Value",
            barmode='group',  # Group bars side by side
            template="plotly_white",
            hovermode="x unified",
            margin=dict(t=28, l = 20),
            width=900,  # Adjust figure width
            height=350,  # Adjust figure height
        )

        # Show the plot
        st.plotly_chart(fig, use_container_width=False)

        max_indices = max_indices.style.pipe(make_large_font)
        st.dataframe(max_indices, use_container_width=True)
    # Read preprocessing_pipeline from file


def plot_spectrals_data(df_list, col_th):
    if df_list:
        fig = go.Figure()

        # Iterate through each uploaded file
        for df in df_list:
            # Check if the file has at least two columns (Wavelength + Intensity)
            if df.shape[1] < 2:
                st.error(f"File {uploaded_file.name} must have at least two columns (Wavelength & Intensity).")
                continue

            # Assume first column is "Wavelength", others are different intensity spectra
            wavelength_col = df.columns[0]
            intensity_cols = df.columns[1:]  # All other columns are spectra

            # Add traces for each spectrum
            for col in intensity_cols:
                fig.add_trace(go.Scatter(
                    x=df[wavelength_col], 
                    y=df[col], 
                    mode='lines',
                    name=f"{uploaded_file.name} - {col}"  # Unique legend entry
                ))
        with col_th:
            # Customize layout
            fig.update_layout(
                title="Multi-File Spectral Data Plot",
                xaxis_title="Wavelength (cm-1)",
                yaxis_title="Intensity",
                template="plotly_white",
                hovermode="x unified",
                legend_title="Files & Spectra",
                width=900,  # Adjust figure width
                height=500,  # Adjust figure height
                margin=dict(t=28),
                legend=dict(
                    x=0.02, y=0.02,  # Move legend inside top-left
                    bgcolor="rgba(255,255,255,0.5)"  # Semi-transparent background
                )
            )

            # Display the plot
            st.plotly_chart(fig, use_container_width=False)

            

with col1:
    uploaded_files = st.file_uploader(
        "Choose a CSV file", accept_multiple_files=True, type=["csv"]
    )

    if uploaded_files:
        # Iterate through each uploaded file
        for uploaded_file in uploaded_files:
            # Read the CSV file
            df_names.append(uploaded_file.name.split(".csv")[0].split(".CSV")[0])
            df = pd.read_csv(uploaded_file, header = None)
            df_list.append(df)
        plot_spectrals_data(deepcopy(df_list), col2)

    if st.button("Classify", use_container_width=True):
        X = pd.concat(df_list, axis = 1).iloc[:, 1::2].T.values
        plot_analysis_result(X, df_names, col2, col3)