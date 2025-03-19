import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
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
lanho_indices = None
col1, col2, col3 = st.columns([3,4,4])

result_analysis_colors_dict = {
    0: "#16C47F",
    1: "#FF9D23",
    2: "#F93827"
}

@st.cache_resource
def get_classification_model():
    preprocessing_pipeline = joblib.load("preprocessing_pipeline.pkl")
    lanho_preprocessing_pipeline = joblib.load("lanho_predict_preprocessing_pipeline.pkl")
    classification_model = load_model("models/250225.h5")
    lanho_classification_model = joblib.load("logistic_regression_model.pkl")
    return preprocessing_pipeline, classification_model, lanho_preprocessing_pipeline, lanho_classification_model

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None

preprocessing_pipeline, classification_model, lanho_preprocessing_pipeline, lanho_classification_model = get_classification_model()
        
def plot_analysis_result(X, name_lst, col2, col3):
    def make_large_font(styler):
        styler.set_table_styles([
            {'selector': 'th', 'props': [('font-size', '20px')]},
            {'selector': 'td', 'props': [('font-size', '20px')]}
        ])
        return styler
    preprocesed_X = preprocessing_pipeline.transform(X)
    print(preprocesed_X.shape)
    y_pred = classification_model.predict(preprocesed_X).reshape((len(X), 3))
    labels = ["Mixed", "Root", "Stem-branch"]
    result_table = pd.DataFrame(deepcopy(y_pred), columns=labels)
    print(result_table)
    max_indices = result_table.idxmax(axis=1)
    
    max_indices.index = name_lst
    max_indices = pd.DataFrame(max_indices, columns=["Classification result"])
    max_indices.index.name = "File name"
    with col3:
        # X-axis: column indices

        # Create bar chart
        fig = go.Figure()
        colors = px.colors.qualitative.Set2

        bar_data = []
        for i in range(y_pred.shape[0]):
            bar_data.append({
                'x': [name_lst[i]],
                'y': [y_pred[i, :].max()],
                'name': labels[np.argmax(y_pred[i, :])],
                'marker_color': result_analysis_colors_dict[np.argmax(y_pred[i, :])]
            })

        # Add all traces
        for data in bar_data:
            fig.add_trace(go.Bar(**data, showlegend=False))

        fig.add_trace(go.Scatter(
            x=[None], y=[None],  # Invisible point
            mode="markers",
            marker=dict(color=result_analysis_colors_dict[0], size=10),
            name="Mixed"
        ))

        fig.add_trace(go.Scatter(
            x=[None], y=[None],  # Invisible point
            mode="markers",
            marker=dict(color=result_analysis_colors_dict[1], size=10),
            name="Root"
        ))

        fig.add_trace(go.Scatter(
            x=[None], y=[None],  # Invisible point
            mode="markers",
            marker=dict(color=result_analysis_colors_dict[2], size=10),
            name="Stem branch"
        ))


        fig.update_layout(
            title="Kết quả phân loại:",
            xaxis_title="Sample names",
            yaxis_title="Probabilities",
            template="plotly_white",
            hovermode="x unified",
            margin=dict(t=28, l = 20, r = 5),
            width=900,  # Adjust figure width
            height=300,  # Adjust figure height
            bargap=0.3,
            legend=dict(
                orientation="h",  # Make legend horizontal
                x=0.5,  # Center legend horizontally
                y=-0.4,  # Move legend below the chart
                xanchor="center",  # Align center
                yanchor="top",  
                bgcolor="rgba(255,255,255,0.5)",  # Semi-transparent background
                itemclick=False,  # Disables clicking on the legend
                itemdoubleclick=False  # Disables double-clicking on the legend
            )
        )

        # Show the plot
        st.plotly_chart(fig, use_container_width=False)

        max_indices = max_indices.style.pipe(make_large_font)
        st.dataframe(max_indices, use_container_width=True)
    # Read preprocessing_pipeline from file


def plot_spectrals_data(df_list, name_lst, col_th):
    if df_list:
        fig = go.Figure()

        # Iterate through each uploaded file
        for i, df in enumerate(df_list):
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
                    name=name_lst[i]  # Unique legend entry
                ))
        with col_th:
            # Customize layout
            fig.update_layout(
                title="Phổ IR:",
                xaxis_title="Wavelength (cm-1)",
                yaxis_title="Intensity",
                template="plotly_white",
                hovermode="x unified",
                legend_title="Files & Spectra",
                width=900,  # Adjust figure width
                height=400,  # Adjust figure height
                margin=dict(t=28, b = 22, r = 5),
                legend=dict(
                    orientation="h",  # Make legend horizontal
                    x=0.5,  # Center legend horizontally
                    y=-0.2,  # Move legend below the chart
                    xanchor="center",  # Align center
                    yanchor="top",  
                    bgcolor="rgba(255,255,255,0.5)"  # Semi-transparent background
                )
            )

            # Display the plot
            st.plotly_chart(fig, use_container_width=False)
            
def plot_lanho_proba(X, name_lst, col2):
    def custom_predict(X, threshold):
        probs = lanho_classification_model.predict_proba(X)
        return (probs[:, 1] > threshold).astype(int)
    lanho_threshold = 0.85
    preprocesed_X = lanho_preprocessing_pipeline.transform(X)
    y_pred = lanho_classification_model.predict_proba(preprocesed_X)
    with col2:
        class_1_probs = y_pred[:, 1]
        sample_indices = name_lst

        # Create Plotly figure
        # Assign colors based on threshold
        colors = ["orange" if prob > lanho_threshold else "blue" for prob in class_1_probs]

        # Create Plotly figure
        fig = go.Figure()

        # Add bars with conditional colors
        for i, (prob, sample, color) in enumerate(zip(class_1_probs, sample_indices, colors)):
            fig.add_trace(go.Bar(
                x=[prob], 
                y=[sample], 
                orientation='h',
                marker=dict(color=color),
                text=f"{prob:.4f} đinh lăng",
                textposition='outside',
                showlegend=False  # Hide individual bar legends
            ))

        # Add vertical threshold line
        fig.add_vline(
            x=lanho_threshold, 
            line=dict(color="red", width=2, dash="dash"),
        )

        # **Add scatter traces for legend (without affecting bars)**
        fig.add_trace(go.Scatter(
            x=[None], y=[None],  # Invisible point
            mode="markers",
            marker=dict(color="orange", size=10),
            name="Là đinh lăng lá nhỏ"
        ))

        fig.add_trace(go.Scatter(
            x=[None], y=[None],  # Invisible point
            mode="markers",
            marker=dict(color="blue", size=10),
            name="Không phải đinh lăng lá nhỏ"
        ))

        # Adjust layout
        fig.update_layout(
            title="Xác suất đinh lăng lá nhỏ",
            xaxis_title="Probability",
            yaxis_title="Sample",
            bargap=0.3,  # Increase spacing between bars
            height=200,
            margin=dict(t=22),
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=-1, 
                xanchor="center", 
                x=0.5,
                itemclick=False,  # Disables clicking on the legend
                itemdoubleclick=False  # Disables double-clicking on the legend
            )  # Horizontal legend
        )

        # Display in Streamlit
        st.plotly_chart(fig, use_container_width=False)

    # Return list indices of which sample is "la nho"
    final_result = custom_predict(preprocesed_X, lanho_threshold)
    indices = np.where(final_result == 1)[0]
    return indices



X = None
with col1:
    uploaded_files = st.file_uploader(
        "Choose a CSV file", accept_multiple_files=True, type=["csv"]
    )
    MAX_FILES = 5

    if len(uploaded_files) > MAX_FILES:

        st.warning(f"Giới hạn {MAX_FILES} files được tải lên một lúc, hệ thống sẽ chỉ nhận diện 5 file đầu tiên")

        uploaded_files = uploaded_files[:MAX_FILES]
    if uploaded_files and uploaded_files != st.session_state.uploaded_files:
        st.session_state.uploaded_files = uploaded_files  # Store files in session state
        st.rerun()  # Rerun the app once after file upload
    if st.session_state.uploaded_files:
        df_list.clear()
        df_names.clear()
        X = None
        lanho_indices = None
        # Iterate through each uploaded file
        for uploaded_file in uploaded_files:
            # Read the CSV file
            df_names.append(uploaded_file.name.split(".csv")[0].split(".CSV")[0])
            df = pd.read_csv(uploaded_file, header = None)
            df_list.append(df)
        if len(df_list) > 0:
            X = pd.concat(df_list, axis = 1).iloc[:, 1::2].T.values
            plot_spectrals_data(df_list, df_names, col2)
            lanho_indices = plot_lanho_proba(X, df_names, col2)
    if st.button("Classify", use_container_width=True):
        if lanho_indices is not None and len(lanho_indices) > 0:
            plot_analysis_result(X[lanho_indices], [df_names[i] for i in lanho_indices], col2, col3)
    else:
        print("ERROR")