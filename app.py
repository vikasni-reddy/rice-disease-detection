# contents of file
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json
import datetime
import plotly.graph_objects as go
import os
import gdown


# ---------------------- Load Model ----------------------
@st.cache_resource
def load_model_and_classes():
    MODEL_PATH = "best_model.h5"
    MODEL_URL = "https://drive.google.com/uc?id=1O4KC7l0jzyZkrUsrWKX-KWGfVCzy7W2v"

    # Download model if not present
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading model (one-time)..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    model = load_model(MODEL_PATH)

    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)

    class_names = {v: k for k, v in class_indices.items()}
    return model, class_names


# ---------------------- Preprocessing ----------------------
# NOTE: This preprocessing matches training that used rescale=1./255.
def preprocess_image(img):
    # Ensure RGB and correct size
    img = img.convert('RGB').resize((224, 224))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------------- Radar Chart ----------------------
def create_radar_chart(predictions, class_names):
    categories = [name.replace('_', ' ').title() for name in class_names.values()]
    fig = go.Figure(data=go.Scatterpolar(
        r=predictions * 100,
        theta=categories,
        fill='toself',
        name='Disease Confidence',
        fillcolor='rgba(99, 110, 250, 0.3)',
        line=dict(color='rgb(99, 110, 250)', width=2)
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
        ),
        showlegend=False,
        margin=dict(l=60, r=60, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, color='#e2e8f0')
    )
    return fig

# ---------------------- Pages ----------------------
def home_page():
    st.markdown('<div class="page-title">🌾 Rice Disease Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.write("### Welcome")
    st.write("This AI-powered system helps identify diseases in rice plants. Upload a clear image of a rice leaf to get instant analysis and treatment recommendations.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("⚡ Fast", "< 2 seconds")
    with col2:
        st.metric("🎯 Accurate", "96.7%")
    with col3:
        st.metric("🔬 Diseases", "5 types")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.write("### Upload Image for Analysis")
    uploaded_file = st.file_uploader("Choose a rice plant leaf image", type=["jpg", "jpeg", "png"], label_visibility="visible")

    if uploaded_file is not None:
        process_image(uploaded_file)

def about_page():
    st.markdown('<div class="page-title">ℹ️ About</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.write("### Our Mission")
    st.write("To empower farmers with AI-driven solutions for early detection of rice plant diseases, helping them take timely actions and improve crop yield.")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.write("### Why Rice Disease Detection?")
    st.write("Rice is a staple food for over half the world's population. Early detection of diseases can reduce crop losses, minimize pesticide use, and increase yield.")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.write("### Team")
    st.write("**Developed by:** @vikasni-reddy @harish @ankit @maanvitha")
    st.write("**Project:** B.Tech Final Year Mini-Project (2025)")

def model_page():
    st.markdown('<div class="page-title">🤖 Model Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Technical Details")
        st.write("**Architecture:** EfficientNetB0")
        st.write("**Framework:** TensorFlow/Keras")
        st.write("**Input Size:** 224×224")
        st.write("**Dataset:** 3,252 train / 817 validation images")
    with col2:
        st.write("### Performance Metrics")
        st.write("**Accuracy:** 96.7%")
        st.write("**Training Time:** 2 hours")
        st.write("**Model Size:** 29 MB")

def help_page():
    st.markdown('<div class="page-title">❓ Help & Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.write("### How to Use")
    st.write("**Step 1:** Navigate to the Home page")
    st.write("**Step 2:** Click on the upload button")
    st.write("**Step 3:** Select a clear image of a rice plant leaf")
    st.write("**Step 4:** Wait for the analysis results")
    st.write("**Step 5:** Review the detection results and recommendations")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.write("### Image Guidelines")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**✅ Good Practices**")
        st.write("• Use clear, well-lit photos")
        st.write("• Capture the entire leaf")
        st.write("• JPG, JPEG, PNG formats")
    with col2:
        st.write("**❌ Avoid**")
        st.write("• Blurry or dark images")
        st.write("• Partial leaf coverage")
        st.write("• Multiple overlapping leaves")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.write("### Troubleshooting")
    st.write("**Upload fails:** Check image format and file size (under 10MB)")
    st.write("**Low confidence:** Try a clearer image with better lighting")
    st.write("**Slow processing:** Ensure stable internet connection")

# ---------------------- Image Processing ----------------------
def process_image(uploaded_file):
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption='Uploaded Image', use_container_width=True)

    with st.spinner('🔍 Analyzing image...'):
        model, class_names = load_model_and_classes()
        processed_image = preprocess_image(image_display)
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction[0])
        predicted_class = class_names[predicted_class_index]
        confidence = float(prediction[0][predicted_class_index])

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.write("### Detection Results")

    is_healthy = predicted_class.lower() == "healthy"
    status_emoji = "✅" if is_healthy else "⚠️"

    st.markdown(f"""
        <div class="result-box">
            <div class="disease-name">{status_emoji} {predicted_class.replace("_", " ").title()}</div>
            <div class="confidence-text">Confidence: {confidence*100:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)

    st.write("#### Confidence Distribution")
    fig = create_radar_chart(prediction[0], class_names)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.write("### Treatment Recommendations")

    recommendations = {
        "brown_spot": "Apply fungicides like **Tricyclazole** or **Mancozeb**. Maintain proper field drainage and avoid water stress.",
        "leaf_blast": "Use **Tricyclazole** (0.6g/lit). Avoid excessive nitrogen application and ensure balanced nutrition.",
        "leaf_scald": "Improve air circulation in the field. Use resistant varieties and apply fungicides if infection is severe.",
        "sheath_blight": "Apply **Validamycin** or **Hexaconazole**. Improve field drainage and remove infected plant debris.",
        "healthy": "✅ Your rice plant is healthy! Continue with current management practices, regular monitoring, and preventive care."
    }

    rec = recommendations.get(predicted_class.lower(), "⚠️ Consult an agricultural expert for further analysis.")
    st.write(rec)

# ---------------------- Main App ----------------------
def main():
    st.set_page_config(
        page_title="🌾 Rice Disease Detection",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ---------------------- Custom CSS with Sidebar ----------------------
    st.markdown("""
        <style>
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1f2e 0%, #0e1117 100%);
            border-right: 1px solid rgba(99, 110, 250, 0.3);
        }
        
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 2rem;
        }
        
        /* Sidebar header */
        .sidebar-header {
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #636efa, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5rem;
            padding: 0 1rem;
        }
        
        .sidebar-subtitle {
            font-size: 0.85rem;
            color: #94a3b8;
            text-align: center;
            margin-bottom: 2rem;
            padding: 0 1rem;
        }
        
        /* Main content area */
        .block-container {
            max-width: 1000px;
            margin: 0 auto !important;
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        
        .page-title {
            font-size: 2rem;
            font-weight: 600;
            color: #e2e8f0;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        .divider {
            height: 1px;
            background: linear-gradient(90deg,transparent,rgba(99,110,250,0.5),transparent);
            margin: 2rem auto;
            width: 70%;
        }
        
        h3, h4, p {
            text-align: center !important;
            color: #cbd5e1;
        }
        
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
            text-align: center !important;
        }
        
        /* Sidebar buttons */
        [data-testid="stSidebar"] .stButton button {
            background: linear-gradient(135deg, #636efa, #8b5cf6);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 500;
            width: 100%;
            height: 3rem;
            transition: all 0.3s ease;
            font-size: 1rem;
            margin-bottom: 0.5rem;
        }
        
        [data-testid="stSidebar"] .stButton button:hover {
            transform: translateX(5px);
            box-shadow: 0 4px 12px rgba(99,110,250,0.4);
            background: linear-gradient(135deg, #8b5cf6, #636efa);
        }
        
        /* Active button state */
        [data-testid="stSidebar"] .stButton button:active {
            background: linear-gradient(135deg, #8b5cf6, #636efa);
        }
        
        /* File uploader */
        [data-testid="stFileUploader"] {
            border: 2px dashed rgba(99,110,250,0.3);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center !important;
            background: rgba(99,110,250,0.05);
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: rgba(99,110,250,0.6);
            background: rgba(99,110,250,0.08);
        }
        
        /* Result box */
        .result-box {
            background: rgba(99,110,250,0.1);
            border: 1px solid rgba(99,110,250,0.3);
            border-radius: 12px;
            padding: 2rem;
            margin: 1.5rem auto;
            width: 80%;
            text-align: center;
        }
        
        .disease-name {
            font-size: 2rem;
            font-weight: 700;
            color: #e2e8f0;
        }
        
        .confidence-text {
            font-size: 1.3rem;
            color: #636efa;
            font-weight: 600;
        }
        
        [data-testid="stImage"] {
            text-align: center;
            display: flex;
            justify-content: center;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem 0;
            color: #64748b;
            border-top: 1px solid rgba(99,110,250,0.2);
            margin-top: 3rem;
        }
        
        .footer p {
            margin: 0.3rem 0;
        }
        
        /* Sidebar footer */
        .sidebar-footer {
            position: fixed;
            bottom: 1rem;
            left: 1rem;
            right: 1rem;
            text-align: center;
            color: #64748b;
            font-size: 0.75rem;
            padding: 0.5rem;
            border-top: 1px solid rgba(99,110,250,0.2);
        }
        
        /* Animations */
        @keyframes fadeIn {
            from {opacity:0; transform:translateY(10px);}
            to {opacity:1; transform:translateY(0);}
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar Navigation
    with st.sidebar:
        st.markdown('<div class="sidebar-header">🌾 Rice Disease Detection</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-subtitle">AI-Powered Agricultural Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        st.write("### Navigation")
        
        # Initialize session state
        if 'page' not in st.session_state:
            st.session_state.page = "🏠 Home"
        
        # Navigation buttons
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "🏠 Home"
        
        if st.button("ℹ️ About", use_container_width=True):
            st.session_state.page = "ℹ️ About"
        
        if st.button("🤖 Model", use_container_width=True):
            st.session_state.page = "🤖 Model"
        
        if st.button("❓ Help", use_container_width=True):
            st.session_state.page = "❓ Help"
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Sidebar info
        st.write("### Quick Info")
        st.metric("Model Accuracy", "96.7%")
        st.metric("Detection Speed", "< 2s")
        st.metric("Diseases", "5 types")

    # Main content area - Display selected page
    pages = {
        "🏠 Home": home_page,
        "ℹ️ About": about_page,
        "🤖 Model": model_page,
        "❓ Help": help_page
    }
    
    pages[st.session_state.page]()

    # Footer
    st.markdown(f"""
        <div class="footer">
            <p>© 2025 Rice Disease Detection System</p>
            <p>Last Updated: {datetime.datetime.utcnow().strftime('%Y-%m-%d')}</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":

    main()

