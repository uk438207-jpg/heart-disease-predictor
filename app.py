import gradio as gr
import joblib
import pandas as pd
import numpy as np
import json

# Load the trained model
try:
    model = joblib.load('heart_disease_model.pkl')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Chest pain type mapping
cp_mapping = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, thalch, exang, oldpeak):
    """
    Predict heart disease risk based on input features
    """
    if model is None:
        return "❌ Model not loaded. Please check the model file."
    
    try:
        # Prepare features
        features = np.array([[
            float(age),
            1 if sex == "Male" else 0,
            cp_mapping[cp],
            float(trestbps),
            float(chol),
            1 if fbs == "Yes" else 0,
            float(thalch),
            1 if exang == "Yes" else 0,
            float(oldpeak)
        ]])
        
        # Feature names
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'thalch', 'exang', 'oldpeak']
        
        # Create DataFrame
        df = pd.DataFrame(features, columns=feature_names)
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1] * 100
        
        # Prepare result
        if prediction == 1:
            result = {
                "prediction": "⚠️ **Heart Disease Detected**",
                "probability": f"{probability:.2f}%",
                "risk_level": "High",
                "color": "#ff4444",
                "icon": "❤️‍🩹",
                "recommendations": [
                    "Consult a cardiologist immediately",
                    "Maintain a heart-healthy diet (low sodium, low cholesterol)",
                    "Exercise regularly (30 minutes daily)",
                    "Monitor blood pressure and cholesterol",
                    "Quit smoking and limit alcohol",
                    "Consider stress management techniques"
                ]
            }
        else:
            result = {
                "prediction": "✅ **No Heart Disease Detected**",
                "probability": f"{probability:.2f}%",
                "risk_level": "Low",
                "color": "#00C851",
                "icon": "💚",
                "recommendations": [
                    "Continue regular check-ups with your doctor",
                    "Maintain a balanced diet and healthy lifestyle",
                    "Exercise at least 30 minutes daily",
                    "Monitor your heart health indicators regularly",
                    "Stay hydrated and maintain healthy sleep patterns"
                ]
            }
        
        # Create result HTML
        html_result = f"""
        <div style="text-align: center; padding: 20px;">
            <h2 style="color: {result['color']};">{result['icon']} {result['prediction']}</h2>
            <h3 style="font-size: 24px; margin: 20px 0;">Probability: <span style="color: {result['color']};">{result['probability']}</span></h3>
            <div style="background-color: {result['color']}15; border: 2px solid {result['color']}; 
                       border-radius: 10px; padding: 15px; margin: 20px 0; display: inline-block;">
                <strong>Risk Level:</strong> {result['risk_level']}
            </div>
            
            <div style="text-align: left; margin-top: 30px; background: #f8f9fa; padding: 20px; border-radius: 10px;">
                <h4>📋 Recommendations:</h4>
                <ul style="padding-left: 20px;">
        """
        
        for rec in result['recommendations']:
            html_result += f'<li style="margin-bottom: 8px;">{rec}</li>'
        
        html_result += """
                </ul>
            </div>
            
            <div style="margin-top: 30px; padding: 15px; background: #fff3cd; border-radius: 10px; font-size: 14px;">
                ⚠️ <strong>Disclaimer:</strong> This prediction is based on machine learning models and 
                should not replace professional medical advice. Always consult with healthcare professionals.
            </div>
        </div>
        """
        
        return html_result
    
    except Exception as e:
        return f"❌ Error making prediction: {str(e)}"

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Heart Disease Prediction") as demo:
    gr.Markdown("""
    # ❤️ Heart Disease Prediction System
    
    Predict the likelihood of heart disease using machine learning based on health indicators.
    
    **Instructions:** Fill in all the fields below and click 'Predict' to see the results.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Image("https://huggingface.co/datasets/huggingface/badges/resolve/main/powered-by-huggingface-dark.svg", 
                    height=30, show_label=False)
            
        with gr.Column(scale=3):
            gr.Markdown("### 📊 Model Information")
            gr.Markdown("""
            - **Model:** Random Forest Classifier
            - **Accuracy:** 82.07% on test data
            - **Training Data:** Heart Disease UCI Dataset (920 samples)
            - **Features Used:** 9 clinical parameters
            """)
    
    with gr.Row():
        with gr.Column():
            # Input Section 1
            age = gr.Number(label="Age (years)", value=55, minimum=20, maximum=100, step=1)
            sex = gr.Radio(["Male", "Female"], label="Sex", value="Male")
            cp = gr.Dropdown(
                ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"], 
                label="Chest Pain Type", 
                value="Non-anginal Pain"
            )
            
        with gr.Column():
            # Input Section 2
            trestbps = gr.Number(label="Resting Blood Pressure (mm Hg)", value=130, minimum=90, maximum=200, step=1)
            chol = gr.Number(label="Cholesterol (mg/dl)", value=250, minimum=100, maximum=600, step=1)
            fbs = gr.Radio(["Yes", "No"], label="Fasting Blood Sugar > 120 mg/dl", value="No")
            
        with gr.Column():
            # Input Section 3
            thalch = gr.Number(label="Maximum Heart Rate Achieved", value=150, minimum=60, maximum=220, step=1)
            exang = gr.Radio(["Yes", "No"], label="Exercise Induced Angina", value="No")
            oldpeak = gr.Number(label="ST Depression", value=1.2, minimum=-3, maximum=7, step=0.1)
    
    # Example inputs
    gr.Markdown("### 💡 Try Example Inputs")
    with gr.Row():
        example1 = gr.Button("Example 1: High Risk Patient")
        example2 = gr.Button("Example 2: Low Risk Patient")
        example3 = gr.Button("Example 3: Female Patient")
    
    # Predict button
    predict_btn = gr.Button("🔍 Predict Heart Disease Risk", variant="primary", size="lg")
    
    # Output
    output = gr.HTML(label="Prediction Result")
    
    # Footer
    gr.Markdown("""
    ---
    ### ℹ️ About the Features
    
    | Feature | Description | Normal Range |
    |---------|-------------|--------------|
    | **Age** | Patient's age in years | 20-100 years |
    | **Sex** | Patient's gender | Male/Female |
    | **Chest Pain Type** | Type of chest pain experienced | 4 types |
    | **Resting BP** | Resting blood pressure in mm Hg | 90-120 mm Hg |
    | **Cholesterol** | Serum cholesterol in mg/dl | < 200 mg/dl |
    | **Fasting BS** | Fasting blood sugar > 120 mg/dl | True/False |
    | **Max Heart Rate** | Maximum heart rate achieved | 60-220 bpm |
    | **Exercise Angina** | Angina induced by exercise | Yes/No |
    | **ST Depression** | Depression induced by exercise | -3 to 7 |
    
    *Built with ❤️ using Random Forest Machine Learning*
    """)
    
    # Example handlers
    example1.click(
        fn=lambda: [63, "Male", "Asymptomatic", 145, 233, "Yes", 150, "No", 2.3],
        inputs=[],
        outputs=[age, sex, cp, trestbps, chol, fbs, thalch, exang, oldpeak]
    )
    
    example2.click(
        fn=lambda: [45, "Female", "Typical Angina", 120, 180, "No", 160, "No", 0.5],
        inputs=[],
        outputs=[age, sex, cp, trestbps, chol, fbs, thalch, exang, oldpeak]
    )
    
    example3.click(
        fn=lambda: [58, "Female", "Atypical Angina", 140, 220, "No", 135, "Yes", 1.8],
        inputs=[],
        outputs=[age, sex, cp, trestbps, chol, fbs, thalch, exang, oldpeak]
    )
    
    # Connect predict button
    predict_btn.click(
        fn=predict_heart_disease,
        inputs=[age, sex, cp, trestbps, chol, fbs, thalch, exang, oldpeak],
        outputs=output
    )
    
    # Auto-clear example buttons
    example1.click(
        fn=lambda: gr.update(value=""),
        inputs=[],
        outputs=output
    )
    example2.click(
        fn=lambda: gr.update(value=""),
        inputs=[],
        outputs=output
    )
    example3.click(
        fn=lambda: gr.update(value=""),
        inputs=[],
        outputs=output
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )