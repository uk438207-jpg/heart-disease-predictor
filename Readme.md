---
title: Heart Disease Prediction
emoji: ❤️
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 3.41.2
app_file: app.py
pinned: false
license: mit
---

# ❤️ Heart Disease Prediction System

A machine learning web application that predicts the likelihood of heart disease based on clinical parameters using a Random Forest classifier.

## 🚀 Features

- **Real-time Prediction**: Get instant heart disease risk assessment
- **Multiple Input Examples**: Pre-filled examples for quick testing
- **Detailed Results**: Probability percentage and risk level
- **Medical Recommendations**: Personalized health recommendations
- **Responsive Design**: Works on desktop and mobile devices

## 📊 Model Information

- **Algorithm**: Random Forest Classifier
- **Accuracy**: 82.07% on test data
- **Training Data**: Heart Disease UCI Dataset (920 samples)
- **Features Used**: 9 clinical parameters

## 🎯 How to Use

1. **Fill in the form** with patient information:
   - Age, Sex, Chest Pain Type
   - Resting Blood Pressure, Cholesterol
   - Fasting Blood Sugar, Maximum Heart Rate
   - Exercise Angina, ST Depression

2. **Click "Predict"** to see results

3. **View detailed results** including:
   - Prediction (Heart Disease/No Heart Disease)
   - Probability percentage
   - Risk level (High/Low)
   - Personalized recommendations

## 🧪 Try Example Inputs

Use the example buttons to quickly test different scenarios:
- **Example 1**: High-risk patient profile
- **Example 2**: Low-risk patient profile
- **Example 3**: Female patient profile

## ⚠️ Important Disclaimer

**This tool is for informational purposes only.** 
- NOT a substitute for professional medical advice
- Always consult with healthcare professionals
- Use at your own discretion

## 🛠️ Technical Details

- **Framework**: Gradio for UI
- **Backend**: Flask-like Gradio interface
- **Model**: Scikit-learn Random Forest
- **Deployment**: Hugging Face Spaces

## 📁 Project Structure

```
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── heart_disease_model.pkl  # Trained ML model
└── README.md          # This file
```

## 🔧 Local Development

```bash
# Clone and setup
git clone https://huggingface.co/spaces/your-username/heart-disease-prediction
cd heart-disease-prediction

# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

## 📚 Dataset Information

The model was trained on the Heart Disease UCI Dataset which contains:
- 920 patient records
- 14 clinical features
- 5 classes (0-4) of heart disease severity
- Binary classification: 0 = No disease, 1-4 = Disease present

## 🤝 Contributing

Feel free to:
- Report issues
- Suggest improvements
- Fork and create pull requests

## 📄 License

MIT License - see LICENSE file for details

---

*Made with ❤️ for better healthcare insights*