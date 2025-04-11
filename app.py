from flask import Flask, render_template, request, jsonify
import pickle
import os
import numpy as np
import pandas as pd

app = Flask(__name__)

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Create static directory if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')

# Define a mapping from user-friendly questions to technical features with weights
user_friendly_questions = {
    'is_suspicious_url': {
        'question': 'Does the website URL look strange or suspicious?',
        'features': ['having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix'],
        'weight': 2.0,  # High impact factor
        'explanation': 'URLs with IP addresses, unusual length, or shortening services are often used in phishing.'
    },
    'domain_issues': {
        'question': 'Is the website domain new or does it seem untrustworthy?',
        'features': ['having_Sub_Domain', 'Domain_registeration_length', 'age_of_domain', 'DNSRecord'],
        'weight': 1.5,
        'explanation': 'Phishing sites often use newly registered domains or suspicious subdomains.'
    },
    'security_concerns': {
        'question': 'Does the website lack security indicators (like a padlock in the browser)?',
        'features': ['SSLfinal_State', 'HTTPS_token'],
        'weight': 2.0,  # High impact factor
        'explanation': 'Legitimate websites typically use HTTPS with valid certificates.'
    },
    'external_resources': {
        'question': 'Does the website load content from other suspicious sites?',
        'features': ['Favicon', 'Request_URL', 'URL_of_Anchor', 'Links_in_tags', 'SFH'],
        'weight': 1.0,
        'explanation': 'Phishing sites often load resources from multiple suspicious domains.'
    },
    'suspicious_behavior': {
        'question': 'Does the website have suspicious behavior (popups, redirects, etc.)?',
        'features': ['Submitting_to_email', 'Abnormal_URL', 'Redirect', 'popUpWidnow', 'Iframe'],
        'weight': 1.5,
        'explanation': 'Unexpected popups, redirects, and iframes can indicate malicious intent.'
    },
    'user_interaction': {
        'question': 'Does the website limit your control (disabled right-click, hidden links)?',
        'features': ['on_mouseover', 'RightClick'],
        'weight': 1.0,
        'explanation': 'Phishing sites often disable user controls to prevent inspection.'
    },
    'low_popularity': {
        'question': 'Does the website appear to have low popularity or visitor count?',
        'features': ['web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page'],
        'weight': 0.5,  # Lower impact factor
        'explanation': 'Legitimate sites usually have established web presence and traffic.'
    },
    'known_threat': {
        'question': 'Have you heard warnings about this website from others?',
        'features': ['Statistical_report'],
        'weight': 2.0,  # High impact factor
        'explanation': 'Reports from security services are strong indicators of phishing.'
    },
    'login_request': {
        'question': 'Does the website unexpectedly ask for your username and password?',
        'features': [],  # This is a new question not directly mapped to original features
        'weight': 2.0,  # High impact factor
        'explanation': 'Unexpected login requests are classic phishing techniques.'
    },
    'urgent_action': {
        'question': 'Does the website claim urgent action is required to avoid a problem?',
        'features': [],  # This is a new question not directly mapped to original features
        'weight': 1.5,
        'explanation': 'Creating a false sense of urgency is a common phishing tactic.'
    }
}

# Default mappings from user-friendly answers to technical features
def get_default_mappings():
    return {
        'having_IP_Address': 0,
        'URL_Length': 0,
        'Shortining_Service': 0,
        'having_At_Symbol': 0,
        'double_slash_redirecting': 0,
        'Prefix_Suffix': 0,
        'having_Sub_Domain': 0,
        'SSLfinal_State': 1,  # Reversed: 1 means secure in original dataset
        'Domain_registeration_length': 0,
        'Favicon': 0,
        'port': 0,
        'HTTPS_token': 0,
        'Request_URL': 0,
        'URL_of_Anchor': 0,
        'Links_in_tags': 0,
        'SFH': 0,
        'Submitting_to_email': 0,
        'Abnormal_URL': 0,
        'Redirect': 0,
        'on_mouseover': 0,
        'RightClick': 0,
        'popUpWidnow': 0,
        'Iframe': 0,
        'age_of_domain': 0,
        'DNSRecord': 0,
        'web_traffic': 0,
        'Page_Rank': 0,
        'Google_Index': 0,
        'Links_pointing_to_page': 0,
        'Statistical_report': 0
    }

# Get the absolute path to the models directory
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, 'models')

# Initialize global variables for models and scaler
models = {}
scaler = None
feature_names = []
models_loaded = False

def load_models():
    """Load all required model files"""
    global models, scaler, feature_names, models_loaded
    
    try:
        # Print working directory for debugging
        print("Working directory:", os.getcwd())
        print("Looking for models in:", models_dir)
        
        # Check if the models directory exists and list its contents
        if os.path.exists(models_dir):
            print("Files in models directory:", os.listdir(models_dir))
        else:
            print("Models directory does not exist!")
        
        # Load the models using absolute paths
        model_files = {
            'Random Forest': os.path.join(models_dir, 'logistic_regression_model.pkl')
        }
        
        models = {}
        for name, file_path in model_files.items():
            print(f"Attempting to load {file_path}...")
            try:
                with open(file_path, 'rb') as f:
                    models[name] = pickle.load(f)
                print(f"Successfully loaded {name} model")
            except Exception as specific_error:
                print(f"Error loading {name} model: {specific_error}")
                raise specific_error
                
        print("Attempting to load scaler.pkl...")
        with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        print("Successfully loaded scaler")
            
        print("Attempting to load feature_names.pkl...")
        with open(os.path.join(models_dir, 'feature_names.pkl'), 'rb') as f:
            feature_names = pickle.load(f)
        print("Successfully loaded feature names")
            
        models_loaded = True
        print("All models loaded successfully!")
        return True
    except Exception as e:
        models_loaded = False
        print(f"Error loading models: {type(e).__name__}: {e}")
        print("Models not yet trained or saved. Please run the training script first.")
        return False

# Try to load models at startup
load_models()

@app.route('/', methods=['GET'])
def index():
    if not models_loaded:
        return render_template('models_not_loaded.html')
    return render_template('index.html', questions=user_friendly_questions)

@app.route('/predict', methods=['POST'])
def predict():
    # Try to load models if not already loaded
    global models_loaded
    if not models_loaded:
        models_loaded = load_models()
        if not models_loaded:
            return jsonify({"error": "Models not loaded yet. Please train the models first."})
    
    # Get form data - the user-friendly responses
    user_responses = {}
    for question_id in user_friendly_questions:
        user_responses[question_id] = int(request.form.get(question_id, '0'))
    
    # APPROACH 1: Direct scoring based on user responses
    # Calculate direct risk score based on weighted questions
    direct_risk_score = 0
    max_possible_score = 0
    
    # Track which factors contributed to the risk assessment
    risk_factors = []
    
    for question_id, response in user_responses.items():
        question_weight = user_friendly_questions[question_id].get('weight', 1.0)
        max_possible_score += question_weight
        if response == 1:  # If user answered "Yes" to a risk factor
            direct_risk_score += question_weight
            risk_factors.append({
                'question': user_friendly_questions[question_id]['question'],
                'explanation': user_friendly_questions[question_id].get('explanation', ''),
                'weight': question_weight
            })
    
    # Convert to percentage (0-100%)
    risk_percentage = (direct_risk_score / max_possible_score) * 100 if max_possible_score > 0 else 0
    
    # APPROACH 2: Model-based prediction (as a reference point)
    technical_features = get_default_mappings()
    
    # Update technical features based on user responses
    for question_id, response in user_responses.items():
        if response == 1:
            for feature in user_friendly_questions[question_id].get('features', []):
                if feature == 'SSLfinal_State':
                    technical_features[feature] = 0
                else:
                    technical_features[feature] = 1
    
    # Prepare feature array for model prediction
    features = [technical_features[feature] for feature in feature_names]
    features_array = np.array(features).reshape(1, -1)
    scaled_features = scaler.transform(features_array)

    model_results = {}
    for name, model in models.items():
        raw_prediction = model.predict(scaled_features)[0]
        raw_probability = model.predict_proba(scaled_features)[0][1]
        model_results[name] = {
            'prediction': int(raw_prediction),
            'probability': float(raw_probability)
        }
    
    # IMPROVED FINAL DETERMINATION: Better threshold handling
    # Define confidence levels based on risk percentage
    if risk_percentage >= 50:  # Very high risk
        confidence = "high"
        verdict = "SUSPICIOUS"
        final_prediction = 1  # Suspicious
        advisory = "This website shows strong indicators of being a phishing attempt. Exercise extreme caution."
    elif risk_percentage >= 30:  # High risk
        confidence = "medium"
        verdict = "SUSPICIOUS"
        final_prediction = 1  # Suspicious
        advisory = "This website shows concerning indicators of being potentially malicious. Proceed with caution."
    elif risk_percentage >= 15:  # Medium risk
        # Check if model agrees with medium risk assessment
        model_votes = sum(result['prediction'] for result in model_results.values())
        if model_votes > 0:
            confidence = "medium"
            verdict = "POTENTIALLY SUSPICIOUS"
            final_prediction = 1  # Suspicious
            advisory = "This website shows some concerning signs. Proceed with caution and verify its legitimacy."
        else:
            confidence = "medium"
            verdict = "LIKELY SAFE"
            final_prediction = 0  # Safe
            advisory = "This website appears mostly safe, but some minor concerns were detected."
    elif risk_percentage >= 5:  # Low risk
        confidence = "medium"
        verdict = "SAFE"
        final_prediction = 0  # Safe
        advisory = "This website appears to be safe based on your responses."
    else:  # Very low risk
        confidence = "high" 
        verdict = "SAFE"
        final_prediction = 0  # Safe
        advisory = "This website appears to be completely safe based on your responses."

    # Create comprehensive results dictionary
    final_results = {
        'Direct Assessment': {
            'prediction': 1 if risk_percentage >= 30 else 0,
            'probability': float(risk_percentage / 100)
        }
    }
    
    # Add model results
    for name, result in model_results.items():
        final_results[name] = result
    
    # Add combined result
    final_results['Final Verdict'] = {
        'prediction': final_prediction,
        'probability': float(risk_percentage / 100),
        'risk_score': float(risk_percentage),
        'confidence': confidence,
        'verdict': verdict,
        'advisory': advisory,
        'risk_factors': risk_factors
    }

    # Add Ensemble key for backward compatibility with templates
    final_results['Ensemble'] = {
        'prediction': final_prediction,
        'probability': float(risk_percentage / 100)
    }

    # Debug output
    print("==== ASSESSMENT RESULTS ====")
    print(f"Direct risk score: {direct_risk_score:.2f}/{max_possible_score:.2f} ({risk_percentage:.2f}%)")
    for name, result in model_results.items():
        print(f"{name}: Prediction={result['prediction']}, Prob={result['probability']:.4f}")
    print(f"Final verdict: {verdict} with {confidence} confidence")
    print("Risk factors identified:")
    for factor in risk_factors:
        print(f"- {factor['question']}")
    print("============================")

    return render_template('result.html', results=final_results)

@app.route('/retry', methods=['GET'])
def retry():
    """Allow users to try another assessment"""
    return index()

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    status = {
        'status': 'ok',
        'models_loaded': models_loaded
    }
    return jsonify(status)

@app.route('/reload_models', methods=['POST'])
def reload_models():
    """Admin endpoint to reload models without restarting the application"""
    if request.form.get('admin_key') == os.environ.get('ADMIN_KEY', 'default_key'):
        success = load_models()
        return jsonify({
            'success': success,
            'message': 'Models reloaded successfully' if success else 'Failed to reload models'
        })
    else:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401

if __name__ == '__main__':
    # Try loading models at startup
    if not models_loaded:
        load_models()
    app.run(debug=True)