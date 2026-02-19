import streamlit as st
import numpy as np
import json

# Load model weights
with open('model_weights.json', 'r') as f:
    layers = json.load(f)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def predict(features):
    x = features
    for i, layer in enumerate(layers):
        weights = np.array(layer['weights'])
        biases = np.array(layer['biases'])
        x = np.dot(x, weights) + biases
        if i < len(layers) - 1:
            x = relu(x)
        else:
            x = sigmoid(x)
    return x[0][0]

st.title('Real-Time Fraud Detection System')
st.write('Enter transaction details to get fraud probability')

amount = st.number_input('Transaction Amount', min_value=0.0, max_value=30000.0, value=100.0)
time_val = st.number_input('Time (seconds)', min_value=0, max_value=180000, value=50000)

st.subheader('PCA Features (V1-V28)')

v_features = []
cols = st.columns(4)
for i in range(28):
    with cols[i % 4]:
        v = st.number_input(f'V{i+1}', value=0.0, format='%.4f', key=f'v{i+1}')
        v_features.append(v)

if st.button('Detect Fraud', type='primary'):
    features = np.array([[time_val] + v_features + [amount]])
    probability = predict(features)
    
    if probability > 0.5:
        st.error(f'FRAUD DETECTED - Probability: {probability:.2%}')
    elif probability > 0.3:
        st.warning(f'SUSPICIOUS - Probability: {probability:.2%}')
    else:
        st.success(f'NORMAL - Fraud Probability: {probability:.2%}')
    
    st.progress(min(float(probability), 1.0))
