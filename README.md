# ML Pipeline Doctor AI

Streamlit app that diagnoses ML pipeline problems (leakage, missing values, imbalance,
overfitting, underfitting, preprocessing, poor validation) from CSV data or pasted training metrics.
All results are computed from your data.

## Run
    pip install -r requirements.txt
    streamlit run app.py

## Optional AI explanations
Set ANTHROPIC_API_KEY or paste a key in the sidebar. Without a key, a built-in rule-based explainer is used.
