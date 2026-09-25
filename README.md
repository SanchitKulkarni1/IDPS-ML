# 🛡️ IDPS-ML: Intrusion Detection with Machine Learning

**AnyScanner IDS** classifies network traffic as normal or as a specific **attack type** (multiclass) using trained ML models. A Streamlit app runs predictions in real time.

## Features

- **Binary and multiclass models:** attack vs. normal, and which attack.
- **Feature-selected models:** trained on a reduced set of the most informative traffic features.
- **Three input modes in the UI:**
  - *Manual entry:* type in the traffic features yourself
  - *Random dataset entry:* sample a real row from the training data
  - *Synthetic entry:* generate a random plausible input

## Tech stack

Python · scikit-learn · pandas · NumPy · Streamlit

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

`python viewParameters.py` prints the feature list the multiclass model expects.

## Structure

```
app.py              Streamlit UI (multiclass detection)
models/             Trained binary and multiclass models + selected feature sets
viewParameters.py   Inspect model input features
```
