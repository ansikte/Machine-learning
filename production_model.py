import pandas as pd
import joblib

# Laddar in testdata
test_samples = pd.read_csv('test_samples.csv')

# Laddar in GBM-modellen och scalern
model = joblib.load('gbm_model.pkl')  # Uppdaterad till GBM-modellen
scaler = joblib.load('scaler.pkl')

# Skalar testdata
X_test = test_samples.drop('cardio', axis=1)  # Antag att 'cardio' är målvariabeln
X_test_scaled = scaler.transform(X_test)

# Gör förutsägelser och beräknar sannolikheter
predictions = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)

# Skapar en DataFrame för att hålla resultatet
results_df = pd.DataFrame(probabilities, columns=['probability class 0', 'probability class 1'])
results_df['prediction'] = predictions

# Exporterar resultaten till en CSV-fil
results_df.to_csv('prediction_gbm.csv', index=False)
