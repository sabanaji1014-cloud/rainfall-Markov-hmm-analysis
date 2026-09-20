# NOTE: kept exactly as originally written (Part 3 of the report). This
# script was developed in Google Colab against an external weather dataset
# ('/content/data_new.xlsx', with an added 'Weather_Code' column) that is
# not included in this repository. It also still references a couple of
# undefined variables (df_new, df_x_matrix) from that notebook. Left
# unedited per the "don't change the code" instruction -- see the README
# for details.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
import pandas as pd

df = pd.read_excel('/content/data_new.xlsx', index_col=0) 
# display(df)

df['Weather_Code'] = df['Weather_Code'].replace(np.nan, 0)
df = df.replace(np.nan, 0)
df.replace(np.nan, 0, inplace=True)
df_x = df_new.drop('Rain',axis = 1)

model = hmm.GaussianHMM(n_components=6, covariance_type='full', n_iter=1000)

# Fit the model to the data
model.fit(df_x_matrix)

# Calculate the emission probabilities for the sample data
emission_probs = model.predict_proba(df_x_matrix)

# Display the first few emission probabilities
print(emission_probs[:6])

# Define the state space
states = ["Rainy"]
n_states = len(states)
print('Number of hidden states :',n_states)
# Define the observation space
observations = ['Min',  'Wind', 'Humidity', 'Cloud',  'Pressure', 'Weather_Code']
n_observations = len(observations)
print('Number of observations  :',n_observations)

# Define the initial state distribution
state_probability = np.array([[0.1, 0.2, 0.2, 0.1, 0.2, 0.2],[0.2, 0.3, 0.1, 0.1, 0.1, 0.2]])
print("State probability: ", state_probability)

# Define the state transition probabilities
transition_probability = np.array([[0.859649, 0.114035, 0.017544, 0.00, 0.0, 0.008772],
[0.619048, 0.333333, 0.047619, 0.00, 0.0, 0.000000],
[0.500000, 0.000000, 0.250000, 0.25, 0.0, 0.000000],
[0.000000, 0.000000, 0.000000, 0.00, 1.0, 0.000000],
[0.000000, 1.000000, 0.000000, 0.00, 0.0, 0.000000],
[0.000000, 0.000000, 0.000000, 1.00, 0.0, 0.00000]])
print("\nTransition probability:\n", transition_probability)
# Define the observation likelihoods
emission_probability= emission_probs[:6]
print("\nEmission probability:\n", emission_probability)

model = hmm.CategoricalHMM(n_components=6)
model.startprob_ = state_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

rainy_state = (df['Rain'] > 0).astype(int)

features = df[['Max', 'Min', 'Wind', 'Humidity', 'Cloud', 'Pressure']]
target = df['Rain']

# Step 5: Train the HMM
# Initialize the HMM
model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000)

# Fit the model
model.fit(features)

# Step 6: Predict the sequence
# Predict the hidden states
hidden_states = model.predict(features)

# Step 7: Map the hidden states to 0 and 1 based on the rainy state
# Assuming 'Rain' > 0 indicates a rainy state
rainy_state = (df['Rain'] > 0).astype(int)

# Map the hidden states to 0 and 1
state_mapping = {0: 0, 1: 1}  # Adjust this mapping based on the model's output
predicted_sequence = [state_mapping[state] for state in hidden_states]

# Print the predicted sequence
print(predicted_sequence)
print("Most likely hidden states:", hidden_states)