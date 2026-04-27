DeepGuard: Multi-Layered Neural Fraud Detection
1. Project Overview & Mission
DeepGuard is a deep learning framework designed to solve one of the most persistent challenges in the financial sector: Credit Card Fraud.

Traditional fraud detection systems rely on "Static Rules" (e.g., Flag transaction if Amount > $5,000). These systems are rigid and easily bypassed by modern cybercriminals. DeepGuard moves beyond rules into Behavioral Intelligence. By analyzing the relationship between a user’s geographic history, transaction timing, and spending velocity, the model identifies "impossible" patterns that no human could spot in real-time.

2. The Real-World Problem vs. Solution
The Problem: The "Silent" Fraud
Modern fraud doesn't always look like a massive spending spree. Attackers often perform:

Probing Attacks: Small, innocuous transactions to verify if a card is active.

Velocity Attacks: Rapid-fire transactions across different physical locations using stolen credentials.

Geographic Anomaly: Transactions occurring in a location physically unreachable given the user's previous location.

The Solution: Temporal-Geospatial Analysis
DeepGuard solves this by treating every transaction as part of a Sequence. It calculates the physical feasibility of every swipe. If a user buys coffee in London and then tries to buy a laptop in New York 30 minutes later, the model identifies a Velocity Violation and flags the event.

3. Data Engineering & Mathematical Foundation
Raw data is rarely useful for Deep Learning. This project features a robust feature engineering pipeline that transforms raw timestamps and GPS coordinates into "High-Signal" features.

A. The Haversine Formula (Geospatial Logic)
To understand distance on a sphere (Earth), we cannot use simple Euclidean distance. We implement the Haversine Formula to calculate the shortest distance between two points over the earth's surface:

d=2rarcsin( 
sin 
2
 ( 
2
Δϕ
​
 )+cosϕ 
1
​
 cosϕ 
2
​
 sin 
2
 ( 
2
Δλ
​
 )

​
 )
Where:

ϕ is Latitude, λ is Longitude, and r is the Earth's radius (6,371 km).

B. Engineered Features
Feature	Description	Fraud Signal
Distance (km)	Distance from the last transaction.	High values indicate stolen card movement.
Time Delta (min)	Minutes since the last transaction.	Low values indicate bot-driven "burst" attacks.
Velocity (km/min)	Distance divided by Time.	Values > 15 km/min suggest "impossible travel."
Amount Z-Score	Deviation from the user's average spend.	Identifies "out-of-character" high-value fraud.

Export to Sheets

4. Deep Learning Architecture
The core of DeepGuard is a Multi-Layer Perceptron (MLP) built using the Keras Sequential API. It is designed to find non-linear correlations between the features listed above.

Neural Network Layers:
Input Layer: Accepts the 4 engineered behavioral features.

Dense Hidden Layer (128 units): Uses ReLU activation to learn complex patterns.

Dropout Layer (0.3): A regularization technique that randomly "shuts off" 30% of neurons during training to prevent the model from memorizing the data (Overfitting).

Dense Hidden Layer (64 units): Refines the features extracted by the previous layer.

Output Layer (1 unit): Uses a Sigmoid activation function to output a probability between 0.0 (Legitimate) and 1.0 (Fraud).

5. Challenges & Technical Solutions
Challenge 1: The "Needle in a Haystack" (Class Imbalance)
In a real bank, 99.9% of transactions are legitimate. If a model simply guesses "Not Fraud" every time, it would have 99.9% accuracy but catch zero fraud.

Solution: We prioritized Recall (catching fraud) over Accuracy. We used Stratified Splitting to ensure both our training and testing sets had a proportional representation of fraud cases.

Challenge 2: Scale Variance
A transaction amount might be $500, but a time-delta might be 0.5 minutes. If fed raw into the network, the $500 would "overpower" the 0.5.

Solution: Applied StandardScaler to normalize all inputs to a mean of 0 and a standard deviation of 1.

Challenge 3: Cold Start Problem
What happens when a user makes their very first transaction? There is no "previous location" to calculate distance.

Solution: Implemented logic to handle NaN values for first-time transactions, ensuring the model doesn't crash or generate false alerts for new users.

6. Performance Evaluation
We don't just look at Accuracy. We use three specific metrics to judge success:

ROC-AUC Score: Measures how well the model distinguishes between the two classes. A score near 1.0 indicates near-perfect separation.

Confusion Matrix: A visual breakdown showing:

True Positives: Fraud we caught.

False Positives: Innocent customers we accidentally blocked.

False Negatives: The "Dangerous" category—fraud we missed.

Recall: The percentage of total fraud cases that our model successfully flagged.

7. Installation & Usage
Bash

# 1. Clone the repository
git clone https://github.com/yourusername/DeepGuard-Fraud-Detection.git

# 2. Install required libraries
pip install tensorflow pandas numpy scikit-learn seaborn matplotlib

# 3. Run the pipeline
python fraud_detection_main.py
8. Future Roadmap
LSTM Integration: Upgrade to Long Short-Term Memory networks to analyze the last 10 transactions as a sequence rather than just the last 1.

SHAP Explainability: Adding a layer to explain why a transaction was blocked (e.g., "Blocked due to high velocity").

FastAPI Deployment: Turning this model into a live API that can process transactions in milliseconds.
