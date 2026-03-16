Mobile Phone Price Prediction

This project uses Machine Learning to predict the price range of a mobile phone based on its specifications such as battery power, RAM, processor, camera, and screen size.

The model analyzes different hardware features and classifies the phone into one of four price categories.

🚀 Features

Data preprocessing and scaling

Machine learning model training

Price range prediction

Command line input for phone specifications

Probability output for each price category

📊 Price Categories
Class	Price Range
0	Low Cost
1	Medium Cost
2	High Cost
3	Very High Cost
⚙️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

Joblib

📁 Project Files
mobile_phone_pricing/
│
├── mobile_data.csv
├── train_model.py
├── predict_price.py
├── requirements.txt
│
├── mobile_price_model.pkl
├── scaler.pkl
├── feature_names.pkl
└── mobile_price_results.png
▶️ How to Run

Install dependencies:

pip install -r requirements.txt

Train the model:

python train_model.py

Run prediction:

python predict_price.py

Enter phone specifications and the model will predict the price range.

👨‍💻 Author

Sanjay Vanol
