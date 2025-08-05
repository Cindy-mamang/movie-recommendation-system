#  Movie Recommendation System

##  Group 5 – DSA 1080 Project

This project was developed as part of the **DSA 1080** coursework by **Group 2**.

###  Team Members
- Cindy Mamang
- Lokomar samuel
- Precious Katkun
- Owen Nyameino


##  Overview

This Movie Recommendation System suggests movies based on **genre similarity**, using **content-based filtering** powered by **cosine similarity**.

The system was built using:
- Python (Pandas, Scikit-learn)
- Streamlit (for web deployment)
- GitHub (for version control and collaboration)

The model and application pipeline include:
- Data cleaning & feature engineering
- Genre-based encoding
- Similarity computation
- Recommender logic
- Model evaluation
- Streamlit web interface

---

##  Project Objective

> "Build a working machine learning pipeline that can recommend similar movies using real-world data and deploy it as an interactive web app."

### Core Steps:
-  Data preprocessing & visualization
-  Feature engineering & binning
- Model training (Random Forest)
-  Movie similarity using cosine similarity
-  Model evaluation (accuracy, precision, recall, F1-score)
-  Deployment using Streamlit Cloud



## 🗂 Project Structure

```bash
movie-recommendation-system/
│
├── streamlit_app.py           # Main app file (for Streamlit)
├── app.ipynb                  # Jupyter notebook used for development
├── tmdb_5000_movies.csv       # Dataset (from TMDB via Kaggle)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation (this file)
📚 Dataset
We used the TMDB 5000 Movies Dataset from Kaggle, which contains information on:

Titles

Genres

Ratings

Release dates

Keywords

Cast & crew

Dataset Source – Kaggle

🖥 How to Run the App Locally
Clone this repository:

bash
Copy code
git clone https://github.com/Cindy-mamang/Movie-Recommendation-System.git
cd Movie-Recommendation-System
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy code
streamlit run streamlit_app.py
The app will launch in your default web browser at http://localhost:8501.

 Model Evaluation
Metric	Value (Example)
Accuracy	0.78
Precision	0.80
Recall	0.79
F1 Score	0.78
MSE	1.12
R² Score	0.83

These scores were obtained using a Random Forest model on binned rating data (vote_average_binned).

 Live Deployment
Our app is hosted on Streamlit Cloud:

 Click here to try the app
(Replace this with your actual link)

Acknowledgments
TMDB for the dataset

Kaggle for data hosting

Streamlit for app deployment

DSA 1080 faculty for guidance

 Contact
For suggestions or collaboration, feel free to reach out:

GitHub: @Cindy-mamang

Email: [mkcindy99@gmail.com]
