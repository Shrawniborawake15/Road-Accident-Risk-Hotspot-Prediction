**Road Accident Risk Hotspot Prediction**


Identifies road locations and time periods with higher accident risk so safety teams can prioritise inspection. Dept. of AI & Data Science, Sanjivani University - PBL project.

**Dataset**


Demo mode uses synthetic accident records generated inside the app. For real results, upload a row-level accident CSV (date/time, latitude, longitude; optional weather, road type, severity). Descriptive national statistics come from MoRTH, Road Accidents in India 2024, Table 3.8.

**Pipeline**


Problem definition -> data understanding -> cleaning -> EDA -> feature engineering (grid cells, monthly counts, lag features) -> Linear Regression (accident counts) / Logistic Regression (high-risk label, chronological split) -> evaluation (MAE, RMSE, R2, precision, recall, F1, confusion matrix) -> hotspot map -> Streamlit app.

**Results (test set)**


To be filled after running on the final real dataset.

**Run**


pip install -r requirements.txt

streamlit run app.py


**Limitations**


Weather totals ignore traffic exposure and do not prove causation.
The high-risk label is a proxy based on historical counts.
Demo data is synthetic; report only metrics from real data.

**Team contributions**


Nikita Jori :- implementation (data cleaning, EDA, feature engineering, Linear and Logistic Regression models, hotspot map, Streamlit app), GitHub repository

Shrawani Borawake :- project idea, problem definition, presentation (PPT), dataset search and testing

Samruddhi Bodakhe :- documentation support, app testing, screenshots, proofreading
