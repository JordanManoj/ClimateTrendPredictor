#  GlobalTempInsight

### Predicting Global Temperature Trends Using Machine Learning & Deep Learning

**GlobalTempInsight** is a data-driven project focused on analyzing historical global temperature data to predict future temperature trends. By combining traditional Machine Learning models with advanced Deep Learning architectures, this project provides insights into long-term climate patterns and global warming.

---

##  Dataset

**Dataset:** [GlobalLandTemperaturesByCountry.csv](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data)

The dataset includes:
- **dt** – Date of record (from 1743 to 2013)  
- **AverageTemperature** – Average recorded temperature  
- **Country** – Country name for which data was recorded  

The dataset contains over 500,000 records of temperature readings across countries.

---

##  Data Preprocessing

Preprocessing plays a crucial role in improving model accuracy and stability. The following steps were applied:

1. **Missing Value Handling:**  
   Missing temperature values were filled using **linear interpolation**.

2. **Outlier Removal:**  
   Applied **Z-score filtering** to remove anomalous readings.

3. **Datetime Conversion:**  
   Converted `dt` into datetime format and extracted **Year** and **Month** features.

4. **Aggregation:**  
   Grouped data to compute **monthly** and **yearly average temperatures**.

5. **Feature Engineering:**  
   - **Rolling Mean & Standard Deviation (12-month window)** – Captures trends  
   - **Cyclical Month Encoding** using sine and cosine transformations  
   - **Lag Features** to capture temporal dependencies  
   - **MinMax Scaling** to normalize input values

---

##  Models Trained

The project evaluates both **Machine Learning** and **Deep Learning** approaches:

| Model | MAE | RMSE | R² Score | Description |
|:--|:--|:--|:--|:--|
| **Linear Regression** | 3.894 | 3.960 | -122.505 | Simple linear relationship baseline |
| **Random Forest Regressor** | 0.329 | 0.425 | -0.422 | Ensemble-based tree model |
| **Support Vector Regressor (SVR)** | 1.214 | 1.571 | -19.156 | Captures non-linear patterns |
| **LSTM (Deep Learning)** | **0.214** | **0.258** | **0.964** | Learns temporal dependencies effectively |

 **Result:**  
The **LSTM model** achieved the best performance, demonstrating its ability to capture long-term temperature trends and sequential patterns in climate data.

---

##  Results & Visualization

- Plotted **Historical vs Predicted Temperature Trends**.  
- **LSTM** model shows a smooth and realistic upward temperature forecast.  
- Future predictions indicate a **continued warming trend** globally.  

---

##  Future Scope

This project can be extended into a **Streamlit web application** to make temperature forecasting **interactive and accessible**.

**App Features (Proposed):**
- Upload temperature datasets  
- Visualize preprocessing results  
- Choose model (ML/DL)  
- Display predictions and trend graphs in real-time  

---

##  Tech Stack

- **Languages:** Python  
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow/keras  
- **Visualization:** Matplotlib, Seaborn  


---

##  Summary

This project demonstrates how combining **data preprocessing, feature engineering**, and **hybrid ML-DL techniques** can improve climate trend predictions.  
It highlights how AI can assist researchers and policymakers in understanding **climate change patterns** with data-driven insights.

---

### 👨‍💻 Author
**Jordan Manoj**  
*Mini Project – Global Temperature Prediction Using Machine Learning & Deep Learning*

---
