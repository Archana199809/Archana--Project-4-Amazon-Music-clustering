# Archana--Project-4-Amazon-Music-clustering
Amazon Music clustering

# Amazon-Music-Clustering
**Problem Statement:**
---

With millions of songs available on platforms like Amazon, manually categorizing tracks into genres is impractical. 

The goal of this project is to automatically group similar songs based on their audio characteristics such as tempo, energy, danceability, etc., using clustering techniques.

This helps in Playlist curation.

**Skills TakeAway from this Project:**
---
--> Data Exploration & Cleaning

--> Feature Engineering & Data Normalization

--> K-Means Clustering & Elbow Method

--> Cluster Evaluation (Silhouette Score and Davies–Bouldin Index)

--> Dimensionality Reduction (PCA)

--> Cluster Visualization & Interpretation

--> Python for Machine Learning (Pandas, Numpy, Scikit learn, Matplotlib, Seaborn)

--> Streamlit



 Tech Stack
---
--> Python 3.10+

--> Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, streamlit, joblib

--> IDE: Jupyter Notebook / VS Code

--> Visualization: Streamlit Dashboard

**Import Libraries:**
---
**  Data Handling **

--> import pandas as pd

--> import numpy as np

** Visualization **

--> import matplotlib.pyplot as plt

--> import seaborn as sns

Machine Learning & Clustering 
---
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score, davies_bouldin_score

**Approach:**
---
--> Data Loading & Exploration – Imported the Amazon Music dataset and explored song-level attributes like energy, tempo, and valence using pandas and seaborn.

--> Data Cleaning & Feature Engineering – Removed irrelevant columns,Data is already non null.

--> Feature Scaling & Transformation – Normalized all numeric features using StandardScaler to prepare for clustering.

--> Dimensionality Reduction (PCA) – Reduced high-dimensional data into 2 principal components to visualize patterns and variance among songs.

--> Clustering Model Development – Applied KMeans to group songs

--> Cluster Evaluation & Profiling – Evaluated models using Silhouette Score and Davies–Bouldin Index

--> Visualization & Insights – Visualized clusters through PCA scatter plots and heatmaps; extracted feature-wise summaries for each cluster.

--> Dashboard Development – Built an interactive Streamlit app to explore clusters dynamically, vis
ualize feature comparisons, and display top 10 songs per cluster.

**Snapshot:**
---
<img width="1902" height="1001" alt="1" src="https://github.com/user-attachments/assets/bac726a2-ac6c-4bbe-abe5-b78d50f49eb8" />
<img width="965" height="757" alt="2" src="https://github.com/user-attachments/assets/f9862680-b063-4469-82a4-ac0544e528c3" />

<img width="955" height="753" alt="3" src="https://github.com/user-attachments/assets/1497be13-a2bd-494e-9e9e-3d634fba8619" />

<img width="1892" height="941" alt="4" src="https://github.com/user-attachments/assets/645ecde3-242a-4a20-9282-2eaedcf52388" />




**Insights of this project**
--


**User Interface/Implementation:**
The design and functionality of the interactive interface, which allows users to select clusters, apply feature filters, and view summary statistics and comparisons.
**Results/Insights:**
The key findings from the analysis, such as how different clusters appeal to different moods or use cases, and how the feature means help define these distinctions.
**Potential Applications:**
Real-world applications of this project, such as personalized music recommendations or playlist generation for specific events










