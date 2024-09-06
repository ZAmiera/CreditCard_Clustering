import pandas as pd
import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
 
df = pd.read_csv('data5_clean_w5.csv')

def main():
    st.title("Credit Card Customers Clustering Dashboard")
    st.write("This app visualizes clustering data for credit card customers.")

    # Display  
    st.subheader("Customer Data")
    st.dataframe(df)

    # Elbow Method  
    st.subheader("Elbow Method")
    max_k = st.slider("Select the maximum number of clusters (k)", 1, 10, 5)
    
    # Within-Cluster Sum of Squares
    wcss = []
    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(df[['BALANCE', 'PURCHASES', 'CREDIT_LIMIT']])
        wcss.append(kmeans.inertia_)

    # Plotting the Elbow Method
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, max_k + 1), wcss, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS')
    st.pyplot(plt)

    # Scatter plot of the clusters
    st.subheader("Scatter Plot of Clusters")
    k = st.slider("Select number of clusters for scatter plot", 1, max_k, 3)
    
    # Fit KMeans with the selected number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['BALANCE', 'PURCHASES', 'CREDIT_LIMIT']])

    # Plotting the scatter plot
    plt.figure(figsize=(10, 5))
    plt.scatter(df['BALANCE'], df['CREDIT_LIMIT'], c=df['Cluster'], cmap='viridis', alpha=0.6)
    plt.title(f'Scatter Plot of Credit Card Customers (k={k})')
    plt.xlabel('Balance')
    plt.ylabel('Credit Limit')
    plt.colorbar(label='Cluster')
    st.pyplot(plt)

if __name__ == "__main__":
    main()
