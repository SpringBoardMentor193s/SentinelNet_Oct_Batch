import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

# Set a custom style
sns.set_theme(style="darkgrid")

def display_prediction_stats(predictions):
    """
    Displays a bar chart of prediction counts.
    """
    unique_classes, counts = np.unique(predictions, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=unique_classes, y=counts, palette="viridis", ax=ax)
    
    ax.set_xlabel("Detected Class", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Traffic Classification Results", fontsize=14, fontweight='bold')
    
    st.pyplot(fig)
