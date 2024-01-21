# Created by trilo at 21-01-2024
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Installation
# pip install streamlit

# 2. Hello World Example
st.title("Hello Streamlit!")
st.write("This is a basic Streamlit app.")

# 3. Widgets
# Text Input
user_input = st.text_input("Enter your name", "Triloki Gupta")
st.write("You entered:", user_input)

# Slider
age = st.slider("Select your age", 0, 100, 25)
st.write("Your age:", age)

# Button
if st.button("Click me"):
    st.write("Button clicked!")

# 4. Data Visualization
data = np.random.randn(100)
st.line_chart(data)

# 5. DataFrame Display
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
st.dataframe(df)

# 6. Layout Customization
# Sidebar
st.sidebar.title("Options")
option = st.sidebar.selectbox("Choose an option", ["Option 1", "Option 2"])

# Main content
st.title("Main Content")
st.write(f"You selected: {option}")
