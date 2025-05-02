import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Streamlit Cloud Demo",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add a title
st.title("Streamlit Cloud Demo Application")

# Add sidebar
st.sidebar.header("Dashboard Settings")

# Add user input parameters in sidebar
option = st.sidebar.selectbox(
    'Select visualization type:',
    ['Line Chart', 'Bar Chart', 'Scatter Plot', 'Histogram']
)

num_points = st.sidebar.slider('Number of data points', 10, 1000, 100)
random_seed = st.sidebar.number_input('Random seed', min_value=0, max_value=1000, value=42)

# Main content
st.header("Interactive Data Visualization")
st.write("This is a demo application showing how to create a Streamlit app for cloud deployment.")

# Generate random data
np.random.seed(random_seed)
data = pd.DataFrame({
    'x': range(num_points),
    'y': np.random.randn(num_points).cumsum(),
    'category': np.random.choice(['A', 'B', 'C', 'D'], size=num_points)
})

# Display the data
with st.expander("View Raw Data"):
    st.dataframe(data)

# Allow users to download the data
st.download_button(
    label="Download data as CSV",
    data=data.to_csv().encode('utf-8'),
    file_name='data.csv',
    mime='text/csv',
)

# Create different visualizations based on user selection
st.subheader(f"Visualization: {option}")

if option == 'Line Chart':
    fig = px.line(data, x='x', y='y', color='category',
                 title='Line Chart of Random Data')
    st.plotly_chart(fig, use_container_width=True)
    
elif option == 'Bar Chart':
    fig = px.bar(data, x='category', y='y', color='category',
                title='Bar Chart by Category')
    st.plotly_chart(fig, use_container_width=True)
    
elif option == 'Scatter Plot':
    fig = px.scatter(data, x='x', y='y', color='category', size=np.abs(data['y'])*3,
                    title='Scatter Plot with Size Based on Value')
    st.plotly_chart(fig, use_container_width=True)
    
elif option == 'Histogram':
    fig = px.histogram(data, x='y', color='category',
                      title='Distribution of Values by Category')
    st.plotly_chart(fig, use_container_width=True)

# Add metrics section
st.header("Summary Metrics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Mean Value", round(data['y'].mean(), 2), 
              round(data['y'].mean() - data['y'].median(), 2))
with col2:
    st.metric("Max Value", round(data['y'].max(), 2))
with col3:
    st.metric("Min Value", round(data['y'].min(), 2))

# Add an interactive section
st.header("Interactive Filter")
selected_category = st.multiselect("Select categories to display:", 
                                  options=data['category'].unique(),
                                  default=data['category'].unique())

if selected_category:
    filtered_data = data[data['category'].isin(selected_category)]
    st.line_chart(filtered_data.pivot(index='x', columns='category', values='y'))
else:
    st.warning("Please select at least one category.")

# Add information for deployment
st.header("Deployment Instructions")
st.info("""
To deploy this app to Streamlit Cloud:
1. Save this code in a file named `app.py`
2. Create a `requirements.txt` file with the necessary dependencies
3. Push the code to a GitHub repository
4. Sign in to [Streamlit Cloud](https://streamlit.io/cloud)
5. Deploy your app by connecting to your GitHub repository
""")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit")
