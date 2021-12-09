import streamlit as st
from multiapp import MultiApp
from apps import home, data, model # import your app modules here

app = MultiApp()

st.markdown("""
# Professor
""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Upload Data Saham", data.app)
app.add_app("Prediksi Saham", model.app)
# The main app
app.run()
