import streamlit as st
from path import Path
import os
from scoring_app import run_scoring_app


print('current directory: ',os.getcwd())

def main():
    st.title("Your Marketing App for Customer´s Churn Prediction!")

    menu = ["About", "Churn Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "About":
        st.subheader("About")
        st.markdown(Path('.\web_app\About.md').read_text())
    
    elif choice == "Churn Prediction":
        st.subheader('Predict Churn')
        run_scoring_app()
    
if __name__ == "__main__":
    main()



