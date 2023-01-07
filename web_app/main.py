import streamlit as st
from path import Path
from scoring_app import run_scoring_app

def main():
    st.title("Your Marketing App for Customer´s Churn Prediction!")

    menu = ["About", "Churn Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "About":
        st.subheader("About")
        # depending on deployment i.e. local, docker or streamlit clout try different paths
        try:
            st.markdown(Path('About.md').read_text())
        except:
            st.markdown(Path('/app/marketing-sales-customer-churn-prediction-webapp/web_app/About.md').read_text())
    
    elif choice == "Churn Prediction":
        st.subheader('Predict Churn')
        run_scoring_app()
    
if __name__ == "__main__":
    main()



