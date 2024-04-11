import streamlit as st

from predict import predict


def main():
    st.set_page_config(layout='wide')
    st.title("HSE Criminal Cases")
    st.write("Welcome")
    st.write("Let's predict something...")
    text_input = st.text_area("To Predict:", "Your prompt here")
    prediction = predict(text_input)
    st.write(prediction)


if __name__ == "__main__":
    main()
