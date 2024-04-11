import streamlit as st
import pandas as pd

from predict import predict


def main():
    st.title("HSE Criminal Cases")
    st.write("Welcome ")
    st.write("Let's predict something...")
    text_input = st.text_input("Enter your input here")
    prediction = predict(text_input)
    st.write("The prediction is:", prediction)


if __name__ == "__main__":
    main()
