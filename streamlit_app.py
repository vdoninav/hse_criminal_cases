import streamlit as st
import html

from predict import predict
from word_count import word_count
from summarize import summarize


def page_about():
    st.title("HSE Criminal Cases")
    st.subheader("About this coursework")

    st.subheader("Created by :green[***Lebedyuk Eva***], :green[***Vdonin Aleksei***]")

    st.subheader("[GitHub](https://github.com/vdoninav/hse_criminal_cases)")


def page_summarize():
    # st.set_page_config(layout='wide')
    st.title("HSE Criminal Cases")
    st.write("Welcome")
    st.write("Let's summarize something...")

    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    user_input = st.text_area("Input:", value=st.session_state.user_input)
    st.session_state.user_input = user_input

    if user_input:
        # st.markdown("---")
        summarized_text = summarize([user_input])
        s_len = len(summarized_text)
        u_len = len(user_input)
        st.write(summarized_text)
        st.markdown("---")
        st.write(
            f"original text length - :blue[{u_len}] vs :green[{s_len}] - summarized text length")
        # st.write(f"summarized text length: :green[{len(summarized_text)}]")
        st.write(f"Shortened by :green[{100 * (1 - s_len / u_len)}%]")


def page_nlp():
    # st.set_page_config(layout='wide')
    st.title("HSE Criminal Cases")
    st.write("Welcome")
    st.write("Let's predict something...")

    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    user_input = st.text_area("Input:", value=st.session_state.user_input)
    st.session_state.user_input = user_input

    if user_input:
        # st.markdown("---")
        # define dictionary to map entity groups to color
        groups_to_color = {"IND": "green", "LE": "blue", "PEN": "orange", "LAW": "red", "CR": "purple"}
        groups_to_label = {"IND": "Individual", "LE": "Legal Entity", "PEN": "Penalty", "LAW": "Law", "CR": "Crime"}

        # Doesn't work as expected
        # user_input = summarize(user_input)
        # Prediction result as dictionary list
        predict_result = predict(user_input)

        output_parts = []
        last_end = 0
        words_predicted = []
        word_groups = {}

        # Add color to entity words
        for result in predict_result:
            start = result['start']
            end = result['end']
            word = html.escape(user_input[start:end])
            words_predicted.append(word)

            # save group of the word
            word_groups[word] = result['entity_group']

            color = groups_to_color.get(result['entity_group'], "black")
            colored_part = f'<span style="color:{color};">{word}</span>'

            # Add nonentity text and entity text (colored) to output_parts
            output_parts.append(user_input[last_end:start])
            output_parts.append(colored_part)

            last_end = end

        # Append remaining text
        output_parts.append(user_input[last_end:])
        # Join all parts
        html_output = ''.join(output_parts)

        # Legend generation
        legend_parts = []
        for group, color in groups_to_color.items():
            label = groups_to_label.get(group, group)
            legend_part = f'<span style="color:{color};">{group} - {label}</span>'
            legend_parts.append(legend_part)
        st.markdown('&nbsp; | &nbsp;'.join(legend_parts), unsafe_allow_html=True)

        # Render it in markdown with html enabled
        st.markdown(html_output, unsafe_allow_html=True)

        if predict_result:
            st.markdown("---")
        # Legend generation
        if len(user_input) >= 500:

            legend_parts = []
            for group, color in groups_to_color.items():
                label = groups_to_label.get(group, group)
                legend_part = f'<span style="color:{color};">{group} - {label}</span>'
                legend_parts.append(legend_part)
            st.markdown('&nbsp; | &nbsp;'.join(legend_parts), unsafe_allow_html=True)

        # Word count displaying
        word_counts = word_count(words_predicted)
        words_predicted_lower = [word.lower() for word in words_predicted]
        words_encountered = []
        for word in set(words_predicted):
            if word.lower() in words_encountered:
                continue
            words_encountered.append(word.lower())
            color = groups_to_color.get(word_groups[word], "black")
            colored_word = f'<span style="color:{color};">{word.upper()}</span>'
            st.markdown(
                f'слово {colored_word} было найдено в тексте {words_predicted_lower.count(word.lower())} раз, в аналогичных текстах {word_counts.get(word, 0)} раз',
                unsafe_allow_html=True)
            # TODO: заменить 'слово' на IND = личность, CR = преступление и т.д.
        # for word in set(words_predicted_lower):
        #     st.write(
        #         f'слово {word.upper()} было найдено в тексте {words_predicted.count(word)} раз, в аналогичных текстах {word_counts.get(word, 0)} раз')


def main():
    st.set_page_config(layout='wide')
    st.sidebar.title("Navigation")

    nav_option = st.sidebar.radio("Go To", ["***NLP***", "***Summarize***", "***About***"],
                                  captions=["Classify identities in text", "Summarize large text",
                                            "About this project"], label_visibility="hidden")

    if nav_option == "***NLP***":
        page_nlp()
    elif nav_option == "***Summarize***":
        page_summarize()
    elif nav_option == "***About***":
        page_about()
    else:
        page_nlp()


if __name__ == "__main__":
    main()
