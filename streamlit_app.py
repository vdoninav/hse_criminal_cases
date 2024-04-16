import streamlit as st
from pymystem3 import Mystem
import html

from predict import predict
from word_count import word_count


# from summarize import summarize


def main():
    st.set_page_config(layout='wide')
    st.title("HSE Criminal Cases")
    st.write("Welcome")
    st.write("Let's predict something...")

    user_input = st.text_area("To Predict:")
    # morph analysis
    # might take time
    morph = Mystem()

    if user_input:
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
        # TODO: Remove special symbols, i.e. '\n', '\t'
        st.markdown(html_output, unsafe_allow_html=True)

        # Legend generation
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


if __name__ == "__main__":
    main()
