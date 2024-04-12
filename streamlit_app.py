import streamlit as st

from predict import predict


def main():
    st.set_page_config(layout='wide')
    st.title("HSE Criminal Cases")
    st.write("Welcome")
    st.write("Let's predict something...")

    user_input = st.text_area("To Predict:")

    if user_input:
        # define dictionary to map entity groups to color
        groups_to_color = {"IND": "green", "LE": "blue", "PEN": "orange", "LAW": "red", "CR": "purple"}
        groups_to_label = {"IND": "Individual", "LE": "Legal Entity", "PEN": "Penalty", "LAW": "Law", "CR": "Crime"}

        # Prediction result as dictionary list
        predict_result = predict(user_input)

        output_parts = []
        last_end = 0

        # Add color to entity words
        for result in predict_result:
            start = result['start']
            end = result['end']

            color = groups_to_color.get(result['entity_group'], "black")
            colored_part = f'<span style="color:{color};">{user_input[start:end]}</span>'

            # Add nonentity text and entity text (colored) to output_parts
            output_parts.append(user_input[last_end:start])
            output_parts.append(colored_part)

            last_end = end

        # Append remaining text
        output_parts.append(user_input[last_end:])

        # Join all parts
        html_output = ''.join(output_parts)
        # Render it in markdown with html enabled
        st.markdown(html_output, unsafe_allow_html=True)

        # legend
        for group, color in groups_to_color.items():
            label = groups_to_label.get(group, group)
            st.markdown(f'<span style="color:{color};">{group} - {label}</span>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
