from clickbait import *
import streamlit as st


def main():
    st.title('Clickbait Classification')

    #input
    text_input = st.text_input('Headline News')

    #prediction
    if st.button('Predict'):
        list_text_input = []
        list_text_input.append(text_input)
        pred_result = predict_spam(list_text_input)[0][0]
        # st.success('{headline} \n**{pred_result:.2%}** clickbait'.format(headline=text_input,pred_result=pred_result))

        st.markdown('<p style="font-size:50px">{pred_result:.2%} clickbait</p>'.format(pred_result=pred_result), unsafe_allow_html=True)

if __name__=='__main__':
    main()