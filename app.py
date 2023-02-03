from predeploy import *

# test_text = ['Viral Video Review Beli Udara Bekasi Cuma Goceng, yang Nonton 8,5 Juta Netizen!']
# print(predict_spam(test_text)[0][0])

import streamlit as st
st.title('Clickbait Classification for News Headline')
text_input = st.text_input('News Headline')

if st.button('Predict'):
        list_text_input = []
        list_text_input.append(text_input)
        pred_result = predict_spam(list_text_input)[0][0]
        # st.success('{headline} \n**{pred_result:.2%}** clickbait'.format(headline=text_input,pred_result=pred_result))

        st.markdown('<p style="font-size:50px">{pred_result:.2%} clickbait</p>'.format(pred_result=pred_result), unsafe_allow_html=True)