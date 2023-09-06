import streamlit as st
import joblib as jb
def main(model,cv):
    st.title('Spam Filter'.upper())
    st.markdown("<h1 style = 'text-align : center; font-size: 25px; color: blue ;'></h1>",unsafe_allow_html=True)
    hide_github_icon = """
    <style>
    .css-ch5dnh{
    display: none;
    }
    </style>"""
    st.markdown(hide_github_icon, unsafe_allow_html=True)
    result =''

    text_message = ''
    text_message = st.text_input('Enter your text message')
    if st.button('Predict'):
        if text_message =='':
            st.write("Please Enter any text")
        else:
            text = cv.transform([text_message])
            prediction = model.predict(text)
            if prediction[0] == 1:
                result = 'Spam'
            else:
                result = 'Ham(Not Spam)'

            st.success('Prediction : {}'.format(result))
            # st.write(st.__version__) # give current version of pkg that is being used
            # st.write(jb.__version__)



model_load = jb.load('Spam_model.pkl')
cv = jb.load('C_vector.pkl')
main(model_load,cv)
