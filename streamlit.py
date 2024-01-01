# Created by triloki at 01-01-2024

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space


# sidebar contents
with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM_Powered chatbot built using:
    - [Streamlit](https://streamlit.io)
    - [LangChain](https://python.langchain.com)
    - [OpenAI](https://platform.openai.com/docs/models) LLM Model
    ''')
    add_vertical_space(5)
    st.write('Made with by Promt Engineer')