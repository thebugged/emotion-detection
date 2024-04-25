import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import streamlit as st

from apps.face import face_page
from apps.speech import speech_page
from streamlit_option_menu import option_menu



st.set_page_config(
    page_title="Emotions",
    page_icon="😑",
    layout="centered",
    initial_sidebar_state="collapsed",
  )

pages = {
    # "Home": home_page,
    "Face Emotion Detection": face_page,
    "Audio Emotion Detection": speech_page,
}

selected_page = option_menu(
        menu_title = None,
        options = list(pages.keys()),
        icons=['person', 'filetype-wav'],
        orientation="horizontal",
    )

with st.sidebar:
    
    st.title("Emotions")
    st.sidebar.markdown("---")
    st.sidebar.markdown("made by [thebugged](https://github.com/thebugged)")


if selected_page in pages:
    pages[selected_page]()
else:
    st.markdown("### Invalid Page Selected")