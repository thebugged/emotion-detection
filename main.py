import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import streamlit as st

from apps.face import face_page
from apps.audio import audio_page
from streamlit_option_menu import option_menu



st.set_page_config(
    page_title="Emotions",
    page_icon="😑",
    layout="centered",
    initial_sidebar_state="auto",
  )

pages = {
    # "Home": home_page,
    "Face Emotion Detection": face_page,
    "Audio Emotion Detection": audio_page,
}

with st.sidebar:
    
    st.title("Emotions 😑 😁 😭")
    st.sidebar.markdown("---")

    selected_page = option_menu(
            menu_title = None,
            options = list(pages.keys()),
            icons=['person', 'filetype-wav'],
            orientation="vertical",
        )
    st.sidebar.markdown("---")
    st.sidebar.markdown("👨🏾‍💻 by [thebugged](https://github.com/thebugged)")


if selected_page in pages:
    pages[selected_page]()
else:
    st.markdown("### Invalid Page Selected")