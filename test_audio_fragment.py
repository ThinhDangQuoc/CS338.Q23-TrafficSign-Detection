import streamlit as st
import time

if "start" not in st.session_state:
    st.session_state.start = time.time()

st.write("Main app")

@st.fragment(run_every="1s")
def audio_frag():
    st.write(f"Fragment time: {time.time()}")
    st.audio("test.mp3", autoplay=True)

audio_frag()
