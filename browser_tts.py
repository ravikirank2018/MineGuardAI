import streamlit as st
import streamlit.components.v1 as components

def speak_text(text: str):
    """Use browser TTS to speak the given text via JS injection."""
    js = f'''
    <script>
    var msg = new SpeechSynthesisUtterance({text!r});
    window.speechSynthesis.speak(msg);
    </script>
    '''
    components.html(js, height=0, width=0)
