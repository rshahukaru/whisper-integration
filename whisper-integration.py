import streamlit as st
from openai import OpenAI
import os
from audio_recorder_streamlit import audio_recorder
import base64
import time

# Dictionary of countries and their primary languages
COUNTRY_LANGUAGES = {
    "Spain": "Spanish",
    "France": "French",
    "Germany": "German",
    "Italy": "Italian",
    "Japan": "Japanese",
    "China": "Chinese",
    "Brazil": "Portuguese",
    "North India": "Hindi",
    "Telangana / Andhra Pradesh": "Telugu",
    # Add more countries and their languages as needed
}

# Initialize OpenAI client
def setup_openai_client():
    if 'openai_client' not in st.session_state:
        api_key = st.secrets["openai_api_key"]
        st.session_state.openai_client = OpenAI(api_key=api_key)
    return st.session_state.openai_client

def cleanup_audio_files():
    """Clean up any temporary audio files"""
    try:
        for file in os.listdir():
            if (file.startswith("audio_input_") or file.startswith("audio_response_")) and file.endswith(".mp3"):
                try:
                    os.remove(file)
                except Exception as e:
                    st.warning(f"Could not remove audio file {file}: {str(e)}")
    except Exception as e:
        st.warning(f"Error during audio cleanup: {str(e)}")

def transcribe_audio(client, audio_path):
    """Transcribe audio file to text"""
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcript.text

def text_to_audio(client, text, audio_path, voice="nova"):
    """Convert text to speech"""
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text
    )
    response.stream_to_file(audio_path)

def auto_play_audio(audio_file):
    """Create an auto-playing audio element"""
    if os.path.exists(audio_file):
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        audio_html = f'<audio src="data:audio/mp3;base64,{base64_audio}" controls autoplay>'
        st.markdown(audio_html, unsafe_allow_html=True)

def translate_text(client, text, target_language):
    """Translate text using GPT-4"""
    messages = [
        {"role": "system", "content": f"You are a translator. Translate the following text to {target_language}. Maintain the tone and meaning of the original text. Only respond with the translation, no additional text. Also, do not talk too fast"},
        {"role": "user", "content": text}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.75
    )
    
    return response.choices[0].message.content

def process_input(text, target_language, is_voice=False):
    """Process input text or voice and return translation"""
    client = setup_openai_client()
    
    # Translate the text
    translated_text = translate_text(client, text, target_language)
    
    if is_voice:
        # Create audio response
        audio_file = f"audio_response_{int(time.time())}.mp3"
        text_to_audio(client, translated_text, audio_file)
        return translated_text, audio_file
    
    return translated_text, None

def main():
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'last_recorded_audio' not in st.session_state:
        st.session_state.last_recorded_audio = None
    if 'target_language' not in st.session_state:
        st.session_state.target_language = None
    if 'cleanup_on_start' not in st.session_state:
        cleanup_audio_files()
        st.session_state.cleanup_on_start = True

    st.title("Travel Translation Assistant")

    # Country/Language Selection
    st.sidebar.header("Translation Settings")
    country_selection = st.sidebar.selectbox(
        "Where are you traveling to?",
        options=list(COUNTRY_LANGUAGES.keys()),
        key="country_selection"
    )
    
    # Update target language based on country selection
    st.session_state.target_language = COUNTRY_LANGUAGES[country_selection]
    st.sidebar.write(f"Translation will be provided in: {st.session_state.target_language}")

    # Chat interface
    chat_container = st.container()

    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "translation" in message:
                    st.write(f"🔄 {message['translation']}")
                if "audio" in message:
                    auto_play_audio(message["audio"])

    # Voice and text input
    col1, col2 = st.columns([8, 2])
    
    with col2:
        recorded_audio = audio_recorder(
            text="",
            recording_color="#e74c3c",
            neutral_color="#95a5a6",
            key="voice_recorder"
        )

    with col1:
        text_input = st.chat_input("Type your message or use voice input...")

    # Handle text input
    if text_input:
        translation, _ = process_input(text_input, st.session_state.target_language)
        
        st.session_state.messages.append({
            "role": "user",
            "content": text_input,
            "translation": translation
        })
        st.rerun()

    # Handle voice input
    if recorded_audio is not None and recorded_audio != st.session_state.last_recorded_audio:
        st.session_state.last_recorded_audio = recorded_audio
        
        # Save and transcribe audio
        audio_file = f"audio_input_{int(time.time())}.mp3"
        with open(audio_file, "wb") as f:
            f.write(recorded_audio)

        # Transcribe the audio
        client = setup_openai_client()
        transcribed_text = transcribe_audio(client, audio_file)
        os.remove(audio_file)

        # Get translation and audio response
        translation, response_audio = process_input(
            transcribed_text, 
            st.session_state.target_language, 
            is_voice=True
        )

        # Update chat history
        st.session_state.messages.append({
            "role": "user",
            "content": f"🎤 {transcribed_text}",
            "translation": translation,
            "audio": response_audio
        })
        
        st.rerun()

if __name__ == "__main__":
    main()