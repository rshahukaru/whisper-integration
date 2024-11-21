# import sys
# import streamlit as st
# from openai import OpenAI
# from bs4 import BeautifulSoup
# import os
# from collections import deque
# import numpy as np
# from audio_recorder_streamlit import audio_recorder
# import base64
# import time

# # Workaround for sqlite3 issue
# __import__('pysqlite3')
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# import chromadb

# # Initialize OpenAI client
# def setup_openai_client():
#     if 'openai_client' not in st.session_state:
#         api_key = st.secrets["openai_api_key"]
#         st.session_state.openai_client = OpenAI(api_key=api_key)
#     return st.session_state.openai_client

# def cleanup_audio_files():
#     """Clean up any temporary audio files"""
#     try:
#         for file in os.listdir():
#             if (file.startswith("audio_input_") or file.startswith("audio_response_")) and file.endswith(".mp3"):
#                 try:
#                     os.remove(file)
#                 except Exception as e:
#                     st.warning(f"Could not remove audio file {file}: {str(e)}")
#     except Exception as e:
#         st.warning(f"Error during audio cleanup: {str(e)}")

# # Audio Processing Functions
# def transcribe_audio(client, audio_path):
#     with open(audio_path, "rb") as audio_file:
#         transcript = client.audio.transcriptions.create(
#             model="whisper-1",
#             file=audio_file
#         )
#         return transcript.text

# def text_to_audio(client, text, audio_path):
#     response = client.audio.speech.create(
#         model="tts-1",
#         voice="nova",
#         input=text
#     )
#     response.stream_to_file(audio_path)

# def auto_play_audio(audio_file):
#     if os.path.exists(audio_file):
#         with open(audio_file, "rb") as f:
#             audio_bytes = f.read()
#         base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
#         audio_html = f'<audio src="data:audio/mp3;base64,{base64_audio}" controls autoplay>'
#         st.markdown(audio_html, unsafe_allow_html=True)

# # ChromaDB Functions
# def create_su_club_collection():
#     if 'HW_URL_Collection' not in st.session_state:
#         persist_directory = os.path.join(os.getcwd(), "chroma_db")
#         client = chromadb.PersistentClient(path=persist_directory)
#         collection = client.get_or_create_collection("HW_URL_Collection")

#         directory_path = os.path.join(os.getcwd(), "su_orgs")
#         if not os.path.exists(directory_path):
#             st.error(f"Directory not found: {directory_path}")
#             return None

#         if collection.count() == 0:
#             with st.spinner("Processing content and preparing the system..."):
#                 client = setup_openai_client()

#                 for root, dirs, files in os.walk(directory_path):
#                     for filename in files:
#                         if filename.endswith('.html'):
#                             file_path = os.path.join(root, filename)
#                             try:
#                                 with open(file_path, 'r', encoding='utf-8') as f:
#                                     content = f.read()
                                    
#                                 soup = BeautifulSoup(content, 'html.parser')
#                                 text = soup.get_text(separator=' ', strip=True)

#                                 response = client.embeddings.create(
#                                     input=text,
#                                     model="text-embedding-3-small"
#                                 )
#                                 embedding = response.data[0].embedding

#                                 collection.add(
#                                     documents=[text],
#                                     metadatas=[{"filename": filename}],
#                                     ids=[filename],
#                                     embeddings=[embedding]
#                                 )
#                             except Exception as e:
#                                 st.error(f"Error processing {filename}: {str(e)}")

#         st.session_state.HW_URL_Collection = collection

#     return st.session_state.HW_URL_Collection

# def get_relevant_info(query):
#     collection = st.session_state.HW_URL_Collection
#     client = setup_openai_client()

#     try:
#         response = client.embeddings.create(
#             input=query,
#             model="text-embedding-3-small"
#         )
#         query_embedding = response.data[0].embedding
#     except Exception as e:
#         st.error(f"Error creating OpenAI embedding: {str(e)}")
#         return "", []

#     query_embedding = np.array(query_embedding) / np.linalg.norm(query_embedding)

#     try:
#         results = collection.query(
#             query_embeddings=[query_embedding.tolist()],
#             n_results=3
#         )
#         relevant_texts = results['documents'][0]
#         relevant_docs = [result['filename'] for result in results['metadatas'][0]]
#         return "\n".join(relevant_texts), relevant_docs
#     except Exception as e:
#         st.error(f"Error querying the database: {str(e)}")
#         return "", []

# def call_llm(model, messages, temp, query, tools=None):
#     """Call OpenAI's LLM with streaming and tool support"""
#     client = setup_openai_client()
#     try:
#         response = client.chat.completions.create(
#             model=model,
#             messages=messages,
#             temperature=temp,
#             tools=tools,
#             tool_choice="auto" if tools else None,
#             stream=True
#         )
#     except Exception as e:
#         st.error(f"Error calling OpenAI API: {str(e)}")
#         return "", "Error occurred while generating response."

#     tool_called = None
#     full_response = ""
#     tool_usage_info = ""

#     try:
#         for chunk in response:
#             if hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
#                 for tool_call in chunk.choices[0].delta.tool_calls:
#                     if hasattr(tool_call, 'function'):
#                         tool_called = tool_call.function.name
#                         if tool_called == "get_club_info":
#                             extra_info, _ = get_relevant_info(query)
#                             tool_usage_info = f"Tool used: {tool_called}"
#                             for msg in messages:
#                                 if msg["role"] == "system":
#                                     msg["content"] += f"\n\nAdditional context: {extra_info}"
#                             recursive_response, recursive_tool_info = call_llm(
#                                 model, messages, temp, query)
#                             full_response += recursive_response
#                             tool_usage_info += "\n" + recursive_tool_info
#                             return full_response, tool_usage_info
#             elif hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
#                 full_response += chunk.choices[0].delta.content

#         if tool_called:
#             tool_usage_info = f"Tool used: {tool_called}"
#         else:
#             tool_usage_info = "No tools were used in generating this response."

#         return full_response, tool_usage_info

#     except Exception as e:
#         st.error(f"Error in streaming response: {str(e)}")
#         return "I encountered an error while generating the response.", "Error in response generation"

# def process_message(input_text, context, conversation_memory, is_voice=False):
#     system_message = """You are an AI assistant specialized in providing information about student organizations and clubs at Syracuse University. 
#     Your primary source of information is the context provided. Please be concise and natural in your responses, as they may be converted to speech."""

#     condensed_history = "\n".join(
#         [f"Human: {exchange['question']}\nAI: {exchange['answer']}" 
#          for exchange in conversation_memory]
#     )

#     messages = [
#         {"role": "system", "content": system_message},
#         {"role": "user", "content": f"Context: {context}\n\nConversation history:\n{condensed_history}\n\nQuestion: {input_text}"}
#     ]

#     tools = [
#         {
#             "type": "function",
#             "function": {
#                 "name": "get_club_info",
#                 "description": "Get information about a specific club or organization",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "club_name": {
#                             "type": "string",
#                             "description": "The name of the club or organization to look up"
#                         }
#                     },
#                     "required": ["club_name"]
#                 }
#             }
#         }
#     ]

#     response, tool_usage_info = call_llm(
#         "gpt-4", messages, 0.7, input_text, tools)
    
#     if is_voice:
#         audio_file = f"audio_response_{int(time.time())}.mp3"
#         text_to_audio(setup_openai_client(), response, audio_file)
#         return response, tool_usage_info, audio_file
    
#     return response, tool_usage_info, None

# def main():
#     # Initialize session state
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []
#     if 'last_recorded_audio' not in st.session_state:
#         st.session_state.last_recorded_audio = None
#     if 'awaiting_response' not in st.session_state:
#         st.session_state.awaiting_response = False
#     if 'temp_message' not in st.session_state:
#         st.session_state.temp_message = None
#     if 'conversation_memory' not in st.session_state:
#         st.session_state.conversation_memory = deque(maxlen=5)
#     if 'system_ready' not in st.session_state:
#         st.session_state.system_ready = False
#     if 'cleanup_on_start' not in st.session_state:
#         cleanup_audio_files()
#         st.session_state.cleanup_on_start = True

#     st.title("iSchool Voice-Enabled RAG Chatbot")

#     # Chat history container
#     chat_container = st.container()

#     # Initialize ChromaDB
#     if not st.session_state.system_ready:
#         with st.spinner("Processing documents and preparing the system..."):
#             st.session_state.collection = create_su_club_collection()
#             if st.session_state.collection:
#                 st.session_state.system_ready = True
#                 st.success("AI ChatBot is Ready!")

#     # Custom CSS with fixed positions
#     st.markdown("""
#         <style>
#             /* Chat container */
#             .chat-container {
#                 margin-bottom: 100px;
#             }
            
#             /* Input area container */
#             .stChatInput, .input-container {
#                 position: fixed !important;
#                 bottom: 0 !important;
#                 background: white !important;
#                 z-index: 99 !important;
#                 padding: 1rem !important;
#                 margin-right: 1rem !important;
#             }
            
#             /* Voice recorder container */
#             .voice-recorder-container {
#                 position: fixed !important;
#                 bottom: 5rem !important;
#                 right: 6rem !important;
#                 z-index: 100 !important;
#                 background: transparent !important;
#             }
            
#             /* Ensure space at bottom */
#             .main {
#                 padding-bottom: 100px !important;
#             }
            
#             /* Stacked elements */
#             .stStackedContainer {
#                 margin-bottom: 100px !important;
#             }
#         </style>
#     """, unsafe_allow_html=True)

#     # Display chat interface
#     if st.session_state.system_ready:
#         with chat_container:
#             for message in st.session_state.messages:
#                 with st.chat_message(message["role"]):
#                     # Display the message content
#                     content = message["content"]
                    
#                     # If it's an assistant message with documents, format nicely
#                     if message["role"] == "assistant" and "Relevant Documents:" in content:
#                         # Split content and documents
#                         main_content, docs_section = content.split("**Relevant Documents:**")
                        
#                         # Display main response
#                         st.markdown(main_content)
                        
#                         # Display documents in a cleaner format
#                         st.markdown("**📚 Source Documents:**")
#                         docs = [doc.strip("- ") for doc in docs_section.strip().split("\n")]
#                         for doc in docs:
#                             st.markdown(f"- {doc}")
#                     else:
#                         st.markdown(content)
                    
#                     # Handle audio if present
#                     if "audio" in message:
#                         auto_play_audio(message["audio"])

#         # Footer with input elements
#         with st.container():
#             # Voice recorder
#             col1, col2 = st.columns([8, 2])
#             with col2:
#                 st.markdown('<div class="voice-recorder-container">', unsafe_allow_html=True)
#                 recorded_audio = audio_recorder(
#                     text="",
#                     recording_color="#e74c3c",
#                     neutral_color="#95a5a6",
#                     key="voice_recorder"
#                 )
#                 st.markdown('</div>', unsafe_allow_html=True)
            
#             # Text input
#             with col1:
#                 text_input = st.chat_input("Type your message or use voice input...")

#         # Handle text input
#         if text_input and not st.session_state.awaiting_response:
#             st.session_state.awaiting_response = True
#             relevant_texts, relevant_docs = get_relevant_info(text_input)
#             response, tool_usage_info, _ = process_message(
#                 text_input, relevant_texts, st.session_state.conversation_memory)
            
#             # Add user message
#             st.session_state.messages.append({
#                 "role": "user", 
#                 "content": text_input
#             })
            
#             # Add assistant message with relevant documents
#             st.session_state.messages.append({
#                 "role": "assistant", 
#                 "content": f"{response}\n\n*{tool_usage_info}*\n\n" + 
#                           "**Relevant Documents:**\n" + 
#                           "\n".join([f"- {doc}" for doc in relevant_docs])
#             })
            
#             st.session_state.conversation_memory.append({
#                 "question": text_input,
#                 "answer": response
#             })
#             st.session_state.awaiting_response = False
#             st.rerun()

#         # Handle voice input
#         if recorded_audio is not None and recorded_audio != st.session_state.last_recorded_audio:
#             st.session_state.awaiting_response = True
#             st.session_state.last_recorded_audio = recorded_audio
            
#             # Save and transcribe audio
#             audio_file = f"audio_input_{int(time.time())}.mp3"
#             with open(audio_file, "wb") as f:
#                 f.write(recorded_audio)

#             transcribed_text = transcribe_audio(setup_openai_client(), audio_file)
#             os.remove(audio_file)

#             # Get response
#             relevant_texts, relevant_docs = get_relevant_info(transcribed_text)
#             response, tool_usage_info, response_audio = process_message(
#                 transcribed_text, relevant_texts, st.session_state.conversation_memory, is_voice=True)

#             # Update chat history with relevant documents
#             st.session_state.messages.append({
#                 "role": "user", 
#                 "content": f"🎤 {transcribed_text}"
#             })
#             st.session_state.messages.append({
#                 "role": "assistant", 
#                 "content": f"{response}\n\n*{tool_usage_info}*\n\n" +
#                           "**Relevant Documents:**\n" + 
#                           "\n".join([f"- {doc}" for doc in relevant_docs]),
#                 "audio": response_audio
#             })
            
#             st.session_state.conversation_memory.append({
#                 "question": transcribed_text,
#                 "answer": response
#             })
            
#             st.session_state.awaiting_response = False
#             st.rerun()

# if __name__ == "__main__":
#     main()




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
        {"role": "system", "content": f"You are a translator. Translate the following text to {target_language}. Maintain the tone and meaning of the original text. Only respond with the translation, no additional text."},
        {"role": "user", "content": text}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7
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