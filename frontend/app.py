import streamlit as st
import requests
import json
import time
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Drug Agentic Chatbot",
    page_icon="💊",
    layout="wide"
)

# API base URL
API_BASE_URL = "http://localhost:8000"

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for indexing controls
st.sidebar.title("Indexing Controls")

# User ID and Conversation ID inputs
user_id = st.sidebar.text_input("User ID", value="user_001", key="user_id")
conversation_id = st.sidebar.text_input("Conversation ID", value="conv_001", key="conversation_id")

# Index type selection
index_type = st.sidebar.selectbox(
    "Select Index Type",
    ["intent", "knowledge"],
    key="index_type"
)

# File upload for indexing
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File for Indexing",
    type=['csv'],
    key="csv_file"
)

# Run Indexing button
if st.sidebar.button("Run Indexing", type="primary"):
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_file_path = f"/tmp/{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Prepare indexing request
        indexing_data = {
            "index_type": index_type,
            "csv_file_path": temp_file_path
        }
        
        with st.sidebar:
            with st.spinner("Running indexing..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/indexing/run",
                        json=indexing_data,
                        timeout=300
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {result['message']}")
                        st.info(f"Documents indexed: {result['document_count']}")
                    else:
                        st.error(f"❌ Indexing failed: {response.text}")
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Connection error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
    else:
        st.sidebar.error("Please upload a CSV file first!")

# Main chat interface
st.title("💊 Drug Agentic Chatbot")
st.markdown("Ask me anything about drugs, medications, and medical information!")

# Display chat history
chat_container = st.container()
with chat_container:
    for i, (role, message) in enumerate(st.session_state.chat_history):
        if role == "user":
            st.chat_message("user").write(message)
        else:
            st.chat_message("assistant").write(message)

# Chat input
query = st.chat_input("Type your medical question here...")

if query:
    # Add user message to chat history
    st.session_state.chat_history.append(("user", query))
    
    # Display user message
    with st.chat_message("user"):
        st.write(query)
    
    # Show thinking indicator
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.write("🤔 Searching and Thinking...")
        
        # Prepare medical query request
        medical_data = {
            "query": query,
            "user_id": user_id,
            "conversation_id": conversation_id
        }
        
        try:
            # Make API call
            response = requests.post(
                f"{API_BASE_URL}/medical/run",
                json=medical_data,
                timeout=360
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result["answer"]
                intent = result["intent"]
                
                # Clear thinking indicator and show response
                thinking_placeholder.empty()
                st.write(f"**Intent:** {intent}")
                st.write(answer)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append(("assistant", f"**Intent:** {intent}\n\n{answer}"))
                
            else:
                thinking_placeholder.empty()
                error_message = f"❌ API Error: {response.text}"
                st.error(error_message)
                st.session_state.chat_history.append(("assistant", error_message))
                
        except requests.exceptions.RequestException as e:
            thinking_placeholder.empty()
            error_message = f"❌ Connection error: {str(e)}"
            st.error(error_message)
            st.session_state.chat_history.append(("assistant", error_message))
        except Exception as e:
            thinking_placeholder.empty()
            error_message = f"❌ Error: {str(e)}"
            st.error(error_message)
            st.session_state.chat_history.append(("assistant", error_message))

# Add some styling and info
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 12px;'>
        Drug Agentic Chatbot v1.0 | API Running on Port 8000
    </div>
    """,
    unsafe_allow_html=True
)