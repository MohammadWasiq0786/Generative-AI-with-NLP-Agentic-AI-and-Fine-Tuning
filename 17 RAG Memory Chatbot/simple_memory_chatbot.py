import os
import json
import faiss
import numpy as np
import streamlit as st
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =====================================================================
# Configuration
# =====================================================================

class Settings:
    """Application settings."""
    
    # API Keys
    api_key = os.getenv("EURIAI_API_KEY", "")
    
    # Model Configuration
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model = os.getenv("CHAT_MODEL", "gpt-4.1-nano")
    
    # Generation Settings
    temperature = 0.7
    max_tokens = 800
    
    # Memory Settings
    memory_k = 5  # Number of relevant memories to retrieve
    history_limit = 20  # Maximum number of conversation turns to keep
    
    # Vector Store Settings
    vector_dim = 1536  # Dimension of embedding vectors
    index_path = "data/faiss_index"
    
    # User Identity
    user_identity = os.getenv("USER_IDENTITY", "")
    
    # Application Settings
    app_name = os.getenv("APP_NAME", "Memory Chatbot")

settings = Settings()

# Create a data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# =====================================================================
# Embedding Model (Direct API)
# =====================================================================

class EuriaiEmbeddings:
    """Wrapper for Euriai embeddings API."""
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
    ):
        """Initialize the Euriai embeddings model."""
        self.api_key = api_key or settings.api_key
        self.model = model or settings.embedding_model
        self.embed_url = "https://api.euron.one/api/v1/euri/alpha/embeddings"
        
        if not self.api_key:
            raise ValueError("Euriai API key not provided")
            
    def _get_headers(self):
        """Get API request headers."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        if not texts:
            return []
            
        # Process in batches if needed for large text collections
        all_embeddings = []
        
        # For now, process all at once
        payload = {
            "input": texts,
            "model": self.model
        }
        
        try:
            response = requests.post(
                self.embed_url, 
                headers=self._get_headers(), 
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract embeddings from the response
            embeddings = [item["embedding"] for item in data["data"]]
            return embeddings
            
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
            
    def embed_query(self, text: str) -> List[float]:
        """Generate an embedding for a single query text."""
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else []

# =====================================================================
# Chat Model (with EuriaiClient)
# =====================================================================

class EuriaiChat:
    """Wrapper for Euriai chat completion API."""
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ):
        """Initialize the Euriai chat model."""
        self.api_key = api_key or settings.api_key
        self.model = model or settings.chat_model
        self.temperature = temperature or settings.temperature
        self.max_tokens = max_tokens or settings.max_tokens
        
        # Import and initialize the Euriai client
        try:
            from euriai import EuriaiClient
            self.client = EuriaiClient(
                api_key=self.api_key,
                model=self.model
            )
            self.client_available = True
        except ImportError:
            st.warning("Euriai client not installed. Using direct API calls instead.")
            self.client_available = False
        except Exception as e:
            st.warning(f"Error initializing Euriai client: {str(e)}. Using direct API calls instead.")
            self.client_available = False
        
    def generate_completion(self, prompt: str, **kwargs) -> str:
        """Generate a completion from the model."""
        # Extract parameters
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        if self.client_available:
            try:
                # Use the Euriai client to generate a completion
                response = self.client.generate_completion(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response
            except Exception as e:
                st.warning(f"Error with Euriai client: {str(e)}. Falling back to direct API.")
                return self._generate_completion_direct(prompt, temperature, max_tokens)
        else:
            return self._generate_completion_direct(prompt, temperature, max_tokens)
            
    def _generate_completion_direct(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate a completion using direct API call."""
        try:
            # Prepare the API request
            url = "https://api.euron.one/api/v1/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Make the API request
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Extract the completion text
            return data["choices"][0]["text"]
            
        except Exception as e:
            # If that fails, try the chat endpoint
            try:
                url = "https://api.euron.one/api/v1/chat/completions"
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                
                return data["choices"][0]["message"]["content"]
                
            except Exception as e2:
                raise Exception(f"Error generating completion: {str(e)}. Additional error: {str(e2)}")

# =====================================================================
# Vector Store
# =====================================================================

class MemoryVectorStore:
    """Vector store for chatbot memories using FAISS."""
    
    def __init__(
        self,
        embedding_model: Optional[EuriaiEmbeddings] = None,
        index_path: str = None
    ):
        """Initialize the vector store."""
        self.embedding_model = embedding_model or EuriaiEmbeddings()
        self.index_path = index_path or settings.index_path
        self.metadata_path = f"{self.index_path}_metadata.json"
        self.vector_dim = settings.vector_dim
        
        # Initialize or load the index
        self._initialize_index()
        
    def _initialize_index(self):
        """Initialize a new FAISS index or load an existing one."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Load existing index and metadata if available
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(f"Loaded existing index with {len(self.metadata)} entries")
            except Exception as e:
                print(f"Error loading index: {str(e)}. Creating new index.")
                self._create_new_index()
        else:
            self._create_new_index()
            
    def _create_new_index(self):
        """Create a new FAISS index."""
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.metadata = []
        print("Created new FAISS index")
        
    def save(self):
        """Save the index and metadata to disk."""
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f)
            print(f"Saved index with {len(self.metadata)} entries")
        except Exception as e:
            print(f"Error saving index: {str(e)}")
            
    def add_memory(
        self,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> int:
        """Add a memory to the vector store."""
        # Generate embedding
        embedding = self.embedding_model.embed_query(text)
        
        # Convert to numpy array of correct shape and type
        embedding_np = np.array([embedding]).astype(np.float32)
        
        # Add to FAISS index
        self.index.add(embedding_np)
        
        # Create metadata entry
        memory_id = len(self.metadata)
        timestamp = datetime.now().isoformat()
        
        memory_metadata = {
            "id": memory_id,
            "text": text,
            "timestamp": timestamp,
            **(metadata or {})
        }
        
        # Add to metadata store
        self.metadata.append(memory_metadata)
        
        # Save the updated index
        self.save()
        
        return memory_id
        
    def search(
        self,
        query: str,
        k: int = None
    ) -> List[Dict[str, Any]]:
        """Search for similar memories."""
        if len(self.metadata) == 0:
            return []
            
        k = k or settings.memory_k
        k = min(k, len(self.metadata))
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        query_embedding_np = np.array([query_embedding]).astype(np.float32)
        
        # Search the index
        distances, indices = self.index.search(query_embedding_np, k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                memory = self.metadata[idx].copy()
                memory["distance"] = float(distances[0][i])
                memory["similarity"] = 1.0 / (1.0 + float(distances[0][i]))
                results.append(memory)
                
        return results
        
    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Get all memories in the store."""
        return self.metadata.copy()
        
    def clear(self):
        """Clear all memories from the store."""
        self._create_new_index()
        self.save()

# =====================================================================
# Conversation Memory
# =====================================================================

class ConversationMemory:
    """Store for conversation history."""
    
    def __init__(self, history_limit: int = None, storage_path: str = "data/conversation_history.json"):
        """Initialize the conversation memory."""
        self.history_limit = history_limit or settings.history_limit
        self.storage_path = storage_path
        self.history = []
        
        # Load existing history if available
        self._load_history()
        
    def _load_history(self):
        """Load conversation history from disk."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    self.history = json.load(f)
                print(f"Loaded conversation history with {len(self.history)} entries")
            except Exception as e:
                print(f"Error loading conversation history: {str(e)}")
                self.history = []
                
    def save(self):
        """Save conversation history to disk."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.history, f)
            print(f"Saved conversation history with {len(self.history)} entries")
        except Exception as e:
            print(f"Error saving conversation history: {str(e)}")
            
    def add_user_message(self, message: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add a user message to the conversation history."""
        return self._add_message("user", message, metadata)
        
    def add_assistant_message(self, message: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add an assistant message to the conversation history."""
        return self._add_message("assistant", message, metadata)
        
    def _add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add a message to the conversation history."""
        timestamp = datetime.now().isoformat()
        
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp,
            **(metadata or {})
        }
        
        self.history.append(message)
        
        # Trim history if it exceeds the limit
        if self.history_limit > 0 and len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit:]
            
        # Save the updated history
        self.save()
        
        return message
        
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        if limit is None or limit <= 0:
            return self.history.copy()
        
        return self.history[-limit:].copy()
        
    def get_formatted_history(self, limit: Optional[int] = None) -> str:
        """Get formatted conversation history as a string."""
        history = self.get_history(limit)
        formatted = ""
        
        for entry in history:
            role = "Human" if entry["role"] == "user" else "Assistant"
            formatted += f"{role}: {entry['content']}\n"
            
        return formatted
        
    def clear(self):
        """Clear the conversation history."""
        self.history = []
        self.save()

# =====================================================================
# LangChain Integration (Optional)
# =====================================================================

def create_langchain_llm(api_key=None, model=None, temperature=None, max_tokens=None):
    """Create a LangChain LLM using Euriai (if available)."""
    try:
        from euriai import EuriaiLangChainLLM
        
        llm = EuriaiLangChainLLM(
            api_key=api_key or settings.api_key,
            model=model or settings.chat_model,
            temperature=temperature or settings.temperature,
            max_tokens=max_tokens or settings.max_tokens
        )
        return llm
    except ImportError:
        st.warning("LangChain integration not available: euriai package not installed correctly.")
        return None
    except Exception as e:
        st.warning(f"Error initializing LangChain LLM: {str(e)}")
        return None

# =====================================================================
# Memory-Based Chatbot
# =====================================================================

class MemoryChatbot:
    """Main chatbot class that ties everything together."""
    
    def __init__(
        self,
        conversation_memory: Optional[ConversationMemory] = None,
        vector_store: Optional[MemoryVectorStore] = None,
        chat_model: Optional[EuriaiChat] = None,
        user_identity: str = None
    ):
        """Initialize the memory chatbot."""
        self.conversation_memory = conversation_memory or ConversationMemory()
        self.vector_store = vector_store or MemoryVectorStore()
        self.chat_model = chat_model or EuriaiChat()
        self.user_identity = user_identity or settings.user_identity
        
        # Try to initialize LangChain integration (optional)
        self.langchain_llm = create_langchain_llm()
        
    def process_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input and generate a response."""
        # Add user message to conversation history
        self.conversation_memory.add_user_message(user_input)
        
        # Get conversation history for context
        formatted_history = self.conversation_memory.get_formatted_history()
        
        # Get relevant memories
        relevant_memories = self.vector_store.search(user_input, settings.memory_k)
        
        # Format relevant memories
        formatted_memories = self._format_memories(relevant_memories)
        
        # Create the prompt
        prompt = self._create_prompt(user_input, formatted_history, formatted_memories)
        
        # Generate response
        try:
            # Try with LangChain if available (may have more features)
            if self.langchain_llm:
                response = self.langchain_llm.invoke(prompt)
            else:
                # Fallback to direct client
                response = self.chat_model.generate_completion(prompt)
                
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            response = f"I apologize, but I encountered an error while processing your request. Please try again or check your API configuration."
        
        # Add assistant message to conversation history
        self.conversation_memory.add_assistant_message(response)
        
        # Add the interaction to memory
        self._add_interaction_to_memory(user_input, response)
        
        # Prepare response object
        response_obj = {
            "response": response,
            "conversation_history": self.conversation_memory.get_history(),
            "metadata": {
                "timestamp": self.conversation_memory.history[-1]["timestamp"]
            }
        }
        
        return response_obj
        
    def _format_memories(self, memories: List[Dict[str, Any]]) -> str:
        """Format retrieved memories for the prompt."""
        if not memories:
            return "No relevant memories found."
            
        formatted = ""
        for i, memory in enumerate(memories):
            formatted += f"{i+1}. {memory['text']} (Relevance: {memory['similarity']:.2f})\n"
            
        return formatted
        
    def _create_prompt(self, query: str, conversation_history: str, relevant_memories: str) -> str:
        """Create the prompt for the language model."""
        return f"""
You are a helpful, intelligent memory-based assistant that responds based on the conversation history and relevant memories.

USER IDENTITY:
{self.user_identity}

RELEVANT MEMORIES:
{relevant_memories}

CONVERSATION HISTORY:
{conversation_history}

CURRENT QUERY:
Human: {query}

Your response should be helpful, personalized based on the user's identity and conversation history, and incorporate relevant memories when appropriate.
Assistant: """
        
    def _add_interaction_to_memory(self, query: str, response: str, metadata: Dict[str, Any] = None) -> int:
        """Add an interaction to the memory store."""
        # Create a memory entry from the interaction
        memory_text = f"Human: {query}\nAssistant: {response}"
        
        # Add to vector store
        memory_id = self.vector_store.add_memory(
            text=memory_text,
            metadata={
                "type": "interaction",
                "query": query,
                "response": response,
                **(metadata or {})
            }
        )
        
        return memory_id
        
    def set_user_identity(self, identity: str):
        """Set or update the user identity."""
        self.user_identity = identity
        
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        return self.conversation_memory.get_history(limit)
        
    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation_memory.clear()
        
    def add_to_memory(self, text: str, metadata: Dict[str, Any] = None) -> int:
        """Add a custom memory entry."""
        return self.vector_store.add_memory(text, metadata)
        
    def get_suggestions(self, query: str = "", k: int = 3) -> List[Dict[str, Any]]:
        """Get suggestions based on conversation history and memories."""
        # If query is empty, use the last user message
        if not query and self.conversation_memory.history:
            for message in reversed(self.conversation_memory.history):
                if message["role"] == "user":
                    query = message["content"]
                    break
                    
        if not query:
            return []
            
        # Get relevant memories
        memories = self.vector_store.search(query, k)
        
        # Format as suggestions
        suggestions = []
        for memory in memories:
            suggestion = {
                "text": memory["text"],
                "relevance": memory["similarity"],
                "id": memory["id"]
            }
            suggestions.append(suggestion)
            
        return suggestions

# =====================================================================
# Helper Functions
# =====================================================================

def format_date(timestamp: str) -> str:
    """Format ISO timestamp as a human-readable date string."""
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%b %d, %Y %I:%M %p")
    except:
        return timestamp

def save_user_identity(identity: str) -> bool:
    """Save user identity to environment variables."""
    try:
        # Update environment variable
        os.environ["USER_IDENTITY"] = identity
        settings.user_identity = identity
        
        # Update .env file if available
        env_file = ".env"
        if os.path.exists(env_file):
            with open(env_file, "r") as f:
                lines = f.readlines()
                
            # Look for existing USER_IDENTITY line
            updated = False
            for i, line in enumerate(lines):
                if line.startswith("USER_IDENTITY="):
                    lines[i] = f'USER_IDENTITY="{identity}"\n'
                    updated = True
                    break
                    
            # Add new line if not found
            if not updated:
                lines.append(f'USER_IDENTITY="{identity}"\n')
                
            # Write back to file
            with open(env_file, "w") as f:
                f.writelines(lines)
                
        return True
    except Exception as e:
        print(f"Error saving user identity: {str(e)}")
        return False

# =====================================================================
# Demo Embedding Function (For API testing)
# =====================================================================

def test_embeddings(api_key=None):
    """Test function for Euriai embeddings API."""
    url = "https://api.euron.one/api/v1/euri/alpha/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key or settings.api_key}"
    }
    payload = {
        "input": "The food was delicious and the service was excellent.",
        "model": "text-embedding-3-small"
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Convert to numpy array for vector operations
        embedding = np.array(data['data'][0]['embedding'])
        
        result = {
            "success": True,
            "shape": embedding.shape,
            "first_5_values": embedding[:5].tolist(),
            "norm": float(np.linalg.norm(embedding))
        }
        
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# =====================================================================
# Streamlit UI 
# =====================================================================

# Page configuration
st.set_page_config(
    page_title=settings.app_name,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Key check
if not settings.api_key:
    st.error("⚠️ Euriai API key not found. Please add it to your .env file or provide it below:")
    api_key = st.text_input("Euriai API Key", type="password")
    if api_key:
        settings.api_key = api_key
        os.environ["EURIAI_API_KEY"] = api_key
        st.success("API key saved for this session!")
        
        # Test the API key
        with st.spinner("Testing API key..."):
            test_result = test_embeddings(api_key)
            
        if test_result["success"]:
            st.success("✅ API key is valid! Embeddings test successful.")
            st.json(test_result)
        else:
            st.error(f"❌ API key test failed: {test_result['error']}")
            st.stop()
            
        st.experimental_rerun()
    st.stop()

# Initialize session state
if "chatbot" not in st.session_state:
    st.session_state.chatbot = MemoryChatbot()
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []

def display_message(role, content, timestamp=None):
    """Display a chat message."""
    if role == "user":
        with st.chat_message("user"):
            st.write(content)
            if timestamp:
                st.caption(format_date(timestamp))
    else:
        with st.chat_message("assistant"):
            st.write(content)
            if timestamp:
                st.caption(format_date(timestamp))

def handle_user_input(user_input):
    """Process user input and generate a response."""
    if user_input and user_input.strip():
        # Display user message
        display_message("user", user_input)
        
        # Add to session chat history for display
        st.session_state.chat_history.append({"role": "user", "content": user_input, "timestamp": datetime.now().isoformat()})
        
        # Process with chatbot
        with st.spinner("Thinking..."):
            response_obj = st.session_state.chatbot.process_input(user_input)
            
        # Display assistant response
        display_message("assistant", response_obj["response"], response_obj["metadata"]["timestamp"])
        
        # Add to session chat history for display
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": response_obj["response"], 
            "timestamp": response_obj["metadata"]["timestamp"]
        })
        
        # Update suggestions
        st.session_state.suggestions = st.session_state.chatbot.get_suggestions()
        
def load_conversation_history():
    """Load and display conversation history."""
    history = st.session_state.chatbot.get_conversation_history()
    
    if history:
        st.session_state.chat_history = history
        
        # Display existing messages
        for message in history:
            display_message(message["role"], message["content"], message.get("timestamp"))

def save_identity():
    """Save user identity."""
    if st.session_state.identity_input and st.session_state.identity_input.strip():
        identity = st.session_state.identity_input
        
        # Save to environment and settings
        success = save_user_identity(identity)
        
        # Update chatbot
        st.session_state.chatbot.set_user_identity(identity)
        
        if success:
            st.sidebar.success("Identity saved successfully!")
        else:
            st.sidebar.error("Failed to save identity.")

def add_custom_memory():
    """Add a custom memory entry."""
    if st.session_state.memory_input and st.session_state.memory_input.strip():
        memory_text = st.session_state.memory_input
        
        # Add to vector store via memory chain
        memory_id = st.session_state.chatbot.add_to_memory(
            memory_text,
            {"type": "custom", "source": "user_input"}
        )
        
        if memory_id >= 0:
            st.sidebar.success(f"Memory added successfully! (ID: {memory_id})")
            # Clear memory input (using a different approach)
            st.session_state.memory_input = ""
            
            # Update suggestions
            st.session_state.suggestions = st.session_state.chatbot.get_suggestions()
        else:
            st.sidebar.error("Failed to add memory.")

# Sidebar
with st.sidebar:
    st.title(f"🧠 {settings.app_name}")
    st.markdown("---")
    
    # API key display (masked)
    api_key_masked = settings.api_key[:4] + "..." + settings.api_key[-4:] if len(settings.api_key) > 8 else "****"
    st.success(f"✅ Using API key: {api_key_masked}")
    
    # Test API button
    if st.button("Test API Connection"):
        with st.spinner("Testing API..."):
            test_result = test_embeddings()
            
        if test_result["success"]:
            st.success("✅ API connection successful!")
            with st.expander("API Test Results"):
                st.json(test_result)
        else:
            st.error(f"❌ API test failed: {test_result['error']}")
    
    # Model information
    st.info(f"📝 Embedding model: {settings.embedding_model}\n🤖 Chat model: {settings.chat_model}")
    
    st.markdown("---")
    
    # User identity section
    st.subheader("User Identity")
    identity = st.session_state.chatbot.user_identity or "Not set"
    st.text_area("Current Identity", value=identity, height=100, disabled=True)
    
    st.text_area("Update Identity", key="identity_input", 
                placeholder="Enter information about yourself...", height=150)
    st.button("Save Identity", on_click=save_identity)
    
    st.markdown("---")
    
    # Custom memory section
    st.subheader("Add Custom Memory")
    st.text_area("Memory Content", key="memory_input", 
                placeholder="Enter a memory to store...", height=150)
    st.button("Add Memory", on_click=add_custom_memory)
    
    st.markdown("---")
    
    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.chatbot.clear_conversation()
        st.session_state.chat_history = []
        st.experimental_rerun()

# Main chat interface
st.title(f"Chat with {settings.app_name}")

# Load and display conversation history
if not st.session_state.chat_history:
    load_conversation_history()
else:
    # Display existing messages from session
    for message in st.session_state.chat_history:
        display_message(message["role"], message["content"], message.get("timestamp"))

# Chat input - Fixed for newer Streamlit versions
user_input = st.chat_input("Type your message here...")
if user_input:
    handle_user_input(user_input)

# Suggestions panel
if st.session_state.suggestions:
    st.markdown("---")
    st.subheader("Suggestions & Related Memories")
    
    for i, suggestion in enumerate(st.session_state.suggestions):
        with st.expander(f"Suggestion {i+1} (Relevance: {suggestion['relevance']:.2f})"):
            st.markdown(suggestion["text"])