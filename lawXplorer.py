import os
import time
import requests
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
from chroma_retriever import LawXplorerChromaStore

# --- Configuration ---
# Reads from Streamlit Secrets first, falls back to environment variable
API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
MODEL_NAME = "gemini-2.5-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

# --- Resource Caching ---
@st.cache_resource(show_spinner="Initializing ChromaDB Vector Store...")
def load_chroma():
    return LawXplorerChromaStore()

@st.cache_resource(show_spinner="Loading InLegalBERT tokenizer & weights...")
def load_bert():
    model_name = "law-ai/InLegalBERT"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        return tokenizer, model
    except Exception as e:
        return None, None

vector_db = load_chroma()
tokenizer, bert_model = load_bert()

def extract_bert_features(snippet: str):
    """
    Extracts token representations and mean-pooled sentence embeddings
    using domain-adapted InLegalBERT.
    """
    if not tokenizer or not bert_model or not snippet.strip():
        return None
    inputs = tokenizer(snippet, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return {
        "embedding_dim": embeddings.shape[0],
        "sample_vector": [round(float(x), 4) for x in embeddings[:5]],
        "token_count": inputs["input_ids"].shape[1]
    }

# --- Backend Logic (Gemini API with Grounding) ---

def call_gemini_api(topic, article, query, language, retrieved_cases):
    """
    Calls the Gemini API with Google Search Grounding to generate cited legal analysis
    and defense points in the requested language.
    """
    if not API_KEY:
        st.error("Gemini API Key is missing. Please set the GEMINI_API_KEY in Streamlit Secrets or Environment Variables.")
        return None, []

    precedent_context = "\n".join([
        f"- Case: {c['metadata']['case_name']} ({c['metadata']['citation']})\n"
        f"  Holding: {c['text']}\n"
        f"  Citator Status: {c['metadata']['citator_status']}"
        for c in retrieved_cases
    ])

    system_prompt = (
        f"You are LawXplorer, a specialized AI assistant focused on Indian Constitutional Law and legal compliance. "
        f"MANDATORY REQUIREMENT: You MUST generate your ENTIRE output strictly in {language}. "
        f"Base your analysis strictly on the provided ChromaDB precedents and retrieved evidence. "
        f"Format your response into EXACTLY two sections separated by the invariant raw delimiter '### ---DEFENSE_SPLIT---':\n"
        f"[Section 1]: Detailed legal analysis, statutory evaluation, and constitutional interpretation written in {language}.\n"
        f"### ---DEFENSE_SPLIT---\n"
        f"[Section 2]: Actionable bulleted legal arguments, relevant constitutional articles, and case laws in {language}."
    )

    context = f"Topic: {topic}. Article/Section: {article if article else 'General Inquiry'}"
    user_payload = f"{context}\n\nRETRIEVED PRECEDENTS FROM CHROMADB:\n{precedent_context}\n\nLEGAL QUERY: '{query}'"

    # Fixed: snake_case 'system_instruction' schema for REST v1beta compliance
    payload = {
        "contents": [{"parts": [{"text": user_payload}]}],
        "tools": [{"google_search": {}}],
        "system_instruction": {"parts": [{"text": system_prompt}]},
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_URL,
                headers={'Content-Type': 'application/json'},
                params={'key': API_KEY},
                json=payload,
                timeout=60
            )
            if response.status_code == 429:
                time.sleep((2 ** attempt) + 2)
                continue
            response.raise_for_status()

            result = response.json()
            candidate = result.get('candidates', [{}])[0]
            generated_text = candidate.get('content', {}).get('parts', [{}])[0].get('text', 'No response generated.')

            sources = []
            metadata = candidate.get('groundingMetadata', {})
            for chunk in metadata.get('groundingChunks', []):
                web = chunk.get('web')
                if web and web.get('uri'):
                    sources.append({'uri': web['uri'], 'title': web.get('title', web['uri'])})

            if not sources:
                for attr in metadata.get('groundingAttributions', []):
                    web = attr.get('web')
                    if web and web.get('uri'):
                        sources.append({'uri': web['uri'], 'title': web.get('title', web['uri'])})

            return generated_text, sources
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"API Error: {e}")
                return None, []
    return None, []

@st.cache_data(ttl=86400, show_spinner=False)
def get_latest_amendments(api_key_for_cache, language="English"):
    """Fetches the latest 5 constitutional amendments with Google Search Grounding."""
    if not api_key_for_cache:
        return "Amendment info unavailable (API Key missing)."

    amendment_query = (
        f"List the latest 5 significant amendments to the Indian Constitution in {language}. "
        f"For each, provide the amendment number, year, and a concise summary of its impact."
    )
    
    # Fixed: snake_case 'system_instruction' schema
    payload = {
        "contents": [{"parts": [{"text": amendment_query}]}],
        "tools": [{"google_search": {}}], 
        "system_instruction": {"parts": [{"text": f"You are a legal summarizer. Output the list of amendments directly using a numbered list in {language}. Do not add introductory conversational fluff."}]},
    }

    try:
        response = requests.post(
            API_URL,
            headers={'Content-Type': 'application/json'},
            params={'key': api_key_for_cache}, 
            json=payload,
            timeout=25
        )
        if response.status_code == 429:
            return "Rate limit exceeded when fetching amendments. Click to retry later."
        response.raise_for_status()
        result = response.json()
        return result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'Could not retrieve information.')
    except Exception as e:
        return f"Could not load updates: {e}"

@st.cache_data(ttl=43200, show_spinner=False)
def get_law_news(api_key_for_cache, language="English"):
    """Fetches top legal news headlines in India with Google Search Grounding."""
    if not api_key_for_cache:
        return "Top law news unavailable (API Key missing)."

    news_query = f"Summarize the top 3 most significant legal news headlines in India today in {language}."
    
    # Fixed: snake_case 'system_instruction' schema
    payload = {
        "contents": [{"parts": [{"text": news_query}]}],
        "tools": [{"google_search": {}}], 
        "system_instruction": {"parts": [{"text": f"You are a legal news aggregator. Present 3 headlines as a markdown bulleted list in {language} without introductory conversational text."}]},
    }

    try:
        response = requests.post(
            API_URL,
            headers={'Content-Type': 'application/json'},
            params={'key': api_key_for_cache},
            json=payload,
            timeout=25
        )
        if response.status_code == 429:
            return "Rate limit exceeded when fetching news. Click to retry later."
        response.raise_for_status()
        result = response.json()
        return result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'Could not retrieve top news.')
    except Exception as e:
        return f"Could not load news: {e}"

# --- Frontend/UI Logic ---

def main():
    st.set_page_config(page_title="LawXplorer: Multilingual Legal Assistant", layout="wide")
    st.title("⚖️ LawXplorer: Multilingual Legal Intelligence")
    st.markdown("ChromaDB Precedent Retrieval • InLegalBERT Feature Extraction • Grounded Gemini Synthesis")
    st.markdown("---")

    # Complete 22 Scheduled Indian Languages + Major International Languages
    ALL_LANGUAGES = sorted([
        "Assamese", "Bengali", "Bodo", "Dogri", "Gujarati", "Hindi", 
        "Kannada", "Kashmiri", "Konkani", "Maithili", "Malayalam", 
        "Manipuri", "Marathi", "Nepali", "Odia", "Punjabi", 
        "Sanskrit", "Santali", "Sindhi", "Tamil", "Telugu", "Urdu",
        "Arabic", "English", "French", "German", "Italian", 
        "Japanese", "Mandarin Chinese", "Portuguese", "Russian", "Spanish"
    ])

    with st.sidebar:
        st.header("⚙️ Configuration")
        
        selected_language = st.selectbox(
            "🌐 Select Output Language", 
            ALL_LANGUAGES, 
            index=ALL_LANGUAGES.index("English")
        )

        st.markdown("---")
        st.markdown("### 🗄️ ChromaDB Status")
        st.success(f"Indexed Precedents: {vector_db.collection.count()}")

        st.markdown("### 🧠 Legal-BERT Status")
        if bert_model is not None:
            st.success("`law-ai/InLegalBERT` active")
        else:
            st.info("InLegalBERT running in fallback mode")

        st.markdown("---")
        
        # Constitutional Amendments Section (Cached)
        with st.expander("🏛️ Latest 5 Amendments", expanded=False):
            if API_KEY:
                if st.button("Fetch Amendments", key="btn_amendments"):
                    with st.spinner("Fetching constitutional updates..."):
                        amendment_list = get_latest_amendments(API_KEY, language=selected_language)
                        st.markdown(amendment_list)
            else:
                st.info("API Key missing.")

        # Top Legal News Section (Cached)
        with st.expander("📰 Top Legal News", expanded=False):
            if API_KEY:
                if st.button("Fetch Latest News", key="btn_news"):
                    with st.spinner("Fetching legal headlines..."):
                        news_list = get_law_news(API_KEY, language=selected_language)
                        st.markdown(news_list)
            else:
                st.info("API Key missing.")

        st.markdown("---")
        st.markdown(
            "**DISCLAIMER:** LawXplorer provides legally grounded analysis with live citations. "
            "It is for informational purposes only and is **not a substitute for professional legal counsel**."
        )

    # Main Query Workspace
    col1, col2 = st.columns([1, 1])
    with col1:
        topic = st.selectbox(
            "Area of Law / Topic", 
            ["Fundamental Rights", "Directive Principles", "Union & State Relations", "Constitutional Amendments", "Judiciary & Courts", "Contract & Commercial Law", "Other"],
            help="Select the broad area of the inquiry."
        )
    with col2:
        article = st.text_input(
            "Relevant Article / Section (Optional)", 
            placeholder="e.g., Article 21, Section 27 Indian Contract Act",
            help="Specify an article or statutory provision."
        )

    query = st.text_area(
        "Your Specific Legal Question or Compliance Scenario", 
        height=130, 
        placeholder="e.g., What are the constitutional protections against arbitrary state action under Article 21? Cite relevant landmark Supreme Court precedents."
    )

    if st.button("Analyze & Synthesize Guidance", type="primary"):
        if not query.strip():
            st.warning("Please enter your legal query before analyzing.")
            return

        col_left, col_right = st.columns([1, 1])

        # Step 1: ChromaDB Retrieval & BERT Embeddings
        with st.spinner("Searching ChromaDB and extracting InLegalBERT features..."):
            matched_cases = vector_db.query_precedents(query, n_results=3, topic_filter=topic)
            bert_features = extract_bert_features(query)

        with col_left:
            st.subheader("📚 ChromaDB Retrieved Precedents")
            for case in matched_cases:
                status = case["metadata"]["citator_status"]
                badge = "red" if "OVERRULED" in status else "green"
                st.markdown(f"**{case['metadata']['case_name']}** (`{case['metadata']['citation']}`)")
                st.markdown(f"Similarity Score: `{case['similarity_score']}` | Citator Status: :{badge}[{status}]")
                st.markdown(f"> {case['text']}")
                st.markdown("---")

            if bert_features:
                with st.expander("🔬 InLegalBERT Tensor Representation"):
                    st.write(f"Sequence Length: `{bert_features['token_count']}` tokens")
                    st.write(f"Hidden Dimension: `{bert_features['embedding_dim']}`")
                    st.write("**Latent Representation Sample:**")
                    st.code(str(bert_features["sample_vector"]), language="python")

        # Step 2: Gemini Grounded Synthesis in Target Language
        with col_right:
            st.subheader(f"⚖️ Legal Analysis ({selected_language})")
            with st.spinner(f"Synthesizing grounded analysis in {selected_language}..."):
                response_text, sources = call_gemini_api(topic, article, query, selected_language, matched_cases)

            if response_text:
                if "### ---DEFENSE_SPLIT---" in response_text:
                    analysis_part, defense_part = response_text.split("### ---DEFENSE_SPLIT---", 1)
                    st.markdown(analysis_part.strip())
                    st.markdown("---")
                    st.markdown(f"### 🛡️ Defense Points ({selected_language})")
                    st.markdown(defense_part.strip())
                else:
                    st.markdown(response_text)

                if sources:
                    st.markdown("---")
                    st.markdown("#### 🔗 Grounded Legal Citations")
                    seen_urls = set()
                    idx = 1
                    for s in sources:
                        if s['uri'] not in seen_urls:
                            st.markdown(f"{idx}. [{s['title']}]({s['uri']})")
                            seen_urls.add(s['uri'])
                            idx += 1

                st.balloons()

if __name__ == "__main__":
    main()
