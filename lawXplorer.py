import streamlit as st
import requests
import os
import time
import torch
from transformers import AutoTokenizer, AutoModel
from chroma_retriever import LawXplorerChromaStore

# --- API Configuration ---
# Reads from Streamlit Secrets first, falls back to environment variable
API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
MODEL_NAME = "gemini-2.5-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

# --- Resource Caching ---
@st.cache_resource(show_spinner="Initializing ChromaDB Vector Engine...")
def load_chroma():
    return LawXplorerChromaStore()

@st.cache_resource(show_spinner="Loading law-ai/InLegalBERT...")
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

def call_gemini_api(topic, article, query, language, retrieved_cases):
    if not API_KEY:
        st.error("Gemini API Key missing. Set GEMINI_API_KEY in Streamlit Secrets or Environment Variables.")
        return None, []

    precedent_context = "\n".join([
        f"- Case: {c['metadata']['case_name']} ({c['metadata']['citation']})\n"
        f"  Holding: {c['text']}\n"
        f"  Citator Status: {c['metadata']['citator_status']}"
        for c in retrieved_cases
    ])

    system_prompt = (
        f"You are LawXplorer, an AI assistant specialized in Indian Law and Constitutional analysis. "
        f"MANDATORY REQUIREMENT: You MUST generate your ENTIRE response strictly in {language}. "
        f"Base your analysis strictly on the provided ChromaDB precedents and retrieved evidence. "
        f"Structure the response into two sections: "
        f"1. ANALYSIS: A clear summary of the legal position in {language}. "
        f"2. DEFENSE POINTS: A section titled 'DEFENSE POINTS' containing bulleted legal arguments, relevant articles, and precedents in {language}."
    )

    context = f"Topic: {topic}. Article/Section: {article if article else 'General Inquiry'}"
    user_payload = f"{context}\n\nRETRIEVED PRECEDENTS:\n{precedent_context}\n\nLEGAL QUERY: '{query}'"

    payload = {
        "contents": [{"parts": [{"text": user_payload}]}],
        "tools": [{"google_search": {}}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
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
            if metadata and metadata.get('groundingAttributions'):
                sources = [
                    {'uri': attr['web']['uri'], 'title': attr['web']['title']}
                    for attr in metadata['groundingAttributions'] if attr.get('web')
                ]

            return generated_text, sources
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"API Error: {e}")
                return None, []
    return None, []

# --- Streamlit Frontend ---
def main():
    st.set_page_config(page_title="LawXplorer: Multilingual Legal Assistant", layout="wide")
    st.title("⚖️ LawXplorer: Multilingual Legal Intelligence")
    st.markdown("ChromaDB Semantic Precedent Retrieval • InLegalBERT Feature Extraction • Gemini Synthesis")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️️ Configuration")
        languages = ["English", "French", "Spanish", "German", "Hindi", "Tamil"]
        selected_language = st.selectbox("🌐 Select Language", languages, index=0)

        st.markdown("---")
        st.markdown("### 🗄️ ChromaDB Status")
        st.success(f"Indexed Precedents: {vector_db.collection.count()}")

        st.markdown("### 🧠 Legal-BERT Status")
        if bert_model is not None:
            st.success("`law-ai/InLegalBERT` active")
        else:
            st.info("BERT running in fallback mode")

        st.markdown("---")
        st.markdown("**DISCLAIMER:** For informational and research purposes only. Not formal legal advice.")

    col1, col2 = st.columns([1, 1])
    with col1:
        topic = st.selectbox("Area of Law / Topic", ["Fundamental Rights", "Directive Principles", "Union & State Relations", "Contract & Commercial Law", "Other"])
    with col2:
        article = st.text_input("Relevant Article / Section (Optional)", placeholder="e.g., Article 21, Section 27")

    query = st.text_area("Legal Question or Compliance Scenario", height=130, placeholder="Explain the constitutional validity of arbitrary executive detention under Article 21.")

    if st.button("Run ChromaDB Search & Synthesize Guidance", type="primary"):
        if not query.strip():
            st.warning("Please enter your legal query.")
            return

        col_left, col_right = st.columns([1, 1])

        # Step 1: ChromaDB Retrieval & BERT Feature Extraction
        with st.spinner("Searching ChromaDB and extracting InLegalBERT features..."):
            matched_cases = vector_db.query_precedents(query, n_results=3, topic_filter=topic)
            bert_features = extract_bert_features(query)

        with col_left:
            st.subheader("📚 ChromaDB Retrieved Precedents")
            for case in matched_cases:
                status = case["metadata"]["citator_status"]
                badge = "red" if "OVERRULED" in status else "green"
                st.markdown(f"**{case['metadata']['case_name']}** (`{case['metadata']['citation']}`)")
                st.markdown(f"Similarity: `{case['similarity_score']}` | Citator: :{badge}[{status}]")
                st.markdown(f"> {case['text']}")
                st.markdown("---")

            if bert_features:
                with st.expander("🔬 InLegalBERT Tensor Representation"):
                    st.write(f"Tokens: `{bert_features['token_count']}` | Dimension: `{bert_features['embedding_dim']}`")
                    st.code(str(bert_features["sample_vector"]), language="python")

        # Step 2: Gemini Synthesis in Target Language
        with col_right:
            st.subheader(f"⚖️ Legal Analysis ({selected_language})")
            with st.spinner(f"Generating grounded analysis in {selected_language}..."):
                response_text, sources = call_gemini_api(topic, article, query, selected_language, matched_cases)

            if response_text:
                if "DEFENSE POINTS" in response_text:
                    parts = response_text.split("DEFENSE POINTS", 1)
                    st.markdown(parts[0].replace("ANALYSIS:", "").strip())
                    st.markdown("---")
                    st.markdown(f"### 🛡️ DEFENSE POINTS\n{parts[1].strip()}")
                else:
                    st.markdown(response_text)

                if sources:
                    st.markdown("---")
                    st.markdown("#### 🔗 Grounded Legal Citations")
                    for i, s in enumerate(sources, 1):
                        st.markdown(f"{i}. [{s['title']}]({s['uri']})")

if __name__ == "__main__":
    main()
