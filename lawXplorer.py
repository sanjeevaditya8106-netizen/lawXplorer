import streamlit as st
import requests
import os
import time
import json

# --- Configuration ---
# Uses the environment variable GEMINI_API_KEY for security
API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_NAME = "gemini-2.5-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

# --- Backend Logic (API Call with Grounding) ---

def call_gemini_api(topic, article, query):
    """
    Calls the Gemini API with Google Search Grounding to get cited legal analysis.
    Implements exponential backoff for resilience.
    """
    if not API_KEY:
        st.error("Gemini API Key is missing. Please set the GEMINI_API_KEY environment variable.")
        return None, []

    # 1. Build the system prompt
    system_prompt = (
        "You are LawXplorer, a specialized AI assistant focused on the Indian Constitution. "
        "Your task is to provide accurate, concise, and professional legal analysis based on your search results. "
        "Strictly cite all sources used. The response MUST be structured into two sections: "
        "1. ANALYSIS: A clear, accessible summary of the legal situation. "
        "2. DEFENSE POINTS: A section titled 'Defense Points' containing bulleted legal arguments, relevant articles, and case laws that can be used to resolve any potential limitations or defend the legal position."
    )

    # 2. Build user prompt and payload
    context = f"Topic: {topic}. Article/Section: {article if article else 'General Inquiry'}"
    user_query = f"Analyze the following legal query within the context of the Indian Constitution: '{query}'"

    payload = {
        "contents": [{"parts": [{"text": f"{context}\n\n{user_query}"}]}],
        "tools": [{"google_search": {}}],  # Enable Google Search Grounding
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    # 3. API Request with Exponential Backoff
    max_retries = 4
    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_URL,
                headers={'Content-Type': 'application/json'},
                params={'key': API_KEY},
                json=payload,
                timeout=60
            )
            
            # Handle Rate Limits specifically before general error raising
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + 2  # Exponential backoff + jitter delay
                    time.sleep(wait_time)
                    continue
                else:
                    st.error("Rate limit (429) exceeded. You have made too many API requests in a short time. Please wait a minute and try again.")
                    return None, []

            response.raise_for_status()
            
            result = response.json()
            candidate = result.get('candidates', [{}])[0]
            
            # Extract text
            generated_text = candidate.get('content', {}).get('parts', [{}])[0].get('text', 'No response generated.')

            # Extract grounding sources
            sources = []
            grounding_metadata = candidate.get('groundingMetadata', {})
            if grounding_metadata and grounding_metadata.get('groundingAttributions'):
                sources = [
                    {'uri': attr['web']['uri'], 'title': attr['web']['title']}
                    for attr in grounding_metadata['groundingAttributions'] if attr.get('web')
                ]
            
            return generated_text, sources

        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP Error: {e}. Status Code: {response.status_code}")
            return None, []
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred during the API request: {e}")
            return None, []
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return None, []
            
    return None, []

@st.cache_data(ttl=86400, show_spinner=False)  # Cache for 24 hours to reduce background quota usage
def get_latest_amendments(api_key_for_cache):
    """
    Fetches the latest 5 constitutional amendments.
    """
    if not api_key_for_cache:
        return "Amendment info unavailable (API Key missing)."

    amendment_query = (
        "List the latest 5 significant amendments to the Indian Constitution. "
        "For each, provide the amendment number and a concise summary of its impact."
    )
    
    payload = {
        "contents": [{"parts": [{"text": amendment_query}]}],
        "tools": [{"google_search": {}}], 
        "systemInstruction": {"parts": [{"text": "You are a legal summarizer. Output the list of amendments directly using a numbered list. Do not add intro text."}]},
    }

    try:
        response = requests.post(
            API_URL,
            headers={'Content-Type': 'application/json'},
            params={'key': api_key_for_cache}, 
            json=payload,
            timeout=15 
        )
        if response.status_code == 429:
            return "Rate limit exceeded when fetching amendments. Click to retry later."
        response.raise_for_status()
        
        result = response.json()
        return result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'Could not retrieve information.')
    except Exception as e:
        return f"Could not load updates: {e}"

@st.cache_data(ttl=43200, show_spinner=False)  # Cache for 12 hours
def get_law_news(api_key_for_cache):
    """
    Fetches top legal news headlines in India.
    """
    if not api_key_for_cache:
        return "Top law news unavailable (API Key missing)."

    news_query = "Summarize the top 3 most significant legal news headlines in India today."
    
    payload = {
        "contents": [{"parts": [{"text": news_query}]}],
        "tools": [{"google_search": {}}], 
        "systemInstruction": {"parts": [{"text": "You are a legal news aggregator. Present 3 headlines as a markdown bulleted list without intro text."}]},
    }

    try:
        response = requests.post(
            API_URL,
            headers={'Content-Type': 'application/json'},
            params={'key': api_key_for_cache},
            json=payload,
            timeout=15 
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
    st.set_page_config(page_title="LawXplorer: Indian Constitution Assistant", layout="wide")

    st.title("⚖️ LawXplorer: Law and Compliance Assistant")
    st.markdown("A specialized AI tool for legal analysis and guidance on Law and Compliance.")
    st.markdown("---")

    # Sidebar for API Key check and optional manual fetching
    with st.sidebar:
        st.header("Configuration & Disclaimer")
        
        if not API_KEY:
            st.warning("Please set the GEMINI_API_KEY environment variable to use the assistant.")
        
        st.markdown("---")
        
        # Load secondary calls inside expandable blocks to prevent automatic quota depletion
        with st.expander("🏛️ Latest 5 Amendments", expanded=False):
            if API_KEY:
                if st.button("Fetch Amendments", key="btn_amendments"):
                    with st.spinner("Fetching constitutional updates..."):
                        amendment_list = get_latest_amendments(API_KEY)
                        st.markdown(amendment_list)
            else:
                st.info("API Key missing.")

        with st.expander("📰 Top Legal News", expanded=False):
            if API_KEY:
                if st.button("Fetch Latest News", key="btn_news"):
                    with st.spinner("Fetching today's legal headlines..."):
                        news_list = get_law_news(API_KEY)
                        st.markdown(news_list)
            else:
                st.info("API Key missing.")

        st.markdown("---")
        st.markdown(
            "**DISCLAIMER:** LawXplorer provides legally relevant information and cites sources using Google Search grounding. "
            "It is for informational purposes only and is **not a substitute for professional legal advice** from a qualified lawyer."
        )

    st.markdown("### Your Legal Query")

    col1, col2 = st.columns([1, 1])

    with col1:
        topic = st.selectbox(
            "Area of Law/Topic",
            ["Fundamental Rights", "Directive Principles", "Union & State Relations", "Constitutional Amendments", "Judiciary & Courts", "Other"],
            help="Select the broad area of the query."
        )

    with col2:
        article = st.text_input(
            "Relevant Article/Section (Optional)",
            placeholder="e.g., Article 21, Article 19",
            help="Specify a relevant article for focused research."
        )

    query = st.text_area(
        "Your Specific Legal Question or Compliance Scenario",
        placeholder="e.g., What are the current limitations on the Right to Freedom of Speech under Article 19(1)(a)? Provide recent Supreme Court judgments.",
        height=150
    )

    if st.button("Analyze & Get Guidance", type="primary"):
        if not query:
            st.warning("Please enter your legal question before analyzing.")
        else:
            with st.spinner("LawXplorer is consulting the Constitution and legal sources..."):
                response_text, sources = call_gemini_api(topic, article, query)

            if response_text:
                parts = response_text.split("Defense Points", 1)
                analysis_text = parts[0].replace("ANALYSIS:", "").strip()
                defense_text = f"Defense Points{parts[1].strip()}" if len(parts) > 1 else None

                st.markdown("---")
                st.subheader("LawXplorer Analysis & Guidance")
                st.markdown(analysis_text)

                if defense_text:
                    st.markdown("---")
                    st.subheader("🛡️ Legal Defense and Resolution Points")
                    st.markdown(defense_text)

                st.markdown("---")
                st.subheader("📚 Cited Sources (Grounding)")
                
                if sources:
                    source_markdown = ""
                    for i, source in enumerate(sources, 1):
                        source_markdown += f"{i}. [{source['title']}]({source['uri']})\n"
                    st.markdown(source_markdown)
                else:
                    st.info("No external sources were cited for this specific response.")
                
                st.balloons()

if __name__ == "__main__":
    main()
