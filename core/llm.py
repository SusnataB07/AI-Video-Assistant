"""One place to choose the LLM. Change LLM_PROVIDER in .env to switch."""
import os


def get_llm(temperature: float = 0.3):
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    model = os.getenv("LLM_MODEL", "").strip()

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model or "llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=temperature,
            max_retries=6,
        )

    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(
            model=model or "mistral-small-latest",
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            temperature=temperature,
            max_retries=6,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model or "gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
            max_retries=6,
        )

    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use groq, mistral or gemini.")
