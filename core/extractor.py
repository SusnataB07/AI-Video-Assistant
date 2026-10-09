# Action items, decisions, open questions

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.llm import get_llm


def build_chain(system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}"),
    ])
    return prompt | get_llm(0.2) | StrOutputParser()


def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the transcript, "
        "extract all action items, including tasks that were agreed or implied. "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )
    return chain.invoke({"text": transcript})


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return chain.invoke({"text": transcript})


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return chain.invoke({"text": transcript})
