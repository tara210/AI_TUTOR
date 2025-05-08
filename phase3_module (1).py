import os
import getpass
import json
import random
from typing import List, Literal
from transformers import pipeline
import functools

# Define global variables and load the API key
os.environ["GROQ_API_KEY"] = getpass.getpass("🔑 Paste your Groq API key: ")

# Groq setup (for LLM interaction)
from groq import Groq

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
LLM_MODEL = "llama-3.3-70b-versatile"  # fast 70B model

ChatMessage = dict  # simple alias

def chat_completion(messages: List[ChatMessage], temperature: float = 0.3, model: str = LLM_MODEL) -> str:
    """
    Thin wrapper around Groq chat completions.
    """
    resp = groq_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


# Prompt templates
BLOOM_PROMPT = """You are an educational psychologist.
Classify the student's utterance by Bloom’s level:
Remember / Understand / Apply / Analyze / Evaluate / Create.
Utterance: "{u}"
Level:"""

ICAP_PROMPT = """Label the learning behavior by ICAP framework (Interactive, Constructive, Active, Passive).
Utterance: "{u}"
ICAP:"""

MISCON_PROMPT = """Detect specific algebra misconceptions; return JSON list of codes.
Utterance: "{u}"
Misconceptions:"""


# Classifiers for Bloom, ICAP, and Misconception
def classify_bloom(u: str) -> str:
    return chat_completion([{"role":"user","content":BLOOM_PROMPT.format(u=u)}])

def classify_icap(u: str) -> str:
    return chat_completion([{"role":"user","content":ICAP_PROMPT.format(u=u)}])

def detect_miscon(u: str) -> list[str]:
    raw = chat_completion([{"role":"user","content":MISCON_PROMPT.format(u=u)}])
    try:
        return json.loads(raw)
    except Exception:
        return []


# Emotion / sentiment detector (HF pipeline, CPU‑friendly)
device = 0 if torch.cuda.is_available() else -1

@functools.lru_cache(maxsize=1)
def _sent_pipe():
    return pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment", device=device)

@functools.lru_cache(maxsize=1)
def _emo_pipe():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, device=device)

def emotion_sentiment(u: str) -> dict:
    s = _sent_pipe()(u)[0]  # {'label': 'NEG', 'score': …}
    e = {d["label"]: d["score"] for d in _emo_pipe()(u)}
    return {"sentiment": s, "emotion": e}


# Knowledge-Tracing stub (Track student mastery)
class ConceptState:
    def __init__(self, mastery=0.5):
        self.mastery = mastery
        self.t = 0

def update_state(state: ConceptState, correct: bool) -> ConceptState:
    state.t += 1
    delta = 0.2 if correct else -0.1
    state.mastery = min(max(state.mastery + delta, 0.01), 0.99)
    return state

_student_cache: dict[tuple[str,str], ConceptState] = {}

def get_state(student_id: str, concept: str) -> ConceptState:
    return _student_cache.setdefault((student_id, concept), ConceptState())


# Risk score helper
def ras_score(affect: dict, miscon: list[str]) -> int:
    base = 10 * len(miscon)
    if affect["sentiment"]["label"] == "NEGATIVE":
        base += 15 * affect["sentiment"]["score"]
    frustration = affect["emotion"].get("frustration", 0)
    base += 20 * frustration
    return int(min(base, 100))


# Main analysis + decision logic
def analyse(student_id: str, utterance: str, concept: str, correct: bool | None = False):
    bloom = classify_bloom(utterance)
    icap = classify_icap(utterance)
    miscon = detect_miscon(utterance)
    affect = emotion_sentiment(utterance)

    state = get_state(student_id, concept)
    state = update_state(state, bool(correct))

    ras = ras_score(affect, miscon)

    decision = {
        "need_visual": "ALG_DIV_DISTRIB" in miscon,
        "tone": "encourage" if affect["sentiment"]["label"] == "NEGATIVE" else "neutral"
    }

    return {
        "tags": dict(bloom=bloom, icap=icap, misconceptions=miscon, affect=affect, ras=ras),
        "state": vars(state),
        "decision": decision
    }


# Response generator (MM-RAG-lite)
VISUAL_GIF = "https://cdn.mathvisuals.com/perfect_square.gif"

def generate_reply(utterance: str, analysis: dict) -> dict:
    sys = "You are an expert psychologist-tutor. Use Socratic guidance first, then clarify."
    usr = f"""Student: "{utterance}"

JSON analysis:
{json.dumps(analysis['tags'], indent=2)}

Respond with:
1. Socratic question if bloom is lower than Analyze.
2. Correct explanation.
3. Motivational tip if sentiment negative."""
    reply = chat_completion([{"role": "system", "content": sys},
                             {"role": "user", "content": usr}], temperature=0.4)

    assets = [VISUAL_GIF] if analysis["decision"]["need_visual"] else []
    return dict(reply=reply, ras=analysis["tags"]["ras"], assets=assets)
