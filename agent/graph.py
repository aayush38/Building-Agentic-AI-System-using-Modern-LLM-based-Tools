"""ReAct agent built with LangGraph, using an OpenRouter-hosted LLM."""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from .logger import TraceLogger
from .prompts import SYSTEM_PROMPT
from .tools import TOOLS

load_dotenv()


# All providers expose an OpenAI-compatible API. name -> (key env var, base URL, model env var, default model)
PROVIDERS = {
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1",
                   "MODEL_NAME", "nvidia/nemotron-3-super-120b-a12b:free"),
    "gemini": ("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/",
               "GEMINI_MODEL", "gemini-3.8-flash"),
    "groq": ("GROQ_API_KEY", "https://api.groq.com/openai/v1",
             "GROQ_MODEL", "openai/gpt-oss-120b"),
    "xai": ("XAI_API_KEY", "https://api.x.ai/v1", "XAI_MODEL", "grok-4"),
}


def build_llm():
    """Primary LLM with fallbacks. Order comes from LLM_CHAIN (default: all providers that
    have a key set). If a call fails (rate limit, outage), the next provider is tried."""
    if os.getenv("GROK_API_KEY") and not os.getenv("XAI_API_KEY"):  # accept either name
        os.environ["XAI_API_KEY"] = os.environ["GROK_API_KEY"]
    order = [p.strip() for p in os.getenv("LLM_CHAIN", ",".join(PROVIDERS)).split(",")]
    models = []
    for name in order:
        key_var, url, model_var, default = PROVIDERS[name]
        key = os.getenv(key_var)
        if not key or key.startswith("your_"):
            continue
        llm = ChatOpenAI(model=os.getenv(model_var, default), api_key=key,
                         base_url=url, temperature=0,
                         max_retries=0, timeout=60)  # fail fast; the fallback chain takes over
        models.append(llm.bind_tools(TOOLS))
    if not models:
        raise RuntimeError("No API key found. Set at least one of: "
                           + ", ".join(v[0] for v in PROVIDERS.values()))
    return models[0].with_fallbacks(models[1:]) if len(models) > 1 else models[0]


def build_agent():
    return create_react_agent(build_llm(), TOOLS, prompt=SYSTEM_PROMPT)


def run_agent(goal: str, run_name: str = "run", max_steps: int = 12):
    """Run the agent; yield one trace record per Thought/Action/Observation/Final step."""
    agent = build_agent()
    logger = TraceLogger(run_name)
    yield logger.log("goal", goal)
    config = {"recursion_limit": max_steps}
    for chunk in agent.stream({"messages": [("user", goal)]}, config, stream_mode="updates"):
        for node, update in chunk.items():
            for msg in update.get("messages", []):
                if node == "agent":
                    model = (msg.response_metadata or {}).get("model_name")
                    tokens = (msg.usage_metadata or {}).get("total_tokens")  # one LLM call
                    if msg.content:
                        kind = "thought" if msg.tool_calls else "final_answer"
                        yield logger.log(kind, msg.content, model=model, tokens=tokens)
                        tokens = None  # count each LLM call's tokens once
                    for call in msg.tool_calls:
                        yield logger.log("action", call["name"], args=call["args"],
                                         model=model, tokens=tokens)
                        tokens = None
                elif node == "tools":
                    yield logger.log("observation", msg.content, tool=msg.name)
