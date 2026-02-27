"""
LLM Generation Layer
- Prompt engineering with templates, few-shot, chain-of-thought
- Guardrails (topic relevance, hallucination reduction, output constraints)
- LangChain agents with tool-calling (SQL query + web search)
- Executive summary + theme extraction
"""

import time
import logging
import re
from typing import List, Dict, Optional, Any

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.core.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

client = OpenAI(api_key=settings.OPENAI_API_KEY)


# ──────────────────────────────────────────────
# PROMPT TEMPLATES
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert analyst summarizing survey responses for executives.
Your job is to:
1. Identify the key themes in the responses
2. Write concise, factual summaries grounded ONLY in the provided responses
3. Never hallucinate or add information not present in the responses
4. Flag uncertainty when evidence is limited

Format your response as structured JSON only."""

SUMMARIZE_TEMPLATE = """Analyze the following survey responses related to the query: "{query}"

SURVEY RESPONSES:
{context}

INSTRUCTIONS:
- Identify up to {max_themes} distinct themes
- For each theme provide: theme name, 1-2 sentence summary, 2-3 supporting quotes
- Write an executive summary (3-4 sentences) covering all themes
- Confidence: high (5+ responses), medium (2-4), low (1)
- Only use information from the provided responses

GUARDRAILS:
- Do NOT speculate beyond the provided text
- If a theme has fewer than 2 supporting responses, mark confidence as "low"
- Do NOT include personally identifiable information

Respond ONLY with valid JSON in this format:
{{
  "executive_summary": "...",
  "themes": [
    {{
      "theme": "...",
      "summary": "...",
      "supporting_responses": ["...", "..."],
      "response_count": 0,
      "confidence": 0.0
    }}
  ]
}}"""

FEW_SHOT_EXAMPLES = [
    {
        "query": "What do employees think about remote work?",
        "context": "I love the flexibility... saves 2 hours commuting... miss team collaboration... productivity is up...",
        "output": '{"executive_summary": "Employees broadly value remote work for flexibility and productivity gains, though collaboration remains a concern.", "themes": [{"theme": "Flexibility & Work-Life Balance", "summary": "Employees appreciate the ability to manage their schedules.", "supporting_responses": ["I love the flexibility"], "response_count": 1, "confidence": 0.6}]}'
    }
]


def build_few_shot_prefix() -> str:
    """Build few-shot examples string."""
    examples = []
    for ex in FEW_SHOT_EXAMPLES:
        examples.append(f"EXAMPLE:\nQuery: {ex['query']}\nContext: {ex['context']}\nOutput: {ex['output']}")
    return "\n\n".join(examples)


def apply_guardrails(text: str, query: str) -> Dict[str, Any]:
    """
    Pre/post generation guardrails:
    - Input: check query is survey-related
    - Output: validate JSON structure, check for hallucination markers
    """
    guardrail_flags = []

    # Input guardrail: reject off-topic queries
    off_topic_patterns = [
        r'\b(password|credit card|ssn|social security)\b',
        r'\b(hack|exploit|inject)\b',
    ]
    for pattern in off_topic_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            guardrail_flags.append(f"Input blocked: matched pattern '{pattern}'")

    # Output guardrail: check for uncertainty markers
    uncertainty_phrases = ["I think", "I believe", "probably", "might be", "could be", "I'm not sure"]
    for phrase in uncertainty_phrases:
        if phrase.lower() in text.lower():
            guardrail_flags.append(f"Uncertainty detected: '{phrase}' — verify with source data")

    return {
        "flagged": len(guardrail_flags) > 0,
        "flags": guardrail_flags
    }


# ──────────────────────────────────────────────
# CORE GENERATION
# ──────────────────────────────────────────────

class SummaryGenerator:
    """
    Generates structured executive summaries from retrieved chunks.
    """

    def __init__(self):
        self.model = settings.OPENAI_MODEL
        self.strong_model = settings.OPENAI_MODEL_STRONG

    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks as numbered context."""
        lines = []
        for i, chunk in enumerate(chunks, 1):
            lines.append(f"[{i}] {chunk.text}")
        return "\n".join(lines)

    def generate(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        max_themes: int = 5,
        use_strong_model: bool = False
    ) -> Dict[str, Any]:
        """
        Generate executive summary with theme extraction.

        Returns structured dict with executive_summary + themes.
        """
        start = time.perf_counter()
        model = self.strong_model if use_strong_model else self.model

        context = self._format_context(chunks)

        # Guardrail check on input
        guardrail_result = apply_guardrails("", query)
        if guardrail_result["flagged"]:
            logger.warning(f"Guardrail triggered: {guardrail_result['flags']}")
            return {
                "executive_summary": "Query blocked by content guardrails.",
                "themes": [],
                "guardrail_flags": guardrail_result["flags"],
                "model": model,
                "latency_ms": 0
            }

        prompt = SUMMARIZE_TEMPLATE.format(
            query=query,
            context=context,
            max_themes=max_themes
        )

        few_shot = build_few_shot_prefix()

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{few_shot}\n\nNOW ANALYZE:\n{prompt}"}
                ],
                temperature=0.1,       # Low temp for factual consistency
                max_tokens=2000,
                response_format={"type": "json_object"}  # Force JSON output
            )

            raw_output = response.choices[0].message.content
            import json
            result = json.loads(raw_output)

            # Post-generation guardrail
            post_guard = apply_guardrails(raw_output, query)
            result["guardrail_flags"] = post_guard.get("flags", [])

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            result = {
                "executive_summary": f"Generation error: {str(e)}",
                "themes": []
            }

        latency_ms = (time.perf_counter() - start) * 1000
        result["model"] = model
        result["latency_ms"] = latency_ms
        return result


# ──────────────────────────────────────────────
# LANGCHAIN AGENT WITH TOOL-CALLING
# ──────────────────────────────────────────────

def make_sql_tool(db_connection=None):
    """Tool: query a SQL database for structured survey metadata."""
    def run_sql(query: str) -> str:
        # Stub: in production, connect to PostgreSQL
        return f"[SQL Tool] Query executed: {query[:100]}. Results: (stub - connect PostgreSQL)"
    return Tool(name="sql_query", func=run_sql,
                description="Query survey metadata from PostgreSQL. Use for counts, filters, date ranges.")


def make_rag_tool(retriever):
    """Tool: semantic search over survey responses."""
    def run_rag(query: str) -> str:
        chunks, _ = retriever.retrieve(query, top_k=5, mode="hybrid")
        return "\n".join([f"- {c.text[:200]}" for c in chunks])
    return Tool(name="semantic_search", func=run_rag,
                description="Semantic search over survey responses. Use for open-ended theme discovery.")


class SurveyAnalysisAgent:
    """
    LangChain agent with tool-calling for multi-step survey analysis.
    Tools: semantic_search, sql_query
    """

    def __init__(self, retriever=None):
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0,
            api_key=settings.OPENAI_API_KEY
        )
        self.retriever = retriever

        tools = []
        if retriever:
            tools.append(make_rag_tool(retriever))
        tools.append(make_sql_tool())

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a survey analysis agent. Use your tools to:
1. Search for relevant survey responses using semantic_search
2. Query metadata using sql_query when you need counts or filters
3. Synthesize findings into a clear executive answer

Always cite which tool provided the evidence."""),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])

        agent = create_openai_tools_agent(self.llm, tools, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )

    def run(self, query: str, chat_history: List = None) -> Dict[str, Any]:
        """Run multi-step agent reasoning."""
        start = time.perf_counter()
        try:
            result = self.executor.invoke({
                "input": query,
                "chat_history": chat_history or []
            })
            output = result.get("output", "")
        except Exception as e:
            logger.error(f"Agent error: {e}")
            output = f"Agent error: {str(e)}"

        latency_ms = (time.perf_counter() - start) * 1000
        return {"answer": output, "latency_ms": latency_ms}
