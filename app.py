"""Personal HR Benefits Assistant - capstone demo.

Five coordinated roles are implemented as components in one LangChain Runnable
workflow. This is a local demonstration, NOT a production-authenticated HR system.
"""
from __future__ import annotations

import os, re, csv, json, hashlib
from pathlib import Path
from datetime import datetime, timezone, date
from typing import TypedDict, Any

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
POLICY_DIR = DATA_DIR / "policies"
EMPLOYEE_FILE = DATA_DIR / "employees.csv"
CURRENT_POLICY_YEAR = int(os.getenv("POLICY_YEAR", str(date.today().year)))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

st.set_page_config(page_title="Personal HR Benefits Assistant", page_icon="🍊", layout="wide")
st.title("🍊 Personal HR Benefits Assistant")
st.caption("Capstone demonstration • Synthetic employee records and sample policies only")

# -------------------------- Structured workflow state --------------------------
class HRState(TypedDict, total=False):
    question: str
    route: str
    employee_id: str
    demo_authorized: bool
    employee: dict | None
    employee_retrieved_at: str | None
    policy_results: list[dict]
    candidate_branches: list[dict]
    evaluations: list[dict]
    hard_rule_checks: list[dict]
    conflicts: list[str]
    needs_hr: bool
    answer: str
    final_sources: list[dict]

# -------------------------- Document ingestion / RAG ---------------------------
def file_signature() -> str:
    """Changes whenever a supported policy file is added, edited, or removed."""
    entries = []
    for p in DATA_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".pdf", ".docx", ".html", ".htm"}:
            stat = p.stat()
            entries.append(f"{p.relative_to(DATA_DIR)}:{stat.st_mtime_ns}:{stat.st_size}")
    return hashlib.sha256("|".join(sorted(entries)).encode()).hexdigest()

def read_policy_files() -> list[dict]:
    docs = []
    for path in sorted(POLICY_DIR.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == ".txt":
                docs.append({"source": path.name, "text": path.read_text(encoding="utf-8"), "pages": ""})
            elif suffix == ".html" or suffix == ".htm":
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
                for tag in soup(["script", "style", "noscript"]): tag.decompose()
                docs.append({"source": path.name, "text": soup.get_text("\n", strip=True), "pages": ""})
            elif suffix == ".pdf":
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                for n, page in enumerate(reader.pages, 1):
                    text = page.extract_text() or ""
                    if text.strip(): docs.append({"source": path.name, "text": text, "pages": str(n)})
            elif suffix == ".docx":
                from docx import Document
                doc = Document(str(path))
                text = "\n".join(p.text for p in doc.paragraphs)
                for table in doc.tables:
                    for row in table.rows:
                        text += "\n" + " | ".join(cell.text for cell in row.cells)
                docs.append({"source": path.name, "text": text, "pages": ""})
        except Exception as exc:
            st.warning(f"Could not read {path.name}: {type(exc).__name__}. Check the relevant parser dependency.")
    return docs

def parse_metadata(text: str) -> dict:
    def get(pattern, default=""):
        m = re.search(pattern, text, re.I)
        return m.group(1).strip() if m else default
    year_text = get(r"Policy Year:\s*(\d{4})", str(CURRENT_POLICY_YEAR))
    try: year = int(year_text)
    except ValueError: year = CURRENT_POLICY_YEAR
    return {
        "policy_year": year,
        "status": get(r"Status:\s*(Active|Expired|Draft)", "Active").title(),
        "effective_date": get(r"Effective Date:\s*(\d{4}-\d{2}-\d{2})", f"{year}-01-01"),
        "expiration_date": get(r"Expiration Date:\s*(\d{4}-\d{2}-\d{2})", f"{year}-12-31"),
    }

def chunk_documents(docs: list[dict], size=150, overlap=30) -> list[dict]:
    chunks = []
    for doc in docs:
        words = doc["text"].split()
        meta = parse_metadata(doc["text"])
        step = max(1, size-overlap)
        for start in range(0, len(words), step):
            text = " ".join(words[start:start+size]).strip()
            if not text: continue
            page_match = re.search(r"\[?PAGE\s+(\d+)\]?", text, re.I)
            page = page_match.group(1) if page_match else doc.get("pages", "")
            chunks.append({"source": doc["source"], "text": text, "page": page, **meta})
            if start + size >= len(words): break
    return chunks

@st.cache_data(show_spinner="Loading and indexing policy documents…")
def cached_index(_signature: str):
    chunks = chunk_documents(read_policy_files())
    if not chunks: return chunks, None, None
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform([c["text"] for c in chunks])
    return chunks, vectorizer, matrix

def active_policy(c: dict) -> bool:
    today = date.today().isoformat()
    try:
        return (c["status"].lower() == "active" and c["policy_year"] == CURRENT_POLICY_YEAR
                and c["effective_date"] <= today <= c["expiration_date"])
    except (ValueError, TypeError):
        return False

def retrieve(query: str, k=5) -> list[dict]:
    chunks, vectorizer, matrix = cached_index(file_signature())
    if not chunks or vectorizer is None: return []
    # Hard metadata filtering is applied BEFORE relevance ranking.
    eligible = [i for i, c in enumerate(chunks) if active_policy(c)]
    if not eligible: return []
    scores = cosine_similarity(vectorizer.transform([query]), matrix[eligible]).flatten()
    ranked = sorted(zip(eligible, scores), key=lambda x: x[1], reverse=True)
    return [{**chunks[i], "score": float(score)} for i, score in ranked[:k] if score > 0]

# -------------------------- Employee data access -------------------------------
def load_employee(employee_id: str) -> dict | None:
    """Read CSV fresh on each lookup; ID alone is not treated as authentication."""
    if not EMPLOYEE_FILE.exists(): return None
    with EMPLOYEE_FILE.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("employee_id", "").strip().casefold() == employee_id.strip().casefold():
                return row
    return None

# -------------------------- Five coordinated roles -----------------------------
def coordinator_agent(state: HRState) -> HRState:
    q = state["question"].lower()
    personal_terms = ("my ", "i am", "i'm", "am i", "do i", "did i", "my contribution", "my coverage", "which plan")
    general_terms = ("what does", "what is", "explain", "company offer", "policy", "how does", "plan options")
    mixed_terms = ("eligible", "eligibility", "qualify", "can i enroll", "am i covered")
    personal = any(x in q for x in personal_terms)
    eligibility = any(x in q for x in mixed_terms)
    # Personal wording takes precedence over generic phrases such as "what is".
    # Eligibility questions need both policy evidence and the employee record.
    route = "mixed" if personal and eligibility else ("personal" if personal else "general")
    state["route"] = route
    state["needs_hr"] = False
    return state

def policy_research_agent(state: HRState) -> HRState:
    state["policy_results"] = retrieve(state["question"], 8) if state["route"] in ("general", "mixed") else []
    return state

def employee_data_agent(state: HRState) -> HRState:
    state["employee"] = None
    state["employee_retrieved_at"] = None
    if state["route"] in ("personal", "mixed"):
        # Demo checkbox is only a demo gate, NOT identity verification.
        if not state.get("demo_authorized"):
            state["needs_hr"] = True
        else:
            state["employee"] = load_employee(state.get("employee_id", ""))
            state["employee_retrieved_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
            if state["employee"] is None: state["needs_hr"] = True
    return state

def detect_policy_conflicts(results: list[dict]) -> list[str]:
    """Flag differing values for the same labelled rule across distinct active files."""
    by_topic: dict[str, dict[str, dict[str, set[str]]]] = {}
    for r in results:
        name = r["source"].lower()
        topic = next((t for t in ["medical", "dental", "vision", "hsa", "401k"] if t in name), None)
        if not topic: continue
        for label in ["deductible", "per paycheck", "annual maximum", "annual max", "frame allowance", "employer match"]:
            for m in re.finditer(rf"{label}[^.\n]{{0,70}}", r["text"], re.I):
                value = re.sub(r"\s+", " ", m.group(0).lower()).strip()
                by_topic.setdefault(topic, {}).setdefault(label, {}).setdefault(r["source"], set()).add(value)
    conflicts = []
    for topic, labels in by_topic.items():
        for label, source_values in labels.items():
            # Multiple chunks from one file are not independent conflicting authorities.
            if len(source_values) > 1:
                distinct = {v for values in source_values.values() for v in values}
                if len(distinct) > 1:
                    conflicts.append(f"Potential conflicting {topic} policy statements across {len(source_values)} active files for '{label}'. HR must confirm the authoritative wording.")
    return conflicts

def thought_generator_agent(state: HRState) -> HRState:
    q = state["question"]
    if not any(t in q.lower() for t in ["compare", "which is best", "best plan", "options", "difference"]):
        state["candidate_branches"] = []
        return state
    # Selective ToT: three query branches, depth 2, retain best two by evidence score.
    branches = [q, q + " eligibility costs employee contributions", q + " coverage deductible limits employer contribution"]
    candidates = []
    for branch in branches[:3]:
        hits = retrieve(branch, 4)
        candidates.append({"branch": branch, "evidence": hits, "score": sum(x["score"] for x in hits)})
    candidates.sort(key=lambda x: x["score"], reverse=True)
    state["candidate_branches"] = candidates[:2]
    return state

def evaluation_validation_agent(state: HRState) -> HRState:
    results = state.get("policy_results", [])
    conflicts = detect_policy_conflicts(results)
    state["conflicts"] = conflicts
    evaluations = []
    for candidate in state.get("candidate_branches", []):
        supported = bool(candidate["evidence"])
        evaluations.append({"branch": candidate["branch"], "evidence_found": supported,
                            "evidence_count": len(candidate["evidence"]), "score": round(candidate["score"], 3)})
    state["evaluations"] = evaluations
    # Deterministic eligibility check for an explicit full-time/30-hour rule.
    # If the employee CSV lacks weekly hours, do not assume the threshold is met.
    checks = []
    employee = state.get("employee")
    policy_text = " ".join(r["text"].lower() for r in results)
    if employee and ("30 hours" in policy_text or "30+ hours" in policy_text):
        status = employee.get("employment_status", "").strip().lower()
        hours = employee.get("hours_per_week", "").strip()
        if status == "part-time":
            checks.append({"rule": "Full-time / 30-hour eligibility", "result": "Not met based on recorded part-time status"})
        elif hours:
            try:
                checks.append({"rule": "Full-time / 30-hour eligibility", "result": "Meets threshold" if float(hours) >= 30 else "Does not meet threshold"})
            except ValueError:
                checks.append({"rule": "Full-time / 30-hour eligibility", "result": "Cannot validate: hours field is not numeric"})
        else:
            checks.append({"rule": "Full-time / 30-hour eligibility", "result": "Cannot confirm: weekly hours are not present in the sample employee record"})
            state["needs_hr"] = True
    state["hard_rule_checks"] = checks
    if conflicts: state["needs_hr"] = True
    if state["route"] in ("general", "mixed") and not results: state["needs_hr"] = True
    return state

def build_context(state: HRState) -> str:
    bits = []
    for r in state.get("policy_results", []):
        bits.append(f"SOURCE: {r['source']} | POLICY YEAR: {r['policy_year']} | EFFECTIVE: {r['effective_date']} | EXPIRES: {r['expiration_date']} | PAGE: {r['page'] or 'not detected'}\n{r['text']}")
    emp = state.get("employee")
    if emp: bits.append("SYNTHETIC EMPLOYEE RECORD: " + json.dumps(emp, ensure_ascii=False))
    if state.get("conflicts"): bits.append("CONFLICT WARNINGS: " + " ".join(state["conflicts"]))
    return "\n\n".join(bits)

def generate_answer(state: HRState) -> str:
    q, route = state["question"], state["route"]
    if route in ("personal", "mixed") and not state.get("demo_authorized"):
        return ("Personal employee information was not accessed. The local demo checkbox is not authentication. "
                "In a real deployment, sign-in and authorization must be completed through company SSO and role-based access controls. Please contact HR for your personal record.")
    if route in ("personal", "mixed") and not state.get("employee"):
        return "I could not locate the requested synthetic employee record. Please verify the demo ID or contact HR."
    if state.get("conflicts"):
        return "I found policy statements that may conflict. I will not choose one or guess which is authoritative. Please ask HR to confirm the current policy before relying on this information."
    context = build_context(state)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and context:
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage
            llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1, api_key=api_key)
            system = ("You are a careful HR benefits information assistant for a synthetic capstone demo. "
                      "Use ONLY the supplied evidence. Do not invent facts or infer enrollment from eligibility. "
                      "Distinguish policy eligibility from the employee's recorded election. If evidence is missing, "
                      "unclear, or insufficient, say so and recommend HR. Never submit or recommend a binding election. "
                      "Cite filenames and page numbers. Use plain, friendly English.")
            response = llm.invoke([SystemMessage(content=system), HumanMessage(content=f"Question type: {route}\nQuestion: {q}\nEvidence:\n{context}")])
            return str(response.content)
        except Exception as exc:
            st.info(f"LLM unavailable ({type(exc).__name__}); using evidence-only fallback.")
    # Conservative fallback: show retrieved facts without pretending to interpret them.
    if state.get("policy_results"):
        excerpts = "\n\n".join(f"**{r['source']}** (page {r['page'] or 'not detected'}): {r['text'][:650]}" for r in state["policy_results"][:3])
        answer = "I found the following current-policy excerpts. Please review them with HR if your situation requires an eligibility determination.\n\n" + excerpts
    else:
        answer = "I do not have enough current policy evidence to answer confidently. Please contact HR."
    emp = state.get("employee")
    if emp:
        answer += (f"\n\n**Synthetic employee record:** {emp.get('employee_name','Employee')} — "
                   f"medical: {emp.get('medical_plan')}; dental: {emp.get('dental_plan')}; "
                   f"vision: {emp.get('vision_plan')}; HSA contribution: {emp.get('hsa_contribution')}; "
                   f"401(k) contribution: {emp.get('401k_contribution')}.\n\n"
                   "This shows the sample CSV record, not proof of eligibility or a live benefits-system election.")
    return answer

def final_controller_agent(state: HRState) -> HRState:
    state["answer"] = generate_answer(state)
    return state

# LangChain is used to compose the five roles into one sequential workflow.
def build_workflow():
    try:
        from langchain_core.runnables import RunnableLambda
        return (RunnableLambda(coordinator_agent)
                | RunnableLambda(policy_research_agent)
                | RunnableLambda(employee_data_agent)
                | RunnableLambda(thought_generator_agent)
                | RunnableLambda(evaluation_validation_agent)
                | RunnableLambda(final_controller_agent))
    except ImportError:
        return None

WORKFLOW = build_workflow()
def run_workflow(state: HRState) -> HRState:
    if WORKFLOW is not None: return WORKFLOW.invoke(state)
    # Safe sequential fallback if LangChain dependencies were not installed.
    for role in [coordinator_agent, policy_research_agent, employee_data_agent,
                 thought_generator_agent, evaluation_validation_agent, final_controller_agent]:
        state = role(state)
    return state

# ----------------------------------- Sidebar ------------------------------------
with st.sidebar:
    st.header("Demo settings")
    st.write(f"Policy year filter: **{CURRENT_POLICY_YEAR}**")
    st.caption("Only Active policies for the selected year and current effective dates are retrieved.")
    st.divider()
    st.subheader("Synthetic employee lookup")
    employee_id = st.text_input("Synthetic Employee ID", value="E1001")
    demo_authorized = st.checkbox("I understand this is synthetic demo data and want to enable the sample lookup", value=False)
    st.warning("This checkbox is only a demo gate. An employee ID or checkbox is NOT authentication. Production requires SSO, authorization, and audit logging.")
    st.divider()
    st.write("**LLM status**")
    st.write("OpenAI key configured" if os.getenv("OPENAI_API_KEY") else "No API key: evidence-only fallback")
    if st.button("Refresh HR Data", use_container_width=True):
        st.cache_data.clear()
        st.success("Policy index cache cleared. Files will be re-read on the next question; employee CSV is read fresh per lookup.")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------ Chat --------------------------------------
if "messages" not in st.session_state: st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

prompt = st.chat_input("Ask about medical, dental, vision, HSA, 401(k), or your sample enrollment…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    initial: HRState = {"question": prompt, "employee_id": employee_id, "demo_authorized": demo_authorized}
    state = run_workflow(initial)
    answer = state["answer"]
    if state.get("employee_retrieved_at"):
        answer += f"\n\n**Synthetic record retrieval time:** {state['employee_retrieved_at']}"
    if state.get("policy_results"):
        answer += "\n\n**Policy sources retrieved**\n" + "\n".join(
            f"- {r['source']} | policy year {r['policy_year']} | effective {r['effective_date']} to {r['expiration_date']} | page {r['page'] or 'not detected'} | relevance {r['score']:.2f}"
            for r in state["policy_results"][:5])
    if state.get("hard_rule_checks"):
        answer += "\n\n**Deterministic rule checks (demo):**\n" + "\n".join(
            f"- {item['rule']}: {item['result']}" for item in state["hard_rule_checks"])
    if state.get("evaluations"):
        answer += "\n\n**ToT evaluation trace (selective, demo):**\n" + "\n".join(
            f"- Candidate branch: {e['branch']} — evidence items: {e['evidence_count']}, score: {e['score']}"
            for e in state["evaluations"])
        answer += "\n\nThe workflow considers up to three query branches, depth two, and retains the two strongest evidence paths. This is a limited evidence-search strategy, not autonomous benefit decision-making."
    if state.get("needs_hr"):
        answer += "\n\n**Human intervention:** HR review is recommended because authorization, source evidence, or record completeness needs confirmation."
    with st.chat_message("assistant"): st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.expander("Architecture and safety notes"):
    st.markdown("""
**Five roles in one LangChain workflow**
1. **Coordinator/Controller:** classifies general, personal, and mixed requests and manages workflow completion.
2. **Policy Research:** ingests PDF, DOCX, TXT, and HTML documents; filters status, policy year, and effective dates before TF-IDF ranking.
3. **Employee Data:** reads the synthetic CSV only after the demo gate; the CSV is re-read for each lookup.
4. **Thought Generator:** selectively explores up to three evidence-query branches for comparison questions and retains the best two.
5. **Evaluation/Validation:** checks evidence availability, flags potential policy conflicts, and routes uncertain cases to HR.

**Production requirements not implemented by this local capstone:** company SSO, verified identity, role-based access control, encryption/key management, secure HRIS integration, immutable audit logs, approved policy publishing, and security testing. The demo employee ID and checkbox do not authenticate anyone. The app does not change enrollments or employee records.

**Evaluation plan:** test policy-answer accuracy and citations against HR-reviewed questions; measure eligibility-rule validation, retrieval precision, conflict/escalation accuracy, response time, and authorization denial. Suggested targets (not measured results): ≥90% answer accuracy on a verified test set, ≥95% source-grounded policy answers, and 100% denial of unauthorized record access in security tests.
""")
