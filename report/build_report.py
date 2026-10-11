"""Fill the K. J. Somaiya IA2 report template with the ML Model Advisor Agent report.

    python report/make_diagrams.py     # diagrams + screenshot crops (run once)
    python report/build_report.py      # writes the .docx next to the template

Edit MEMBERS below for the cover page. All numbers quoted in the text come from the saved logs
(logs/) and the evaluation run (eval/results/).
"""
import json
from pathlib import Path

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.text.paragraph import Paragraph

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASSETS = HERE / "assets"
TEMPLATE = ROOT.parent / "AI IA2 report format.docx"
OUT = ROOT.parent / "AI IA2 Report - ML Model Advisor Agent.docx"
TRACE_LOG = ROOT / "logs" / "churn_20261011_213812.jsonl"   # final-design run: first turn + follow-up
REPO_URL = "https://github.com/aayush38/Building-Agentic-AI-System-using-Modern-LLM-based-Tools"

# Cover page: (roll number, name). Replace the placeholders, or edit them directly in Word.
MEMBERS = [("[Roll No]", "Aayush Patil"), ("[Roll No]", "[Team member 2]"), ("[Roll No]", "[Team member 3]")]

FONT = "Times New Roman"
CENTER, JUSTIFY, LEFT = WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.LEFT


# ---------------------------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------------------------
def style_run(run, size=12, bold=None, italic=None, name=FONT, color=None):
    run.font.name = name
    rpr = run._r.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), name)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def shade_cell(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcpr.append(shd)


def remove(par):
    par._p.getparent().remove(par._p)


def set_runs(par, texts):
    """Replace the text of a paragraph's existing runs (keeps their formatting); drop extra runs."""
    for run, text in zip(par.runs, texts):
        run.text = text
    for run in par.runs[len(texts):]:
        run._r.getparent().remove(run._r)


class Flow:
    """Insert paragraphs, bullets, images and tables one after another, after an anchor paragraph."""

    def __init__(self, doc, anchor_par):
        self.doc, self.anchor, self.parent = doc, anchor_par._p, anchor_par._parent

    def _new_p(self, bullet=False, shade=None):
        p = OxmlElement("w:p")
        self.anchor.addnext(p)
        self.anchor = p
        ppr = p.get_or_add_pPr()
        if bullet:  # reuse the template's bullet list (numId 2); numPr must precede spacing/jc
            numpr = OxmlElement("w:numPr")
            ilvl, numid = OxmlElement("w:ilvl"), OxmlElement("w:numId")
            ilvl.set(qn("w:val"), "0")
            numid.set(qn("w:val"), "2")
            numpr.append(ilvl)
            numpr.append(numid)
            ppr.append(numpr)
        if shade:
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear")
            shd.set(qn("w:color"), "auto")
            shd.set(qn("w:fill"), shade)
            ppr.append(shd)
        return Paragraph(p, self.parent)

    def para(self, parts, size=11, align=JUSTIFY, after=4, before=0, line=1.0, keep_next=False,
             bullet=False, shade=None, font=FONT):
        par = self._new_p(bullet=bullet, shade=shade)
        fmt = par.paragraph_format
        fmt.space_after, fmt.space_before, fmt.line_spacing = Pt(after), Pt(before), line
        if keep_next:
            fmt.keep_with_next = True
        par.alignment = align
        if isinstance(parts, str):
            parts = [(parts, {})]
        for text, kw in parts:
            style_run(par.add_run(text), size=kw.get("size", size), bold=kw.get("b"),
                      italic=kw.get("i"), name=kw.get("font", font))
        return par

    def label(self, text):
        """Bold-italic sub-heading in the template's own style."""
        return self.para([(text, {"b": True, "i": True})], size=12, align=LEFT, before=6, after=2, keep_next=True)

    def bullet(self, parts, size=11):
        return self.para(parts, size=size, after=2, bullet=True, align=LEFT)

    def code(self, lines, size=7.5):
        for text in lines:
            self.para(text, size=size, align=LEFT, after=0, line=1.0, shade="F2F2F2", font="Courier New")
        self.para("", size=4, after=2)

    def image(self, path, width_in, caption=None):
        par = self._new_p()
        par.alignment = CENTER
        par.paragraph_format.space_after = Pt(2)
        par.paragraph_format.keep_with_next = bool(caption)
        par.add_run().add_picture(str(path), width=Inches(width_in))
        if caption:
            self.para([(caption, {"i": True})], size=10, align=CENTER, after=8)

    def table(self, rows, widths, size=9.5, header=True, bold_first_col=False):
        t = self.doc.add_table(rows=len(rows), cols=len(rows[0]))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                cell = t.cell(r, c)
                cell.width = Inches(widths[c])
                par = cell.paragraphs[0]
                par.paragraph_format.space_after = Pt(1)
                par.paragraph_format.space_before = Pt(1)
                par.paragraph_format.line_spacing = 1.0
                style_run(par.add_run(str(text)), size=size,
                          bold=(header and r == 0) or (bold_first_col and c == 0 and r > 0))
                if header and r == 0:
                    shade_cell(cell, "D9E2F3")
        self.anchor.addnext(t._tbl)
        self.anchor = t._tbl
        self.para("", size=4, after=4)
        return t


# ---------------------------------------------------------------------------------------------
# data pulled from the real log
# ---------------------------------------------------------------------------------------------
def load_trace():
    recs = [json.loads(line) for line in TRACE_LOG.read_text(encoding="utf-8").splitlines()]
    return recs


def excerpt_lines(recs, steps=(2, 3, 6, 7, 8, 9, 10, 12)):
    """Selected JSONL lines, content trimmed so each fits one line of the report."""
    out = []
    for r in recs:
        if r["step"] not in steps:
            continue
        slim = {k: v for k, v in r.items() if k != "t"}
        if isinstance(slim.get("content"), str) and len(slim["content"]) > 58:
            slim["content"] = slim["content"][:58] + "..."
        out.append(json.dumps(slim, ensure_ascii=False))
    return out


# ---------------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------------
def build():
    doc = docx.Document(str(TEMPLATE))
    P = list(doc.paragraphs)  # snapshot: indexes below refer to the untouched template
    trace = load_trace()
    tokens = [r["tokens"] for r in trace if r.get("tokens")]

    # ---- cover page -------------------------------------------------------------------------
    set_runs(P[3], ["“ML Model Advisor Agent”"])
    for par, (roll, name) in zip(P[6:9], MEMBERS):
        set_runs(par, [f"Roll No: {roll}   ", name])
    remove(P[9])  # template has four member lines; the group has three
    set_runs(P[22], ["ML Model Advisor Agent"])
    set_runs(P[61], ["ML Model Advisor Agent - Conclusion"])
    # The template ends the "Discussion" title with a page break (one title per page). Our content
    # follows the title directly, so drop that break or the title would be stranded on its own page.
    for br in P[60]._p.xpath('.//w:br[@w:type="page"]'):
        run = br.getparent()
        run.getparent().remove(run)

    flows = {name: Flow(doc, P[i]) for name, i in
             dict(intro=25, frameworks=30, tools=34, a=37, b=48, disc=60, concl=61, refs=62).items()}

    # ---- introduction / abstract ------------------------------------------------------------
    f = flows["intro"]
    f.para("Choosing a baseline machine-learning model for a new dataset is a repetitive, multi-step task: "
           "inspect the data, clean it, try several algorithms, compare them fairly and explain the choice. "
           "This project builds an agentic AI system that does this autonomously. The user gives a goal and a "
           "CSV file; an LLM-driven agent plans the analysis, calls tools, reads each observation and decides "
           "what to do next until it can recommend a model with evidence.", before=3)
    f.para("Unlike a classical rule-based agent, the plan is not hard-coded. The LLM chooses the target column, "
           "which identifier columns to drop, which models to compare and which metric to use (macro-F1 when "
           "the classes are imbalanced), and it recovers from tool errors. The system also answers follow-up "
           "questions by reusing the conversation (contextual memory). No model training of the LLM takes "
           "place: the work is reasoning, planning and tool orchestration.")
    f.para([("Problem statement: ", {"b": True}),
            ("given a tabular dataset and a goal such as “recommend a baseline model”, produce a "
             "justified recommendation using only evidence returned by tools.", {})])

    # ---- frameworks and tools -----------------------------------------------------------------
    f = flows["frameworks"]
    f.bullet([("LangGraph", {"b": True}), (" (create_react_agent): ReAct agent loop, plus the MemorySaver "
              "checkpointer that gives per-session conversation memory.", {})])
    f.bullet([("LangChain", {"b": True}), (" (langchain-openai, tool decorators): model wrappers, tool "
              "definitions and the provider fallback chain.", {})])
    f.bullet([("LLM APIs", {"b": True}), (" (OpenAI-compatible): Groq with openai/gpt-oss-120b for all reported "
              "runs; Google Gemini (gemini-3.8-flash) and OpenRouter free models as fallbacks.", {})])
    f.bullet([("scikit-learn, pandas, NumPy", {"b": True}), (": the numerical work done inside the tools. ", {}),
              ("Streamlit", {"b": True}), (": the demo interface.", {})])
    f.para("", size=4, after=2)

    f = flows["tools"]
    f.bullet([("File/document reader: ", {"b": True}), ("CSV loader and profiler (pandas), for the datasets in "
              "data/ or an uploaded file.", {})])
    f.bullet([("Python tools: ", {"b": True}), ("preprocessing, multi-model comparison, hyper-parameter tuning "
              "and permutation feature importance (scikit-learn).", {})])
    f.bullet([("Calculator tool: ", {"b": True}), ("safe arithmetic evaluator based on Python's ast module "
              "(no eval).", {})])
    f.para("", size=4, after=2)

    # ---- part A ---------------------------------------------------------------------------------
    f = flows["a"]
    f.label("Goal of the agent")
    f.para("Given a tabular dataset and a user goal, recommend a baseline ML model and justify it with measured "
           "evidence: cross-validated scores, a tuned score and the features that drive the predictions. "
           "Example goal: “Recommend a baseline model for data/churn.csv”.")
    f.label("Agent role")
    f.para("An autonomous ML consultant and orchestrator. The LLM is the controller: it decides the target "
           "column, the columns to drop, the 4–5 models to compare, the metric, and when it has enough "
           "evidence to answer. The tools do all the numeric work, so every score in the answer comes from a "
           "tool observation and not from the LLM.")
    f.label("Tools used")
    f.para("profile_dataset, preprocess_data, compare_models, analyze_best_model, train_and_evaluate (single "
           "model, for retries) and calculator. Details are in Part B.")
    f.label("Memory type (short-term / context)")
    f.para("Short-term, contextual memory. (1) The conversation, meaning the system prompt, user messages, tool "
           "calls and observations, is stored per session by a LangGraph MemorySaver keyed by a thread id, so a "
           "follow-up such as “Why not random forest?” is answered from earlier results. (2) A shared "
           "workspace in the tool module keeps the loaded DataFrame and the preprocessed X and y between tool "
           "calls. There is no long-term memory across sessions.")
    f.label("Planning strategy (ReAct / step-by-step)")
    f.para("ReAct (Reason, Act, Observe) with a step-by-step plan given in the system prompt: profile, "
           "preprocess, compare models in one batched call, tune and explain the best model, then answer. The "
           "LLM may adapt the plan, for example by retrying after a tool error. Safeguards: one tool call per "
           "turn, a retry after an error, and a recursion limit of 12 steps.")

    # ---- part B ---------------------------------------------------------------------------------
    f = flows["b"]
    f.label("LLM-based agent")
    f.para("The agent is one LLM behind LangGraph's ReAct loop. All reported runs used openai/gpt-oss-120b, an "
           "open-weight 120B model served by Groq through an OpenAI-compatible API. We chose it because it "
           "supports tool calling reliably, answers quickly (a full run took about 7–9 seconds) and its "
           "free tier allowed repeated runs. Temperature is 0 for repeatable behaviour. Gemini "
           "(gemini-3.8-flash) and OpenRouter free models are fallbacks, with retries disabled so a "
           "rate-limited provider is skipped at once. Free-tier limits were the main constraint: OpenRouter "
           "allowed only 50 requests per day, and Groq and Gemini returned rate-limit errors during testing.")
    f.label("Tools used")
    f.para("Table 1. Tools available to the agent.", size=10, align=CENTER, after=2, keep_next=True)
    f.table([
        ["Tool", "What it does", "Why it is needed"],
        ["profile_dataset", "Loads a CSV; reports rows and, per column, dtype, missing count and unique count.",
         "Lets the LLM find the target, ID columns and missing data without seeing raw rows."],
        ["preprocess_data", "Sets the target, drops columns, imputes (median/mode), one-hot encodes, detects "
                            "classification vs regression and class imbalance.",
         "Makes the data model-ready and tells the LLM which metric to use."],
        ["compare_models", "5-fold cross-validation for several models in one call, ranked best first. 8 "
                           "classifiers and 9 regressors are available.",
         "One call instead of one per model; cuts LLM calls and tokens."],
        ["analyze_best_model", "Small grid search (3-fold CV) on the winner plus permutation feature importance.",
         "Adds tuned performance and an explanation of what drives predictions."],
        ["train_and_evaluate", "Cross-validates a single model.", "Retries or adds one model."],
        ["calculator", "Evaluates arithmetic expressions.", "Exact comparison of scores."],
    ], widths=[1.25, 3.0, 2.25], size=9, bold_first_col=True)

    f.label("Multi-step reasoning")
    f.para("Table 2 shows one real run on data/churn.csv (from logs/). The agent needed five LLM calls: four to "
           "decide the next action and one to write the answer. Each observation was fed back before the next "
           "decision, and the agent adapted to what it saw: it dropped customer_id because every value was "
           "unique, and it chose macro-F1 because preprocess_data reported imbalance (22.3% churn).")
    rows = [["#", "Agent action (decision)", "Observation (summary)", "Tokens"],
            ["1", "profile_dataset(data/churn.csv)", "1,500 rows; customer_id unique=1500; monthly_charge 121 "
                                                    "missing; support_calls 120 missing", f"{tokens[0]:,}"],
            ["2", "preprocess_data(target=churn, drop=[customer_id])", "classification, 7 features, class "
                                                                     "balance 0.777 / 0.223, imbalanced", f"{tokens[1]:,}"],
            ["3", "compare_models(5 models, metric=f1_macro)", "gradient_boosting 0.5904 > random_forest "
                                                              "0.5815 > knn 0.5759 > decision_tree 0.5713 > "
                                                              "logistic_regression 0.5703", f"{tokens[2]:,}"],
            ["4", "analyze_best_model(gradient_boosting)", "baseline 0.608, tuned 0.610 (3-fold); top features "
                                                          "tenure_months 0.056, support_calls 0.021, "
                                                          "contract_two_year 0.016", f"{tokens[3]:,}"],
            ["5", "Final answer", "Score table, recommendation: gradient boosting, caveat", f"{tokens[4]:,}"],
            ["", "", "Total (5 LLM calls)", f"{sum(tokens[:5]):,}"]]
    f.para("Table 2. Reasoning trace for one run (Groq, openai/gpt-oss-120b).", size=10, align=CENTER, after=2,
           keep_next=True)
    f.table(rows, widths=[0.3, 2.2, 3.35, 0.65], size=9)
    f.para("The agent also recovers from errors. In an earlier build of the tool set, asked to compare "
           "gradient_boosting, it called a tool that returned “Unknown model 'gradient_boosting' ... Choose "
           "from [...]”, switched to random_forest and finished, and its answer stated that the requested "
           "model was unavailable instead of inventing a score (log: logs/churn_20261011_130035.jsonl). Because "
           "the full history is re-sent on each call, tokens per call grow as the run proceeds "
           f"({tokens[0]:,} to {tokens[4]:,}); the design therefore keeps observations compact.")

    f.label("Architecture diagram")
    f.image(ASSETS / "diagram_architecture.png", 5.3,
            "Figure 1. Architecture: orchestrator agent (LLM, planner, memory, trace logger) calling tool APIs "
            "that work on the data workspace and scikit-learn.")
    f.label("Tool flow diagram")
    f.image(ASSETS / "diagram_toolflow.png", 5.8,
            "Figure 2. Tool flow: Goal → Plan → Tool use → Observation → Final output, with "
            "the error-retry branch.")

    f.label("Observable agent behavior (logs / traces)")
    f.para("Every Thought, Action, Observation and Final answer is written as one JSON line to logs/*.jsonl, with "
           "the step number, the model that produced it and the tokens used by that LLM call. The Streamlit "
           "app shows the same trace live. Excerpt from the run above (content trimmed):", after=4)
    f.code(excerpt_lines(trace))

    f.label("Code repository / notebook")
    f.para([("GitHub: ", {"b": True}), (REPO_URL, {})], align=LEFT, after=3)
    f.para("agent/ (graph.py, tools.py, prompts.py, logger.py), app.py (Streamlit), run_cli.py (terminal demo), "
           "eval/run_eval.py (evaluation harness), data/ (sample CSVs), logs/ (traces). Run with: "
           "python run_cli.py data/churn.csv “Recommend a baseline model”  or  "
           "python -m streamlit run app.py.", align=LEFT)

    f.label("Agent interaction logs")
    f.para("The logs/ folder holds the JSONL trace of every test run. The last two steps of the log excerpted "
           "above show memory at work: the follow-up “Why not random forest?” (step 11) was answered "
           f"in step 12 with a single LLM call ({tokens[5]:,} tokens) and no tool call, using the earlier "
           "results. The follow-up answer is quoted in Figure 6.")

    f.label("Prompts and output screenshots")
    f.para("System prompt (abridged):", after=2, keep_next=True)
    f.code([
        "You are an ML Model Advisor agent. Recommend a baseline model using tools.",
        "Steps (one tool call per turn):",
        "1. profile_dataset(path). Pick the target; drop ID columns.",
        "2. preprocess_data(target, drop_columns).",
        "3. compare_models ONCE with 4-5 models from different families; metric f1_macro if",
        "   imbalanced, else accuracy (classification) or r2 (regression).",
        "4. analyze_best_model ONCE on the top-ranked model to tune it and get top features.",
        "5. If a tool returns an error, fix the arguments and retry once.",
        "6. Final answer under 170 words: summary, score table, recommendation, top 3 features, caveat.",
        "Only report numbers returned by tools.",
    ])
    f.image(ASSETS / "fig_trace.png", 4.6, "Figure 3. Streamlit app, reasoning trace for data/churn.csv "
                                           "(profile, preprocess, compare, analyze).")
    f.image(ASSETS / "fig_answer.png", 4.6, "Figure 4. Final answer: 5 LLM calls, 6,287 tokens (Groq, "
                                            "openai/gpt-oss-120b).")
    f.image(ASSETS / "fig_charts.png", 3.5, "Figure 5. Model comparison (5-fold CV macro-F1) and tuning / "
                                            "feature importance charts.")
    f.image(ASSETS / "fig_followup.png", 4.6, "Figure 6. Follow-up question answered from memory: 1 LLM call, "
                                              "1,953 tokens, no tool calls.")

    # ---- discussion / part C ---------------------------------------------------------------------
    f = flows["disc"]
    f.label("Classical agent versus modern agent")
    f.para("Table 3. Classical agent versus the modern agent built here.", size=10, align=CENTER, after=2, keep_next=True)
    f.table([
        ["Aspect", "Classical agent (syllabus)", "Modern agent (this project)"],
        ["Decision logic", "Rule-based: conditions written by the designer.",
         "LLM-driven: the LLM chose target, models and metric from observations."],
        ["Plan", "Static logic; same steps every time.",
         "Dynamic planning: chose macro-F1 for imbalanced churn data, accuracy for iris; retried after an error."],
        ["Memory", "No memory between percepts and actions.",
         "Contextual memory: the follow-up reused earlier scores without calling a tool."],
        ["Steps", "Single-step percept to action.",
         "Multi-step reasoning: profile, preprocess, compare, tune, answer (5 LLM calls)."],
    ], widths=[1.1, 2.4, 3.0], size=9, bold_first_col=True)

    f.label("Evaluation against a brute-force search")
    f.para("For each dataset we scored every available model with the same 5-fold cross-validation the tools use "
           "(eval/run_eval.py) and compared that true best with the model the agent recommended in one live run.",
           after=3)
    f.para("Table 4. Agent pick versus brute-force best (one live run per dataset).", size=10, align=CENTER, after=2, keep_next=True)
    f.table([
        ["Dataset", "Metric", "True best (all models)", "Agent pick", "Gap", "LLM calls", "Tokens"],
        ["iris", "accuracy", "random_forest 0.9667", "random_forest", "0.0000", "5", "6,082"],
        ["breast_cancer", "accuracy", "logistic_regression 0.9807", "logistic_regression", "0.0000", "5", "7,534"],
        ["churn", "macro-F1", "naive_bayes 0.5945", "gradient_boosting (0.5904)", "0.0041", "5", "6,287"],
    ], widths=[0.95, 0.75, 1.55, 1.45, 0.55, 0.6, 0.65], size=8.5, bold_first_col=True)
    f.para("The agent picked the true best model on 2 of 3 datasets and was within 0.01 of the best on all 3, "
           "using 5 LLM calls and about 6,600 tokens per run on average. This is one run per dataset, so it shows "
           "that the approach works, not how reliable it is.", after=4)

    f.label("Observations and limitations")
    f.bullet("Limited search: on churn the agent compared 5 of 8 classifiers and never tried naive Bayes, the "
             "true best. The agent is only as thorough as the LLM's choice of candidates.", size=11)
    f.bullet("Close calls: on churn the models score between 0.570 and 0.595 macro-F1, so the recommendation is "
             "weakly supported. Tuning raised gradient boosting only from 0.608 to 0.610, and the tuning score "
             "comes from the same folds it was selected on, so it is slightly optimistic.", size=11)
    f.bullet("Free-text claims need checking: in the follow-up answer the LLM called the gap between gradient "
             "boosting and random forest “≈ 1.4% absolute”, but 0.5904 − 0.5815 = 0.009 "
             "absolute (about 1.4% relative), and it said boosting “focuses on the minority class”, "
             "which no tool measured. The numbers from tools were correct; the explanation around them was not "
             "always.", size=11)
    f.bullet("Cost and rate limits: free tiers limited how often we could run the agent. Batching the model "
             "comparison and keeping observations compact reduced a run from 7–9 LLM calls to 5.", size=11)
    f.bullet("Scope of testing: three small datasets, one live run each, and only one kind of follow-up question "
             "tested.", size=11)
    f.label("Takeaway")
    f.para("The LLM is useful as a planner and orchestrator, and deterministic tools keep the numbers honest. "
           "Guardrails (a step cap, error messages the LLM can act on, “only report numbers from tools”) "
           "mattered as much as the model, and verification of the LLM's free text remains necessary.")

    # ---- conclusion --------------------------------------------------------------------------------
    f = flows["concl"]
    f.para("We built an LLM-driven agent that turns a goal and a CSV file into a justified baseline-model "
           "recommendation. It plans with a ReAct loop, uses six tools, remembers the conversation, recovers "
           "from tool errors, logs every step and runs from a terminal or a Streamlit app. On three datasets it "
           "matched the brute-force best model twice and came within 0.004 of it once, with about five LLM calls "
           "per run. Future work: compare all available models by default or let the agent justify skipping "
           "some, evaluate on more datasets and repeated runs, and add a check that compares claims in the "
           "answer with tool outputs.", before=3)

    # ---- references --------------------------------------------------------------------------------
    f = flows["refs"]
    refs = [
        "Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. ICLR 2023.",
        "LangGraph documentation, LangChain Inc. https://langchain-ai.github.io/langgraph/",
        "LangChain documentation. https://python.langchain.com/",
        "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning "
        "Research, 12, 2825–2830.",
        "Streamlit documentation. https://docs.streamlit.io/",
        "Groq, OpenRouter and Google AI Studio API documentation (OpenAI-compatible endpoints). "
        "https://console.groq.com/docs, https://openrouter.ai/docs, https://ai.google.dev/",
        "Hakim, A. AI – IA 2: Agentic AI, course slides and activity instructions, K. J. Somaiya School of "
        "Engineering (2026).",
        f"Project repository: {REPO_URL}",
    ]
    for i, text in enumerate(refs, 1):
        f.para(f"[{i}]  {text}", size=11, align=LEFT, after=3, before=3 if i == 1 else 0)

    # ---- remove template placeholder paragraphs -------------------------------------------------
    for i in [23, 24, *range(26, 30), *range(31, 34), 35, 36, *range(38, 48), *range(49, 59), *range(63, 71)]:
        remove(P[i])

    doc.save(str(OUT))
    print("saved:", OUT)


if __name__ == "__main__":
    build()
