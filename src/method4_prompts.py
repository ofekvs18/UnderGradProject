"""
method4_prompts.py — Prompt templates for LLM-based biomarker generation (issue #20).

Provides blind and seeded prompt strategies for Med-Gemma 4B to produce
CBC-based biomarker formulas for Rheumatoid Arthritis prediction.

No inference is executed here; this module is imported by the runner script.
"""

# ── CBC feature definitions ────────────────────────────────────────────────────
CBC_FEATURES = {
    "hct": {
        "name": "Hematocrit",
        "description": "Percentage of red blood cells in total blood volume",
        "unit": "%",
        "normal_range": "37–52%",
        "ra_relevance": "Often decreased in RA due to anemia of chronic disease",
    },
    "hgb": {
        "name": "Hemoglobin",
        "description": "Oxygen-carrying protein in red blood cells",
        "unit": "g/dL",
        "normal_range": "12–17 g/dL",
        "ra_relevance": "Reduced in anemia of chronic inflammation common in RA",
    },
    "mch": {
        "name": "Mean Corpuscular Hemoglobin",
        "description": "Average amount of hemoglobin per red blood cell",
        "unit": "pg",
        "normal_range": "27–33 pg",
        "ra_relevance": "May be reduced in iron-deficiency anemia secondary to RA",
    },
    "mchc": {
        "name": "Mean Corpuscular Hemoglobin Concentration",
        "description": "Average concentration of hemoglobin in red blood cells",
        "unit": "g/dL",
        "normal_range": "32–36 g/dL",
        "ra_relevance": "Decreased in hypochromic anemias associated with RA",
    },
    "mcv": {
        "name": "Mean Corpuscular Volume",
        "description": "Average size/volume of red blood cells",
        "unit": "fL",
        "normal_range": "80–100 fL",
        "ra_relevance": "Normocytic or microcytic pattern common in RA anemia",
    },
    "plt": {
        "name": "Platelets",
        "description": "Cell fragments involved in clotting and inflammation",
        "unit": "10^9/L",
        "normal_range": "150–400 × 10⁹/L",
        "ra_relevance": "Reactive thrombocytosis is a well-documented marker of active RA",
    },
    "rbc": {
        "name": "Red Blood Cell Count",
        "description": "Number of red blood cells per volume of blood",
        "unit": "10^12/L",
        "normal_range": "4.2–5.8 × 10¹²/L",
        "ra_relevance": "Decreased count observed in RA-associated anemia",
    },
    "rdw": {
        "name": "Red Cell Distribution Width",
        "description": "Variation in red blood cell size (anisocytosis)",
        "unit": "%",
        "normal_range": "11.5–14.5%",
        "ra_relevance": "Elevated RDW is a systemic inflammation marker linked to RA severity",
    },
    "wbc": {
        "name": "White Blood Cell Count",
        "description": "Total count of immune cells in blood",
        "unit": "10^9/L",
        "normal_range": "4.5–11.0 × 10⁹/L",
        "ra_relevance": "Can be elevated in active RA inflammation or lowered by immunosuppressive treatment",
    },
}

# ── Internal helpers ───────────────────────────────────────────────────────────
_FEATURE_BLOCK = "\n".join(
    f"  - {var}: {info['name']} ({info['unit']}) — {info['normal_range']}. "
    f"{info['ra_relevance']}"
    for var, info in CBC_FEATURES.items()
)

_FORMAT_SPEC = """\
OUTPUT FORMAT (strict):
Return exactly {n_formulas} formula(s), one per line, using this syntax:
  FORMULA: <python expression>

Rules:
- Use only these variable names: hct, hgb, mch, mchc, mcv, plt, rbc, rdw, wbc
- Allowed operators: +  -  *  /  **
- Allowed functions: np.sqrt()  np.log1p()  np.abs()
- No conditionals, no loops, no external symbols
- Each formula must be a single expression (no assignment, no def)
- Use 3 to 7 features per formula
- Higher output should correspond to higher probability of Rheumatoid Arthritis

Example (style only, do not reuse):
  FORMULA: rdw * plt / (hgb + 0.01)"""

_COT_INSTRUCTION = """\
Before each formula, briefly explain (2–3 sentences) the clinical rationale
for the combination you chose. Label each block:
  REASONING: <your reasoning>
  FORMULA: <python expression>"""


def _format_spec_filled(n_formulas: int) -> str:
    return _FORMAT_SPEC.format(n_formulas=n_formulas)


# ── Public prompt builders ─────────────────────────────────────────────────────

def build_blind_prompt(n_formulas: int = 5, chain_of_thought: bool = False) -> str:
    """
    Build a prompt that relies solely on clinical/medical knowledge.

    No data-driven hints are included — the model must reason purely from
    domain knowledge about CBC features and Rheumatoid Arthritis pathophysiology.

    Parameters
    ----------
    n_formulas : int
        Number of distinct formulas to request.
    chain_of_thought : bool
        If True, ask the model to explain its reasoning before each formula.

    Returns
    -------
    str
        Ready-to-send prompt string.
    """
    cot_section = f"\n{_COT_INSTRUCTION}\n" if chain_of_thought else ""

    prompt = f"""\
You are a clinical data scientist with deep expertise in hematology and \
autoimmune disease. Your task is to design mathematical biomarker formulas \
that can identify patients likely to have Rheumatoid Arthritis (RA) using \
Complete Blood Count (CBC) measurements alone.

CLINICAL CONTEXT:
Rheumatoid Arthritis is a systemic autoimmune disease causing chronic joint \
inflammation. It commonly induces anemia of chronic disease, reactive \
thrombocytosis, and elevated markers of red cell heterogeneity. CBC panels \
are routinely collected and may carry subtle RA-related signals.

AVAILABLE CBC FEATURES:
{_FEATURE_BLOCK}

TASK:
Using your medical knowledge of how RA affects hematological parameters, \
propose {n_formulas} distinct formula(s) that compute a continuous score \
where higher values indicate a greater likelihood of RA.
{cot_section}
{_format_spec_filled(n_formulas)}
"""
    return prompt.strip()


def build_seeded_prompt(n_formulas: int = 5, chain_of_thought: bool = False) -> str:
    """
    Build a prompt seeded with data-driven insights from prior experiments.

    Includes feature importance ranking and guidance on effective transforms
    and formula complexity — without revealing exact top-performing formulas
    or specific metric values (to avoid anchoring bias).

    Parameters
    ----------
    n_formulas : int
        Number of distinct formulas to request.
    chain_of_thought : bool
        If True, ask the model to explain its reasoning before each formula.

    Returns
    -------
    str
        Ready-to-send prompt string.
    """
    cot_section = f"\n{_COT_INSTRUCTION}\n" if chain_of_thought else ""

    prompt = f"""\
You are a clinical data scientist with deep expertise in hematology and \
autoimmune disease. Your task is to design mathematical biomarker formulas \
that can identify patients likely to have Rheumatoid Arthritis (RA) using \
Complete Blood Count (CBC) measurements alone.

CLINICAL CONTEXT:
Rheumatoid Arthritis is a systemic autoimmune disease causing chronic joint \
inflammation. It commonly induces anemia of chronic disease, reactive \
thrombocytosis, and elevated markers of red cell heterogeneity. CBC panels \
are routinely collected and may carry subtle RA-related signals.

AVAILABLE CBC FEATURES:
{_FEATURE_BLOCK}

DATA-DRIVEN GUIDANCE (derived from prior experiments on this dataset):
1. Feature importance ranking (most to least discriminative):
   RDW > PLT > MCHC > HCT > MCV > WBC > RBC
   (hgb and mch tend to be redundant with hct and mchc respectively)

2. Effective transformations observed on this heavily imbalanced dataset (~1% positive rate):
   - np.log1p() on skewed count features (e.g. plt, wbc, rbc) reduces outlier influence
   - np.sqrt() on ratio features helps compress variance
   - Ratios between features (e.g. rdw / mcv) can amplify subtle group differences

3. Optimal formula complexity: 3–7 features per formula. Simpler formulas
   generalise better; overly complex formulas tend to overfit.

TASK:
Using the above clinical knowledge and data-driven guidance, propose \
{n_formulas} distinct formula(s) that compute a continuous score where \
higher values indicate a greater likelihood of RA. Aim for diverse \
combinations — avoid repeating the same feature pairings.
{cot_section}
{_format_spec_filled(n_formulas)}
"""
    return prompt.strip()


# ── Prompt configurations ──────────────────────────────────────────────────────

def get_all_prompt_configs() -> list[dict]:
    """
    Return 6 prompt configurations covering both strategies at 3 temperatures.

    Each config is a dict with keys:
      - name        : human-readable identifier
      - strategy    : "blind" or "seeded"
      - temperature : float (0.3 / 0.7 / 1.0)
      - n_formulas  : int
      - chain_of_thought : bool
      - prompt      : the full prompt string

    Temperature schedule:
      0.3 — deterministic / conservative (low diversity, high reliability)
      0.7 — balanced (default creative generation)
      1.0 — exploratory (high diversity, more risk of invalid syntax)
    """
    configs = []

    for strategy, builder in [("blind", build_blind_prompt), ("seeded", build_seeded_prompt)]:
        for temperature in [0.3, 0.7, 1.0]:
            cot = temperature >= 0.7  # use chain-of-thought for creative runs
            n_formulas = 5
            configs.append({
                "name": f"{strategy}_temp{temperature}",
                "strategy": strategy,
                "temperature": temperature,
                "n_formulas": n_formulas,
                "chain_of_thought": cot,
                "prompt": builder(n_formulas=n_formulas, chain_of_thought=cot),
            })

    return configs


# ── Quick sanity check (run this file directly) ────────────────────────────────
if __name__ == "__main__":
    configs = get_all_prompt_configs()
    print(f"Total prompt configs: {len(configs)}\n")
    for cfg in configs:
        print(f"=== {cfg['name']} (temp={cfg['temperature']}, cot={cfg['chain_of_thought']}) ===")
        print(cfg["prompt"][:300], "...\n")
