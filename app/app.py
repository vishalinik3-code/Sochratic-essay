"""
Streamlit app: AI-Powered Argument Mining and Socratic Feedback System.

Automatically loads whichever model src/select_best_model.py determined to
be the best (by Macro F1), based on REAL results from run_all.py. It never
hard-codes which model is "best" - it always reads models/results/best_model.json.

Run with:
    python -m streamlit run app/app.py
(after running `python run_all.py` at least once to train the models.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import config  # noqa: E402
import common  # noqa: E402
import socratic_feedback  # noqa: E402  (kept separate from prediction, as required)

st.set_page_config(page_title="Argument Mining & Socratic Feedback", page_icon="🧭", layout="wide")


@st.cache_resource(show_spinner="Loading the selected best model...")
def get_predictor():
    from prediction import load_predictor

    return load_predictor()


def load_best_model_info():
    if not config.BEST_MODEL_JSON.is_file():
        return None
    return common.load_json(config.BEST_MODEL_JSON)


def load_comparison_table():
    if not config.MODEL_COMPARISON_CSV.is_file():
        return None
    import pandas as pd

    return pd.read_csv(config.MODEL_COMPARISON_CSV)


# =====================================================================================
# HEADER
# =====================================================================================
st.title("🧭 AI-Powered Argument Mining & Socratic Feedback")
st.caption(
    "Predicts the effectiveness (and, where available, the type) of an argumentative "
    "text segment, then asks reflective questions to help you improve it - "
    "it never simply gives you the answer."
)

best_info = load_best_model_info()

if best_info is None:
    st.error(
        "No trained model was found yet. Please run the full pipeline first from a "
        "terminal, in the project's root folder:\n\n"
        "    python run_all.py\n\n"
        "This trains the available models, compares them, and automatically selects the best "
        "one. Then reload this page."
    )
    st.stop()

# =====================================================================================
# SECTION 1: ARGUMENT INPUT
# =====================================================================================
st.header("1. Argument Input")
default_text = (
    "I believe that schools should start later in the morning. Teenagers need more "
    "sleep because their biological clocks naturally shift later during puberty."
)
user_text = st.text_area("Enter an argumentative text segment (a sentence or short passage):",
                          value="", height=140, placeholder=default_text)

analyze_clicked = st.button("🔍 Analyze", type="primary")

if analyze_clicked:
    if not user_text.strip():
        st.warning("Please enter some text first.")
        st.stop()

    try:
        predictor = get_predictor()
        with st.spinner("Analyzing..."):
            result = predictor.predict(user_text)
    except FileNotFoundError as exc:
        st.error(f"⚠️ {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Unexpected error while analyzing the text: {exc}")
        st.stop()

    # ------------------------------------------------------------------- SECTION 2: PREDICTION
    st.header("2. Prediction")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Effectiveness", result.effectiveness)
    with col2:
        if result.discourse_type:
            st.metric("Predicted Discourse Type", result.discourse_type)
        else:
            st.metric("Predicted Discourse Type", "N/A")
            st.caption("The auxiliary discourse-type classifier was not trained "
                       "(no discourse-type column was found in the dataset).")

    # ------------------------------------------------------------------- SECTION 3: MODEL USED
    st.header("3. Model Used")
    st.write(f"**{best_info['best_model']}** (automatically selected as the best of the available "
             f"compared models, based on real evaluation results)")

    # ------------------------------------------------------------------- SECTION 4: CONFIDENCE / SCORE
    st.header("4. Confidence / Score")
    if result.effectiveness_probs:
        import pandas as pd

        probs_df = pd.DataFrame(
            sorted(result.effectiveness_probs.items(), key=lambda kv: kv[1], reverse=True),
            columns=["Effectiveness", "Probability"],
        )
        st.bar_chart(probs_df.set_index("Effectiveness"))
        st.dataframe(probs_df, width="stretch", hide_index=True)
    else:
        st.info("The selected model does not provide class probabilities.")

    # ------------------------------------------------------------------- SECTION 5: SOCRATIC FEEDBACK
    st.header("5. Socratic Feedback")
    st.caption(socratic_feedback.get_feedback_intro(result.discourse_type, result.effectiveness))
    for question in socratic_feedback.get_questions(result.discourse_type, result.effectiveness):
        st.markdown(f"- {question}")

# =====================================================================================
# SECTION 6: ABOUT THE MODEL
# =====================================================================================
st.header("6. About the Model")
col1, col2, col3 = st.columns(3)
col1.metric("Best Model", best_info["best_model"])
col2.metric("Selection Metric", "Macro F1")
col3.metric("Macro F1 Score", f"{best_info['score']:.4f}")
st.caption(f"Accuracy: {best_info.get('accuracy', 'N/A')} | Weighted F1: {best_info.get('weighted_f1', 'N/A')} "
           f"(all read live from models/results/best_model.json - never hard-coded)")

comparison_df = load_comparison_table()
if comparison_df is not None:
    with st.expander("See how the models compared"):
        st.dataframe(
            comparison_df.drop(columns=["model_key"], errors="ignore"),
            width="stretch", hide_index=True,
        )
    st.caption("Models are ranked by Macro F1 - the primary selection metric for this "
                   "multiclass, potentially imbalanced classification task.")
