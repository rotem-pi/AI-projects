#!/usr/bin/env python3
"""definiData - ask the database questions in plain English.

Run via install.sh (native app) or init.sh (browser tab), or directly with:
    streamlit run app.py
"""
import json
import os
import shutil
import subprocess
import sys
import time

import boto3
import streamlit as st
from botocore.exceptions import (
    ClientError,
    LoginError,
    LoginInsufficientPermissions,
    NoCredentialsError,
    SSOTokenLoadError,
    TokenRetrievalError,
    UnauthorizedSSOTokenError,
)
from PIL import Image

# Error codes that mean "not signed in / session expired", as opposed to
# e.g. AccessDenied (signed in fine, just lacks IAM permission).
_AUTH_ERROR_CODES = {
    "ExpiredToken",
    "ExpiredTokenException",
    "InvalidClientTokenId",
    "UnrecognizedClientException",
    "AuthFailure",
}


def is_auth_error(e: Exception) -> bool:
    """True if e means 'you're not signed in to AWS', not some other failure
    (e.g. LoginInsufficientPermissions means signed in fine but missing an
    IAM policy - clicking "sign in" again won't fix that, an admin needs to)."""
    if isinstance(e, LoginInsufficientPermissions):
        return False
    if isinstance(e, (NoCredentialsError, TokenRetrievalError, UnauthorizedSSOTokenError, SSOTokenLoadError, LoginError)):
        return True
    if isinstance(e, ClientError):
        return e.response.get("Error", {}).get("Code") in _AUTH_ERROR_CODES
    return False


def find_aws_cli() -> str | None:
    """Locates the aws CLI. Apps launched by double-clicking (vs. a terminal)
    get a minimal PATH that often excludes where aws is actually installed
    (Homebrew, ~/.local/bin, etc.), so PATH alone isn't reliable here."""
    found = shutil.which("aws")
    if found:
        return found
    for candidate in (
        os.path.expanduser("~/.local/bin/aws"),
        "/opt/homebrew/bin/aws",
        "/usr/local/bin/aws",
        "/usr/bin/aws",
    ):
        if os.path.exists(candidate):
            return candidate
    return None

REGION = "eu-north-1"
AWS_ACCOUNT_ID = "412550564892"
# Always target this account/profile, regardless of whatever AWS_PROFILE or
# default profile happens to be set on a given machine.
AWS_PROFILE = "dev-admin"
os.environ["AWS_PROFILE"] = AWS_PROFILE

LAMBDA_FUNCTION_NAME = "BedrockAthenaSQLExecutor"
# Application inference profile tagged Project=definidata-tool, so Bedrock spend
# is attributable/filterable in Cost Explorer.
MODEL_ID = "arn:aws:bedrock:eu-north-1:412550564892:application-inference-profile/718rksnz6bq9"

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
# Set by desktop_launcher.py when running as the installed native app, so we
# know whether "update" should relaunch the .app or just tell the user to
# restart their browser-mode process.
NATIVE_MODE = os.environ.get("DEFINIDATA_NATIVE") == "1"

ASSETS_DIR = os.path.join(REPO_DIR, "assets")
LOGO = Image.open(os.path.join(ASSETS_DIR, "logo.png"))


@st.cache_data(ttl="120m", show_spinner=False)
def check_for_update() -> bool:
    """True if origin has commits we don't have yet. Fails closed (False) on
    any error - offline, no upstream configured, git missing, etc."""
    try:
        subprocess.run(
            ["git", "fetch", "--quiet"], cwd=REPO_DIR, timeout=10, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        local = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, timeout=5,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        remote = subprocess.run(
            ["git", "rev-parse", "@{u}"], cwd=REPO_DIR, timeout=5,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return local != remote
    except Exception:
        return False


def perform_update() -> tuple[bool, str]:
    """Pulls the latest commits (fast-forward only - refuses rather than ever
    clobbering local changes) and reinstalls. Returns (success, log)."""
    log_parts = []

    pull = subprocess.run(
        ["git", "pull", "--ff-only"], cwd=REPO_DIR, capture_output=True, text=True, timeout=60,
    )
    log_parts.append(f"$ git pull --ff-only\n{pull.stdout}{pull.stderr}")
    if pull.returncode != 0:
        return False, "\n\n".join(log_parts)

    if NATIVE_MODE and sys.platform == "darwin":
        install = subprocess.run(
            ["./install.sh"], cwd=REPO_DIR, capture_output=True, text=True, timeout=300,
        )
        log_parts.append(f"$ ./install.sh\n{install.stdout}{install.stderr}")
        if install.returncode != 0:
            return False, "\n\n".join(log_parts)
        subprocess.Popen(["open", "-a", "definiData"])
    else:
        pip_install = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
            cwd=REPO_DIR, capture_output=True, text=True, timeout=300,
        )
        log_parts.append(f"$ pip install -r requirements.txt\n{pip_install.stdout}{pip_install.stderr}")
        if pip_install.returncode != 0:
            return False, "\n\n".join(log_parts)

    return True, "\n\n".join(log_parts)

TEAL_LIGHT = "#23AEB1"
TEAL_DARK = "#007C7C"

LIGHT_PALETTE = {
    "bg": "#FFFFFF",
    "text": "#192530",
    "card_border": "#bfe6e6",
    "input_bg": "#FFFFFF",
    "caption": "#5b6b73",
}
DARK_PALETTE = {
    "bg": "#12181D",
    "text": "#E8EEF1",
    "card_border": "#2A3B42",
    "input_bg": "#1B252B",
    "caption": "#9fb0b6",
}


def build_css(p: dict) -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Ubuntu:wght@400;500;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Ubuntu', sans-serif !important;
}}

.stApp {{
    background-color: {p["bg"]} !important;
    color: {p["text"]} !important;
}}

h1, h2, h3 {{
    color: {p["text"]} !important;
    font-weight: 500 !important;
}}

p, span, label, div {{
    color: {p["text"]};
}}

.stCaption, [data-testid="stCaptionContainer"] {{
    color: {p["caption"]} !important;
}}

/* Responsive width instead of a fixed narrow centered column, with less
   wasted padding. */
.block-container {{
    max-width: min(1100px, 94vw) !important;
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    margin: 0 auto !important;
}}

div[data-testid="stTextInput"] input {{
    border-radius: 24px;
    border: 1px solid {p["card_border"]};
    padding: 10px 18px;
    background-color: {p["input_bg"]};
    color: {p["text"]};
}}

div[data-testid="stTextInput"] input:focus {{
    border-color: {TEAL_LIGHT};
    box-shadow: 0 0 0 1px {TEAL_LIGHT};
}}

.stButton button[kind="primary"] {{
    background: linear-gradient(90deg, {TEAL_LIGHT}, {TEAL_DARK}) !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 0.5rem 2rem !important;
    font-weight: 500 !important;
    color: #FFFFFF !important;
}}

.stButton button[kind="primary"]:hover {{
    opacity: 0.9;
}}

[data-testid="stVerticalBlockBorderWrapper"] {{
    border-color: {p["card_border"]} !important;
    border-radius: 20px !important;
}}

code, pre, .stCodeBlock {{
    color: {p["text"]};
}}

/* Hide the Streamlit toolbar (Deploy button, etc.) - config.toml's
   toolbarMode already does this; this is a fallback in case that
   setting isn't honored by a given Streamlit version. */
[data-testid="stToolbar"] {{
    display: none !important;
}}
</style>
"""


st.set_page_config(page_title="definiData", page_icon=LOGO, layout="wide")

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

header_col1, header_col2, header_col3 = st.columns([1, 6, 1], vertical_alignment="center")
with header_col1:
    st.image(LOGO, width=56)
with header_col2:
    st.markdown("<h1 style='margin-bottom:0;'>definiData</h1>", unsafe_allow_html=True)
with header_col3:
    st.toggle("Dark mode", key="dark_mode")

st.markdown(build_css(DARK_PALETTE if st.session_state.dark_mode else LIGHT_PALETTE), unsafe_allow_html=True)
st.caption("Ask the database a question in plain English.")

if check_for_update():
    with st.container(border=True):
        st.info("A new version of definiData is available.")
        button_label = "Update & Restart" if NATIVE_MODE else "Update now"
        if st.button(button_label):
            with st.spinner("Updating definiData..."):
                ok, log = perform_update()
            if ok:
                check_for_update.clear()
                if NATIVE_MODE:
                    st.success("Updated! Restarting definiData...")
                    time.sleep(1)
                    os._exit(0)
                else:
                    st.success(
                        "Updated! Stop this process (Ctrl+C) and re-run ./init.sh "
                        "to use the new version."
                    )
            else:
                st.error("Update failed:")
                st.code(log, language="text")


# Matches the sentinel the Lambda returns when it decides the input isn't
# actually a database question (greeting, off-topic, gibberish, prompt
# injection attempt, etc.) - see BedrockAthenaSQLExecutor's generate_sql().
NOT_A_DB_QUESTION = "NOT_A_DB_QUESTION"
FALLBACK_ANSWER = (
    'That doesn\'t look like a question about the dev_app_analytics database. '
    'Try something like "top 5 tenants by activity" or "how many tasks does '
    'grammarly.com have".'
)


def run_query(question: str) -> str:
    """Invokes the Lambda (schema lookup -> SQL generation -> Athena execution)."""
    lambda_client = boto3.Session(profile_name=AWS_PROFILE, region_name=REGION).client("lambda")
    response = lambda_client.invoke(
        FunctionName=LAMBDA_FUNCTION_NAME,
        Payload=json.dumps({"inputText": question}).encode(),
    )
    payload = json.loads(response["Payload"].read())
    if "FunctionError" in response:
        raise RuntimeError(f"Lambda error: {payload}")
    return payload["response"]["responseBody"]["TEXT"]["body"]


def synthesize_answer(question: str, query_result: str) -> str:
    bedrock = boto3.Session(profile_name=AWS_PROFILE, region_name=REGION).client("bedrock-runtime")
    prompt = f"""You are a data analyst assistant.
The user asked: "{question}"

{query_result}

Provide a clear, helpful, and concise answer summarizing these findings for the user."""

    response = bedrock.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 1024, "temperature": 0.2},
    )
    return response["output"]["message"]["content"][0]["text"]


with st.container(border=True):
    # A form binds Enter-in-field to the submit button - a plain st.text_input
    # + st.button does NOT do this on its own (Enter only "applies" the text).
    with st.form("ask_form", border=False):
        question = st.text_input("Ask a question", placeholder="e.g. top 5 tenants by activity")
        ask_clicked = st.form_submit_button("Ask", type="primary")
    # Kept outside the form so it stays instantly reactive (cosmetic toggles
    # like this and dark mode should never require re-running the question).
    show_sql = st.checkbox("Show SQL", key="show_sql")

if ask_clicked:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Querying the database..."):
                query_result = run_query(question)
                is_db_question = query_result != NOT_A_DB_QUESTION
                answer = synthesize_answer(question, query_result) if is_db_question else FALLBACK_ANSWER
        except Exception as e:
            st.session_state.pop("last_result", None)
            if is_auth_error(e):
                st.session_state.auth_error = str(e)
            else:
                st.session_state.pop("auth_error", None)
                st.error(f"Something went wrong: {e}")
        else:
            st.session_state.pop("auth_error", None)
            st.session_state.last_result = {
                "answer": answer,
                "query_result": query_result,
                "is_db_question": is_db_question,
            }

# Rendered from session_state (not the transient ask_clicked flag) so this
# stays visible - and the sign-in button stays clickable - across cosmetic
# reruns like toggling dark mode.
if st.session_state.get("auth_error"):
    with st.container(border=True):
        st.error(
            "Nothing ran because you're not signed in to AWS (or your session "
            "expired) - that's the only reason, not a bug in your question."
        )
        st.caption(f"Details: {st.session_state.auth_error}")
        if st.button(f"Sign in with aws sso login (profile: {AWS_PROFILE})"):
            aws_cli = find_aws_cli()
            if not aws_cli:
                st.error(
                    "Couldn't find the aws CLI on this machine. Install it, or run "
                    f"`aws sso login --profile {AWS_PROFILE}` yourself in a terminal."
                )
            else:
                try:
                    with st.spinner("Opening AWS SSO sign-in - approve it in your browser..."):
                        result = subprocess.run(
                            [aws_cli, "sso", "login", "--profile", AWS_PROFILE],
                            capture_output=True, text=True, timeout=180,
                        )
                except subprocess.TimeoutExpired:
                    st.error(f"Sign-in timed out. Try `aws sso login --profile {AWS_PROFILE}` in a terminal instead.")
                else:
                    if result.returncode == 0:
                        st.session_state.pop("auth_error", None)
                        st.success("Signed in! Try asking your question again.")
                    else:
                        st.error(f"Sign-in failed:\n\n{result.stdout}{result.stderr}")

if "last_result" in st.session_state:
    with st.container(border=True):
        st.markdown("### Answer")
        st.write(st.session_state.last_result["answer"])
        if show_sql and st.session_state.last_result.get("is_db_question", True):
            st.markdown("### SQL & raw results")
            st.code(st.session_state.last_result["query_result"], language="text")
