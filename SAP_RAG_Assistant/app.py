# app.py
import os
import sys

# Ensure the app's directory is in the python path for robust module loading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify
from utils.loader import load_sap_documents
from utils.rag_engine import RAGEngine
from utils import agent
from dotenv import load_dotenv

# optional azure integration
from utils.azure_client import is_azure_configured, synthesize_with_azure

load_dotenv()

app = Flask(__name__)

# Load data and initialize RAG
documents = load_sap_documents()
rag = RAGEngine()
rag.build_index(documents)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask_question():
    """
    POST body expects JSON:
    {
      "question": "<user question>",
      "agent_action": "<optional action name: summarize|extract_financial_lines|run_report_simulation>"
    }
    """
    body = request.get_json(force=True)
    question = (body.get("question") or "").strip()
    agent_action = body.get("agent_action")

    if not question:
        return jsonify({"answer": "Please provide a question."}), 400

    # Step 1: retrieve relevant docs (top_k=3)
    retrieved = rag.search(question, documents, top_k=3)

    # Step 2: run agentic action if requested
    agent_output = None
    if agent_action:
        if agent_action == "summarize":
            agent_output = agent.summarize_docs(retrieved, max_sentences=4)
        elif agent_action == "extract_financial_lines":
            agent_output = agent.extract_financial_lines(retrieved)
        elif agent_action == "run_report_simulation":
            agent_output = agent.run_report_simulation(retrieved)
        else:
            agent_output = f"Unknown agent action: {agent_action}"

    # Step 3: If Azure configured, ask Azure OpenAI to synthesize a final answer using retrieved docs
    final_answer = None
    if is_azure_configured():
        system_prompt = "You are a helpful SAP assistant. Use the provided context to answer concisely."
        try:
            # include agent output in prompt if present
            user_prompt = question
            if agent_output:
                user_prompt += "\n\nAgent output:\n" + (agent_output if isinstance(agent_output, str) else ("\n".join(agent_output)))
            final_answer = synthesize_with_azure(system_prompt, user_prompt, retrieved)
        except Exception as e:
            final_answer = f"[Azure call failed] {str(e)}"

    # If Azure not configured, produce a simple combined answer locally
    if not final_answer:
        # If agent requested, prefer agent_output
        if agent_output:
            if isinstance(agent_output, list):
                final_answer = "\n".join(agent_output)
            else:
                final_answer = agent_output
        else:
            # local synthesis: join top retrieved docs
            final_answer = "\n\n".join(retrieved)

    return jsonify({
        "answer": final_answer,
        "retrieved": retrieved,
        "agent_output": agent_output
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)

