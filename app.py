# -------------------------------------------------------------------
# 1. Import libraries
# -------------------------------------------------------------------
# For environment variable management
from dotenv import load_dotenv

# For data manipulation
import os
import re
import json
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

# For LangChain agent and tools
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage, FunctionMessage

# Use OpenAI for embeddings and LLM
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# Use FAISS for vector store
from langchain_community.vectorstores import FAISS

# For loading CSV files
import csv

# For type hints
from typing import List, Dict, Any, Tuple

# For gradio UI
import gradio as gr

# Load environment variables from .env file
load_dotenv()
os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Expect OPENAI_API_KEY to be set in HF Space secrets
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

# Constants and config
## Dataset folder
DATA_DIR = "./data"
## FAISS vector store cache path
FAISS_PATH = "./faiss_store"

## Dataset file paths
DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")
SYMPTOM_SEVERITY_PATH = os.path.join(DATA_DIR, "Symptom-severity.csv")
SYMPTOM_DESC_PATH = os.path.join(DATA_DIR, "symptom_Description.csv")
SYMPTOM_PREC_PATH = os.path.join(DATA_DIR, "symptom_precaution.csv")

# -------------------------------------------------------------------
# 2. Load data
# -------------------------------------------------------------------
# Load the disease and related symptom data
dataset = pd.read_csv(DATASET_PATH)
# Load the symptom severity data
symptom_severity = pd.read_csv(SYMPTOM_SEVERITY_PATH)
# Load the symptom description data
symptom_desc = pd.read_csv(SYMPTOM_DESC_PATH)
# Load the symptom precaution data
symptom_prec = pd.read_csv(SYMPTOM_PREC_PATH)

# Display part of the data
dataset.head()
dataset.info()

"""### Dataset insights
- The dataset has 4,920 entries
- The majority of diseases in the dataset have 3 or 4 symptoms, so the data is very sparse
"""

# Display part of the symptom severity data
symptom_severity.head()
symptom_severity.info()
print()
print(f"Min weight: {symptom_severity['weight'].min()}")
print()
print(f"Max weight: {symptom_severity['weight'].max()}")

"""### Symptom severity insights
- There are 133 symptoms and their associated severity values (numeric)
- The severity values range from 1 to 7 (1 being low severity, 7 being high severity)
"""

# Display part of the symptom description data
symptom_desc.head()
# Display part of the symptom precautions data
symptom_prec.head()

"""# Data cleanup and standardization"""

# Canonicalize the symptom data (clean it so it has consistent and compatible formatting)
def canon_symptom(x: str) -> str:
  # Convert nulls to blanks ("")
  if pd.isna(x):
    return ""
  # Remove spaces and convert to lowercase
  x = str(x).strip().lower()
  # The symptoms may include underscores and some also have spaces (i.e. "_ "), so replace these with a single space
  x = re.sub(r"\s*_\s*", " ", x)
  # Replace multiple spaces with a single space
  x = re.sub(r" +", " ", x)
  return x

# Generic text canonicalization (for disease names, descriptions, and precautions)
def canon_text(x: str) -> str:
  # Convert nulls to blanks ("")
  if pd.isna(x):
    return ""
  # Strip spaces
  return str(x).strip()

"""## Format the disease and symptom data"""

# Create a list of all symptom columns
symp_cols = [c for c in dataset.columns if c.lower().startswith("symptom_")]

# Convert the dataset to a long format with "Disease" and "symptom" columns
long = (
    dataset.melt(id_vars=["Disease"], value_vars=symp_cols, value_name="symptom").drop(columns=["variable"])
)

# Clean up the disease column
long["Disease"] = long["Disease"].map(canon_text)
# Clean up the symptoms
long["symptom"] = long["symptom"].map(canon_symptom)
# Drop duplicate symptoms
long = long[long["symptom"] != ""].drop_duplicates()

# Create a dictionary of diseases (key) each with a set of symptoms (value)
disease_to_symptoms = long.groupby("Disease")["symptom"].apply(set).to_dict()

# Clean up the symptom values
symptom_severity["Symptom"] = symptom_severity["Symptom"].map(canon_symptom)
# Set the symptom weights - if the weight is null, set to 0
symptom_severity["weight"] = pd.to_numeric(symptom_severity["weight"], errors="coerce").fillna(0).astype(int)
# Convert to a dictionary of symptom (key) and weight (value) for lookups
symptom_to_weight = dict(zip(symptom_severity["Symptom"], symptom_severity["weight"]))

"""## Join symptom description and precautions data"""

# Clean the disease names and descriptions
symptom_desc["Disease"] = symptom_desc["Disease"].map(canon_text)
symptom_prec["Disease"] = symptom_prec["Disease"].map(canon_text)
symptom_desc["Description"] = symptom_desc["Description"].map(canon_text)

# Create a dictionary of disease names (key) and symptom descriptions (value)
disease_to_desc = dict(zip(symptom_desc["Disease"], symptom_desc["Description"]))
# Get a list of the precaution columns (excludes the "Disease" column)
prec_cols = [c for c in symptom_prec.columns if c.lower().startswith("precaution_")]

# Create a dictionary and map the precautions to the diseases
disease_to_prec = {}
for _, row in symptom_prec.iterrows():
  dis = row["Disease"]
  # Get a list of the precautions (not blank) for the current disease
  prec_list = [canon_text(row[c]) for c in prec_cols if canon_text(row[c])]
  disease_to_prec[dis] = prec_list

# Get the top K candidate diagnoses (diseases) for given symptoms
# This function gives highly explainable results for diagnoses
def disease_candidates(user_symptoms: list[str], top_k: int = 5):
  # Clean the symptom list
  clean_symp = {canon_symptom(s) for s in user_symptoms}
  # Only keep the non-blank symptoms
  clean_symp = {s for s in clean_symp if s}
  # If the symptom list is empty, return an empty list
  if not clean_symp:
    return []

  # The denominator should be the sum of symptom weights (if the symptom is not in the dictionary, then just use 1 for the weight)
  denominator = sum(symptom_to_weight.get(s, 0) for s in clean_symp) or 1

  candidate_list = []

  for disease, symp in disease_to_symptoms.items():
    # Find the intersection of user symptoms w/symptoms for the current disease
    inter = clean_symp.intersection(symp)
    # If we didn't find matching symptoms, this is not a good candidate
    if not inter:
      continue
    # The numerator should be the sum of symptom weights for matching symptoms
    numerator = sum(symptom_to_weight.get(s, 0) for s in inter)
    candidate_score = numerator / denominator
    # Add the candidate disease, score, and sorted maching symptom list
    candidate_list.append((disease, candidate_score, sorted(inter)))

  # Sort the list in descending order (highest scored first)
  candidate_list.sort(key=lambda x: x[1], reverse=True)
  # Return the top K candidates by score
  return candidate_list[:top_k]

# Get a report of severity based on symptoms
def severity_report(user_symptoms: list[str]):
  # Clean the symptoms and only keep non-blank symptoms
  clean_symp = [canon_symptom(s) for s in user_symptoms]
  clean_symp = [s for s in clean_symp if s]

  # Get weights per symptom
  weight_per_symp = [(s, symptom_to_weight.get(s, 0)) for s in clean_symp]
  # Get the total weight for all symptoms
  total = sum(w for _, w in weight_per_symp)

  # Give severity levels based on total symptom weight
  if total <= 5:
    level = "mild"
  elif total <= 12:
    level = "moderate"
  else:
    level = "high"

  # Track the symptoms that were not found in the severity table (i.e. have no weight)
  no_severity = [s for s in clean_symp if s not in symptom_to_weight]

  return {
      "Total": total,
      "Level": level,
      "Per symptom": weight_per_symp,
      "Symptoms not in severity table": no_severity
  }

# Check data quality
all_symptoms = set().union(*disease_to_symptoms.values())
# Get all symptoms w/out a severity weight
missing_severity = sorted([s for s in all_symptoms if s not in symptom_to_weight])

all_diseases = set(disease_to_symptoms.keys())
# Get diseases missing a description
missing_desc = sorted([d for d in all_diseases if d not in disease_to_desc])
# Get diseases missing precautions
missing_prec = sorted([d for d in all_diseases if d not in disease_to_prec])

print(f"Symptom count: {len(all_symptoms)}")
print(f"Symptoms missing severity: {missing_severity}")
print(f"Diseases missing description: {missing_desc}")
print(f"Diseases missing precautions: {missing_prec}")

# Create disease documents
def disease_doc(disease: str) -> str:
  # Get the symptoms for this disease (from the dataset)
  symptoms = sorted(disease_to_symptoms.get(disease, set()))
  # Get the description for this disease (from the dataset)
  description = disease_to_desc.get(disease, "")
  # Get the precaustions for this disease (from the dataset)
  precautions = disease_to_prec.get(disease, [])

  return "\n".join([
      f"Disease: {disease}",
      f"Description: {description}",
      "Symptoms: " + ", ".join(symptoms),
      "Precautions: " + ", ".join(precautions) if precautions else "Precautions: (none listed)"
  ])

docs: List[Document] = []
for d in sorted(disease_to_symptoms.keys()):
  docs.append(
      Document(
          page_content = disease_doc(d),
          metadata={"disease": d}
      )
  )

# Created embeddings of the disease docs and store in a vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# If we don't have embeddings in cache, create them
def load_or_create_faiss(docs, faiss_path="faiss_store"): 
  """ Load FAISS index if it exists, otherwise create it from docs and save it. """ 
  index_file = os.path.join(faiss_path, "index.faiss") 
  store_file = os.path.join(faiss_path, "index.pkl") 
  # If FAISS cache exists, load it 
  if os.path.exists(index_file) and os.path.exists(store_file): 
    print("Loading FAISS index from cache...") 
    return FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True) 
  # Otherwise create a new FAISS index 
  print("⚙️ Building new FAISS index...") 
  vectorstore = FAISS.from_documents(docs, embeddings) 
  # Ensure directory exists 
  os.makedirs(faiss_path, exist_ok=True) 
  # Save to disk 
  vectorstore.save_local(faiss_path) 
  print("💾 Saved FAISS index to disk.") 
  return vectorstore

# Create or load the FAISS vector store for the disease documents
vector_store = load_or_create_faiss(docs, faiss_path="faiss_store")

# Define schemas to use for the tools

# Define the input schema for each tool using Pydantic BaseModel 
class DiagnoseInput(BaseModel):
  # List of symptoms
  symptoms: List[str] = Field(..., description="List of symptoms the user is experiencing")
  # Top K candidates to return (default to 5, but should be between 1 and 10)
  top_k: int = Field(5, ge=1, le=10, description="Number of candidate diseases to return")

class SeverityInput(BaseModel):
  # List of symptoms
  symptoms: List[str] = Field(..., description="List of symptoms the user is experiencing")

class DiseaseInfoInput(BaseModel):
  # Disease name
  disease: str = Field(..., description="Disease name (must match name in dataset)")

class RetrieveDocsInput(BaseModel):
  # Search query
  query: str = Field(..., description="A search query (i.e. symptoms list or disease name)")
  # Top K docs to retrieve (default to 3, but should be between 1 and 10)
  k: int = Field(3, ge=1, le=10, description="How many disease docs to retrieve")

# Tools (using RAG, so they are deterministic)

# Create a diagnosic tool, using our schema
@tool(args_schema=DiagnoseInput)
def diagnose_diseases(symptoms: List[str], top_k: int=5) -> str:
  """
  Suggest likely diseases based on matching symptoms (using severity weights) from the dataset.
  Returns a JSON string of ranked candidates.
  """
  # Get the top K disease diagnoses
  results = disease_candidates(symptoms, top_k=top_k)
  # Return a JSON formatted string from the list
  return json.dumps({"candidates": results}, indent=2)

# Create a symptom severity calculation tool, using our schema
@tool(args_schema=SeverityInput)
def calc_symptom_severity(symptoms: List[str]) -> str:
  """
  Calculate a symptom severity score using weights from the Symptom-severity.csv dataset.
  Returns a JSON string with the total score, severity level, and weights per symptom.
  """
  # Get the symptoms severity summary
  report = severity_report(symptoms)
  # Return a JSON formatted string
  return json.dumps(report, indent=2)

# Create a tool to get disease description, using our schema
@tool(args_schema=DiseaseInfoInput)
def get_disease_description(disease: str) -> str:
  """
  Get the description for a disease (from the symptom_Description.csv dataset)
  """
  # Clean the disease name
  disease = canon_text(disease)
  # Lookup the disease description and return
  return disease_to_desc.get(disease, "")

# Create a tool to get disease precaution advice, using our schema
@tool(args_schema=DiseaseInfoInput)
def get_precaution_advice(disease: str) -> str:
  """
  Get precaution steps for a disease (from the symptom_precaution.csv dataset)
  """
  # Clean the disease name
  disease = canon_text(disease)
  # Lookup the disease precautions and return
  return json.dumps(disease_to_prec.get(disease, []), indent=2)

# Create a tool to retrieve disease documents from the vector store, using our schema
@tool(args_schema=RetrieveDocsInput)
def retrieve_disease_docs(query: str, k: int = 3):
  """
  Retrieve disease documents from the vector store for "ground truth" and explanations.
  Returns (serialized_text, retrieved_documents).)
  """
  # Get K similar docs from the vector store, based on the query
  retrieved = vector_store.similarity_search(query, k=k)
  # Get a serialized string of the top K matching disease docs
  serialized = "\n\n".join(
      f"Source: {doc.metadata}\nContent\n{doc.page_content}"
      for doc in retrieved
  )
  return serialized, retrieved

tools = [
    diagnose_diseases,
    calc_symptom_severity,
    get_disease_description,
    get_precaution_advice,
    retrieve_disease_docs,
]

# LLM + orchestrator agent

# Use a deterministic model since we are using tools for diagnosis and severity calculation and use the mini model for faster responses
model = init_chat_model(
    "gpt-4.1-mini",
    temperature=0,
    model_provider="openai",
    api_key=OPENAI_API_KEY,
)
system_prompt = """
You are a medical decision support assistant, not to be used as a replacement for professional medical care.
In your responses, always:
- Use tools to identify diagnosis candidates and symptom severity
- Provider a disclaimer that the user should consult with a doctor
- Give precaution advice and recommend professional medical care when appropriate (i.e. moderate to severe symptoms)
- If symptoms suggests a serious emergency risk (e.g. chest pain, difficulty breathing, serious bleeding), then recommend urgent care or emergency room services

Workflow to follow:
1) Ask for symptoms if not provided by user
2) Call calc_symptom_severity(symptoms)
3) Call diagnose_diseases(symptoms, top_k=5)
4) For the top 1-3 diseases, call retreive_disease_docs using the disease name or symptoms.
5) Return: top candidates also with why (matched symptoms), severity level, description + precautions for the top candidate(s), and provide follow-up questions to refine.

Responses should always be concise.
""".strip()

agent = create_agent(model, tools, system_prompt=system_prompt)

# Multi-turn chat
def chat_once(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
  """
  messages: [{"role": "user"|"assistant", "content": "..."}]
  Returns updated messages with the agent's final response added to the end.
  """

  final_state = None
  # Stream the agent's "thought" process, one step at a time - uses the "OPENAI_API_KEY" environment variable to get the API key
  for step in agent.stream({"messages": messages}, stream_mode="values"):
    print(f"Current step: {step}")
    final_state = step

  # Get the role and content from the final state of the agent
  new_messages = final_state["messages"]
  # Parse out the role and content
  out = []
  for m in new_messages:
    # Get the type (if available) or the role/assistant
    role = getattr(m, "type", None) or getattr(m, "role", "assistant")
    # Get the content if available (otherwise use "")
    content = getattr(m, "content", "")
    # Add to the list
    out.append({"role": role, "content": content})
  return out

# Redefine chat_once to correctly handle messages in multi-turn conversations
def chat_once(messages: List[BaseMessage]) -> List[BaseMessage]:
  """
  messages: List of BaseMessage objects for the conversation history.
  Returns updated list of BaseMessage objects with the agent's final response added.
  """
  final_state = None
  for step in agent.stream({"messages": messages}, stream_mode="values"):
    print(f"Current step: {step}")
    final_state = step

  # Return the actual BaseMessage objects from the final state
  return final_state["messages"]

# For the Gradio UI, we want to run the agent and get the updated LangChain message history, so we can extract the assistant's final response and also maintain the full conversation state (including tool calls) in the UI.
def run_agent_once(lc_history: List[BaseMessage]) -> List[BaseMessage]:
    """
    Runs the agent on the current conversation and returns the updated LangChain message history.
    """
    final_state = None
    for step in agent.stream({"messages": lc_history}, stream_mode="values"):
        final_state = step
    return final_state["messages"]

# Helper function to extract the last assistant response content from the LangChain message history
def get_last_ai_content(lc_history: List[BaseMessage]) -> str:
    """
    Extract the last assistant response content from the LangChain message history.
    """
    for m in reversed(lc_history):
        if isinstance(m, AIMessage):
            return (m.content or "").strip()
    return ""

# Gradio UI handlers
def submit_chat(user_text: str, chat_history: List[dict], lc_state: Optional[List[BaseMessage]]):
    """
    Gradio handler:
      - user_text: current textbox input
      - chat_history: list of {"role": "...", "content": "..."} used by gr.Chatbot(type="messages")
      - lc_state: LangChain BaseMessage history stored in gr.State

    Returns:
      - updated chat_history (for UI)
      - cleared textbox
      - updated lc_state
    """
    user_text = (user_text or "").strip()
    if not user_text:
        return chat_history, "", lc_state

    if lc_state is None:
        lc_state = []

    # 1) Add user message to LangChain state
    lc_state.append(HumanMessage(content=user_text))

    # 2) Run agent, get updated LangChain history (includes tool calls, etc.)
    lc_state = run_agent_once(lc_state)

    # 3) Extract the assistant’s final natural-language response
    assistant_text = get_last_ai_content(lc_state)
    if not assistant_text:
        assistant_text = "I couldn't generate a response. Please try rephrasing your symptoms."

    # 4) Update the Gradio chat history (messages format)
    chat_history = chat_history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]

    return chat_history, "", lc_state

# Handler to reset the chat - clears the Gradio chat history and the LangChain state
def reset_chat():
    return [], [],  # chat_history, lc_state

# Build the Gradio UI
with gr.Blocks(title="MediBot – Disease & Symptom Assistant") as demo:
    gr.Markdown(
        """
        # MediBot (RAG + Tools)
        Enter your symptoms in plain language (comma-separated is fine).
        **Note:** This is not medical advice—consult a clinician for diagnosis and treatment.
        """.strip()
    )

    chatbot = gr.Chatbot(height=420) #type="messages", height=420)
    lc_state = gr.State([])

    # User input row
    with gr.Row():
        user_box = gr.Textbox(
            label="Your message",
            placeholder="e.g., fever, headache, chills, fatigue",
            scale=8,
        )
        send_btn = gr.Button("Send", scale=2)

    with gr.Row():
        examples = gr.Examples(
            examples=[
                ["I have fever, headache, chills, and fatigue. What could it be?"],
                ["I also have nausea and joint pain."],
                ["Sore throat, runny nose, mild cough for 2 days."],
                ["Chest pain and shortness of breath."],
            ],
            inputs=user_box,
            label="Examples",
        )

    clear_btn = gr.Button("Clear chat")

    # Wire events
    
    # When the send button is clicked, run the submit_chat function with the current user input, chat history, and LangChain state, and update the chatbot and clear the user input box
    send_btn.click(
        fn=submit_chat,
        inputs=[user_box, chatbot, lc_state],
        outputs=[chatbot, user_box, lc_state],
    )

    # Allow pressing Enter in the textbox to submit the chat as well
    user_box.submit(
        fn=submit_chat,
        inputs=[user_box, chatbot, lc_state],
        outputs=[chatbot, user_box, lc_state],
    )

    # Clear button resets both the Gradio chat history and the LangChain state
    clear_btn.click(
        fn=lambda: ([], [], []),
        inputs=[],
        outputs=[chatbot, user_box, lc_state],
    )

# Launch the UI
demo.launch(share=True)