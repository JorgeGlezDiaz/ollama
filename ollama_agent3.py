import os
from langchain_ollama import OllamaLLM
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import SystemMessage

# Define AI models
llm = OllamaLLM(model="qwen2.5:0.5b")  # Main AI Assistant (Classifier)
nlp_model = OllamaLLM(model="phi:latest")  # Natural Language Model
code_model = OllamaLLM(model="qwen2.5-coder:0.5b")  # Code Model
checker = OllamaLLM(model="qwen2.5:0.5b")  # Final Checker

# Define state class
class State(MessagesState):
    query: str
    category: str = None
    response: str = None

# Function to classify query
def classify_query(state: State) -> State:
    print("🧠 Using AI Assistant (Classifier): qwen2.5:0.5b")  # Log model used
    system_message = SystemMessage(content="You are a classifier. Identify if the query is related to 'code' or 'natural language'.")
    classification = llm.invoke([system_message, {"role": "user", "content": state["query"]}])
    
    category = "code" if "code" in classification.lower() else "natural_language"
    print(f"✅ Classified as: {category}")  # Log classification result
    state["category"] = category
    return state

# Function to process the request based on category
def process_query(state: State) -> State:
    model = code_model if state["category"] == "code" else nlp_model
    print(f"⚡ Using Model: {model.model}")  # Log model used
    response = model.invoke(state["query"])
    state["response"] = response
    return state

# Function to check the response
def check_response(state: State) -> State:
    print("🔍 Using Checker Model: qwen2.5:0.5b")  # Log model used
    system_message = SystemMessage(content="You are a response checker. Improve or validate the given response.")
    checked_response = checker.invoke([system_message, {"role": "user", "content": state["response"]}])
    state["response"] = checked_response
    return state

# Build LangGraph
builder = StateGraph(State)
builder.add_node("classify_query", classify_query)
builder.add_node("process_query", process_query)
builder.add_node("check_response", check_response)

# Define state transitions
builder.add_edge(START, "classify_query")
builder.add_edge("classify_query", "process_query")
builder.add_edge("process_query", "check_response")
builder.add_edge("check_response", END)

# Compile the graph
graph = builder.compile()

# Run the AI Assistant
user_input = input("AI Assistant: How can I assist you today?\n> ")
state = {"query": user_input}
final_state = graph.invoke(state)

# Display response
print("\nAI Assistant Response:", final_state["response"])
