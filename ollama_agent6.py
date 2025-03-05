import os
from langchain_ollama import OllamaLLM
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Define AI models
classifier_model = OllamaLLM(model="mistral:latest")  
nlp_model = OllamaLLM(model="phi:latest")  
code_model = OllamaLLM(model="codellama:latest")  
checker = OllamaLLM(model="mistral:latest")  

# Define state class with memory
class State(MessagesState):
    query: str
    category: str = None
    response: str = None
    history: list = []  # Store chat history

# Function to classify query using the classifier_model
def classify_query(state: State) -> State:
    print(f"🧠 Using AI Assistant (Classifier): {classifier_model.model}")  

    system_message = SystemMessage(content=
        """
        You are a classifier that determines whether a user's query is about 'code' or 'natural_language'.
        
        **Rules:**
        - If the query contains programming-related words such as: Python, JavaScript, Java, C++, C#, SQL, database, algorithm, function, loop, debugging, API, programming, or coding, classify it as: **code**.
        - Otherwise, classify it as: **natural_language**.
        - **ONLY return one word: "code" or "natural_language".**
    """
    )

    classification = classifier_model.invoke([
        system_message, 
        HumanMessage(content=state["query"])  
    ])
    
    state["category"] = classification.strip()
    print(f"✅ The query has been classified as: {state['category']}.")  
    return state

# Function to process the request based on category
def process_query(state: State) -> State:
    model = code_model if state["category"] == "code" else nlp_model
    print(f"🛠️ Using Model: {model.model}")

    system_message = SystemMessage(content=
        """
        You are an AI assistant. Keep responses clear and concise.
        - For code questions, provide code snippets and explanations.
        - For general questions, give structured answers.
        """
    )

    messages = [system_message] + state["history"] + [HumanMessage(content=state["query"])]

    response = model.invoke(messages)
    state["response"] = response.strip()

    # Update memory
    state["history"].append(HumanMessage(content=state["query"]))
    state["history"].append(AIMessage(content=state["response"]))

    return state

# Function to check the response
def check_response(state: State) -> State:
    print(f"🔍 Using Checker Model: {checker.model}")  

    system_message = SystemMessage(content=
        """
        You are a response checker. Read the response and improve it if necessary.
        Keep it clear and concise.
        """
    )

    messages = [system_message] + state["history"] + [AIMessage(content=state["response"])]

    checked_response = checker.invoke(messages)
    state["response"] = checked_response.strip()

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

# Continuous conversation loop
state = State(history=[])  # Initialize state with memory
first_time = True  # Flag to track the first interaction

while True:
    if first_time:
        user_input = input("AI Assistant: How can I assist you today?\n> ")
        first_time = False  # Ensure the prompt doesn't repeat
    else:
        user_input = input("Do you need anything else?\n> ")  # Change prompt for subsequent interactions

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("\n👋 Goodbye! Chat history saved.")
        break
    
    state["query"] = user_input
    final_state = graph.invoke(state)

    print("\nAI Assistant Response:", final_state["response"])
