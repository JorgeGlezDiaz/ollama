import os
from langchain_ollama import OllamaLLM
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
### import tools
"""
@tool
def slang_checker: 
    dict = {
        'wawawa' : 'definitions...'
    }
    return ''
"""

# Define AI models
classifier_model = OllamaLLM(model="mistral:latest") #, tools=[slang_checker])  # Updated AI Assistant (Classifier)
nlp_model = OllamaLLM(model="phi:latest")  # Natural Language Model
code_model = OllamaLLM(model="codellama:latest")  # Code Model (Fixed space issue)
checker = OllamaLLM(model="mistral:latest")  # Final Checker

# Define state class
class State(MessagesState):
    query: str
    category: str = None
    response: str = None
    history: list = []     # Chat history

# Function to classify query using the classifier_model
def classify_query(state: State) -> State:

    print(f"🧠 Using AI Assistant (Classifier): {classifier_model.model}")  # Log model used

    # Classification prompt
    system_message = SystemMessage(content=
        """
        You are a classifier that determines whether a user's query is about 'code' or 'natural_language'.
        
        **Rules:**
        - If the query contains programming-related words such as: Python, JavaScript, Java, C++, C#, SQL, database, algorithm, function, loop, debugging, API, programming, or coding, classify it as: **code**.
        - If the query does NOT contain programming-related words and is about history, science, music, language, daily life, or general topics, classify it as: **natural_language**.
        - **ONLY return one word: "code" or "natural_language".**
    """
    )

    classification = classifier_model.invoke([system_message, {"role": "user", "content": state["query"]}])
    state["category"] = classification
    
    print(f"✅ The query has been classified as: {state['category']}.")  # Log classification result
    return state


# Function to process the request based on category
def process_query(state: State) -> State:

    if state["category"] == "code":
        model = code_model
        print(f"🛠️ Using Model: {model.model}")
        system_message = SystemMessage(content=
            """
            You are an AI code assistant. Answer programming questions concisely with examples when necessary.
            - Provide clear explanations with code snippets.
            - If asked to debug, explain errors and suggest fixes.
            - For SQL queries, ensure correct syntax and optimization.
            """
        )
        response = model.invoke([system_message, {"role": "user", "content": state["query"]}])
        state["response"] = response.strip()

    else:
        model = nlp_model
        print(f"🛠️ Using Model: {model.model}")
        system_message = SystemMessage(content=
            """
            You are an AI that answers general knowledge questions clearly and concisely.
            - Provide well-structured explanations.
            - If answering historical or scientific questions, use reliable knowledge.
            """
        )
        response = model.invoke([system_message, {"role": "user", "content": state["query"]}])
        state["response"] = response.strip()

    return state


# Function to check the response
def check_response(state: State) -> State:

    print(f"🔍 Using Checker Model: {checker.model}")  # Log model used
    system_message = SystemMessage(content=
        """
        You are a response checker. Read the given response and check if it needs improvement. If so, improve it.
        The answer must be **clear and concise**.
        """
    )
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
