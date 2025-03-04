import os
from langchain_ollama import OllamaLLM
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import SystemMessage

# Define AI models
classifier_model = OllamaLLM(model="deepseek-r1:latest")  # Updated AI Assistant (Classifier)
nlp_model = OllamaLLM(model="phi:latest")  # Natural Language Model
code_model = OllamaLLM(model="qwen2.5-coder:0.5b")  # Code Model
checker = OllamaLLM(model="deepseek-r1:latest")  # Final Checker

# Define state class
class State(MessagesState):
    query: str
    category: str = None
    response: str = None

# Function to classify query using DeepSeek-R1
def classify_query(state: State) -> State:
    print("🧠 Using AI Assistant (Classifier): deepseek-r1")  # Log model used

    # Stronger classification instructions
    system_message = SystemMessage(content="""
        You are a classifier that determines whether a user's query is about 'code' or 'natural_language'.
        
        **Rules:**
        - If the query contains programming-related words such as Python, JavaScript, Java, C++, C#, SQL, database, algorithm, function, loop, debugging, API, programming, or coding, classify it as: **code**.
        - If the query does NOT contain programming-related words and is about history, science, music, language, daily life, or general topics, classify it as: **natural_language**.
        - **ONLY return one word: "code" or "natural_language".**
        - Do not provide explanations or additional words. Just return **"code"** or **"natural_language"**.

        **Examples:**
        - "How do I write a function in Python?" → code
        - "Explain recursion in JavaScript." → code
        - "What is the Big Bang Theory?" → natural_language
        - "Tell me about the history of Rome." → natural_language
        - "How do I debug a Python script?" → code
        - "How does gravity work?" → natural_language
        - "Write a SQL query to join two tables." → code
        - "How do I play the guitar?" → natural_language
    """)

    classification = classifier_model.invoke([system_message, {"role": "user", "content": state["query"]}])

    # Ensure strict classification output
    classification = classification.strip().lower()

    # 🚨 Fallback: If DeepSeek misclassifies, use keyword detection
    code_keywords = ["python", "java", "c++", "c#", "javascript", "sql", "database", 
                     "algorithm", "function", "loop", "debugging", "api", "class", 
                     "object", "variable", "programming", "coding"]
    
    if any(word in state["query"].lower() for word in code_keywords):
        classification = "code"

    state["category"] = classification

    print(f"✅ Classified as: {state['category']}")  # Log classification result
    return state


# Function to process the request based on category
def process_query(state: State) -> State:
    model = code_model if state["category"] == "code" else nlp_model
    print(f"⚡ Using Model: {model.model}")  # Log model used

    # Shorten the response: Directly ask for minimal output
    system_message = SystemMessage(content="""
        You are a concise AI assistant. 
        - If the query is about **coding**, respond with **only** the correct code snippet.
        - Do **not** provide explanations, reasoning, or analysis—just output the necessary code.
        - If a question is about a concept, answer in **one or two sentences max**.
        - Keep responses as short and direct as possible.

        **Examples:**
        - "How do I print numbers 1 to 10 in Python?" → 
          ```python
          for i in range(1, 11):
              print(i)
          ```
        - "What is a Python class?" → 
          ```python
          class Car:
              def __init__(self, make, model):
                  self.make = make
                  self.model = model
          ```
        - "What is recursion?" → "Recursion is when a function calls itself."
        - "Tell me about Einstein." → "Einstein developed the theory of relativity."
    """)

    response = model.invoke([system_message, {"role": "user", "content": state["query"]}])
    
    state["response"] = response.strip()
    return state


# Function to check the response
def check_response(state: State) -> State:
    print("🔍 Using Checker Model: deepseek-r1")  # Log model used
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
