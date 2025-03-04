import os
from langchain_ollama import OllamaLLM
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import SystemMessage
from graphviz import Digraph
from IPython.display import Image, display


# Load the locally available Ollama model
llm = OllamaLLM(model="qwen2.5:0.5b")  # Change model if needed


# Test the model
response = llm.invoke("Hello, world!")
print(response)

# Define state class
class State(MessagesState):
    my_var: str
    customer_name: str

# Node function
def node_llm(state: State) -> State:
    print(state)
    system_message = SystemMessage(content="You are a helpful assistant that can answer questions.")
    return {"messages": [llm.invoke([system_message] + state["messages"])]}

# Build state graph
builder = StateGraph(State)
builder.add_node("node_llm", node_llm)
builder.add_edge(START, "node_llm")
builder.add_edge("node_llm", END)

# Compile graph
graph = builder.compile()

# Render the graph using Graphviz
dot = Digraph()
dot.node("START", "Start")
dot.node("node_llm", "LLM Node")
dot.node("END", "End")
dot.edge("START", "node_llm")
dot.edge("node_llm", "END")

# Save and display the image
dot.render("state_graph", format="png", cleanup=False)
display(Image(filename="state_graph.png"))
