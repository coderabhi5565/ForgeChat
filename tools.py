from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()


@tool
def calculator(a: int, b: int, operation: str) -> int:
    """Perform basic arithmetic operations."""
    
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a // b
    else:
        raise ValueError("Invalid operation")

tools = [calculator, search]