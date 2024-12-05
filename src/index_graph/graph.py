"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""

import json
import os
from typing import Optional
from oros.schema import *
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langchain_core.tools import tool
from langchain_openai.chat_models import AzureChatOpenAI

from index_graph.configuration import AgentConfiguration
from index_graph.state import AgentState
from shared import retrieval
from langsmith import Client
from dotenv import load_dotenv
from shared.state import reduce_docs

load_dotenv()

client = Client()

model = AzureChatOpenAI(temperature=0,
                            seed=42,
                            streaming=True,
                            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                            openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
                            openai_api_key=os.getenv('OPENAI_API_KEY'),
                            deployment_name=os.getenv('AZURE_OPENAI_CHATGPT_DEPLOYMENT'),
                            openai_api_type=os.getenv('OPENAI_API_TYPE')
                            )
def generate_questions(state: list): 
    """Generate interview questions using an LLM.
    Args:
        state (dict): The input state containing required fields.
    Returns:
        dict: A dictionary containing generated questions.
    """
    try:
        # Ensure the state contains all required fields
        required_fields = ["resume", "job_description", "job_title", "company_name", "history"]
        for field in required_fields:
            if field not in state:
                raise ValueError(f"Missing required field: {field}")

        document_prompt = client.pull_prompt("interviewer_prompt")
        chain = (document_prompt | model.with_structured_output(Question))
        response = chain.invoke({
            "resume": state["resume"], 
            "job_description": state["job_description"],
            "job_title": state["job_title"],
            "company_name": state["company_name"],
            "history": state["history"]
        }) 
        
        return {"agent_out": response}
    except Exception as e:
        print("Error:", e)
        return {"error": f"Could not generate questions. Details: {str(e)}"}
@tool
def evaluate_answers(state: AgentState):
    """Generate interview questions using an LLM.
    Args:
        resume (str): The candidate's resume.
        job_description (str): The job description for the role.
        job_title (str): The job title of the role
        company_name (str): The name of the company
        history Optional[str]: History of previously asked questions and answers
    Returns:
        dict: A dictionary containing generated questions categorized into technical, personal, HR, and logical sections.
    """
    try:
        document_prompt = client.pull_prompt("quiz_from_jd")
        chain = (document_prompt | model.with_structured_output(Question))
        response = chain.invoke({"resume_text": state.resume, 
                           "job_description": state.job_description,
                           "job_title": state.job_title,
                           "company_name": state.company_name,
                           "history": state.history})
        return response
    except Exception as e:
        print("Error:", e)
        return {"error": f"Could not generate questions. Details: {str(e)}"}
@tool
   
def evaluate_candidate(state: AgentState):
    """Generate interview questions using an LLM.
    Args:
        resume (str): The candidate's resume.
        job_description (str): The job description for the role.
        job_title (str): The job title of the role
        company_name (str): The name of the company
        history Optional[str]: History of previously asked questions and answers
    Returns:
        dict: A dictionary containing generated questions categorized into technical, personal, HR, and logical sections.
    """
    try:
        document_prompt = client.pull_prompt("quiz_from_jd")
        chain = (document_prompt | model.with_structured_output(Question))
        response = chain.invoke({"resume_text": state.resume, 
                           "job_description": state.job_description,
                           "job_title": state.job_title,
                           "company_name": state.company_name,
                           "history": state.history})
        return response
    except Exception as e:
        print("Error:", e)
        return {"error": f"Could not generate questions. Details: {str(e)}"}
    
def count_questions(state: AgentState):
    """
    Determines whether to finish.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

# Define the graph
builder = StateGraph(AgentState)
builder.add_node(generate_questions)
builder.add_node(evaluate_answers)
builder.add_edge(START, "generate_questions")
builder.add_edge("generate_questions", "evaluate_answers")
builder.add_edge("evaluate_answers", "generate_questions")
builder.add_edge("evaluate_answers", END)
# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "IndexGraph"


agent_state = AgentState(
    resume="John Doe's resume text here...",
    job_description="Job description for a data scientist role...",
    job_title="Data Scientist",
    company_name="Tech Corp",
    history=[{"question": "What are your strengths?", "answer": "Problem-solving and teamwork"}]
)

result = graph.invoke({
    "resume": "John Doe's resume text here...",
    "job_description": "Job description for a data scientist role...",
    "job_title": "Data Scientist",
    "company_name": "Tech Corp",
    "history": [{"question": "What are your strengths?", "answer": "Problem-solving and teamwork"}]
})
agent_state["history"].append({"question": "What is your experience with Python?", "answer": "5 years"})

