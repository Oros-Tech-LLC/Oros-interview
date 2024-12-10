"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""

import json
import os
from typing import Optional
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langchain_core.tools import tool
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.tools import HumanInputRun

from index_graph.configuration import AgentConfiguration
from index_graph.state import AgentState
from shared import retrieval
from langsmith import Client
from dotenv import load_dotenv
from shared.state import reduce_docs
from pydantic import BaseModel, Field
from typing import List, Optional

class Question(BaseModel):
    topic: str = Field(description="Topic of the questions that has been generated")
    question: str = Field(description="Interview question for the candidate")
    answer: str = Field(description="Suggested answer for the question")

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
        return {"agent_out": f"Could not generate questions. Details: {str(e)}"}
def evaluate_answers(state: AgentState):
    """Generate interview questions and get human answers.
    Args:
        resume (str): The candidate's resume.
        job_description (str): The job description for the role.
        job_title (str): The job title of the role
        company_name (str): The name of the company
        history Optional[str]: History of previously asked questions and answers
    Returns:
        dict: A dictionary containing the question and human answer.
    """
    try:
        document_prompt = client.pull_prompt("quiz_from_jd")
        chain = (document_prompt | model.with_structured_output(Question))
        question_response = chain.invoke({"resume_text": state.resume, 
                           "job_description": state.job_description,
                           "job_title": state.job_title,
                           "company_name": state.company_name,
                           "history": state.history})
        
        # Get human input for the answer
        human_input = HumanInputRun()
        print(f"\nQuestion: {question_response.question}")
        answer = human_input.run("Please provide your answer: ")
        
        return {
            "question": question_response.question,
            "answer": answer,
            "topic": question_response.topic
        }
    except Exception as e:
        print("Error:", e)
        return {"error": f"Could not process question and answer. Details: {str(e)}"}
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
    
def count_questions(state: list):
    """
    Determines whether to finish.

    Args:
        state (list): The current graph state

    Returns:
        str: Next node to call
    """
    if len(state["history"]) >= 40:
        return "evaluate_candidate"
    else:
        return "generate_questions"

# Define the graph
builder = StateGraph(AgentState)
builder.add_node("generate_questions", generate_questions)
builder.add_node("evaluate_answers", evaluate_answers)
builder.add_node("evaluate_candidate", evaluate_candidate)

# Add edges using string identifiers
builder.add_edge(START, "generate_questions")
builder.add_edge("generate_questions", "evaluate_answers")

builder.add_conditional_edges(
    "evaluate_answers", 
    count_questions,   
    {                   
        "generate_questions": "generate_questions", 
        "evaluate_candidate": "evaluate_candidate"
    }
)

builder.add_edge("evaluate_candidate", END)

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "IndexGraph"


#agent_state = AgentState(
#    resume="John Doe's resume text here...",
#    job_description="Job description for a data scientist role...",
#    job_title="Data Scientist",
#    company_name="Tech Corp",
#    history=[{"question": "What are your strengths?", "answer": "Problem-solving and teamwork"}]
#)
#
#result = graph.invoke({
#    "resume": "John Doe's resume text here...",
#    "job_description": "Job description for a data scientist role...",
#    "job_title": "Data Scientist",
#    "company_name": "Tech Corp",
#    "history": [{"question": "What are your strengths?", "answer": "Problem-solving and teamwork"}]
#})
#agent_state["history"].append({"question": "What is your experience with Python?", "answer": "5 years"})

