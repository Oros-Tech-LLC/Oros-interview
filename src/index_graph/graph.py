"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""

import json
import os
from typing import Optional
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langchain_core.tools import tool
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.tools import HumanInputRun
from langchain_core.messages.ai import AIMessage
from langchain_core.runnables.base import Runnable
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

class EvaluateAnswers(BaseModel):
    question: str = Field(description="Question for which the evaluation is made")
    answer: str = Field(description="Question for which the evaluation is made")
    evaluation_summary: str = Field(description="Evaluation summary for the given question and answer that has been generated")
    relevance: int = Field(description="Relevance score of the answer")
    correctness: int = Field(description="Correctness score of the answer")
    leadership: int = Field(description="Leadership score assessed from the answer")
    team_work: int = Field(description="Team work score assessed from the answer")
    technical_strength: int = Field(description="Technical strength score assessed from the answer")
    communication: int = Field(description="Communication score assessed from the answer")

load_dotenv()

client = Client()

model = ChatOpenAI(temperature=0,
                            seed=42,
                            streaming=True,
                            api_key=os.environ.get("OPENAI_API_KEY")
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
        required_fields = ["resume", "job_description", "job_title", "company_name"]
        for field in required_fields:
            if field not in state:
                raise ValueError(f"Missing required field: {field}")

        # Assign history as an empty list if it is empty
        if not state.get("history"):
            state["history"] = []

        document_prompt = client.pull_prompt("interviewer_prompt")
        chain = (document_prompt | model.with_structured_output(Question))
        response = chain.invoke({
            "resume": state["resume"], 
            "job_description": state["job_description"],
            "job_title": state["job_title"],
            "company_name": state["company_name"],
            "history": state["history"]
        }) 
        return {"generated_question": response}
    except Exception as e:
        print("Error:", e)
        return {"generated_question": f"Could not generate questions. Details: {str(e)}"}
    
def human_input(state: list) -> Runnable:
    """
    Accepts a human response as an argument and appends it to the state.

    Args:
        state (list): The current state of the conversation or interaction.

    Returns:
        Runnable: The updated state after appending the human's input.
    """
    max_retries = 3
    retries = 0

    while retries < max_retries:
        try:
            response = input()
            
            if not response.strip():
                print("Input cannot be empty. Please provide a valid response.")
                retries += 1
                continue
            
            state.human_answer = response
            return state
        
        except EOFError:
            print("EOFError: Input stream is closed or unavailable.")
            retry = input()
            if retry != 'y':
                raise ValueError("User chose to exit after encountering an EOFError.")
            retries += 1

    raise ValueError("Maximum retries reached. Unable to get valid input.") 

def reset_entries(state: list):
    """Resets specific entries in the state."""
    state["generated_question"] = None
    state["answer_evaluation"] = None
    state["human_answer"] = None

def evaluate_answers(state: list):
    """Generate evaluation  questions and get human answers.
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
        # Assign history as an empty list if it is empty
        if not state.get("history"):
            state["history"] = []

        document_prompt = client.pull_prompt("interviewer_answer_evaluator_prompt")
        chain = (document_prompt | model.with_structured_output(EvaluateAnswers))
        evaluation_response = chain.invoke({
                           "resume": state["resume"], 
                           "job_description": state["job_description"],
                           "job_title": state["job_title"],
                           "company_name": state["company_name"],
                           "question": state["generated_question"],
                           "answer": state["human_answer"],
                           "history": state["history"]})
        
        state["history"].append(evaluation_response) 
        return {"answer_evaluation": evaluation_response}
    except Exception as e:
        print("Error:", e)
        return {"answer_evaluation": f"Could not process question and answer. Details: {str(e)}"}
    
def evaluate_candidate(state: list):
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
builder.add_node("human_input", human_input)
builder.add_node("evaluate_answers", evaluate_answers)
builder.add_node("reset_entries", reset_entries)
builder.add_node("evaluate_candidate", evaluate_candidate)

# Add edges using string identifiers
builder.add_edge(START, "generate_questions")
builder.add_edge("generate_questions", "human_input")
builder.add_edge("human_input", "evaluate_answers")
builder.add_edge("reset_entries", "generate_questions")

builder.add_conditional_edges(
    "evaluate_answers", 
    count_questions,   
    {                   
        "generate_questions": "reset_entries", 
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

