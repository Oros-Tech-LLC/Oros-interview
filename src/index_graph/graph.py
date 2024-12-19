"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""

import json
import os
from typing import Literal, Optional, Union
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
from pydantic import BaseModel, Field
from pydantic import ValidationError
from typing import Optional, List, Dict

class BaseQuestion(BaseModel):
    topic: str = Field(description="Topic of the question that has been generated")
    question: str = Field(description="Interview question for the candidate")
    answer: str = Field(description="Suggested answer or evaluation criteria for the question")
    complexity: int = Field(description="A number from 1 to 5, where 1 is least complex and 5 is most complex")

class PersonalQuestion(BaseModel):
    topic: str = Field(description="Topic of the question('Personal')")
    question: str = Field(description="Personal Interview question for the candidate")
    answer: str = Field(description="Suggested answer or evaluation criteria for the question")
    complexity: int = Field(description="A number from 1 to 5, where 1 is least complex and 5 is most complex")

class HRQuestion(BaseModel):
    topic: str = Field(description="Topic of the question that has been generated")
    question: str = Field(description="The full text of the HR question")
    answer: str = Field(description="A suggested model answer or evaluation criteria for the question")
    complexity: int = Field(description="A number from 1 to 5, where 1 is least complex and 5 is most complex")

class CriticalQuestion(BaseModel):
    topic: str = Field(description="Topic of the question that has been generated")
    scenario: str = Field(description="A detailed description of the situation or problem")
    question: str = Field(description="The specific question or task for the candidate based on the scenario")
    key_considerations: List[str] = Field(description="A list of important factors the candidate should consider in their response")
    evaluation_criteria: str = Field(description="Key points or approaches the candidate should demonstrate in their answer")
    follow_up: Optional[str] = Field(description="An optional follow-up question to probe deeper into the candidate's thinking")
    complexity: int = Field(description="A number from 1 to 5, where 1 is least complex and 5 is most complex")

class TechnicalQuestion(BaseModel):
    topic: str = Field(description="Topic of the question that has been generated")
    question_type: Literal["Theoretical", "Coding Challenge", "System Architecture"] = Field(description="Type of technical question")
    question: str = Field(description="The full text of the technical question or coding challenge")
    instructions: str = Field(description="Clear, step-by-step instructions for the candidate")
    example_input: Optional[str] = Field(description="Sample input for the problem, if applicable")
    example_output: Optional[str] = Field(description="Expected output for the sample input, if applicable")
    answer_criteria: str = Field(description="Key points or approach the candidate should demonstrate in their answer or solution")
    complexity: int = Field(description="A number from 1 to 5, where 1 is least complex and 5 is most complex")


class EvaluateAnswers(BaseModel):
    question: str = Field(description="Question for which the evaluation is made")
    answer: str = Field(description="Answer for which the evaluation is made(human entered answer)")
    evaluation_summary: str = Field(description="Evaluation summary for the given question and answer that has been generated")
    relevance: int = Field(description="Relevance score of the answer")
    correctness: int = Field(description="Correctness score of the answer")
    leadership: int = Field(description="Leadership score assessed from the answer")
    team_work: int = Field(description="Team work score assessed from the answer")
    technical_strength: int = Field(description="Technical strength score assessed from the answer")
    communication: int = Field(description="Communication score assessed from the answer")

class CategoryEvaluation(BaseModel):
    score: int = Field(..., description="Score from 1 (Poor) to 5 (Excellent)")
    examples: List[str] = Field(..., description="Specific examples supporting the evaluation")

class CandidateEvaluation(BaseModel):
    overall_assessment: str = Field(..., description="Concise summary of the candidate's suitability for the job role")
    
    technical_proficiency: CategoryEvaluation = Field(
        ..., description="Evaluation of technical knowledge, problem-solving skills, and technical challenge handling (Weight: 25%)"
    )
    communication_skills: CategoryEvaluation = Field(
        ..., description="Evaluation of communication clarity, effectiveness, and ability to explain complex concepts (Weight: 20%)"
    )
    cultural_fit: CategoryEvaluation = Field(
        ..., description="Assessment of alignment with company culture and potential contribution to the team (Weight: 15%)"
    )
    leadership_teamwork: CategoryEvaluation = Field(
        ..., description="Evaluation of leadership potential and ability to collaborate effectively (Weight: 15%)"
    )
    adaptability_learning_agility: CategoryEvaluation = Field(
        ..., description="Assessment of adaptability, learning agility, openness to feedback, and continuous improvement (Weight: 15%)"
    )
    relevant_experience: CategoryEvaluation = Field(
        ..., description="Evaluation of alignment between past experiences and job role requirements (Weight: 10%)"
    )
    
    strengths: List[str] = Field(..., description="3-5 key strengths of the candidate with specific examples")
    areas_for_improvement: List[str] = Field(..., description="2-3 areas where the candidate could improve or need support")
    
    final_recommendation: str = Field(
        ..., description="Final recommendation: Strongly Recommend Hire, Recommend Hire, Recommend Additional Interview, or Do Not Recommend"
    )
    
    additional_comments: Optional[str] = Field(None, description="Additional insights or observations relevant to the hiring decision")
    
load_dotenv()

client = Client()

model = ChatOpenAI(temperature=0,
                            seed=42,
                            streaming=True,
                            api_key=os.environ.get("OPENAI_API_KEY")
                            )

from typing import Dict, Union
from pydantic import ValidationError

def personal_questions(state: AgentState) -> Dict[str, Union[PersonalQuestion, str]]:
    """Generate interview questions using an LLM."""
    try:
        # Pull and execute the prompt chain
        document_prompt = client.pull_prompt("interviewer_personal_question")
        chain = (document_prompt | model.with_structured_output(PersonalQuestion))
        
        # Invoke the chain and get raw response
        raw_response = chain.invoke({
            "resume": state.resume,
            "job_description": state.job_description,
            "job_title": state.job_title,
            "company_name": state.company_name,
            "history": state.history
        })

        # Debug raw response
        print(f"Raw LLM Output: {raw_response}")

        # Validate and parse the response
        response = PersonalQuestion.parse_obj(raw_response)

        # Ensure all required fields are present and valid
        required_fields = ["topic", "question", "answer", "complexity"]
        for field in required_fields:
            if not getattr(response, field, None):
                raise ValueError(f"Missing required field: {field}")

        return {"generated_question": response.dict()}

    except ValidationError as ve:
        print(f"Validation Error: {ve}")
        return {"generated_question": f"Could not generate questions. Details: {ve}"}
    except Exception as e:
        print(f"Error in personal_questions: {e}")
        return {"generated_question": f"Could not generate questions. Details: {str(e)}"}

    
def critical_questions(state: AgentState) -> Dict[str, Union[CriticalQuestion, str]]:
    """Generate interview questions using an LLM."""
    try:
        question_count = len(state.history)

        document_prompt = client.pull_prompt("interviewer_critical_questions")
        chain = (document_prompt | model.with_structured_output(CriticalQuestion))
        response = chain.invoke({
            "resume": state.resume,
            "job_description": state.job_description,
            "job_title": state.job_title,
            "company_name": state.company_name,
            "history": state.history,
            "question_count": question_count
        })

        if not all(hasattr(response, field) for field in ["topic", "question"]):
            raise ValueError("Generated question is missing required fields")

        return {"generated_question": response}
    except Exception as e:
        print(f"Error in generate_questions: {e}")
        return {"generated_question": f"Could not generate questions. Details: {str(e)}"}

def technical_questions(state: AgentState) -> Dict[str, Union[TechnicalQuestion, str]]:
    """Generate interview questions using an LLM."""
    try:
        question_count = len(state.history)

        document_prompt = client.pull_prompt("interviewer_technical_questions")
        chain = (document_prompt | model.with_structured_output(TechnicalQuestion))
        response = chain.invoke({
            "resume": state.resume,
            "job_description": state.job_description,
            "job_title": state.job_title,
            "company_name": state.company_name,
            "history": state.history,
            "question_count": question_count
        })

        if not all(hasattr(response, field) for field in ["topic", "question"]):
            raise ValueError("Generated question is missing required fields")

        return {"generated_question": response}
    except Exception as e:
        print(f"Error in generate_questions: {e}")
        return {"generated_question": f"Could not generate questions. Details: {str(e)}"}
    
def hr_questions(state: AgentState) -> Dict[str, Union[HRQuestion, str]]:
    """Generate interview questions using an LLM."""
    try:
        question_count = len(state.history)

        document_prompt = client.pull_prompt("interviewer_hr_questions")
        chain = (document_prompt | model.with_structured_output(HRQuestion))
        response = chain.invoke({
            "resume": state.resume,
            "job_description": state.job_description,
            "job_title": state.job_title,
            "company_name": state.company_name,
            "history": state.history,
            "question_count": question_count
        })

        if not all(hasattr(response, field) for field in ["topic", "question"]):
            raise ValueError("Generated question is missing required fields")

        return {"generated_question": response}
    except Exception as e:
        print(f"Error in generate_questions: {e}")
        return {"generated_question": f"Could not generate questions. Details: {str(e)}"}

def human_input(state: AgentState) -> AgentState:
    """Accepts a human response and updates the state."""
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
            retry = input("Do you want to retry? (y/n): ").lower()
            if retry != 'y':
                raise ValueError("User chose to exit after encountering an EOFError.")
            retries += 1

    raise ValueError("Maximum retries reached. Unable to get valid input.")

def reset_entries(state: AgentState) -> AgentState:
    """Resets specific entries in the state."""
    state.generated_question = None
    state.answer_evaluation = None
    state.human_answer = None
    return state

def evaluate_answers(state: AgentState) -> Dict[str, Union[EvaluateAnswers, str]]:
    """Generate evaluation for the human answer."""
    try:
        document_prompt = client.pull_prompt("interviewer_answer_evaluator_prompt")
        chain = (document_prompt | model.with_structured_output(EvaluateAnswers))
        evaluation_response = chain.invoke({
            "resume": state.resume,
            "job_description": state.job_description,
            "job_title": state.job_title,
            "company_name": state.company_name,
            "question": state.generated_question,
            "answer": state.human_answer,
            "history": state.history
        })
        
        state.history.append(evaluation_response.dict())
        return {"answer_evaluation": evaluation_response}
    except Exception as e:
        print(f"Error in evaluate_answers: {e}")
        return {"answer_evaluation": f"Could not process question and answer. Details: {str(e)}"}

def count_questions(state: AgentState) -> str:
    """Determines whether to finish based on the number of questions asked."""
    if len(state.history) >= 20:
        return "evaluate_candidate"
    else:
        return "generate_questions"
    
def evaluate_candidate(state: AgentState)-> Dict[str, Union[CandidateEvaluation, str]]:
    """Generate candidate evaluation using an LLM."""
    try:
        document_prompt = client.pull_prompt("candidate_evaluation_prompt")
        chain = (document_prompt | model.with_structured_output(CandidateEvaluation))
        response = chain.invoke({ "job_description": state.job_description,
                           "job_role": state.job_title,
                           "company_name": state.company_name,
                           "history": state.history})
        return {"candidate_evaluation": response}
    except Exception as e:
        print("Error:", e)
        return {"candidate_evaluation": f"Could not generate candidate_evaluation. Details: {str(e)}"}
    
def questions_topic_decider(state: AgentState) -> str:
    """Determines which topic to choose based on the number of questions asked."""
    if len(state.history) <= 7:
        return "personal_questions"
    elif len(state.history) <= 20:
        return "technical_questions"
    elif len(state.history) <= 27:
        return "critical_questions"
    else:
        return "hr_questions" 
    
# Define the graph
builder = StateGraph(AgentState)
builder.add_node("technical_questions", technical_questions)
builder.add_node("hr_questions", hr_questions)
builder.add_node("critical_questions", critical_questions)
builder.add_node("personal_questions", personal_questions)
builder.add_node("human_input", human_input)
builder.add_node("evaluate_answers", evaluate_answers)
builder.add_node("reset_entries", reset_entries)
builder.add_node("evaluate_candidate", evaluate_candidate)


builder.add_edge(START, "personal_questions")
builder.add_edge("personal_questions", "human_input")
builder.add_edge("technical_questions", "human_input")
builder.add_edge("critical_questions", "human_input")
builder.add_edge("hr_questions", "human_input")
builder.add_edge("human_input", "evaluate_answers")

builder.add_conditional_edges(
    "reset_entries",
    questions_topic_decider,
    {
        "personal_questions": "personal_questions",
        "technical_questions": "technical_questions",
        "critical_questions": "critical_questions",
        "hr_questions": "hr_questions"
    }
)

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

