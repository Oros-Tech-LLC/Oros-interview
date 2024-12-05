"""State management for the index graph."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Annotated, Optional

from langchain_core.documents import Document

from shared.state import reduce_docs
from shared.configuration import BaseConfiguration
from dataclasses import dataclass, field
from typing import Optional, TypedDict, List, Dict, Union, Annotated
from langchain.schema import AgentAction, AgentFinish
import operator


class AgentState(TypedDict):

    resume: str = field(
        metadata={
            "description": "The resume of the candidate."
        }
    )
    job_description: str = field(
        metadata={
            "description": "The job description for the role, represented as a string."
        }
    )
    job_title: str = field(
        metadata={
            "description": "The job title for which questions and evaluations are generated."
        }
    )
    company_name: str = field(
        metadata={
            "description": "The name of the company associated with the job description."
        }
    )
    history: Optional[List[Dict[str, str]]] = field(
        default=None,
        metadata={
            "description": "History of previously asked questions and answers as a list of dictionaries."
        }
    )
    agent_out: Union[AgentAction, AgentFinish, None] = field(
        default=None,
        metadata={
            "description": "The output from the agent, which can be an action or a finish signal."
        }
    )
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add] = field(
        default_factory=list,
        metadata={
            "description": "A list of intermediate steps taken by the agent, each represented as a tuple of action and description."
        }
    )