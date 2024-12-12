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

@dataclass
class AgentState:
    resume: str
    job_description: str
    job_title: str
    company_name: str
    history: List[Dict[str, str]] = field(default_factory=list)
    generated_question: Optional[Union[AgentAction, AgentFinish, Dict[str, str]]] = None
    human_answer: Optional[str] = None
    answer_evaluation: Optional[Union[AgentAction, AgentFinish, Dict[str, str]]] = None
    candidate_evaluation: Optional[Union[AgentAction, AgentFinish, Dict[str, str]]] = None