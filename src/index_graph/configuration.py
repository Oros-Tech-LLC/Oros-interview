"""Define the configurable parameters for the index graph."""

from __future__ import annotations
from shared.configuration import BaseConfiguration
from dataclasses import dataclass, field
from typing import Optional, TypedDict, List, Dict, Union, Annotated
from langchain.schema import AgentAction, AgentFinish
import operator

@dataclass(kw_only=True)
class AgentConfiguration(TypedDict):
    """Configuration class for indexing and retrieval operations.

    This class defines the parameters needed for configuring the indexing and
    retrieval processes, including input data for job-specific question generation.
    """

    question: str = field(
        metadata={
            "description": "The interview question for the candidate."
        }
    )
    image: str = field(
        metadata={
            "description": "The URL of the image associated with the question."
        }
    )
    image_type: str = field(
        metadata={
            "description": "The type of the image (e.g., 'png', 'jpg')."
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