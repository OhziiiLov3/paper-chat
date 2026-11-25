from pydantic import BaseModel, Field


# === Stored Data Models (What is Ingested into the Vector Database) ===
class Paper(BaseModel):
    """
    Schema for a research paper.
    """

    id: str = Field(description="Unique identifier for the paper")
    title: str = Field(description="Title of the paper")
    abstract: str | None = Field(default=None, description="Abstract of the paper")
    source: str = Field(description="Source of the paper (e.g., url or file path)")
    authors: list[str] = Field(default_factory=list, description="List of authors of the paper")
    year: int | None = Field(default=None, description="Year of publication")


class Chunk(BaseModel):
    """
    Schema for a chunk of text from the paper.
    """

    id: str = Field(description="Unique identifier for the chunk")
    paper_id: str = Field(description="Identifier of the paper this chunk belongs to")
    page : int | None = Field(default=None, description="0-based page number the chunk came from")
    text: str = Field(description="text of the chunk")
    index: int = Field(description="0-based position of the chunk within its paper")


# === Shared Data Models (reused by several outputs) ===
class Citation(BaseModel):
    """
    Schema for a citation of a paper.
    """

    chunk_id: str = Field(description="Chunk.id of the chunk being cited")
    quote: str = Field(description="Quote from the chunk being cited")


class Resource(BaseModel):
    """
    Schema for a resource related to the paper.
    """

    title: str = Field(description="Title of the resource")
    url: str = Field(description="URL of the resource")
    why: str = Field(description="Reason why the resource is related to the paper")


# === Task-Specific Data Models (used for specific tasks) ===
class Summary(BaseModel):
    """
    Schema for a summary of the paper.
    """

    tldr: str = Field(description="Concise summary of the paper")
    key_points: list[str] = Field(description="List of key points from the paper")
    citations: list[Citation] = Field(description="List of citations from the paper")


class Answer(BaseModel):
    """
    Schema for an answer to a question about the paper.
    """

    answer: str = Field(description="Answer to the question")
    citations: list[Citation] = Field(description="List of citations from the paper")


class PlanStep(BaseModel):
    """
    Schema for a step to achieve a goal related to the paper.
    """

    title: str = Field(description="Title of the step")
    what_to_do: str = Field(description="Description of what to do in this step")
    resources: list[Resource] = Field(
        default_factory=list, description="List of resources for this step"
    )


class LearningPlan(BaseModel):
    """
    Schema for a learning plan related to the paper.
    """

    goal: str = Field(description="Goal of the learning plan")
    steps: list[PlanStep] = Field(min_length=1, description="List of steps to achieve the goal")


# Test code for the Paper model
# paper = Paper(
#     id="1",
#     title="A Comprehensive Study on Machine Learning",
#     abstract="This paper provides an in-depth analysis of various machine learning algorithms and their applications.",
#     source="https://example.com/paper1",
#     authors=["John Doe", "Jane Smith"],
#     year=2023
# )

# print(paper.model_dump())
