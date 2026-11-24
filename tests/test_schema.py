import pytest
from pydantic import ValidationError

from paper_chat.schema import Paper, Chunk, Citation, Answer, LearningPlan, PlanStep, Resource

# Shared test data: a complete, valid paper.
VALID_PAPER = {
    "id": "p1",
    "title": "Sample Paper",
    "abstract": "This is a sample abstract.",
    "authors": ["Author One", "Author Two"],
    "source": "https://example.com/paper1",
    "year": 2023,
}

VALID_CHUNK = {
    "id": VALID_PAPER["id"] + "_c1",
    "paper_id": VALID_PAPER["id"],
    "text": "This is a chunk of text.",
    "index": 0,
}

VALID_CITATION = {"chunk_id": VALID_CHUNK["id"], "quote": "This is a about the citation"}

VALID_ANSWER = {"answer": "This is an exmaple of an answer", "citations": [VALID_CITATION]}


VALID_RESOURCE = {
    "title": "The Illustrated Transformer",
    "url": "https://example.com/illustrated-transformer",
    "why": "Visual walkthrough of the attention mechanism",
}

VALID_PLAN_STEP = {
    "title": "Understand self-attention",
    "what_to_do": "Read section 3 and sketch how queries, keys, and values interact",
    "resources": [VALID_RESOURCE],
}

VALID_LEARNING_PLAN = {
    "goal": "Explain how transformers work",
    "steps": [VALID_PLAN_STEP],
}

# Test Paper Schema


def test_paper_builds_with_all_fields():
    paper = Paper(**VALID_PAPER)
    assert paper.title == "Sample Paper"
    assert paper.authors == ["Author One", "Author Two"]
    assert paper.year == 2023


def test_paper_optional_fields_default_when_missing():
    data = {**VALID_PAPER}
    del data["abstract"]
    del data["authors"]
    del data["year"]

    paper = Paper(**data)
    assert paper.abstract is None
    assert paper.authors == []
    assert paper.year is None


@pytest.mark.parametrize("field", ["id", "title", "source"])
def test_paper_missing_title_raises(field):
    data = {**VALID_PAPER}
    del data[field]

    with pytest.raises(ValidationError):
        Paper(**data)


# Test Chunk Schema


def test_chunk_builds_with_all_fields():

    chunk_data = {**VALID_CHUNK}
    chunk = Chunk(**chunk_data)
    assert chunk.id == "p1_c1"
    assert chunk.text == "This is a chunk of text."
    assert chunk.index == 0


@pytest.mark.parametrize("field", ["id", "paper_id", "text", "index"])
def test_chunk_missing_fields(field):
    chunk_data = {**VALID_CHUNK}
    del chunk_data[field]
    with pytest.raises(ValidationError):
        Chunk(**chunk_data)


@pytest.mark.parametrize("bad_index", ["first", 1.5, None, "index"])
def test_chunk_missing_index(bad_index):
    chunk_data = {**VALID_CHUNK}
    chunk_data["index"] = bad_index
    with pytest.raises(ValidationError):
        Chunk(**chunk_data)


# Test Citation Schema


def test_citation_builds_with_all_fields():
    citation = Citation(**VALID_CITATION)
    assert citation.chunk_id == "p1_c1"
    assert citation.quote == "This is a about the citation"


@pytest.mark.parametrize("field", ["chunk_id", "quote"])
def test_citation_builds_missing_fields(field):
    citation_data = {**VALID_CITATION}
    del citation_data[field]

    with pytest.raises(ValidationError):
        Citation(**citation_data)


# Test Answer Schema


def test_answer_builds_with_all_fields():
    answer_data = {**VALID_ANSWER}
    answer = Answer(**answer_data)
    assert answer.answer
    assert answer.citations


def test_answer_with_no_citations_is_valid():
    answer_data = {**VALID_ANSWER}
    answer_data["citations"] = []

    answer = Answer(**answer_data)
    assert answer.citations == []


def test_answer_nested_citation_missing_quote_raises():
    answer_data = {**VALID_ANSWER}
    answer_data["citations"] = [{"chunk_id": "p1-c0"}]

    with pytest.raises(ValidationError) as exc_info:
        Answer(**answer_data)


@pytest.mark.parametrize("field", ["answer", "citations"])
def test_answer_builds_missing_fields(field):
    answer_data = {**VALID_ANSWER}
    del answer_data[field]
    with pytest.raises(ValidationError):
        Answer(**answer_data)


# Test PlanStep Schema
def test_plan_step_builds_with_all_fields():
    step = PlanStep(**VALID_PLAN_STEP)
    assert step.title == VALID_PLAN_STEP["title"]
    assert isinstance(step.resources[0], Resource)


def test_plan_step_resources_default_to_empty_list():
    data = {**VALID_PLAN_STEP}
    del data["resources"]

    step = PlanStep(**data)
    assert step.resources == []


@pytest.mark.parametrize("field", ["title", "what_to_do"])
def test_plan_step_missing_required_field_raises(field):
    data = {**VALID_PLAN_STEP}
    del data[field]
    with pytest.raises(ValidationError):
        PlanStep(**data)


def test_plan_step_resource_as_plain_string_raises():
    data = {**VALID_PLAN_STEP, "resources": ["just a title"]}

    with pytest.raises(ValidationError) as exc_info:
        PlanStep(**data)

    assert exc_info.value.errors()[0]["loc"] == ("resources", 0)


# Test LearningPlan Schema

def test_plan_step_builds_with_all_fields():
    step = PlanStep(**VALID_PLAN_STEP)
    assert step.title == VALID_PLAN_STEP["title"]
    assert isinstance(step.resources[0], Resource)


def test_plan_step_resources_default_to_empty_list():
    data = {**VALID_PLAN_STEP}
    del data["resources"]

    step = PlanStep(**data)
    assert step.resources == []


@pytest.mark.parametrize("field", ["title", "what_to_do"])
def test_plan_step_missing_required_field_raises(field):
    data = {**VALID_PLAN_STEP}
    del data[field]
    with pytest.raises(ValidationError):
        PlanStep(**data)


def test_plan_step_resource_as_plain_string_raises():
    data = {**VALID_PLAN_STEP, "resources": ["just a title"]}

    with pytest.raises(ValidationError) as exc_info:
        PlanStep(**data)

    assert exc_info.value.errors()[0]["loc"] == ("resources", 0)
