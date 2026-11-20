# Data Schemas: Designing the Contracts for a RAG Chatbot

**Folder:** `src/paper_chat/` (`schema.py`)

## What you'll learn
- How to use Pydantic models as contracts between parts of an AI app
- How to split a RAG app's data into stored, shared, and task-output layers
- How to link models with ids (foreign keys) so answers can cite their sources
- How field descriptions turn into instructions for the LLM
- Why schema validation checks shape, and your own code must check truth

This file defines every piece of data the paper chatbot handles, from the paper you load to the answer the user reads. Getting these shapes right early means each later step (chunking, retrieval, prompting) has a clear target to hit.

---

## 1. A schema is a contract

A **schema** is a description of what valid data looks like. In Python, Pydantic (a data-validation library) lets you write one as a class. When data doesn't match, Pydantic raises an error right away, at the edge of your system, before bad data spreads.

```python
from pydantic import BaseModel, Field

class Paper(BaseModel):
    """
    Schema for a research paper.
    """

    id: str = Field(description="Unique identifier for the paper")
    title: str = Field(description="Title of the paper")
    abstract: str | None = Field(default=None, description="Abstract of the paper")
    source: str = Field(description="Source of the paper (e.g., url or file path)")
    authors: list[str] = Field(description="List of authors of the paper")
    year: int | None = Field(default=None, description="Year of publication")
```

Leaving out a required field gives a clear error (real output, Pydantic 2.14.4):

```
1 validation error for Paper
authors
  Field required [type=missing, input_value={'id': 'p1', 'title': 'X', 'source': 's'}, input_type=dict]
```

> **Key point:** Every model is an agreement between two parts of the system. Ask "who creates this data, and who uses it?" for each one.

---

## 2. Three layers of data

A RAG app (retrieval-augmented generation: the LLM answers using passages fetched from your documents) moves data through three stages. The file groups its models the same way.

| Layer | Models | Created by | Trust level |
|---|---|---|---|
| **Stored**: what gets ingested | `Paper`, `Chunk` | Your code (parsing, chunking) | High |
| **Shared**: reusable pieces | `Citation`, `Resource` | The LLM | Low, needs checking |
| **Task outputs**: what the user sees | `Summary`, `Answer`, `PlanStep`, `LearningPlan` | The LLM | Low, needs checking |

The "created by" column drives the design. Data your code builds can be trusted. Data the LLM writes cannot, even when it has the right shape.

---

## 3. Optional fields: real data is messy

Some PDFs have no abstract or no clear year. If those fields were required, loading such a paper would crash.

```python
abstract: str | None = Field(default=None, description="Abstract of the paper")
year: int | None = Field(default=None, description="Year of publication")
```

| Part | What it does |
|---|---|
| `str \| None` | The value may be a string or empty (`None`) |
| `default=None` | You may leave the field out entirely |

You need both. With only `| None`, you would still have to write `abstract=None` every time.

Real output when both are left out:

```python
Paper(id="p1", title="Attention Is All You Need", source="papers/attention.pdf", authors=["Vaswani"])
# {'id': 'p1', 'title': 'Attention Is All You Need', 'abstract': None,
#  'source': 'papers/attention.pdf', 'authors': ['Vaswani'], 'year': None}
```

> **Key point:** Make a field required only if you can *always* get it. Make it optional if you can only *sometimes* get it.

---

## 4. Chunks and foreign keys

A **chunk** is a small piece of a paper's text. RAG searches chunks, not whole papers, because a small passage matches a question more precisely and fits in the prompt.

```python
class Chunk(BaseModel):
    id: str = Field(description="Unique identifier for the chunk")
    paper_id: str = Field(description="Identifier of the paper this chunk belongs to")
    text: str = Field(description="Text of the chunk")
    index: int = Field(description="0-based position of the chunk within its paper")
```

`paper_id` is a **foreign key**: a field that points to another model's `id`. It's what lets a chunk say which paper it came from.

```
Paper ──< Chunk ──< Citation
 id        id         chunk_id
           paper_id
```

(`──<` means "one to many": one paper has many chunks.)

- **Naming rule:** name a reference after the thing it points to (`paper_id` → `Paper.id`, `chunk_id` → `Chunk.id`).
- **`id` vs `index`:** `id` is unique across all papers. `index` is only unique inside one paper, but it keeps chunks in order.

Pydantic also converts simple types for you. `index="3"` becomes the integer `3`, but `index="first"` fails:

```
index
  Input should be a valid integer, unable to parse string as an integer [type=int_parsing, ...]
```

---

## 5. Shared models: build once, reuse

When the same shape appears in more than one place, give it its own model.

```python
class Citation(BaseModel):
    chunk_id: str = Field(description="Chunk.id of the chunk being cited")
    quote: str = Field(description="Quote from the chunk being cited")

class Resource(BaseModel):
    title: str = Field(description="Title of the resource")
    url: str = Field(description="URL of the resource")
    why: str = Field(description="Reason why the resource is related to the paper")
```

- `Citation` is used by both `Summary` and `Answer`. It's how the user checks where a claim came from.
- `Resource` is used inside `PlanStep`, and it can also be listed on its own.

For citations to work, the chunks you put into the prompt must be labeled with their `id`. Short, readable ids (for example `p1-c3`) are easier for the LLM to copy correctly than long random strings.

---

## 6. Task models: one per job

Each user task gets its own output model.

| User task | Model |
|---|---|
| Simplify / summarize an article | `Summary` |
| Answer follow-up questions | `Answer` |
| Draft a learning plan | `LearningPlan` (made of `PlanStep`s) |
| List resources | `Resource` |
| Provide visuals | not built yet |

```python
class Answer(BaseModel):
    answer: str = Field(description="Answer to the question")
    citations: list[Citation] = Field(description="List of citations from the paper")

class LearningPlan(BaseModel):
    goal: str = Field(description="Goal of the learning plan")
    steps: list[PlanStep] = Field(description="List of steps to achieve the goal")
```

Models can be **nested** (one model holds a list of another). Validation reaches inside nested models too. Here the LLM returned a citation without a quote:

```python
Answer.model_validate_json('{"answer": "Self-attention.", "citations": [{"chunk_id": "p1-c3"}]}')
```
```
1 validation error for Answer
citations.0.quote
  Field required [type=missing, input_value={'chunk_id': 'p1-c3'}, input_type=dict]
```

The path `citations.0.quote` tells you exactly where the problem is: the first citation is missing its quote.

> **Key point:** Small, focused schemas are easier for the LLM to fill correctly than one big schema full of optional fields. Your code already knows which task it's running, so it can ask for the matching model.

---

## 7. Descriptions are prompts

For models the LLM fills in, the class docstring and each `Field(description=...)` are sent to the LLM as part of the JSON schema. Real output of `Citation.model_json_schema()`:

```json
{
  "description": "Schema for a citation of a paper.",
  "properties": {
    "chunk_id": {
      "description": "Chunk.id of the chunk being cited",
      "title": "Chunk Id",
      "type": "string"
    },
    "quote": {
      "description": "Quote from the chunk being cited",
      "title": "Quote",
      "type": "string"
    }
  },
  "required": ["chunk_id", "quote"],
  "title": "Citation",
  "type": "object"
}
```

Nested models are included automatically: `LearningPlan`'s schema carries definitions for `PlanStep` and `Resource`.

So write descriptions the way you'd write instructions. "Chunk.id of the chunk being cited" tells the LLM exactly which value to copy. "An identifier" would not.

---

## 8. Shape vs. truth

Pydantic checks that data has the right **shape**. It cannot check that the data is **true**. LLM-written fields need extra checks in your own code:

| Field | Risk | Check |
|---|---|---|
| `Citation.chunk_id` | Points to a chunk that wasn't retrieved | Must match a chunk id you sent |
| `Citation.quote` | Invented quote that sounds real | Must appear in that chunk's `text` |
| `Resource.url` | Believable URL that doesn't exist | Use web search, or only list papers the source paper cites |

---

## Key ideas
- A schema is a contract between whoever creates the data and whoever uses it.
- Group models by layer: stored, shared, task outputs.
- Link models with ids, and name each reference after what it points to.
- Required means "always available". Everything else is optional with a default.
- Field descriptions are instructions to the LLM. Write them carefully.
- Validation checks shape. Your code must check truth.

## Common mistakes
- **Renaming the wrong field.** While building this file, `paper_id` was renamed to `chunk_id` on `Chunk` by mistake. That broke the link to `Paper`, and the field's name no longer matched its description. Read each field as a sentence after you rename it.
- **Making messy fields required.** `abstract: str` crashes on PDFs with no abstract.
- **Type mismatches between references.** A citation field typed `str` that points to an `int` field.
- **Out-of-date docstrings.** Docstrings are sent to the LLM, so a wrong one misleads both people and the model.
- **Running code at import time.** A `print()` at module level runs every time another file imports the schemas.
- **Trusting LLM output because it validated.** A valid shape can still hold a made-up quote or URL.

## Practice
1. Write a `Visual` model for the visuals task (hint: `kind`, `code`, `caption`, with Mermaid as the simplest option).
2. Write a test in `tests/` that builds a `Paper` without `abstract` or `year`, and checks both are `None`.
3. Write a test that `Answer.model_validate_json(...)` raises `ValidationError` when a citation is missing its `quote`.
4. Write a function `check_citations(answer, chunks)` that returns citations whose `chunk_id` isn't in `chunks` or whose `quote` isn't in that chunk's `text`.
5. Print `LearningPlan.model_json_schema()` and read it as if you were the LLM. Rewrite any description that is unclear.

## Teaching tip
Show learners the same field with two descriptions, `"An identifier"` and `"Chunk.id of the chunk being cited"`, then print `model_json_schema()` for each. Ask, "If you were the LLM and saw only this, which value would you copy?" It makes clear that descriptions are prompts and not comments, and that a schema is something both code and the model have to read.
