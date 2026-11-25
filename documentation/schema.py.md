# Data Schemas: Designing and Testing the Contracts for a RAG Chatbot

**Folder:** `src/paper_chat/` (`schema.py`), tested in `tests/` (`test_schema.py`)

## What you'll learn
- How to use Pydantic models as contracts, grouped into stored, shared, and task-output layers
- How to link models with ids (foreign keys) so answers can cite their sources
- Why field descriptions are instructions to the LLM, and why validation checks shape, not truth
- How to test each schema decision with pytest: start from valid data, copy it, break one thing
- How to use `parametrize` and error locations to write tests that fail for the right reason

This file defines every piece of data the paper chatbot handles, from the paper you load to the answer the user reads. The tests lock those design decisions in place, so a later change can't quietly break them.

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
    authors: list[str] = Field(default_factory=list, description="List of authors of the paper")
    year: int | None = Field(default=None, description="Year of publication")
```

A field with **no default is required automatically**. `Field(description=...)` only adds a description; it doesn't make the field optional. Leaving out `title` gives (real output, Pydantic 2.14.4):

```
1 validation error for Paper
title
  Field required [type=missing, input_value={'id': 'p1', 'source': 's', 'authors': []}, input_type=dict]
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

**Where each model is used today:**

| Model | Built in | Status |
|---|---|---|
| `Paper` | `ingest.py` (`build_paper`, one per PDF) | In use, see [ingest.py.md](ingest.py.md) |
| `Chunk` | `ingest.py` (`split_docs`, one per chunk) | In use |
| `Answer`, `Citation` | `rag.py` (LLM structured output) | In use |
| `Summary`, `LearningPlan`, `PlanStep`, `Resource` | Not yet | Planned features |

A schema nobody builds is a contract nobody has agreed to. Either build its feature or move it to a later branch.

---

## 3. Optional fields: real data is messy

Some PDFs have no abstract, no clear year, or authors you can't parse. If those fields were required, loading such a paper would crash.

```python
abstract: str | None = Field(default=None, description="Abstract of the paper")
authors: list[str] = Field(default_factory=list, description="List of authors of the paper")
year: int | None = Field(default=None, description="Year of publication")
```

| Part | What it does |
|---|---|
| `str \| None` | The value may be a string or empty (`None`) |
| `default=None` | You may leave the field out entirely |
| `default_factory=list` | If left out, call `list()` to make a **new** empty list |

**Why `[]` and not `None` for lists?** Code can always loop over an empty list without checking first. Use `None` only when "unknown" means something different from "empty".

**Why `default_factory=list` and not `default=list`?** `default=` uses the value as-is, so `default=list` stores the `list` *type* itself. Real output:

```python
Paper(id="1", title="t", source="s").authors
# <class 'list'>      ← with default=list (bug)
# []                  ← with default_factory=list (correct)
```

> **Key point:** Make a field required only if you can *always* get it. Make it optional if you can only *sometimes* get it.

---

## 4. Chunks and foreign keys

A **chunk** is a small piece of a paper's text. RAG searches chunks, not whole papers, because a small passage matches a question more precisely and fits in the prompt.

```python
class Chunk(BaseModel):
    id: str = Field(description="Unique identifier for the chunk")
    paper_id: str = Field(description="Identifier of the paper this chunk belongs to")
    page: int | None = Field(default=None, description="0-based page number the chunk came from")
    text: str = Field(description="Text of the chunk")
    index: int = Field(description="0-based position of the chunk within its paper")
```

`page` was added once ingestion started building `Chunk` models. Citations need to say *where* in the paper a quote came from. It is optional (`None` by default) because not every source has pages, and that's why the existing 27 schema tests still pass without changes.

**Why it had to be added to the schema:** Pydantic **silently drops fields it doesn't know about**. Real output from a model without a `page` field:

```python
NoPage(id="x", page=3).model_dump()
# {'id': 'x'}        ← page is gone, no error
```

Without the field, `Chunk(..., page=3)` would have lost the page number and nothing would have warned you.

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


class PlanStep(BaseModel):
    title: str = Field(description="Title of the step")
    what_to_do: str = Field(description="Description of what to do in this step")
    resources: list[Resource] = Field(
        default_factory=list, description="List of resources for this step"
    )


class LearningPlan(BaseModel):
    goal: str = Field(description="Goal of the learning plan")
    steps: list[PlanStep] = Field(min_length=1, description="List of steps to achieve the goal")
```

Three deliberate choices live here:

| Field | Rule | Why |
|---|---|---|
| `Answer.citations` | Required, but may be `[]` | An honest "the paper doesn't cover this" has no sources. Forcing one invites a made-up citation. |
| `PlanStep.resources` | Defaults to `[]` | Not every step needs a link, and fewer links means fewer invented URLs. |
| `LearningPlan.steps` | `min_length=1` | A plan with zero steps is a failure, even though its shape is valid. |

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

## 9. What to test: your decisions, not Pydantic

A schema test proves the model accepts what it should and rejects what it shouldn't. You don't need to prove that a `str` field accepts a string; Pydantic already does that. Test the **choices you made**:

| Decision | Test |
|---|---|
| `abstract`, `year`, `authors` are optional | Leave them out → `None`, `None`, `[]` |
| `id`, `title`, `source` are required | Leave each out → `ValidationError` |
| `index` must be a whole number | `"first"`, `1.5`, `None` → `ValidationError` |
| `citations=[]` is allowed | Empty list → builds |
| `steps` needs at least one item | Empty list → `ValidationError` (`too_short`) |
| Nesting is validated | A bad citation inside an `Answer` → error at `("citations", 0, "quote")` |

Run the tests with:

```
uv run pytest -v                         # all tests (32: 27 schema + 5 ingest)
uv run pytest -v tests/test_schema.py    # schema tests only
uv run pytest -v -k chunk                # tests whose name contains "chunk"
```

None of them need an API key, so they are safe to run as often as you like, including in CI. The ingest tests check that real pipeline code builds valid `Paper` and `Chunk` models; see [ingest.py.md](ingest.py.md).

`-v` (verbose) lists each test by name with `PASSED` or `FAILED`. This is why test names should state the behavior: the output reads like a description of what the code does. Prefer `-v` to `print()`. A print can say "passed" when nothing was checked, but `PASSED` means every assertion ran.

---

## 10. The pattern: valid data, copy, break one thing

Every test starts from one known-good dictionary per model, defined once at the top of the file:

```python
from paper_chat.schema import Answer, Chunk, Citation, LearningPlan, Paper, PlanStep, Resource

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
    "paper_id": VALID_PAPER["id"],  # chunk → paper
    "text": "This is a chunk of text.",
    "index": 0,
}

VALID_CITATION = {"chunk_id": VALID_CHUNK["id"], "quote": "This is about the citation"}
```

The test data is **linked** the same way the models are (`Paper → Chunk → Citation`). If a chunk's id changes, the citation follows.

**Import by package name.** `from paper_chat.schema import ...` works because uv installs the project into `.venv`. Don't write `src.` in the import path, and put imports at the top of the file, not inside tests.

Each test then copies the data, changes **one** thing, and checks the result:

```python
def test_paper_optional_fields_default_when_missing():
    data = {**VALID_PAPER}
    del data["abstract"]
    del data["authors"]
    del data["year"]

    paper = Paper(**data)
    assert paper.abstract is None
    assert paper.authors == []
    assert paper.year is None
```

| Syntax | Meaning |
|---|---|
| `{**VALID_PAPER}` | Unpack into a **new** dict, so the test gets a copy and the shared data stays untouched |
| `Paper(**data)` | Unpack the dict into keyword arguments: `Paper(id="p1", title=...)` |
| `{**VALID_PAPER, "title": "X"}` | Copy and replace one value in one step |

`{**...}` is a **shallow copy**: lists inside are still shared. Replacing a value (`data["citations"] = []`) is safe, but `data["authors"].append(...)` would change `VALID_PAPER` for every later test.

For a failure test, **break the input, then build inside `pytest.raises`**:

```python
def test_chunk_index_wrong_type_raises():
    chunk_data = {**VALID_CHUNK}
    chunk_data["index"] = "first"  # 1. break the data
    with pytest.raises(ValidationError):  # 2. build, which should fail
        Chunk(**chunk_data)
```

Validation only happens **while** the object is being built. Changing the object afterwards (`del chunk.paper_id`) tests nothing.

> **Key point:** Invalid data = valid data minus exactly one thing. Then, when a test fails, you know exactly which field caused it.

---

## 11. `parametrize`: one test, many inputs

When only the input changes and the check stays the same, `pytest.mark.parametrize` runs one test once per value:

```python
@pytest.mark.parametrize("field", ["chunk_id", "quote"])
def test_citation_missing_required_field_raises(field):
    citation_data = {**VALID_CITATION}
    del citation_data[field]

    with pytest.raises(ValidationError):
        Citation(**citation_data)
```

| Part | Meaning |
|---|---|
| `"field"` | The parameter name. It must match the function's argument. |
| `["chunk_id", "quote"]` | The values. The test runs once with each. |

Each value shows up as its own result:

```
test_citation_missing_required_field_raises[chunk_id] PASSED
test_citation_missing_required_field_raises[quote] PASSED
```

You can parametrize **values** as well as field names, for example `["first", 1.5, None]` for bad `index` inputs.

When the *check* is different for each case (for example, missing `authors` → `[]` but missing `year` → `None`), write separate assertions or separate tests instead.

---

## 12. Testing nested models and error details

Nested test data is built from the inside out, just like the models:

```python
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
```

Plain `pytest.raises` passes on **any** `ValidationError`. To make sure a test fails for the right reason, keep the error with `as exc_info` and inspect it:

```python
def test_answer_nested_citation_missing_quote_raises():
    answer_data = {**VALID_ANSWER}
    answer_data["citations"] = [{"chunk_id": "p1-c0"}]  # no "quote"

    with pytest.raises(ValidationError) as exc_info:
        Answer(**answer_data)

    assert exc_info.value.errors()[0]["loc"] == ("citations", 0, "quote")
```

| Expression | Gives you | Real value |
|---|---|---|
| `errors()[0]["loc"]` | **Where** it failed | `("citations", 0, "quote")` |
| `errors()[0]["type"]` | **What kind** of failure | `"missing"`, `"too_short"`, `"model_type"` |

Examples from these models (real output):

| Input | `loc` | `type` |
|---|---|---|
| Citation missing `quote` inside an `Answer` | `("citations", 0, "quote")` | `missing` |
| `PlanStep` with `resources=["just a title"]` | `("resources", 0)` | `model_type` |
| `LearningPlan` with `steps=[]` | `("steps",)` | `too_short` |

The full `too_short` message:

```
steps
  List should have at least 1 item after validation, not 0 [type=too_short, input_value=[], input_type=list]
```

LLMs send text, not dicts, so also test the JSON path. Broken JSON raises `ValidationError` too (type `json_invalid`), so one `except ValidationError` in the app catches both broken JSON and the wrong shape:

```python
answer = Answer.model_validate_json(
    '{"answer": "The paper uses self-attention.", '
    '"citations": [{"chunk_id": "p1-c0", "quote": "This is a chunk"}]}'
)
assert isinstance(answer.citations[0], Citation)  # inner dict became a real Citation
```

---

## 13. See it fail

A test you've never seen fail might not test anything. After a test passes, make a **temporary** change so it should fail, confirm that it does, then undo the change.

| Test | Temporary change | Expected result |
|---|---|---|
| `test_answer_nested_citation_missing_quote_raises` | Add `"quote": "x"` to the nested citation | `DID NOT RAISE ValidationError` |
| A `LearningPlan` empty-steps test | Delete `min_length=1` from `schema.py` | Only that test fails |

The check is always a temporary edit to the **input** or the **code under test**. It never adds a new assertion.

This check caught a real problem while building these tests. An early "missing fields" test passed only because `id` was also missing. When `id` was added back, it failed with `DID NOT RAISE`, which showed that missing `authors` and `year` are valid by design. The schema was right and the test's expectation was wrong.

---

## Key ideas
- A schema is a contract. Group models by layer and link them with ids named after what they point to.
- Required means "always available". Everything else is optional with a default (`None` or `default_factory=list`).
- Field descriptions are instructions to the LLM. Validation checks shape; your code must check truth.
- Test your design decisions: start from valid data, copy it, break exactly one thing.
- Check `loc` and `type` on errors so a test can't pass for the wrong reason.
- Watch every test fail once before you trust it.

## Common mistakes
- **Renaming the wrong field.** While building this file, `paper_id` was renamed to `chunk_id` on `Chunk` by mistake. That broke the link to `Paper`, and the field's name no longer matched its description. Read each field as a sentence after you rename it.
- **Making messy fields required.** `abstract: str` crashes on PDFs with no abstract.
- **`default=list` instead of `default_factory=list`.** The default becomes the `list` type itself, not an empty list.
- **Out-of-date docstrings.** Docstrings are sent to the LLM, so a wrong one misleads both people and the model.
- **Running code at import time.** A `print()` at module level runs every time another file imports the schemas.
- **Trusting LLM output because it validated.** A valid shape can still hold a made-up quote or URL.
- **Passing a dict positionally.** `Paper(data)` raises `TypeError: BaseModel.__init__() takes 1 positional argument but 2 were given`. Use `Paper(**data)` or `Paper.model_validate(data)`.
- **Breaking the object after building it.** `del chunk.paper_id` runs after validation and tests nothing.
- **Removing several fields in one failure test.** The test passes if *any* of them is required, so it can't tell you which one is.
- **A test with no `assert` and no `pytest.raises`.** It can never fail.
- **Duplicate test names.** If two functions in one file share a name, Python keeps only the last one, and pytest silently runs one copy.
- **Test names that don't match the test.** `test_paper_missing_title_raises` that checks `id`, `title`, and `source` misleads whoever reads the `-v` output.
- **Passing a field the schema doesn't define.** Pydantic drops it without an error. Add the field to the schema first (as with `Chunk.page`).
- **Capturing `exc_info` and never checking it.** `with pytest.raises(...) as exc_info:` with no `assert` on `exc_info` afterwards is the same as plain `pytest.raises`: it can pass for the wrong reason.

## Practice
1. Write a `Visual` model for the visuals task (hint: `kind`, `code`, `caption`, with Mermaid as the simplest option), then test it with the valid-copy-break pattern.
2. Write `LearningPlan` tests: one that builds, one where `steps=[]` raises with type `too_short`, and one where a nested step missing `what_to_do` fails at `("steps", 0, "what_to_do")`.
3. Write `Summary` tests, reusing `VALID_CITATION`.
4. Add `Chunk.page` tests: leaving it out gives `None`, and `page="first"` raises `int_parsing`.
5. Write one test that loops over every model's `model_fields` and asserts each field has a `.description`, so a field added later without one fails the build.
6. Write a function `check_citations(answer, chunks)` that returns citations whose `chunk_id` isn't in `chunks` or whose `quote` isn't in that chunk's `text`, and test it.

## Teaching tip
Show learners the same field with two descriptions, `"An identifier"` and `"Chunk.id of the chunk being cited"`, then print `model_json_schema()` for each. Ask, "If you were the LLM and saw only this, which value would you copy?" It makes clear that descriptions are prompts and not comments. For testing, run a failure test that deletes two fields, then add one back and watch it still pass: learners see immediately why "break exactly one thing" matters.
