# Understand the code: LangChain and LangGraph

This guide follows the code in this repository. You can run the same PDF-and-web research assistant with either implementation. Both use Groq to generate responses, FAISS to search a PDF, and Serper to search the web.

## 1. Start with a simple question

Upload the [sample PDF](../sample_documents/research_assistant_test.pdf) and ask:

> What was revenue growth in 2025? Cite the PDF page.

The sample report says revenue grew **34 percent**, on **page 1**. The assistant should search the PDF and use that evidence in its answer.

The important distinction is between the model and the tools:

| Part | Job | Everyday comparison |
|---|---|---|
| Groq chat model | Interprets the question, requests tools, and writes the answer | A researcher deciding what to look up |
| PDF tool | Retrieves relevant passages from your document | Looking through a report |
| Web tool | Gets current search snippets and URLs from Serper | Searching the internet |
| Agent executor or graph | Runs the requested tools and returns results to the model | The application coordinating the research |
| Streamlit | Receives uploads/questions and displays answers | The desk where you work with the researcher |

The model does not execute Python itself. It requests a tool; the application calls the corresponding Python code.

## 2. Where to read the code

Read the files in this order:

| File | What to look for |
|---|---|
| [app.py](../app.py) | Starts the Streamlit interface by calling `run()` |
| [ui.py](../research_assistant/ui.py) | Selects the implementation, processes uploads, and calls `service.ask()` |
| [config.py](../research_assistant/config.py) | Loads API keys, model settings, and resource limits |
| [documents.py](../research_assistant/documents.py) | Turns PDF bytes into a searchable index |
| [tools.py](../research_assistant/tools.py) | Exposes PDF retrieval and web search to the agent |
| [prompts.py](../research_assistant/prompts.py) | Tells the model how to research and cite evidence |
| [agent.py](../research_assistant/agent.py) | Runs the classic LangChain ReAct implementation |
| [langgraph_agent.py](../research_assistant/langgraph_agent.py) | Runs the explicit LangGraph workflow |
| [errors.py](../research_assistant/errors.py) | Defines errors safe to display to users |

## 3. What happens when you process a PDF

This part is shared by both implementations.

1. `ui.run()` receives the uploaded file and reads its bytes with `uploaded.getvalue()`.
2. `build_index()` calls `parse_pdf()` to validate the upload and extract page text.
3. Each page becomes a LangChain `Document` with `page_number` metadata. Numbering starts at 1 so it matches the physical PDF pages.
4. `RecursiveCharacterTextSplitter` splits the text into chunks of 1,000 characters with 200 characters of overlap. Overlap keeps nearby context around chunk boundaries.
5. `FastEmbedEmbeddings` converts each chunk into a numerical vector locally. The embedding model downloads on first use.
6. `FAISS.from_documents()` builds an in-memory index of those vectors.
7. `as_retriever(search_kwargs={"k": 4})` exposes similarity search that retrieves up to four relevant chunks for each query.

The key code in `documents.py` is:

```python
chunks = splitter.split_documents(documents)
embeddings = embeddings or FastEmbedEmbeddings(model_name=settings.embedding_model)
store = FAISS.from_documents(chunks, embeddings)
```

When you later ask about revenue, the retriever embeds the search query and looks for nearby chunk vectors. This is similarity search: it is not simply an exact-word lookup.

**RAG** means retrieving relevant evidence and giving it to the model to help generate an answer. FAISS retrieves passages; Groq writes the answer. Uploading a PDF does not train or fine-tune the Groq model.

Uploaded PDF bytes are not saved to a shared file. The index and chat are kept in Streamlit session memory; relevant retrieved passages are sent to Groq when used in research.

## 4. How the model chooses PDF or web search

`build_tools()` in `tools.py` supplies two named tools:

```python
pdf_knowledge_base  # Search the uploaded PDF for document-specific questions.
web_search          # Search current web facts; results include source URLs.
```

These lines are a simplified illustration of the names and descriptions, not executable tool definitions. The actual PDF tool is created with `create_retriever_tool`; the web tool wraps the `search_web()` function in a LangChain `Tool`.

The prompts in `prompts.py` instruct the model to use:

| Your question | Intended source |
|---|---|
| What does this PDF say about revenue? | PDF |
| What is the latest LangChain release? | Web |
| Compare this PDF's design with current guidance. | PDF and web |

The model reads the question, recent context, instructions, and tool descriptions before making its choice. There is no Python rule such as `if "latest" in question: use_web()`.

These are intended choices, not guaranteed outcomes. The model can choose incorrectly, call a tool more than once, or finish without calling one. **Sources searched** shows the tool names actually recorded for that request.

The web tool sends a query to Serper and returns up to three search results, each with a title, link, and snippet. It does not open and read every linked page.

## 5. LangChain version: the executor manages the loop

Open [agent.py](../research_assistant/agent.py), especially `ResearchService.create()` and `ResearchService.ask()`.

### Build the agent

The key setup is:

```python
tools = build_tools(index, settings)
agent = create_react_agent(llm, tools, RESEARCH_PROMPT)
```

The code then creates an `AgentExecutor` with the agent, tools, error handling, and iteration/time limits. `create_react_agent()` constructs the agent; `AgentExecutor` coordinates repeated model and tool calls.

This project uses the classic API imported from `langchain_classic.agents`. This is one LangChain implementation, not a description of every modern LangChain agent API.

### Ask a question

`ask()` validates the question, builds bounded conversation context, and invokes:

```python
result = self.executor.invoke({
    "input": question,
    "chat_history": context[-self.settings.max_history_chars:],
})
```

The classic model is instructed to request tools in a text format:

```text
Action: pdf_knowledge_base
Action Input: revenue growth in 2025
```

The executor parses this text, finds the matching tool, and executes it. The PDF tool might return:

```text
[PDF page 1] Year-over-year revenue growth in 2025: 34 percent.
```

The executor returns that observation to the model in the next prompt. When enough evidence is available, the model finishes with text such as:

```text
Final Answer: Revenue grew 34 percent in 2025 [PDF page 1].
```

These are illustrative responses. Actual wording and tool inputs vary.

### What is agent_scratchpad?

In `RESEARCH_PROMPT`, this placeholder appears at the end:

```text
Thought:{agent_scratchpad}
```

LangChain fills `agent_scratchpad` with the current request's prior tool actions and observations. It lets the next model call see what has already been searched.

| Variable | Contains |
|---|---|
| `input` | The current question |
| `chat_history` | Earlier completed user/assistant exchanges |
| `agent_scratchpad` | Tool actions and results within the current request |
| `tools` / `tool_names` | Available tool descriptions and names, filled by the agent setup |

You do not manually set `agent_scratchpad` in the Streamlit UI. It is temporary execution context, not permanent memory or a guaranteed description of the model's internal reasoning.

### Return to the UI

The service extracts `result["output"]` and the tool names in `intermediate_steps`. It returns `ResearchAnswer(text, tools_used)` so the UI can display the answer and the sources searched without exposing the full execution trace.

## 6. LangGraph version: the code defines the loop

Open [langgraph_agent.py](../research_assistant/langgraph_agent.py), starting with `ResearchState` and `build_graph()`.

The same research behavior is expressed as two nodes with explicit connections:

```text
START -> model
           |
           +-- tool calls present -> tools -> model
           |
           +-- no tool calls ------> END
```

### State is the information passed between nodes

```python
class ResearchState(MessagesState):
    rounds: int
    deadline: float
```

`MessagesState` provides a `messages` field with a reducer that merges returned messages into the existing list. Normal new messages are appended; messages with an existing ID can replace that message. The other fields track completed tool batches and the cooperative deadline.

### bind_tools tells the model what it can request

```python
bound_model = model.bind_tools(tools)
```

This gives the model structured tool schemas. It does not execute tools automatically. The model returns an `AIMessage` containing tool-call data; our tools node executes those requests.

For PDF retrieval, a simplified tool call looks like:

```python
{
    "name": "pdf_knowledge_base",
    "args": {"query": "revenue growth in 2025"},
    "id": "call_1",
}
```

The `args` fields must match the selected tool's schema. For example, the PDF retriever accepts `query`; the web tool's schema is supplied by its LangChain `Tool` wrapper. This example shows data, not a complete runnable program.

### The model node asks Groq what to do next

`call_model()` invokes the bound model with a `SystemMessage` containing `GRAPH_SYSTEM_PROMPT`, followed by the current state's messages:

```python
response = bound_model.invoke([
    SystemMessage(content=GRAPH_SYSTEM_PROMPT),
    *state["messages"],
])
return {"messages": [response]}
```

The prompt has no `agent_scratchpad`: tool requests and observations are stored as structured messages instead.

### The routing function chooses the next node

`route()` checks whether the latest `AIMessage` has `tool_calls`. If it does, and the round limit has not been reached, execution goes to `tools`. Otherwise, it goes to `END`.

The responsibilities are different:

- **The model** chooses tool names and arguments.
- **The routing function** checks whether any tool calls exist and chooses the next graph node.
- **The tools node** looks up and runs the requested Python tools.

### The tools node returns observations

`call_tools()` looks up each tool in `tools_by_name`, invokes it, and creates a result message:

```python
ToolMessage(
    content=content,
    tool_call_id=call["id"],
    name=call["name"],
)
```

`tool_call_id` matches the result to the original request. Our implementation executes calls sequentially inside the tools node, even if the model requests several at once.

The edge from `tools` back to `model` lets Groq read the results and request more evidence or write the final answer.

### Build and run the graph

The connections are defined explicitly:

```python
builder.add_edge(START, "model")
builder.add_conditional_edges("model", route, {"tools": "tools", END: END})
builder.add_edge("tools", "model")
```

After `builder.compile()`, `LangGraphResearchService.ask()` runs the graph with the question, bounded previous conversation, `rounds=0`, a deadline, and a recursion limit. It accepts a final `AIMessage` without pending tool calls and returns the same `ResearchAnswer` type as the classic service.

Each question starts a fresh graph invocation. This project has no graph checkpointer; earlier completed exchanges come from Streamlit session history, not graph persistence.

## 7. Trace a comparison question in both implementations

Ask:

> Compare the RAG design in this PDF with current official LangChain guidance. Search both sources.

One possible sequence is:

| Step | Classic LangChain | LangGraph |
|---|---|---|
| 1 | Model emits text requesting `pdf_knowledge_base` | Model returns an `AIMessage` with a PDF tool call |
| 2 | Executor runs the PDF retriever | Routing sends execution to `call_tools()` |
| 3 | PDF observation enters `agent_scratchpad` | PDF observation becomes a `ToolMessage` |
| 4 | Model emits text requesting `web_search` | Model returns another structured tool call |
| 5 | Executor runs Serper search | Tools node runs Serper search |
| 6 | Model produces `Final Answer:` text | Model returns answer text with no tool calls |
| 7 | Executor returns the output | Routing reaches `END` |

Both should give a combined explanation with PDF page citations and source URLs. This is an example sequence; the model may change tool order or request both tools in one LangGraph batch. Tool names confirm that a tool ran, not that every generated statement or citation is correct.

## 8. How Streamlit uses both services

In `ui.py`, the selector chooses a service class:

```python
service_class = LangGraphResearchService if backend == "LangGraph" else ResearchService
service = service_class.create(index, settings)
```

Both classes expose:

```python
answer = service.ask(question, messages)
```

That common interface is why the upload, chat rendering, and download code do not need separate implementations. The UI stores successful exchanges in session state and keeps at most 40 messages. Each service also limits previous conversation context to 12,000 characters before sending it to the model. These are character budgets, not token counts.

Changing the implementation clears the previous service and chat. Click **Process PDF** again to build the new service for the selected upload.

## 9. Limits and failures you will see

| Situation | Current behavior |
|---|---|
| Missing Groq or Serper key | Configuration message before indexing or chat |
| PDF too large, encrypted, or unreadable | Safe validation error |
| Image-only scan | Message asking for OCR; this app does not run OCR |
| Serper unavailable | Tool returns an unavailable observation so the model can explain the limitation |
| Groq request fails | Service displays a safe retry/configuration message |
| Classic executor exceeds limits | Service rejects the stopping message as incomplete |
| Graph repeatedly requests tools | Stops after eight completed tool rounds; each round permits up to eight calls |
| Graph exceeds step budget | Recursion error becomes a safe research-limit message |

The graph uses a 20-step recursion ceiling. Both implementations use a cooperative 120-second execution budget and individual provider timeouts. A cooperative limit does not interrupt a call already running.

Prompt instructions guide routing and citations but do not guarantee them. They also do not sandbox malicious PDF contents. The public-deployment limitations in the [README](../README.md#deployment-scope) still apply.

## 10. Try both yourself

Start the app from the project folder:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Configure fresh API keys locally in `.env` or in Streamlit deployment Secrets. Do not commit your keys.

1. Select **LangChain ReAct**, upload the sample PDF, and click **Process PDF**.
2. Ask each question below and check the answer and **Sources searched**.
3. Select **LangGraph**, click **Process PDF** again, and repeat the questions.
4. Compare the evidence and tool activity rather than expecting identical wording.

| Question | What to check |
|---|---|
| What was revenue growth in 2025? Cite the page. | 34 percent, page 1, PDF search |
| What is the 2026 revenue target? Was it achieved? | USD 16 million is a target, not an established result; page 3 |
| What was net profit? | Not provided in the report; no invented number |
| Search the web for recent LangChain news and include URLs. | Web search and source URLs; findings vary over time |
| Compare this PDF's RAG design with current official guidance. | PDF and web evidence with citations |
| Explain your previous answer more simply. | Context from the preceding exchange |

The report's company and financial figures are fictional. Do not search the web expecting to verify them.

## 11. Learn from the offline tests

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

| Test file | What it demonstrates |
|---|---|
| [test_documents.py](../tests/test_documents.py) | Generated PDF parsing, citations, invalid input, and retrieval with fake embeddings |
| [test_agent.py](../tests/test_agent.py) | Classic textual tool requests, bounded context, and sanitized failures |
| [test_langgraph_agent.py](../tests/test_langgraph_agent.py) | Structured tool calls, returned observations, graph limits, and safe failures |
| [test_ui.py](../tests/test_ui.py) | Streamlit chat, download controls, and resetting state when changing implementations |

The fake models have scripted responses, so these tests verify that the application correctly executes a chosen workflow. They do not prove that a live Groq model will always choose the right tool. Test that separately through the UI.

## 12. The main difference to remember

In this project's classic implementation, `AgentExecutor` manages the repeated model/tool steps. In its LangGraph implementation, the application defines those steps, state, and routing explicitly.

LangGraph still uses LangChain components such as messages, tools, and the Groq integration. Both implementations share the same retrieval system; switching workflow libraries alone does not guarantee better answers.
