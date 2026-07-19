\# Architecture — Initial Draft (Sakshi, Week 2 Fri)



\## Pipeline flow

Problem + tests

|

v

Planner  ---- breaks problem into numbered implementation steps

|

v

Coder    ---- writes the solution based on the plan

|

v

Sandbox Tester ---- runs solution + tests inside an isolated Docker container

|

+--- pass ---> Done

|

+--- fail ---> Debugger ---- fixes code using the test error

|

v

back to Sandbox Tester (retry, up to MAX\_RETRIES)



\## Teacher fallback chain



All three LLM-driven agents (Planner, Coder, Debugger) call a single shared

module, `agents/teacher\_router.py`, instead of talking to a model directly.

Planner / Coder / Debugger

|

v

Teacher router

|

tries first     falls back after 3 failures

|                        |

v                        v

Ollama (local 7B)      Groq / Gemini API



This means:

\- Every agent gets automatic failover — if the local Ollama server is down

&#x20; or errors out, the pipeline keeps going using the API teacher instead of

&#x20; crashing.

\- Adding a new teacher source later (e.g. Faiza's 14B) only requires changing

&#x20; `teacher\_router.py`, not every agent file.



\## Sandbox isolation



The Sandbox Tester runs untrusted Coder-agent output inside a hardened Docker

container (`agents/sandbox\_executor.py`):

\- No network access

\- 256MB memory cap, no swap

\- 50% CPU quota, 64 process limit (fork-bomb protection)

\- All Linux capabilities dropped

\- Read-only root filesystem

\- Configurable timeout (default 30s), container force-killed on timeout



\## Notes

\- This is a first draft — will refine as more agents (e.g. Faiza's 14B

&#x20; teacher, dataset compressor) get wired into the same graph.

