---
description: Initiates a deep technical research workflow using a multi-agent orchestration pattern to explore complex ML architectures or domains prior to creating an SDD.
argument-hint: <topic-or-question>
arguments: [topic]
allowed-tools: Bash, Read
---

Initiate the deep research workflow by invoking the specialized `research-planner` subagent.

1. Spawn the `research-planner` subagent using the `invoke_subagent` tool.
2. Send the `research-planner` the following prompt, substituting the requested topic:
   "We are initiating a deep research workflow on the following topic: '$topic'. Please decompose this topic into a search strategy and delegate the execution to the `research-fetcher` subagent. Ensure that the research adheres to the deep-research skill."
3. Wait for the multi-agent workflow (Planner -> Fetcher -> Critic -> Synthesizer) to complete.
4. The final output will be a comprehensive report written to `docs/reports/` and a suggested `/spec-new` command. Present the findings to the user and ask if they are ready to proceed with generating the technical spec.
