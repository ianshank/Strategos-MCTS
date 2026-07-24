---
name: hierarchical-recursive-brainstorm
description: Perform hierarchical recursive brainstorming that decomposes a topic into a tree, expands leaves recursively, and synthesizes upward. Use whenever the user asks to brainstorm, ideate, explore options, generate idea trees, expand concepts hierarchically, or structure thinking on a complex problem, project, research question, or design space — even if they never say the skill name.
---

# Hierarchical Recursive Brainstorm

Decompose any open-ended topic into a controlled hierarchy, recursively expand the leaves, evaluate and prune, then roll the best ideas back up into a coherent tree. Produce a structured, depth-limited idea map rather than a flat list.

## When to activate
Trigger on language that signals structured ideation rather than simple listing:
- “brainstorm”, “ideate”, “explore options”, “generate ideas”, “map out”, “break this down”
- requests for hierarchical, recursive, multi-level, or tree-structured thinking
- complex problems that benefit from top-down decomposition followed by bottom-up synthesis

## Procedure

1. **Clarify the root**  
   Restate the user’s topic or goal as a single, sharp root node. If the request is ambiguous, ask one focused clarifying question and stop. Do not invent a root.

2. **Choose depth and branching limits**  
   Default: max depth 3, max 4–6 children per node.  
   Adjust only if the user explicitly requests deeper or wider exploration. State the limits before expanding.

3. **Hierarchical decomposition (top-down)**  
   At every non-leaf node:
   - Generate 3–6 distinct, non-overlapping child categories or sub-problems.
   - Prefer mutually exclusive, collectively exhaustive partitions when the domain allows.
   - Label each child with a short, concrete title (≤ 8 words).

4. **Recursive expansion (leaves)**  
   For every leaf:
   - Produce 3–7 concrete ideas, options, or next actions.
   - Keep each idea atomic and actionable or testable.
   - Tag each idea with a one-line rationale or expected impact.

5. **Evaluate and prune**  
   After expansion, score every leaf idea on two axes — by default relevance to root × feasibility (use user-supplied criteria such as novelty instead when given; see Constraints).  
   Drop the bottom quartile. Promote only the strongest ideas.  
   If a branch is empty after pruning, collapse it.

6. **Bottom-up synthesis**  
   Starting from the leaves, write a 1–2 sentence synthesis for each parent node that captures the strongest surviving children.  
   Propagate the synthesis all the way to the root so the final tree is self-explanatory.

7. **Output format**  
   Emit a clean Markdown outline (or indented tree) that shows:
   - root
   - every surviving branch with its synthesis
   - every surviving leaf idea with its short rationale
   - explicit depth and branching limits used
   - a one-paragraph “next actions” summary derived from the strongest leaves

## Constraints
- Never expand beyond the stated depth limit.
- Never invent facts or external knowledge; stay inside the user’s stated domain and the ideas generated in the session.
- Prefer concrete, testable leaves over vague abstractions.
- If the user supplies evaluation criteria, use those instead of the default relevance × feasibility axes.
- Keep the entire response under ~800 words unless the user asks for exhaustive coverage.

## Failure modes to avoid
- Flat bullet lists with no hierarchy.
- Runaway depth or branching that produces an unreadable wall of text.
- Synthetic “ideas” that are merely restatements of the parent node.
- Declaring the brainstorm complete before synthesis has been written for every surviving parent.
