# Business Prompts

This directory covers prompts for business workflows: strategy, sales, operations, finance, legal, marketing, HR, customer support, procurement, analytics, and executive decision support.

## Content Map

| Subdirectory | Scope |
|---|---|
| `Legal_Tech/` | Legal drafting, contract review, compliance triage, clause comparison, risk summaries, and legal operations prompts. |

## Prompt Quality Standard

Business prompts should include role, task, audience, constraints, available context, output format, success criteria, verification steps, and escalation rules for uncertainty.

## Source Families

- OpenAI, Anthropic, Google, and Microsoft prompting guides.
- Prompt evaluation tools such as Promptfoo, OpenAI Evals, Ragas, DeepEval, and LangSmith.
- Domain playbooks from legal operations, sales operations, finance operations, and compliance teams.

## Reference Links

- OpenAI prompt engineering: https://developers.openai.com/api/docs/guides/prompt-engineering
- Anthropic prompting overview: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview
- Anthropic business prompting article: https://www.anthropic.com/news/prompt-engineering-for-business-performance
- Google prompt engineering guide: https://cloud.google.com/discover/what-is-prompt-engineering
- Promptfoo: https://www.promptfoo.dev/

## Safety Notes

- Separate creative ideation prompts from decision prompts that affect money, employment, law, health, or regulated activity.
- Require source citation or human review for legal, financial, compliance, or HR outputs.
- Track prompt version, model version, retrieval context, and evaluation result when a prompt is reused.

## Routing Rules

- Put consumer-style prompt collections in sibling topic folders only when they are not business-critical.
- Put prompt-injection and safety controls in `../../Privacy_and_Security/`.
- Put production agent prompts in `../../../04_MLOPS_AND_PRODUCTION_AI/AI_Agents/`.
