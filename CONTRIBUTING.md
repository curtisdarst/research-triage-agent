# Contributing

This is a **reference implementation**, published to demonstrate a pattern
(governed agentic RAG on Google Cloud, with a reproducible grounding eval) —
it is not a supported service, and there is no SLA on issues or PRs.

That said, contributions are welcome, particularly:

- Running the eval harness (see [`eval/`](eval/)) against your own corpus and
  sharing results
- Bug reports with a minimal repro
- Improvements to the grounding validator's prompt/logic, since its own
  false-refusal rate is the project's most important open metric

## Before you open a PR

- This repo pins exact dependency versions (see `requirements.txt`) because
  the Agent Development Kit and the Gemini API have both shipped breaking
  changes across versions during this project's lifetime. If your change
  requires a version bump, call that out explicitly in the PR description.
- No credentials, API keys, or service account files in any commit.
- If you change the synthesis or validation prompts, run the eval harness
  and include the before/after unsupported-claim rate and false-refusal
  rate in the PR description — that is the actual regression surface for
  this project.

## Code of conduct

Be respectful. Assume good faith. Disagree about code, not people.
