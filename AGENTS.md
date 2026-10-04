# AGENTS.md -- python-agent

Guide for AI coding assistants working in this project. Humans: see README.md /
README.zh.md.

## What this is

A complete, standard-library-only Python agent for the GOSIM survey26 hackathon's
telescope-survey challenge. It speaks `participant-agent-protocol-v4` (one JSON
object per line on stdin/stdout), runs a deterministic anchor-search planner, and
uses two bounded LLM stages for sourced message understanding and priority
adaptation. Geometry, completion bookkeeping and final actions are deterministic. Read `agent.py` first -- it is the thin
stdin/stdout loop that dispatches to everything else.

## Module map

```
agent.py                 entry point: stdin/stdout loop, one branch per message type
agent_core/
  protocol.py             transport: read/write one JSON object per line
  state.py                 SurveyState: catalogue + everything tracked across decisions
  geometry.py              public sky maths (sidereal time, alt/az, fibre grid)
  scoring.py               factor/score estimates from PUBLIC scoring config only
  planner.py               bounded pointing/exposure search, completion/anomaly guards
  exposure.py              joint fibre choices and integer exposure breakpoints
  llm_client.py            OpenAI-compatible chat client, defaults to Kimi Coding Plan
  advice.py                source-scoped message interpretation and priority adaptation
  calibration.py           unused experimental helper retained for source identity
  requests.py              valid-window request bundles with required-target protection
  memory.py                optional, best-effort JSONL model/decision audit (off by default)
  validation.py            protocol-legal action checking + a deterministic fallback
observer.project.json      platform project manifest (image, run command, env)
requirements.txt           none needed -- standard library only
pack_agent.py              zip this folder into a project ZIP for submission
experiments/stage3/        rejected urgency/diagnostic prototypes; excluded from ZIP
```

## Changing the strategy

Target ranking, fibre filling and exposure sizing live in `agent_core/planner.py`.
The model calls are issued through `advice.py`, with a per-card limit of 150 seconds,
40 HTTP attempts, 8 seconds per attempt and two attempts per question. Keep the last
60 seconds of wall clock for deterministic actions. Model output may not directly
produce an action or override confirmed progress. Never read private local weather
or branch strategy on card/target IDs. Use official rules, current card configuration,
initialize messages and the original scorer as authority; the upstream example is
a reference. See `PROJECT.md`, `KNOWN_ISSUES.md` and `SCHEDULING.md` for
accepted experiments and evidence. Preserve sequential stage validation: use real
Kimi on all four cards, freeze source, and compare mean/minimum/missing-required
counts before promoting and packaging a candidate. Check both per-card medians
and medians of whole four-card evaluations; verify both model stages actually run.
Candidate exposure evaluation must not mutate feedback predictions or progress.
Commit predictions only for the winning action. Keep the original pointing policy
during its existing quality anomaly phase until a separately evaluated change to
the quality estimator and diagnosis is accepted.

The official alpha-delta export currently lacks full replay files. Local scoring
tools default to official cards and stop on missing products. Choose examples
explicitly with --card-set examples; their scores are engineering regressions,
not official practice, formal or hidden performance.

## Configuring the LLM (Kimi key / base URL / model)

Copy `.env.example` to `.env` and set:

- `OPENAI_API_KEY` (or `KIMI_API_KEY`) -- needed for real model calls; missing keys use deterministic fallback.
- `OPENAI_BASE_URL` -- defaults to the Kimi Coding Plan endpoint
  (`https://api.kimi.com/coding/v1`; use `https://api.kimi.ai/coding/v1` outside
  mainland China) when unset.
- `OPENAI_MODEL` -- defaults to `k3` when unset.

Any other OpenAI-compatible `/chat/completions` endpoint works too -- just point
`OPENAI_BASE_URL` / `OPENAI_MODEL` at it. On the platform, `OPENAI_BASE_URL` /
`OPENAI_API_KEY` are injected automatically for every run (the platform's own model
proxy and a temporary credential); `.env` is never uploaded and is excluded by
`pack_agent.py`.

## Packing and uploading as a complete project

```bash
python3 pack_agent.py --out ../python-agent.zip
```

This zips the project with `observer.project.json` at the root, skipping `.env`,
`__pycache__`, and other local-only files. Upload the resulting ZIP as a complete
project on the Participate page, or push this folder to a GitHub repository and
submit the repo URL instead.

## Protocol & scoring

Full protocol reference and scoring formulas are not duplicated here -- see `docs/`
(bundled in the downloadable ZIP of this example) for the complete participant
guide in Chinese and English.
