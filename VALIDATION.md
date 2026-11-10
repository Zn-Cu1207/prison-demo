# Public-release validation

Checked on 10 November 2026 using Python 3.12.14 and the pinned backend dependencies.

| Check | Result |
| --- | --- |
| Python syntax across the 54 included Python files | Passed |
| JavaScript syntax for `main.js` and `poses.js` | Passed |
| Existing `value-model/test_consistency.py` checks | Passed |
| Model client connected to the heuristic mock through an in-process HTTP transport | Passed |
| Two contexts, four candidate scores, and opposite hunger/fatigue preferences | Passed |
| Legacy mock batch mismatch and individual-request fallback | Passed |
| One simulation-world tick using mock scoring | Passed |
| Backend health route | HTTP 200 |
| Initial WebSocket message | `world_state` received |
| Text scan for credential-like tokens and removed internal infrastructure references | No matches in the public files |

The route/WebSocket checks did not run the application's continuous lifespan loop. These checks do not reproduce GPU training, evaluate the fine-tuned dialogue model, establish production reliability, or validate gameplay over long sessions. Model checkpoints and original data are not included.
