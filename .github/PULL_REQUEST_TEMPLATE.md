## Summary

<!-- One or two sentences: what this PR does, in plain
language. If a reader remembers one line, it should be
this one. -->

## Why

<!-- What problem or need motivates it? Link the issue,
measurement, or record entry behind it. A change without
a why is hard to review and harder to keep. -->

## What changed

<!-- The concrete changes, grouped by area (pipeline /
packs / harness / docs / tests). Name files or surfaces,
not intentions. -->

-

## How verified

<!-- The heart of the PR. Name the commands you ran and
the outputs you observed: full test suite, ruff, coverage.
If behavior changed — or must provably NOT have changed —
attach the before/after equivalence evidence (replay,
golden output, recorded comparison). "Tests pass" alone
is not a verification story for a behavior change. -->

-

## Checklist

- [ ] Full test suite green (`python -m pytest tests/ -q`)
- [ ] `ruff check .` clean
- [ ] Coverage floor held or raised (no silent drop)
- [ ] Behavior changes carry before/after equivalence evidence
- [ ] Pack payload changes bump the pack manifest version
      and add/update a conformance entry
- [ ] No new score claims on front surfaces (D4); no
      per-cell cost detail (D9); no credentials anywhere

## Notes

<!-- Anything reviewers should know: alternatives
considered, follow-ups opened, known limitations. -->
