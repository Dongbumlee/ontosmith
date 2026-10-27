# Security policy

## Supported versions

OntoSmith is pre-1.0. Security fixes land on the latest
release line only; once 0.1.1 is published, that line is
the supported one until a successor ships.

## Reporting a vulnerability

Please do **not** open a public issue for a security
problem. Report it privately through GitHub's
**Security → Report a vulnerability** flow on this
repository (private vulnerability reporting), or to the
maintainer at the address on the GitHub profile of the
repository owner.

Include: the affected version, a minimal reproduction,
and the impact as you understand it. You should receive
an acknowledgement within a few days; a fix or a
mitigation plan follows the severity.

## Scope notes

- **Credentials by design:** OntoSmith never stores API
  keys. Provider keys are supplied at run time through
  the connection layer or environment; a report that a
  key appears in code, fixtures, logs, or docs is treated
  as a security issue.
- **Local-first data:** builds, batches, and the graph
  registry live on the operator's machine; the product
  ships no telemetry. Corpus text is fetched from its
  upstream source under its own license terms.
- **Human gates:** the pipeline's approval points (pack
  approval, review band) are integrity controls, not
  security boundaries — reports are welcome either way,
  but a "the operator can approve their own schema"
  report describes the design, not a vulnerability.
