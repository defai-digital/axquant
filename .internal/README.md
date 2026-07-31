# AXQuant Internal Documentation

This directory is the internal source of truth for AXQuant product and engineering decisions.

## Canonical documents

| Area | Document | Authority |
| --- | --- | --- |
| Product | [Product requirements](product/requirements.md) | Scope, claims, requirements, milestones, and acceptance gates |
| Product | [Release best practices](product/release-best-practices.md) | Gate order, dual-profile completeness, candidate-cycle discipline |
| Architecture | [Decision register](architecture/decision-register.md) | Accepted decisions and rejected alternatives |
| Engineering | [Technical specification](engineering/technical-specification.md) | Normative interfaces, schemas, algorithms, runtime contracts, and test strategy |
| Policy | [Independent implementation](policies/clean-room.md) | Clean-room inputs, evidence, and attribution requirements |

## Directory layout

```text
.internal/
├── README.md
├── product/
│   ├── requirements.md
│   └── archive/                  # Superseded product drafts
├── architecture/
│   ├── decision-register.md
│   └── decisions/                # Supporting numbered ADR narratives
├── engineering/
│   └── technical-specification.md
├── policies/
│   └── clean-room.md
└── tmp/                           # Ignored generated reports
```

## Precedence

The independent implementation policy is a non-negotiable boundary. Within that boundary, use
this order when documents disagree:

1. accepted decisions in `architecture/decision-register.md`;
2. normative engineering contracts in `engineering/technical-specification.md`;
3. product requirements and roadmap in `product/requirements.md`;
4. supporting decision narratives under `architecture/decisions/`;
5. examples and explanatory README text.

Documents under `product/archive/` are retained only for history and are never authoritative.
Code and tests describe the current implementation, but they do not silently supersede an
accepted decision. A deliberate deviation requires updating the decision register and Technical
Specification in the same change.

## Status vocabulary

- **Implemented**: present in the repository and covered by tests.
- **Partially implemented**: a safe boundary exists, but one or more backends or measurements are
  not complete.
- **Planned**: required by the PRD but not yet implemented.
- **Deferred**: intentionally outside the active release scope.
- **Release evidence**: measured, reproducible evidence that may satisfy a production gate.
- **Development evidence**: architecture priors, manual assignments, or smoke tests that cannot
  satisfy a production gate.

## Update policy

- Keep schema names and CLI commands synchronized with `src/axquant/schema.py` and
  `src/axquant/cli.py`.
- Record source revisions, dataset digests, seeds, software versions, and hardware for every
  measured claim.
- Never convert architecture-prior or manual-plan output into a release claim.
- Preserve the clean-room boundary described in `policies/clean-room.md`, the decision register,
  and the Technical Specification.
- Do not place credentials, Hub tokens, private dataset samples, or unreleased model weights in
  this directory.
