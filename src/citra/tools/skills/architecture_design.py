
from citra.context import ExecutionContext
from typing import override
from citra.tools.skills.skill import Skill
class ArchitectureDesign(Skill):
    
    """Represent ArchitectureDesign."""
    def __init__(self):
        """Initialize the instance."""
        super().__init__(
            name = "architecture-design",
            description = "Provides a necessary overview on software systems architecture. Mandatory read for any greenfield or significant task",
            root=None
        )
    
    @override
    def get_md(
        self,
        context : ExecutionContext
    ) -> str:
        """Return get md."""
        return _SKILL


_SKILL="""
# Architecture design and refactoring

Use this skill when a task materially affects system structure, component boundaries,
integrations, quality attributes, deployment, or architectural evolution.

Unless the user asks for design only, architecture is preparation for implementation:
inspect, decide enough, refactor/implement, test, and iterate.

## Core operating rules

- Architecture is important structural decisions and trade-offs, not a diagram or technology list.
- Start from the actual task and repository. Existing structure is evidence, not automatically a constraint.
- **Default: refactoring is allowed.** Unless the user, repository policy, compatibility contract,
  or another explicit constraint says otherwise, you may refactor any code, module/component
  boundary, interface, dependency, data-access path, configuration, test, deployment artifact,
  or infrastructure definition needed for a coherent solution.
- Do not preserve a bad boundary because it already exists. Do not ask permission for ordinary
  refactors covered by this default.
- Preserve externally required behavior/contracts unless the task requires changing them.
- Prefer the simplest structure that satisfies the important requirements and quality attributes.
- Every architectural choice has trade-offs. Do not choose technology or style by fashion.

## Task-state discipline

This mode may not load task-recognition separately. Apply these rules while doing architecture work:

- Inspect existing durable state first when available: constraints, decisions, facts, TODOs,
  working state, checkpoint.
- **Fact:** verified repository/runtime/research conclusion later work depends on.
- **Constraint:** user requirement, compatibility boundary, repository rule, or invariant.
- **Decision:** significant architectural/design choice later work must follow.
- **Working State:** unresolved hypothesis/investigation worth preserving; resolve/discard when settled.
- **TODO:** meaningful remaining work, not individual tool calls.
- **Checkpoint:** compact resume point for substantial unfinished work.

Repository exploration default:

`Tree/Glob -> LSP symbols/references/definitions -> focused Read -> execution/tests`

Use text search for config/strings/dynamic references. Prefer local source/runtime evidence for the
current project. Use web/upstream research only when external evidence materially improves correctness.

---

# 1. Recon the task and current architecture

Establish before editing:

- business/user goal;
- functional requirements;
- actors and external systems;
- important data and ownership;
- hard technical/organizational/legal/compatibility/deployment constraints;
- existing infrastructure and integrations;
- repository/team conventions that actually constrain implementation.

Separate:

- **Functional requirement:** what the system must do.
- **Constraint:** effectively zero freedom for this task.
- **Architecture characteristic / quality attribute:** how the system must be, when the property is
  critical enough to influence structure.

Keep a quality attribute when it is simultaneously:

1. not merely domain functionality;
2. critical/important to success;
3. structurally influential.

Keep the set small. Make accepted characteristics measurable when possible. Do not conflate
performance (response under load) with scalability (handling increased load without redesign/loss).

### Existing-system reconnaissance

Architecture work on an existing system normally begins with reverse engineering and may continue
straight into refactoring.

Map only what the task needs:

- deployable units;
- modules/components and responsibilities;
- public interfaces/contracts;
- direct dependencies and important call paths;
- data stores and who reads/writes/owns them;
- external systems/adapters;
- deployment topology when relevant;
- tests defining current behavior.

Verify relationships with references, definitions, config, tests, and runtime evidence. Names and
old diagrams are hints, not proof.

Persist verified findings as Facts, non-negotiable boundaries as Constraints, clear work as TODOs,
and only unresolved architectural questions as Working State.

### Baseline before broad refactoring

Before an invasive change:

1. identify behavior that must remain stable;
2. run relevant existing tests or add a focused characterization test if necessary;
3. separate pre-existing failures from regressions introduced by the change.

Do not require perfect coverage; obtain enough evidence to refactor the relevant boundary safely.

---

# 2. Form and iterate the component model

A component is an encapsulated, relatively independent part with clear interfaces, typically
packaging one or more related modules. Componentization is iterative.

## Initial partition

Choose deliberately between technical partitioning, domain partitioning, or a combination.
Generate initial components from:

- functional responsibilities;
- existing organizational/system boundaries;
- similar systems/patterns already present;
- domain experience;
- current code when evolving an existing system.

For each component identify:

- responsibility;
- owned behavior/data;
- provided interface/contract;
- required dependencies.

If a components tool exists, keep this model there rather than duplicating it in prose.

## Assign requirements

Map every required behavior to an owner and challenge the partition:

- no natural owner -> component/responsibility may be missing;
- ordinary requirement spans several components -> partition may be too fine-grained;
- one component owns many unrelated requirements -> it may be too coarse-grained.

Use actor/actions to verify which operations each component must support and which collaborations
are actually necessary.

## Re-check quality attributes

For every important quality attribute ask whether the partition supports it. Attributes may change:

- number of components;
- responsibility distribution;
- interaction model;
- data ownership;
- deployment boundaries.

Restructure and iterate until coherent enough to implement. Do not wait for a perfect decomposition.
Record only significant structural commitments as Decisions.

---

# 3. Refactor toward better boundaries

For existing systems, refactoring is a primary architecture activity, not cleanup after design.

## Diagnose modularity

Use these questions together:

1. **Cohesion:** do these elements have a strong reason to live/change together?
2. **Dependency quantity:** where are excessive incoming/outgoing dependencies?
3. **Connascence:** what implicit code-level knowledge must change together?
4. **Integration strength:** what knowledge crosses component boundaries?
5. **Volatility:** how often does the shared knowledge change?

Integration strength, lower to higher risk:

1. **Contract** — minimal explicit integration contract.
2. **Model** — shared domain/business structures.
3. **Functional** — knowledge of the other's operations/business semantics.
4. **Intrusive** — access to internals such as private schemas/classes.

Do not judge coupling only by arrow count; one intrusive dependency may be riskier than several
contract dependencies.

## Refactor aggressively where useful

Typical targets:

- direct reads/writes of another component's private database/schema;
- duplicated knowledge of another component's internal model or magic meanings;
- business semantics leaking across boundaries;
- components with many unrelated outgoing dependencies;
- responsibilities with different change rates bundled together;
- fragmented ownership of one responsibility;
- provider-specific APIs leaking into business components;
- circular dependencies;
- interfaces wider than consumers need;
- technical partitions that force small domain changes through many modules.

Prefer explicit contracts, adapters, encapsulation, higher cohesion, and narrower interfaces when
those reduce shared knowledge.

**Change-together rule:** group what changes together; separate parts with different rates of change.
Granularity is wrong when unrelated change forces coordinated testing/release/deployment.

## Refactor in slices

For each coherent slice:

1. create/update a meaningful TODO;
2. change the boundary;
3. update callers/tests/config as needed;
4. run focused tests and diagnostics;
5. verify behavior and dependency direction;
6. continue.

If implementation disproves an architectural assumption, revise the model/Decision and refactor.
Do not force code into a disproven design.

---

# 4. Apply tactics/assets/styles only when justified

Start from a specific requirement, quality response, or structural problem.

## Prefer reusable architectural knowledge

Before inventing a mechanism, consider suitable existing principles, patterns, tactics, ABBs,
reference architectures, or organization-specific assets. Reuse only after checking contextual fit.

## Tactics

State the target quality attribute and desired response first. Use the smallest sufficient tactic set.
Course examples:

- **Modifiability:** encapsulation, higher cohesion, abstract services/interfaces, refactoring.
- **Performance:** concurrency/parallelism, selective data duplication/desnormalization,
  prioritization/discarding lower-priority work when acceptable.
- **Availability/recovery:** redundancy, voting where replicated results fit, rollback, ignoring faulty
  behavior when safe, timeouts/retries/circuit breakers for transient dependency failures.
- **Security:** limit exposure, authentication, authorization.

A tactic for one attribute is not automatically valid for another. Check trade-offs.

## Styles

Perform explicit style selection only when system-level structure is genuinely undecided.
Evaluate against quality attributes, constraints, communication/data model, deployment, team/ops
complexity, cost, and failure modes.

- **Monolithic family:** one principal deployment unit, potentially well modularized internally
  (Layered, Pipeline, Microkernel).
- **Distributed family:** multiple deployment units communicating by network/protocols
  (Service-Based, Event-Driven, Space-Based, Orchestration-Driven SOA, Microservices).

Do not distribute a system merely to obtain modularity.

Interaction choice:

- **Request-based:** structured/data-driven operation, certainty, explicit workflow control.
- **Event-based:** reacting to occurred actions, responsiveness, scale/elasticity, dynamic processing.

Deterministic queries commonly fit requests better than events.

Record a style as a Decision only when it materially constrains later implementation.

---

# 5. Make significant decisions explicitly

For a choice that constrains future work:

1. state context/problem;
2. identify drivers: requirements, quality attributes, constraints, risks;
3. consider a small set of viable alternatives;
4. compare trade-offs;
5. choose;
6. record desired and accepted negative consequences;
7. supersede/update the Decision if later evidence changes it.

Use Facts for evidence and Constraints for non-negotiables; do not collapse them into a Decision.
Avoid ADR-level ceremony for ordinary implementation details.

---

# 6. Create views only when they answer a real concern

A single diagram is not the architecture. Select the narrowest useful view:

- boundary/actors/external systems -> **Context**;
- responsibilities/dependencies/interfaces -> **Component**;
- temporal message order -> **Sequence**;
- dependency-focused interaction -> **Communication/Collaboration**;
- workflow -> **Activity**;
- lifecycle -> **State**;
- runtime placement/networks -> **Deployment / Operational**.

Keep architectural element names coherent across views. Use the diagram tool only when a visual
artifact improves reasoning, implementation, or communication.

---

# 7. Extend into operational architecture when relevant

When infrastructure, jurisdictions, availability, performance, scalability, security zones, or cost
matter, go from logical to physical:

1. identify **locations**: geography/jurisdiction, site/device type, cardinality, restrictions;
2. identify execution nodes and deployable components/artifacts;
3. identify networks/zones/boundaries/protocols;
4. allocate software to nodes;
5. size/refine resources only when justified by a quality attribute/constraint;
6. introduce concrete technology names only when selected or already constrained.

Operational choices must trace to concerns such as availability, performance, scalability, security,
or cost. Significant platform/sizing choices are Decisions, not unexplained labels.

The default refactoring authority also applies to existing deployment definitions, service topology,
networks, configuration, and infrastructure when no explicit constraint forbids the change.

---

# 8. Implement and continuously review

Unless asked for architecture documentation only, continue into code.

For each meaningful implementation slice:

1. maintain the relevant TODO;
2. implement/refactor the smallest coherent unit;
3. respect component ownership and contracts;
4. isolate volatile provider/infrastructure details behind deliberate boundaries;
5. run tests and LSP/static diagnostics;
6. inspect integration/runtime behavior when relevant;
7. update durable state if evidence changes what is known;
8. continue.

Watch continuously for drift:

- new cross-boundary imports/calls;
- shared internal schema/data access;
- duplicated business semantics;
- responsibilities accumulating in the wrong component;
- new deployment coupling;
- local changes forcing widespread unrelated test/setup changes.

When drift appears, **refactor immediately when practical** rather than preserving it and documenting
around it.

For external libraries/protocols/cloud services, inspect the project's actual version/config first.
Research authoritative docs/upstream source only when needed; digest verified conclusions into Facts.

---

# 9. Completion check

Before reporting completion, verify:

- every required behavior has a clear owner;
- every hard constraint is satisfied;
- important quality attributes have a structural/operational response and measurable criterion where
  appropriate;
- component responsibilities remain cohesive;
- dependency strength is no higher than necessary;
- volatile integrations are isolated;
- intrusive cross-component access is absent unless deliberately justified;
- granularity avoids both excessive communication and giant mixed responsibilities;
- code, decisions, data ownership, diagrams, and deployment do not contradict one another;
- changed behavior and major refactors are supported by tests/diagnostics/runtime evidence.

If a failure can be corrected by refactoring within scope, fix it rather than only reporting it.

Finish valid TODOs, resolve/discard stale Working State, remove stale Facts/Constraints, supersede
obsolete Decisions, and clear/refresh the Checkpoint before the final response.

## Short form

When context is tight:

**Recon real system -> identify requirements/constraints/quality attributes -> form component hypothesis ->
refactor weak boundaries -> apply only justified tactics/styles -> record significant decisions -> implement/test
in slices -> refactor drift -> validate architecture against implemented reality.**
"""