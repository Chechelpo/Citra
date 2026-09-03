"""Shared Mermaid errors, enums, and immutable value objects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

class MermaidError(Exception):
    """Base exception for Mermaid utility failures."""

class MermaidValidationError(MermaidError, ValueError):
    """Raised when Mermaid source or structured builder input is invalid."""

class MermaidRendererUnavailable(MermaidError):
    """Raised when the configured Mermaid renderer executable is unavailable."""

class MermaidRenderError(MermaidError):
    """Raised when Mermaid CLI fails to render a diagram."""

class MermaidDiagramType(str, Enum):
    """Represent MermaidDiagramType."""
    FLOWCHART = 'flowchart'
    SEQUENCE = 'sequence'
    CLASS = 'class'
    STATE = 'state'
    ER = 'er'
    JOURNEY = 'journey'
    GANTT = 'gantt'
    PIE = 'pie'
    QUADRANT = 'quadrant'
    REQUIREMENT = 'requirement'
    GIT_GRAPH = 'git-graph'
    C4 = 'c4'
    MINDMAP = 'mindmap'
    TIMELINE = 'timeline'
    ZENUML = 'zenuml'
    SANKEY = 'sankey'
    XY_CHART = 'xy-chart'
    BLOCK = 'block'
    PACKET = 'packet'
    KANBAN = 'kanban'
    ARCHITECTURE = 'architecture'
    RADAR = 'radar'
    TREEMAP = 'treemap'
    VENN = 'venn'
    ISHIKAWA = 'ishikawa'
    WARDLEY = 'wardley'
    CYNEFIN = 'cynefin'
    TREE_VIEW = 'tree-view'
    EVENT_MODELING = 'event-modeling'
    UNKNOWN = 'unknown'

class MermaidTheme(str, Enum):
    """Represent MermaidTheme."""
    DEFAULT = 'default'
    FOREST = 'forest'
    DARK = 'dark'
    NEUTRAL = 'neutral'

class MermaidOutputFormat(str, Enum):
    """Represent MermaidOutputFormat."""
    SVG = 'svg'
    PNG = 'png'
    PDF = 'pdf'

class FlowDirection(str, Enum):
    """Represent FlowDirection."""
    TOP_DOWN = 'TD'
    TOP_BOTTOM = 'TB'
    BOTTOM_TOP = 'BT'
    LEFT_RIGHT = 'LR'
    RIGHT_LEFT = 'RL'

class FlowNodeShape(str, Enum):
    """Represent FlowNodeShape."""
    RECTANGLE = 'rectangle'
    ROUNDED = 'rounded'
    STADIUM = 'stadium'
    SUBROUTINE = 'subroutine'
    CYLINDER = 'cylinder'
    CIRCLE = 'circle'
    DOUBLE_CIRCLE = 'double-circle'
    DIAMOND = 'diamond'
    HEXAGON = 'hexagon'

class SequenceParticipantKind(str, Enum):
    """Represent SequenceParticipantKind."""
    PARTICIPANT = 'participant'
    ACTOR = 'actor'

@dataclass(frozen=True, slots=True)
class MermaidInfo:
    """Represent MermaidInfo."""
    diagram_type: MermaidDiagramType
    declaration: str
    lines: int
    characters: int
    digest: str

@dataclass(frozen=True, slots=True)
class MermaidRenderResult:
    """Represent MermaidRenderResult."""
    output_path: Path
    output_format: MermaidOutputFormat
    bytes_written: int
    source_digest: str
    command: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class MermaidBundleResult:
    """Represent MermaidBundleResult."""
    source_path: Path
    rendered_path: Path | None
    markdown: str
    diagram_type: MermaidDiagramType

@dataclass(frozen=True, slots=True)
class FlowNode:
    """Represent FlowNode."""
    id: str
    label: str | None = None
    shape: FlowNodeShape = FlowNodeShape.RECTANGLE

@dataclass(frozen=True, slots=True)
class FlowEdge:
    """Represent FlowEdge."""
    source: str
    target: str
    label: str | None = None
    arrow: str = '-->'

@dataclass(frozen=True, slots=True)
class SequenceParticipant:
    """Represent SequenceParticipant."""
    id: str
    label: str | None = None
    kind: SequenceParticipantKind = SequenceParticipantKind.PARTICIPANT

@dataclass(frozen=True, slots=True)
class SequenceMessage:
    """Represent SequenceMessage."""
    sender: str
    receiver: str
    text: str
    arrow: str = '->>'

@dataclass(frozen=True, slots=True)
class ClassSpec:
    """Represent ClassSpec."""
    name: str
    label: str | None = None
    attributes: tuple[str, ...] = ()
    methods: tuple[str, ...] = ()
    annotation: str | None = None

@dataclass(frozen=True, slots=True)
class ClassRelationship:
    """Represent ClassRelationship."""
    left: str
    right: str
    relation: str = '-->'
    label: str | None = None

@dataclass(frozen=True, slots=True)
class StateSpec:
    """Represent StateSpec."""
    id: str
    label: str | None = None

@dataclass(frozen=True, slots=True)
class StateTransition:
    """Represent StateTransition."""
    source: str
    target: str
    label: str | None = None

@dataclass(frozen=True, slots=True)
class ERField:
    """Represent ERField."""
    type: str
    name: str
    key: str | None = None
    comment: str | None = None

@dataclass(frozen=True, slots=True)
class ERRelationship:
    """Represent ERRelationship."""
    left: str
    right: str
    cardinality: str
    label: str

@dataclass(frozen=True, slots=True)
class GanttTask:
    """Represent GanttTask."""
    label: str
    definition: str
    id: str | None = None
    modifiers: tuple[str, ...] = ()

@dataclass(frozen=True, slots=True)
class JourneyTask:
    """Represent JourneyTask."""
    label: str
    score: int
    actors: tuple[str, ...] = ()

@dataclass(frozen=True, slots=True)
class MindmapNode:
    """Represent MindmapNode."""
    text: str
    children: tuple['MindmapNode', ...] = ()

@dataclass(frozen=True, slots=True)
class TimelinePeriod:
    """Represent TimelinePeriod."""
    period: str
    events: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class QuadrantPoint:
    """Represent QuadrantPoint."""
    label: str
    x: float
    y: float

@dataclass(frozen=True, slots=True)
class XYSeries:
    """Represent XYSeries."""
    values: tuple[float, ...]
    kind: str = 'line'
