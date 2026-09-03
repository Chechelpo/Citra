from __future__ import annotations
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from html import escape
from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
from typing import Callable, Final
from ._mermaid_models import MermaidError, MermaidValidationError, MermaidRendererUnavailable, MermaidRenderError, MermaidDiagramType, MermaidTheme, MermaidOutputFormat, FlowDirection, FlowNodeShape, SequenceParticipantKind, MermaidInfo, MermaidRenderResult, MermaidBundleResult, FlowNode, FlowEdge, SequenceParticipant, SequenceMessage, ClassSpec, ClassRelationship, StateSpec, StateTransition, ERField, ERRelationship, GanttTask, JourneyTask, MindmapNode, TimelinePeriod, QuadrantPoint, XYSeries
CommandRunner = Callable[[tuple[str, ...], float], tuple[int, str]]
_NAME_PATTERN: Final[re.Pattern[str]] = re.compile('^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$')
_IDENTIFIER_PATTERN: Final[re.Pattern[str]] = re.compile('^[A-Za-z_][A-Za-z0-9_-]{0,127}$')
_FRONTMATTER_DELIMITER: Final[str] = '---'

class Mermaid:
    """
    Self-contained Mermaid source utility.

    The class intentionally separates two concerns:

    * structured builders generate conservative Mermaid for common diagram
      families;
    * raw Mermaid source remains available for advanced Mermaid features.

    Mermaid CLI (``mmdc``) is optional. Source creation, inspection, loading,
    saving, and Markdown references require only the Python standard library.
    Rendering and authoritative syntax validation use ``mmdc`` when requested.
    """
    _FLOW_ARROWS: Final[frozenset[str]] = frozenset({'-->', '---', '-.->', '==>', '--o', '--x', '<-->'})
    _SEQUENCE_ARROWS: Final[frozenset[str]] = frozenset({'->', '-->', '->>', '-->>', '-x', '--x', '-)', '--)'})
    _CLASS_RELATIONS: Final[frozenset[str]] = frozenset({'<|--', '*--', 'o--', '-->', '--', '..>', '..|>', '..', '<-->'})
    _GANTT_MODIFIERS: Final[frozenset[str]] = frozenset({'active', 'done', 'crit', 'milestone'})
    _GIT_OPERATIONS: Final[frozenset[str]] = frozenset({'commit', 'branch', 'checkout', 'switch', 'merge', 'cherry-pick'})
    _DECLARATIONS: Final[tuple[tuple[re.Pattern[str], MermaidDiagramType], ...]] = ((re.compile('^(?:flowchart|graph)\\b', re.I), MermaidDiagramType.FLOWCHART), (re.compile('^sequenceDiagram\\b', re.I), MermaidDiagramType.SEQUENCE), (re.compile('^classDiagram\\b', re.I), MermaidDiagramType.CLASS), (re.compile('^stateDiagram(?:-v2)?\\b', re.I), MermaidDiagramType.STATE), (re.compile('^erDiagram\\b', re.I), MermaidDiagramType.ER), (re.compile('^journey\\b', re.I), MermaidDiagramType.JOURNEY), (re.compile('^gantt\\b', re.I), MermaidDiagramType.GANTT), (re.compile('^pie\\b', re.I), MermaidDiagramType.PIE), (re.compile('^quadrantChart\\b', re.I), MermaidDiagramType.QUADRANT), (re.compile('^requirementDiagram\\b', re.I), MermaidDiagramType.REQUIREMENT), (re.compile('^gitGraph\\b', re.I), MermaidDiagramType.GIT_GRAPH), (re.compile('^C4(?:Context|Container|Component|Dynamic|Deployment)\\b', re.I), MermaidDiagramType.C4), (re.compile('^mindmap\\b', re.I), MermaidDiagramType.MINDMAP), (re.compile('^timeline\\b', re.I), MermaidDiagramType.TIMELINE), (re.compile('^zenuml\\b', re.I), MermaidDiagramType.ZENUML), (re.compile('^sankey(?:-beta)?\\b', re.I), MermaidDiagramType.SANKEY), (re.compile('^xychart(?:-beta)?\\b', re.I), MermaidDiagramType.XY_CHART), (re.compile('^block(?:-beta)?\\b', re.I), MermaidDiagramType.BLOCK), (re.compile('^packet(?:-beta)?\\b', re.I), MermaidDiagramType.PACKET), (re.compile('^kanban\\b', re.I), MermaidDiagramType.KANBAN), (re.compile('^architecture(?:-beta)?\\b', re.I), MermaidDiagramType.ARCHITECTURE), (re.compile('^radar(?:-beta)?\\b', re.I), MermaidDiagramType.RADAR), (re.compile('^treemap(?:-beta)?\\b', re.I), MermaidDiagramType.TREEMAP), (re.compile('^venn(?:-beta)?\\b', re.I), MermaidDiagramType.VENN), (re.compile('^ishikawa(?:-beta)?\\b', re.I), MermaidDiagramType.ISHIKAWA), (re.compile('^wardley\\b', re.I), MermaidDiagramType.WARDLEY), (re.compile('^cynefin\\b', re.I), MermaidDiagramType.CYNEFIN), (re.compile('^treeView\\b', re.I), MermaidDiagramType.TREE_VIEW), (re.compile('^eventModeling\\b', re.I), MermaidDiagramType.EVENT_MODELING))

    def __init__(self, source: str, *, allow_unknown_type: bool=False, command_runner: CommandRunner | None=None, temporary_root: Path | None=None) -> None:
        """Validate source and configure optional isolated CLI execution."""
        normalized = self.normalize_source(source)
        self._source = normalized
        self._allow_unknown_type = allow_unknown_type
        self._command_runner = command_runner
        self._temporary_root = temporary_root
        self.validate_basic(allow_unknown_type=allow_unknown_type)

    @property
    def source(self) -> str:
        """Handle source."""
        return self._source

    @property
    def declaration(self) -> str:
        """Handle declaration."""
        return self._find_declaration(self._source)

    @property
    def diagram_type(self) -> MermaidDiagramType:
        """Handle diagram type."""
        declaration = self.declaration
        for pattern, diagram_type in self._DECLARATIONS:
            if pattern.match(declaration):
                return diagram_type
        return MermaidDiagramType.UNKNOWN

    @property
    def digest(self) -> str:
        """Handle digest."""
        return sha256(self._source.encode('utf-8')).hexdigest()

    def inspect(self) -> MermaidInfo:
        """Handle inspect."""
        return MermaidInfo(diagram_type=self.diagram_type, declaration=self.declaration, lines=len(self._source.splitlines()), characters=len(self._source), digest=self.digest)

    def validate_basic(self, *, allow_unknown_type: bool | None=None) -> None:
        """
        Perform safe local validation.

        This does not attempt to reimplement Mermaid's parser. When exact
        syntax validation is required, use ``validate_with_cli``.
        """
        if '\x00' in self._source:
            raise MermaidValidationError('Mermaid source cannot contain NUL characters.')
        declaration = self._find_declaration(self._source)
        if not declaration:
            raise MermaidValidationError('Mermaid source is missing a diagram declaration.')
        effective_allow_unknown = self._allow_unknown_type if allow_unknown_type is None else allow_unknown_type
        if self.diagram_type is MermaidDiagramType.UNKNOWN and (not effective_allow_unknown):
            raise MermaidValidationError(f'Unrecognized Mermaid diagram declaration: {declaration!r}. Pass allow_unknown_type=True for new or custom Mermaid syntax.')

    def validate_with_cli(self, *, executable: str='mmdc', timeout: float=30.0) -> None:
        """
        Ask Mermaid CLI to parse and render the diagram.

        The generated SVG is discarded. A successful return means the
        installed renderer accepted the current source.
        """
        with tempfile.TemporaryDirectory(prefix='mermaid-validate-', dir=self._temporary_root) as directory:
            root = Path(directory)
            source_path = root / 'diagram.mmd'
            output_path = root / 'diagram.svg'
            source_path.write_text(self._source, encoding='utf-8')
            self._render_file(source_path=source_path, output_path=output_path, executable=executable, timeout=timeout, theme=None, background=None, width=None, height=None)

    def save_source(self, path: str | Path, *, overwrite: bool=True) -> Path:
        """Handle save source."""
        destination = Path(path)
        if destination.exists() and (not overwrite):
            raise FileExistsError(f'Mermaid source already exists: {destination}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._write_text_atomic(destination, self._source)
        return destination

    def render(self, output_path: str | Path, *, executable: str='mmdc', timeout: float=30.0, theme: MermaidTheme | str | None=None, background: str | None='transparent', width: int | None=None, height: int | None=None) -> MermaidRenderResult:
        """
        Render to SVG, PNG, or PDF with Mermaid CLI.

        Rendering is staged beside the destination and atomically replaces the
        destination only after Mermaid CLI succeeds and the output passes a
        lightweight format check.
        """
        destination = Path(output_path)
        output_format = self._output_format_from_path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='mermaid-render-', dir=destination.parent) as directory:
            staging_root = Path(directory)
            source_path = staging_root / 'diagram.mmd'
            staged_output = staging_root / f'diagram.{output_format.value}'
            source_path.write_text(self._source, encoding='utf-8')
            command = self._render_file(source_path=source_path, output_path=staged_output, executable=executable, timeout=timeout, theme=theme, background=background, width=width, height=height)
            self._validate_rendered_output(staged_output, output_format)
            bytes_written = staged_output.stat().st_size
            os.replace(staged_output, destination)
        return MermaidRenderResult(output_path=destination, output_format=output_format, bytes_written=bytes_written, source_digest=self.digest, command=command)

    def save_bundle(self, directory: str | Path, *, name: str, alt: str, render: bool=True, executable: str='mmdc', timeout: float=30.0, theme: MermaidTheme | str | None=None, background: str | None='transparent') -> MermaidBundleResult:
        """
        Persist editable Mermaid source and, optionally, derived SVG.

        ``<name>.mmd`` is authoritative. ``<name>.svg`` is derived.

        The source file is not replaced until SVG rendering succeeds, so an
        invalid update cannot destroy the previously valid source.
        """
        safe_name = self.validate_asset_name(name)
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        source_path = root / f'{safe_name}.mmd'
        rendered_path = root / f'{safe_name}.svg'
        with tempfile.TemporaryDirectory(prefix=f'.{safe_name}.', dir=root) as temporary_directory:
            temporary_root = Path(temporary_directory)
            staged_source = temporary_root / f'{safe_name}.mmd'
            staged_source.write_text(self._source, encoding='utf-8')
            staged_rendered: Path | None = None
            if render:
                staged_rendered = temporary_root / f'{safe_name}.svg'
                self._render_file(source_path=staged_source, output_path=staged_rendered, executable=executable, timeout=timeout, theme=theme, background=background, width=None, height=None)
                self._validate_rendered_output(staged_rendered, MermaidOutputFormat.SVG)
            self._replace_bundle_files(staged_source=staged_source, source_path=source_path, staged_rendered=staged_rendered, rendered_path=rendered_path, remove_rendered=not render)
        markdown = self.markdown_image(rendered_path.name, alt=alt) if render else self.to_markdown_fence().rstrip()
        return MermaidBundleResult(source_path=source_path, rendered_path=rendered_path if render else None, markdown=markdown, diagram_type=self.diagram_type)

    def to_markdown_fence(self) -> str:
        """Convert the value to markdown fence."""
        return f'```mermaid\n{self._source}```\n'

    @staticmethod
    def markdown_image(path: str | Path, *, alt: str, title: str | None=None) -> str:
        """Handle markdown image."""
        rendered_path = str(path).replace('\\', '/')
        safe_alt = Mermaid._escape_markdown_alt(alt)
        if title is None:
            return f'![{safe_alt}]({rendered_path})'
        safe_title = title.replace('\\', '\\\\').replace('"', '\\"')
        return f'![{safe_alt}]({rendered_path} "{safe_title}")'

    @classmethod
    def load(cls, path: str | Path, *, allow_unknown_type: bool=False, command_runner: CommandRunner | None=None, temporary_root: Path | None=None) -> 'Mermaid':
        """Handle load."""
        source = Path(path).read_text(encoding='utf-8')
        return cls(source, allow_unknown_type=allow_unknown_type, command_runner=command_runner, temporary_root=temporary_root)

    @classmethod
    def flowchart(cls, *, nodes: Iterable[FlowNode], edges: Iterable[FlowEdge], direction: FlowDirection | str=FlowDirection.TOP_DOWN, title: str | None=None) -> 'Mermaid':
        """Handle flowchart."""
        direction_value = cls._enum_value(direction)
        try:
            FlowDirection(direction_value)
        except ValueError as error:
            raise MermaidValidationError(f'Unsupported flowchart direction: {direction_value!r}.') from error
        node_list = tuple(nodes)
        edge_list = tuple(edges)
        known_ids: set[str] = set()
        lines = [f'flowchart {direction_value}']
        for node in node_list:
            node_id = cls._validate_identifier(node.id, field='Flow node id')
            if node_id in known_ids:
                raise MermaidValidationError(f'Duplicate flow node id: {node_id!r}.')
            known_ids.add(node_id)
            label = node.id if node.label is None else node.label
            lines.append('    ' + cls._flow_node_syntax(node_id, label, node.shape))
        for edge in edge_list:
            source = cls._validate_identifier(edge.source, field='Flow edge source')
            target = cls._validate_identifier(edge.target, field='Flow edge target')
            if source not in known_ids:
                raise MermaidValidationError(f'Flow edge references unknown source node: {source!r}.')
            if target not in known_ids:
                raise MermaidValidationError(f'Flow edge references unknown target node: {target!r}.')
            if edge.arrow not in cls._FLOW_ARROWS:
                raise MermaidValidationError(f'Unsupported flow edge arrow: {edge.arrow!r}.')
            label = ''
            if edge.label:
                label = '|' + cls._escape_inline_text(edge.label) + '|'
            lines.append(f'    {source} {edge.arrow}{label} {target}')
        return cls(cls._with_frontmatter_title(lines, title))

    @classmethod
    def sequence(cls, *, participants: Iterable[SequenceParticipant], messages: Iterable[SequenceMessage], title: str | None=None, autonumber: bool=False) -> 'Mermaid':
        """Handle sequence."""
        participant_list = tuple(participants)
        message_list = tuple(messages)
        known_ids: set[str] = set()
        lines = ['sequenceDiagram']
        if autonumber:
            lines.append('    autonumber')
        for participant in participant_list:
            participant_id = cls._validate_identifier(participant.id, field='Sequence participant id')
            if participant_id in known_ids:
                raise MermaidValidationError(f'Duplicate sequence participant id: {participant_id!r}.')
            known_ids.add(participant_id)
            kind = cls._enum_value(participant.kind)
            label = participant.label
            if label is None:
                lines.append(f'    {kind} {participant_id}')
            else:
                lines.append(f'    {kind} {participant_id} as {cls._escape_inline_text(label)}')
        for message in message_list:
            sender = cls._validate_identifier(message.sender, field='Sequence sender')
            receiver = cls._validate_identifier(message.receiver, field='Sequence receiver')
            if sender not in known_ids:
                raise MermaidValidationError(f'Sequence message references unknown sender: {sender!r}.')
            if receiver not in known_ids:
                raise MermaidValidationError(f'Sequence message references unknown receiver: {receiver!r}.')
            if message.arrow not in cls._SEQUENCE_ARROWS:
                raise MermaidValidationError(f'Unsupported sequence arrow: {message.arrow!r}.')
            text = cls._escape_sequence_text(message.text)
            lines.append(f'    {sender}{message.arrow}{receiver}: {text}')
        return cls(cls._with_frontmatter_title(lines, title))

    @classmethod
    def class_diagram(cls, *, classes: Iterable[ClassSpec], relationships: Iterable[ClassRelationship]=(), title: str | None=None, direction: str | None=None) -> 'Mermaid':
        """Handle class diagram."""
        class_list = tuple(classes)
        relationship_list = tuple(relationships)
        known_names: set[str] = set()
        lines = ['classDiagram']
        if direction is not None:
            direction_value = direction.strip().upper()
            if direction_value not in {'TB', 'BT', 'LR', 'RL'}:
                raise MermaidValidationError("Class diagram direction must be one of 'TB', 'BT', 'LR', or 'RL'.")
            lines.append(f'    direction {direction_value}')
        for class_spec in class_list:
            name = cls._validate_identifier(class_spec.name, field='Class name')
            if name in known_names:
                raise MermaidValidationError(f'Duplicate class name: {name!r}.')
            known_names.add(name)
            if class_spec.label is None:
                lines.append(f'    class {name} {{')
            else:
                label = cls._escape_quoted_text(class_spec.label)
                lines.append(f'    class {name}["{label}"] {{')
            if class_spec.annotation:
                annotation = class_spec.annotation.strip().strip('<>')
                if not annotation:
                    raise MermaidValidationError(f'Class annotation for {name!r} cannot be empty.')
                lines.append(f'        <<{annotation}>>')
            for attribute in class_spec.attributes:
                cls._validate_single_line(attribute, field=f'Attribute in class {name!r}')
                lines.append(f'        {attribute.strip()}')
            for method in class_spec.methods:
                cls._validate_single_line(method, field=f'Method in class {name!r}')
                lines.append(f'        {method.strip()}')
            lines.append('    }')
        for relationship in relationship_list:
            left = cls._validate_identifier(relationship.left, field='Class relationship left side')
            right = cls._validate_identifier(relationship.right, field='Class relationship right side')
            if left not in known_names:
                raise MermaidValidationError(f'Class relationship references unknown class: {left!r}.')
            if right not in known_names:
                raise MermaidValidationError(f'Class relationship references unknown class: {right!r}.')
            if relationship.relation not in cls._CLASS_RELATIONS:
                raise MermaidValidationError(f'Unsupported class relationship: {relationship.relation!r}.')
            line = f'    {left} {relationship.relation} {right}'
            if relationship.label:
                line += ' : ' + cls._escape_inline_text(relationship.label)
            lines.append(line)
        return cls(cls._with_frontmatter_title(lines, title))

    @classmethod
    def state_diagram(cls, *, states: Iterable[StateSpec], transitions: Iterable[StateTransition], title: str | None=None, direction: str | None=None) -> 'Mermaid':
        """Handle state diagram."""
        state_list = tuple(states)
        transition_list = tuple(transitions)
        known_ids: set[str] = set()
        lines = ['stateDiagram-v2']
        if direction is not None:
            direction_value = direction.strip().upper()
            if direction_value not in {'TB', 'BT', 'LR', 'RL'}:
                raise MermaidValidationError("State diagram direction must be one of 'TB', 'BT', 'LR', or 'RL'.")
            lines.append(f'    direction {direction_value}')
        for state in state_list:
            state_id = cls._validate_identifier(state.id, field='State id')
            if state_id in known_ids:
                raise MermaidValidationError(f'Duplicate state id: {state_id!r}.')
            known_ids.add(state_id)
            if state.label is None:
                lines.append(f'    state {state_id}')
            else:
                lines.append('    state "' + cls._escape_quoted_text(state.label) + f'" as {state_id}')
        for transition in transition_list:
            source = transition.source
            target = transition.target
            if source != '[*]':
                source = cls._validate_identifier(source, field='State transition source')
                if source not in known_ids:
                    raise MermaidValidationError(f'State transition references unknown source: {source!r}.')
            if target != '[*]':
                target = cls._validate_identifier(target, field='State transition target')
                if target not in known_ids:
                    raise MermaidValidationError(f'State transition references unknown target: {target!r}.')
            line = f'    {source} --> {target}'
            if transition.label:
                line += ' : ' + cls._escape_inline_text(transition.label)
            lines.append(line)
        return cls(cls._with_frontmatter_title(lines, title))

    @classmethod
    def er_diagram(cls, *, entities: Mapping[str, Iterable[ERField]], relationships: Iterable[ERRelationship]=(), title: str | None=None) -> 'Mermaid':
        """Handle er diagram."""
        known_entities: set[str] = set()
        lines = ['erDiagram']
        for raw_name, raw_fields in entities.items():
            name = cls._validate_identifier(raw_name, field='ER entity name')
            if name in known_entities:
                raise MermaidValidationError(f'Duplicate ER entity: {name!r}.')
            known_entities.add(name)
            fields = tuple(raw_fields)
            if not fields:
                lines.append(f'    {name}')
                continue
            lines.append(f'    {name} {{')
            for er_field in fields:
                field_type = cls._safe_er_token(er_field.type, field='ER field type')
                field_name = cls._safe_er_token(er_field.name, field='ER field name')
                parts = ['        ', field_type, ' ', field_name]
                if er_field.key:
                    key = er_field.key.strip().upper()
                    if key not in {'PK', 'FK', 'UK'}:
                        raise MermaidValidationError(f'Unsupported ER key marker: {key!r}.')
                    parts.extend((' ', key))
                if er_field.comment:
                    parts.extend((' "', cls._escape_quoted_text(er_field.comment), '"'))
                lines.append(''.join(parts))
            lines.append('    }')
        for relationship in relationships:
            left = cls._validate_identifier(relationship.left, field='ER relationship left side')
            right = cls._validate_identifier(relationship.right, field='ER relationship right side')
            if left not in known_entities:
                raise MermaidValidationError(f'ER relationship references unknown entity: {left!r}.')
            if right not in known_entities:
                raise MermaidValidationError(f'ER relationship references unknown entity: {right!r}.')
            cardinality = relationship.cardinality.strip()
            if not cardinality or '\n' in cardinality or '\r' in cardinality:
                raise MermaidValidationError('ER relationship cardinality must be a non-empty single-line Mermaid cardinality expression.')
            lines.append(f'    {left} {cardinality} {right} : {cls._escape_inline_text(relationship.label)}')
        return cls(cls._with_frontmatter_title(lines, title))

    @classmethod
    def gantt(cls, *, title: str, sections: Mapping[str, Iterable[GanttTask]], date_format: str='YYYY-MM-DD', axis_format: str | None=None) -> 'Mermaid':
        """Handle gantt."""
        cls._validate_single_line(date_format, field='Gantt date format')
        if axis_format is not None:
            cls._validate_single_line(axis_format, field='Gantt axis format')
        lines = ['gantt', '    title ' + cls._escape_inline_text(title), f'    dateFormat {date_format.strip()}']
        if axis_format:
            lines.append(f'    axisFormat {axis_format.strip()}')
        seen_ids: set[str] = set()
        for section_name, raw_tasks in sections.items():
            cls._validate_single_line(section_name, field='Gantt section name')
            lines.append('    section ' + cls._escape_inline_text(section_name))
            for task in raw_tasks:
                cls._validate_single_line(task.label, field='Gantt task label')
                cls._validate_single_line(task.definition, field='Gantt task definition')
                pieces: list[str] = []
                for modifier in task.modifiers:
                    normalized_modifier = modifier.strip().lower()
                    if normalized_modifier not in cls._GANTT_MODIFIERS:
                        raise MermaidValidationError(f'Unsupported Gantt task modifier: {modifier!r}.')
                    pieces.append(normalized_modifier)
                if task.id:
                    task_id = cls._validate_identifier(task.id, field='Gantt task id')
                    if task_id in seen_ids:
                        raise MermaidValidationError(f'Duplicate Gantt task id: {task_id!r}.')
                    seen_ids.add(task_id)
                    pieces.append(task_id)
                pieces.append(task.definition.strip())
                lines.append('    ' + cls._escape_inline_text(task.label) + ' :' + ', '.join(pieces))
        return cls('\n'.join(lines) + '\n')

    @classmethod
    def pie(cls, *, values: Mapping[str, float], title: str | None=None, show_data: bool=False) -> 'Mermaid':
        """Handle pie."""
        if not values:
            raise MermaidValidationError('Pie chart requires at least one value.')
        declaration = 'pie showData' if show_data else 'pie'
        lines = [declaration]
        if title:
            lines.append('    title ' + cls._escape_inline_text(title))
        for label, value in values.items():
            numeric = float(value)
            if numeric < 0:
                raise MermaidValidationError('Pie chart values cannot be negative.')
            escaped_label = cls._escape_quoted_text(label)
            lines.append(f'    "{escaped_label}" : {numeric:g}')
        return cls('\n'.join(lines) + '\n')

    @classmethod
    def journey(cls, *, title: str, sections: Mapping[str, Iterable[JourneyTask]]) -> 'Mermaid':
        """Handle journey."""
        lines = ['journey', '    title ' + cls._escape_inline_text(title)]
        for section_name, raw_tasks in sections.items():
            cls._validate_single_line(section_name, field='Journey section name')
            lines.append('    section ' + cls._escape_inline_text(section_name))
            for task in raw_tasks:
                if not 0 <= task.score <= 5:
                    raise MermaidValidationError('Journey task score must be between 0 and 5.')
                cls._validate_single_line(task.label, field='Journey task label')
                actor_suffix = ''
                if task.actors:
                    actors: list[str] = []
                    for actor in task.actors:
                        cls._validate_single_line(actor, field='Journey actor')
                        actors.append(cls._escape_inline_text(actor))
                    actor_suffix = ': ' + ', '.join(actors)
                lines.append('        ' + cls._escape_inline_text(task.label) + f': {task.score}' + actor_suffix)
        return cls('\n'.join(lines) + '\n')

    @classmethod
    def mindmap(cls, root: MindmapNode, *, title: str | None=None) -> 'Mermaid':
        """Handle mindmap."""
        lines = ['mindmap']

        def append_node(node: MindmapNode, depth: int) -> None:
            """Handle append node."""
            cls._validate_single_line(node.text, field='Mindmap node text')
            lines.append('    ' * depth + cls._escape_inline_text(node.text))
            for child in node.children:
                append_node(child, depth + 1)
        append_node(root, 1)
        return cls(cls._with_frontmatter_title(lines, title))

    @classmethod
    def timeline(cls, *, periods: Iterable[TimelinePeriod], title: str | None=None) -> 'Mermaid':
        """Handle timeline."""
        period_list = tuple(periods)
        if not period_list:
            raise MermaidValidationError('Timeline requires at least one period.')
        lines = ['timeline']
        if title:
            lines.append('    title ' + cls._escape_inline_text(title))
        for period in period_list:
            cls._validate_single_line(period.period, field='Timeline period')
            if not period.events:
                raise MermaidValidationError(f'Timeline period {period.period!r} has no events.')
            escaped_period = cls._escape_inline_text(period.period)
            escaped_events = [cls._escape_inline_text(event) for event in period.events]
            lines.append('    ' + escaped_period + ' : ' + ' : '.join(escaped_events))
        return cls('\n'.join(lines) + '\n')

    @classmethod
    def git_graph(cls, *, operations: Iterable[str], title: str | None=None) -> 'Mermaid':
        """Handle git graph."""
        operation_list = tuple(operations)
        if not operation_list:
            raise MermaidValidationError('Git graph requires at least one operation.')
        lines = ['gitGraph']
        for operation in operation_list:
            cls._validate_single_line(operation, field='Git graph operation')
            stripped = operation.strip()
            command = stripped.split(maxsplit=1)[0]
            if command not in cls._GIT_OPERATIONS:
                raise MermaidValidationError(f'Unsupported GitGraph operation: {command!r}.')
            lines.append(f'    {stripped}')
        return cls(cls._with_frontmatter_title(lines, title))

    @classmethod
    def quadrant_chart(cls, *, points: Iterable[QuadrantPoint], title: str | None=None, x_axis: tuple[str, str] | None=None, y_axis: tuple[str, str] | None=None, quadrants: tuple[str, str, str, str] | None=None) -> 'Mermaid':
        """Handle quadrant chart."""
        lines = ['quadrantChart']
        if title:
            lines.append('    title ' + cls._escape_inline_text(title))
        if x_axis:
            lines.append('    x-axis ' + cls._escape_inline_text(x_axis[0]) + ' --> ' + cls._escape_inline_text(x_axis[1]))
        if y_axis:
            lines.append('    y-axis ' + cls._escape_inline_text(y_axis[0]) + ' --> ' + cls._escape_inline_text(y_axis[1]))
        if quadrants:
            for index, label in enumerate(quadrants, 1):
                lines.append(f'    quadrant-{index} ' + cls._escape_inline_text(label))
        point_list = tuple(points)
        if not point_list:
            raise MermaidValidationError('Quadrant chart requires at least one point.')
        for point in point_list:
            if not 0 <= point.x <= 1:
                raise MermaidValidationError(f'Quadrant x value must be in [0, 1]: {point.x}.')
            if not 0 <= point.y <= 1:
                raise MermaidValidationError(f'Quadrant y value must be in [0, 1]: {point.y}.')
            lines.append('    ' + cls._escape_inline_text(point.label) + f': [{point.x:g}, {point.y:g}]')
        return cls('\n'.join(lines) + '\n')

    @classmethod
    def xy_chart(cls, *, x_values: Sequence[float], series: Iterable[XYSeries], title: str | None=None, x_label: str | None=None, y_label: str | None=None) -> 'Mermaid':
        """Handle xy chart."""
        if not x_values:
            raise MermaidValidationError('XY chart requires at least one x value.')
        series_list = tuple(series)
        if not series_list:
            raise MermaidValidationError('XY chart requires at least one series.')
        lines = ['xychart-beta']
        if title:
            lines.append('    title "' + cls._escape_quoted_text(title) + '"')
        x_data = ', '.join((f'{float(value):g}' for value in x_values))
        x_axis = f'    x-axis "{cls._escape_quoted_text(x_label)}" ' if x_label else '    x-axis '
        lines.append(x_axis + '[' + x_data + ']')
        if y_label:
            lines.append('    y-axis "' + cls._escape_quoted_text(y_label) + '"')
        for xy_series in series_list:
            kind = xy_series.kind.strip().lower()
            if kind not in {'line', 'bar'}:
                raise MermaidValidationError("XY series kind must be 'line' or 'bar'.")
            if len(xy_series.values) != len(x_values):
                raise MermaidValidationError(f'XY {kind} series has {len(xy_series.values)} values but x-axis has {len(x_values)} values.')
            data = ', '.join((f'{float(value):g}' for value in xy_series.values))
            lines.append(f'    {kind} [{data}]')
        return cls('\n'.join(lines) + '\n')

    @classmethod
    def raw(cls, source: str, *, allow_unknown_type: bool=False) -> 'Mermaid':
        """Handle raw."""
        return cls(source, allow_unknown_type=allow_unknown_type)

    @staticmethod
    def normalize_source(source: str) -> str:
        """
        Normalize user/model Mermaid input.

        A surrounding Markdown `````mermaid`` fence is accepted and removed.
        Internal whitespace is preserved; line endings become ``\\n``.
        """
        normalized = source.replace('\r\n', '\n').replace('\r', '\n').strip()
        if not normalized:
            raise MermaidValidationError('Mermaid source cannot be empty.')
        lines = normalized.splitlines()
        if lines and lines[0].strip().lower() in {'```mermaid', '~~~mermaid'}:
            fence = '```' if lines[0].lstrip().startswith('```') else '~~~'
            if len(lines) < 2 or lines[-1].strip() != fence:
                raise MermaidValidationError('Mermaid Markdown fence is not closed.')
            lines = lines[1:-1]
            normalized = '\n'.join(lines).strip()
            if not normalized:
                raise MermaidValidationError('Mermaid fence contains no source.')
        return normalized + '\n'

    @staticmethod
    def validate_asset_name(name: str) -> str:
        """Handle validate asset name."""
        normalized = name.strip()
        if not _NAME_PATTERN.fullmatch(normalized):
            raise MermaidValidationError("Diagram asset name must begin with an alphanumeric character and contain only letters, numbers, '_' or '-'; maximum length is 128 characters.")
        return normalized

    @classmethod
    def _find_declaration(cls, source: str) -> str:
        """Handle find declaration."""
        lines = source.splitlines()
        index = 0
        while index < len(lines) and (not lines[index].strip()):
            index += 1
        if index < len(lines) and lines[index].strip() == _FRONTMATTER_DELIMITER:
            index += 1
            while index < len(lines):
                if lines[index].strip() == _FRONTMATTER_DELIMITER:
                    index += 1
                    break
                index += 1
            else:
                raise MermaidValidationError('Mermaid frontmatter is not closed.')
        while index < len(lines):
            candidate = lines[index].strip()
            index += 1
            if not candidate:
                continue
            if candidate.startswith('%%'):
                continue
            return candidate
        return ''

    @classmethod
    def _with_frontmatter_title(cls, lines: Sequence[str], title: str | None) -> str:
        """Handle with frontmatter title."""
        body = '\n'.join(lines)
        if title is None:
            return body + '\n'
        cls._validate_single_line(title, field='Diagram title')
        safe_title = title.replace('\\', '\\\\').replace('"', '\\"')
        return f'---\ntitle: "{safe_title}"\n---\n{body}\n'

    @staticmethod
    def _enum_value(value: Enum | str) -> str:
        """Handle enum value."""
        if isinstance(value, Enum):
            return str(value.value)
        return str(value)

    @classmethod
    def _flow_node_syntax(cls, node_id: str, label: str, shape: FlowNodeShape | str) -> str:
        """Handle flow node syntax."""
        try:
            shape_value = FlowNodeShape(cls._enum_value(shape))
        except ValueError as error:
            raise MermaidValidationError(f'Unsupported flow node shape: {shape!r}.') from error
        safe_label = cls._escape_quoted_text(label)
        wrappers = {FlowNodeShape.RECTANGLE: ('["', '"]'), FlowNodeShape.ROUNDED: ('("', '")'), FlowNodeShape.STADIUM: ('(["', '"])'), FlowNodeShape.SUBROUTINE: ('[["', '"]]'), FlowNodeShape.CYLINDER: ('[("', '")]'), FlowNodeShape.CIRCLE: ('(("', '"))'), FlowNodeShape.DOUBLE_CIRCLE: ('((("', '")))'), FlowNodeShape.DIAMOND: ('{"', '"}'), FlowNodeShape.HEXAGON: ('{{"', '"}}')}
        prefix, suffix = wrappers[shape_value]
        return node_id + prefix + safe_label + suffix

    @staticmethod
    def _validate_identifier(value: str, *, field: str) -> str:
        """Handle validate identifier."""
        normalized = value.strip()
        if not _IDENTIFIER_PATTERN.fullmatch(normalized):
            raise MermaidValidationError(f"{field} must begin with a letter or '_' and contain only letters, numbers, '_' or '-': {value!r}.")
        return normalized

    @staticmethod
    def _safe_er_token(value: str, *, field: str) -> str:
        """Handle safe er token."""
        normalized = value.strip()
        if not normalized or not re.fullmatch('[A-Za-z0-9_.<>\\[\\]-]+', normalized):
            raise MermaidValidationError(f'{field} contains unsupported characters: {value!r}.')
        return normalized

    @staticmethod
    def _validate_single_line(value: str, *, field: str) -> None:
        """Handle validate single line."""
        if '\n' in value or '\r' in value:
            raise MermaidValidationError(f'{field} must be a single line.')
        if not value.strip():
            raise MermaidValidationError(f'{field} cannot be empty.')

    @staticmethod
    def _escape_quoted_text(value: str) -> str:
        """Handle escape quoted text."""
        return escape(value, quote=True).replace('\n', '<br/>').replace('\r', '')

    @staticmethod
    def _escape_inline_text(value: str) -> str:
        """Handle escape inline text."""
        return escape(value, quote=True).replace('\n', '<br/>').replace('\r', '')

    @staticmethod
    def _escape_sequence_text(value: str) -> str:
        """Handle escape sequence text."""
        return escape(value, quote=True).replace(';', '#59;').replace('\n', '<br/>').replace('\r', '')

    @staticmethod
    def _escape_markdown_alt(value: str) -> str:
        """Handle escape markdown alt."""
        return value.replace('\\', '\\\\').replace('[', '\\[').replace(']', '\\]').replace('\n', ' ').replace('\r', ' ')

    @staticmethod
    def _output_format_from_path(path: Path) -> MermaidOutputFormat:
        """Handle output format from path."""
        suffix = path.suffix.lower().lstrip('.')
        try:
            return MermaidOutputFormat(suffix)
        except ValueError as error:
            raise MermaidValidationError('Mermaid output must use one of these extensions: .svg, .png, .pdf.') from error

    def _render_file(self, *, source_path: Path, output_path: Path, executable: str, timeout: float, theme: MermaidTheme | str | None, background: str | None, width: int | None, height: int | None) -> tuple[str, ...]:
        """Handle render file."""
        if timeout <= 0:
            raise MermaidValidationError('Render timeout must be greater than zero.')
        if width is not None and width <= 0:
            raise MermaidValidationError('Render width must be greater than zero.')
        if height is not None and height <= 0:
            raise MermaidValidationError('Render height must be greater than zero.')
        if self._command_runner is None:
            executable_path = shutil.which(executable)
            if executable_path is None:
                raise MermaidRendererUnavailable(f'Mermaid renderer is not available: {executable!r}.')
        else:
            executable_path = executable
        command: list[str] = [executable_path, '-i', str(source_path), '-o', str(output_path)]
        if theme is not None:
            try:
                theme_value = MermaidTheme(self._enum_value(theme)).value
            except ValueError as error:
                raise MermaidValidationError(f'Unsupported Mermaid theme: {theme!r}.') from error
            command.extend(('-t', theme_value))
        if background is not None:
            if '\n' in background or '\r' in background or '\x00' in background:
                raise MermaidValidationError('Render background must be a single safe CLI value.')
            command.extend(('-b', background))
        if width is not None:
            command.extend(('-w', str(width)))
        if height is not None:
            command.extend(('-H', str(height)))
        try:
            if self._command_runner is None:
                completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
                returncode = completed.returncode
                diagnostic_output = completed.stderr.strip() or completed.stdout.strip()
            else:
                returncode, diagnostic_output = self._command_runner(tuple(command), timeout)
        except subprocess.TimeoutExpired as error:
            raise MermaidRenderError(f'Mermaid rendering timed out after {timeout:g}s.') from error
        except OSError as error:
            raise MermaidRenderError(f'Could not execute Mermaid renderer: {error}') from error
        if returncode != 0:
            diagnostic = diagnostic_output or 'renderer returned no diagnostic'
            raise MermaidRenderError(f'Mermaid renderer failed with exit code {returncode}: {diagnostic}')
        if not output_path.is_file():
            raise MermaidRenderError(f'Mermaid renderer reported success but did not create the expected output: {output_path}')
        return tuple(command)

    @staticmethod
    def _validate_rendered_output(path: Path, output_format: MermaidOutputFormat) -> None:
        """Handle validate rendered output."""
        if not path.is_file() or path.stat().st_size == 0:
            raise MermaidRenderError(f'Rendered Mermaid output is empty: {path}')
        prefix = path.read_bytes()[:512].lstrip()
        if output_format is MermaidOutputFormat.SVG:
            lowered = prefix.lower()
            if b'<svg' not in lowered and (not (lowered.startswith(b'<?xml') and b'<svg' in lowered)):
                raise MermaidRenderError('Mermaid renderer produced an invalid SVG payload.')
        elif output_format is MermaidOutputFormat.PNG:
            if not prefix.startswith(b'\x89PNG\r\n\x1a\n'):
                raise MermaidRenderError('Mermaid renderer produced an invalid PNG payload.')
        elif output_format is MermaidOutputFormat.PDF:
            if not prefix.startswith(b'%PDF-'):
                raise MermaidRenderError('Mermaid renderer produced an invalid PDF payload.')

    @classmethod
    def _replace_bundle_files(cls, *, staged_source: Path, source_path: Path, staged_rendered: Path | None, rendered_path: Path, remove_rendered: bool) -> None:
        """
        Best-effort two-file transaction.

        Existing files are moved to backups before replacement. If any
        replacement fails, the previous state is restored where possible.
        """
        source_backup = source_path.with_name(f'.{source_path.name}.backup')
        rendered_backup = rendered_path.with_name(f'.{rendered_path.name}.backup')
        for backup in (source_backup, rendered_backup):
            backup.unlink(missing_ok=True)
        source_had_previous = source_path.exists()
        rendered_had_previous = rendered_path.exists()
        try:
            if source_had_previous:
                os.replace(source_path, source_backup)
            if (staged_rendered is not None or remove_rendered) and rendered_had_previous:
                os.replace(rendered_path, rendered_backup)
            os.replace(staged_source, source_path)
            if staged_rendered is not None:
                os.replace(staged_rendered, rendered_path)
            source_backup.unlink(missing_ok=True)
            rendered_backup.unlink(missing_ok=True)
        except Exception:
            source_path.unlink(missing_ok=True)
            if staged_rendered is not None or remove_rendered:
                rendered_path.unlink(missing_ok=True)
            if source_backup.exists():
                os.replace(source_backup, source_path)
            if rendered_backup.exists():
                os.replace(rendered_backup, rendered_path)
            raise

    @staticmethod
    def _write_text_atomic(path: Path, text: str) -> None:
        """Handle write text atomic."""
        descriptor, temporary_raw = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
        temporary = Path(temporary_raw)
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as stream:
                stream.write(text)
            os.replace(temporary, path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
