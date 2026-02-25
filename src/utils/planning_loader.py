"""
Planning YAML Loader Utility.

Provides programmatic access to project planning files (milestones, epics, stories)
with validation and query capabilities.

Based on: universal-dev-agent prompt template
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class StoryStatus(str, Enum):
    """Status of a story."""

    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class MilestoneStatus(str, Enum):
    """Status of a milestone."""

    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass
class Story:
    """Represents a user story."""

    id: str
    name: str
    status: StoryStatus
    subagent: str
    description: str = ""
    acceptance_criteria: list[str] = field(default_factory=list)
    definition_of_done: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Story:
        """Create Story from dictionary."""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            status=StoryStatus(data.get("status", "planned")),
            subagent=data.get("subagent", ""),
            description=data.get("description", ""),
            acceptance_criteria=data.get("acceptance_criteria", []),
            definition_of_done=data.get("definition_of_done", []),
            artifacts=data.get("artifacts", []),
        )

    def is_complete(self) -> bool:
        """Check if story is completed."""
        return self.status == StoryStatus.COMPLETED


@dataclass
class Epic:
    """Represents an epic containing stories."""

    id: str
    name: str
    milestone: str
    status: str
    owner: str
    stories: list[Story] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Epic:
        """Create Epic from dictionary."""
        epic_data = data.get("epic", data)
        stories = [Story.from_dict(s) for s in data.get("stories", [])]

        return cls(
            id=epic_data.get("id", ""),
            name=epic_data.get("name", ""),
            milestone=epic_data.get("milestone", ""),
            status=epic_data.get("status", "planned"),
            owner=epic_data.get("owner", ""),
            stories=stories,
            dependencies=data.get("dependencies", []),
        )

    def completion_percentage(self) -> float:
        """Calculate completion percentage based on stories."""
        if not self.stories:
            return 0.0
        completed = sum(1 for s in self.stories if s.is_complete())
        return (completed / len(self.stories)) * 100


@dataclass
class Milestone:
    """Represents a project milestone."""

    id: str
    name: str
    status: MilestoneStatus
    completion: float
    goal: str
    epics: list[str] = field(default_factory=list)
    deliverables: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Milestone:
        """Create Milestone from dictionary."""
        # Handle epic references
        epic_refs = data.get("epics", [])
        epics = []
        for ref in epic_refs:
            if isinstance(ref, dict) and "$ref" in ref:
                epics.append(ref["$ref"])
            elif isinstance(ref, str):
                epics.append(ref)

        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            status=MilestoneStatus(data.get("status", "planned")),
            completion=float(data.get("completion", 0)),
            goal=data.get("goal", ""),
            epics=epics,
            deliverables=data.get("deliverables", []),
        )


@dataclass
class ProjectPlan:
    """Complete project plan with milestones and epics."""

    name: str
    domain: str
    goal: str
    milestones: list[Milestone] = field(default_factory=list)
    epics: dict[str, Epic] = field(default_factory=dict)
    quality_metrics: dict[str, Any] = field(default_factory=dict)

    def get_milestone(self, milestone_id: str) -> Milestone | None:
        """Get milestone by ID."""
        for m in self.milestones:
            if m.id == milestone_id:
                return m
        return None

    def get_epic(self, epic_id: str) -> Epic | None:
        """Get epic by ID."""
        return self.epics.get(epic_id)

    def get_stories_by_status(self, status: StoryStatus) -> list[Story]:
        """Get all stories with given status."""
        stories = []
        for epic in self.epics.values():
            for story in epic.stories:
                if story.status == status:
                    stories.append(story)
        return stories

    def get_stories_by_subagent(self, subagent: str) -> list[Story]:
        """Get all stories assigned to a subagent."""
        stories = []
        for epic in self.epics.values():
            for story in epic.stories:
                if story.subagent == subagent:
                    stories.append(story)
        return stories

    def overall_completion(self) -> float:
        """Calculate overall project completion percentage."""
        if not self.milestones:
            return 0.0
        total = sum(m.completion for m in self.milestones)
        return total / len(self.milestones)

    def summary(self) -> dict[str, Any]:
        """Generate project summary."""
        total_stories = sum(len(e.stories) for e in self.epics.values())
        completed_stories = len(self.get_stories_by_status(StoryStatus.COMPLETED))
        in_progress_stories = len(self.get_stories_by_status(StoryStatus.IN_PROGRESS))

        return {
            "project": self.name,
            "overall_completion": f"{self.overall_completion():.1f}%",
            "milestones": {
                "total": len(self.milestones),
                "completed": len([m for m in self.milestones if m.status == MilestoneStatus.COMPLETED]),
                "in_progress": len([m for m in self.milestones if m.status == MilestoneStatus.IN_PROGRESS]),
            },
            "epics": {
                "total": len(self.epics),
            },
            "stories": {
                "total": total_stories,
                "completed": completed_stories,
                "in_progress": in_progress_stories,
                "completion_rate": f"{(completed_stories / total_stories * 100) if total_stories > 0 else 0:.1f}%",
            },
        }


class PlanningLoader:
    """
    Loader for project planning YAML files.

    Usage:
        loader = PlanningLoader("planning/")
        plan = loader.load()
        print(plan.summary())
    """

    def __init__(self, planning_dir: str | Path | None = None):
        """
        Initialize loader.

        Args:
            planning_dir: Path to planning directory (default: project_root/planning)
        """
        if planning_dir is None:
            # Default to project root planning directory
            self.planning_dir = Path(__file__).parent.parent.parent / "planning"
        else:
            self.planning_dir = Path(planning_dir)

        self._cached_plan: ProjectPlan | None = None

    def load(self, force_reload: bool = False) -> ProjectPlan:
        """
        Load project plan from YAML files.

        Args:
            force_reload: Force reload even if cached

        Returns:
            Complete project plan
        """
        if self._cached_plan is not None and not force_reload:
            return self._cached_plan

        milestones_file = self.planning_dir / "milestones.yaml"

        if not milestones_file.exists():
            logger.warning(f"Milestones file not found: {milestones_file}")
            return ProjectPlan(name="", domain="", goal="")

        with open(milestones_file) as f:
            data = yaml.safe_load(f)

        # Parse project info
        project = data.get("project", {})
        plan = ProjectPlan(
            name=project.get("name", ""),
            domain=project.get("domain", ""),
            goal=project.get("goal", ""),
            quality_metrics=data.get("quality_metrics", {}),
        )

        # Parse milestones
        for m_data in data.get("milestones", []):
            milestone = Milestone.from_dict(m_data)
            plan.milestones.append(milestone)

            # Load referenced epics
            for epic_ref in milestone.epics:
                epic = self._load_epic(epic_ref)
                if epic:
                    plan.epics[epic.id] = epic

        self._cached_plan = plan
        logger.info(f"Loaded project plan: {plan.name} with {len(plan.milestones)} milestones")

        return plan

    def _load_epic(self, epic_ref: str) -> Epic | None:
        """Load epic from reference path."""
        # Handle $ref format
        if epic_ref.startswith("./"):
            epic_path = self.planning_dir / epic_ref[2:]
        else:
            epic_path = self.planning_dir / epic_ref

        if not epic_path.exists():
            logger.warning(f"Epic file not found: {epic_path}")
            return None

        with open(epic_path) as f:
            data = yaml.safe_load(f)

        return Epic.from_dict(data)

    def get_next_tasks(self, subagent: str | None = None) -> list[Story]:
        """
        Get next tasks to work on.

        Args:
            subagent: Filter by subagent (optional)

        Returns:
            List of stories that are in_progress or planned and ready
        """
        plan = self.load()

        # Get in-progress stories first
        in_progress = plan.get_stories_by_status(StoryStatus.IN_PROGRESS)
        if subagent:
            in_progress = [s for s in in_progress if s.subagent == subagent]

        if in_progress:
            return in_progress

        # Get planned stories with completed dependencies
        planned = plan.get_stories_by_status(StoryStatus.PLANNED)
        if subagent:
            planned = [s for s in planned if s.subagent == subagent]

        return planned[:5]  # Return top 5 planned stories

    def validate(self) -> list[str]:
        """
        Validate planning files for common issues.

        Returns:
            List of validation warnings/errors
        """
        issues = []
        plan = self.load(force_reload=True)

        # Check for orphaned stories (no epic)
        for epic in plan.epics.values():
            for story in epic.stories:
                if not story.id:
                    issues.append(f"Story without ID in epic {epic.id}")
                if not story.acceptance_criteria:
                    issues.append(f"Story {story.id} has no acceptance criteria")

        # Check milestone completion matches story status
        for milestone in plan.milestones:
            epic_ids = [e.split("/")[-1].replace(".yaml", "") for e in milestone.epics]
            related_epics_raw = [plan.epics.get(f"E{eid.split('_')[1]}.{eid.split('_')[2]}") for eid in epic_ids]
            related_epics: list[Epic] = [e for e in related_epics_raw if e is not None]

            total_stories = sum(len(e.stories) for e in related_epics)
            completed = sum(len([s for s in e.stories if s.is_complete()]) for e in related_epics)

            if total_stories > 0:
                actual_completion = (completed / total_stories) * 100
                if abs(actual_completion - milestone.completion) > 10:
                    issues.append(
                        f"Milestone {milestone.id} completion mismatch: "
                        f"stated {milestone.completion}%, actual {actual_completion:.1f}%"
                    )

        return issues


def get_project_plan(planning_dir: str | Path | None = None) -> ProjectPlan:
    """
    Convenience function to get project plan.

    Args:
        planning_dir: Optional path to planning directory

    Returns:
        Loaded project plan
    """
    loader = PlanningLoader(planning_dir)
    return loader.load()


__all__ = [
    "Story",
    "StoryStatus",
    "Epic",
    "Milestone",
    "MilestoneStatus",
    "ProjectPlan",
    "PlanningLoader",
    "get_project_plan",
]
