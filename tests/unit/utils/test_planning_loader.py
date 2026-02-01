"""
Unit tests for Planning Loader utility.

Tests the planning YAML file loader with validation and query capabilities.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from src.utils.planning_loader import (
    Epic,
    Milestone,
    MilestoneStatus,
    PlanningLoader,
    ProjectPlan,
    Story,
    StoryStatus,
    get_project_plan,
)


class TestStory:
    """Tests for Story dataclass."""

    def test_story_creation(self) -> None:
        """Test creating a story from dict."""
        data = {
            "id": "S1.1.1",
            "name": "Test Story",
            "status": "completed",
            "subagent": "coder",
            "description": "A test story",
            "acceptance_criteria": ["AC1", "AC2"],
            "definition_of_done": ["DoD1"],
            "artifacts": ["file.py"],
        }

        story = Story.from_dict(data)

        assert story.id == "S1.1.1"
        assert story.name == "Test Story"
        assert story.status == StoryStatus.COMPLETED
        assert story.subagent == "coder"
        assert len(story.acceptance_criteria) == 2
        assert story.is_complete()

    def test_story_defaults(self) -> None:
        """Test story with minimal data uses defaults."""
        data = {
            "id": "S1",
            "name": "Minimal",
            "status": "planned",
            "subagent": "planner",
        }

        story = Story.from_dict(data)

        assert story.description == ""
        assert story.acceptance_criteria == []
        assert story.definition_of_done == []
        assert story.artifacts == []
        assert not story.is_complete()

    def test_story_status_enum(self) -> None:
        """Test all story status values."""
        for status in ["planned", "in_progress", "completed", "blocked"]:
            data = {"id": "S1", "name": "Test", "status": status, "subagent": "test"}
            story = Story.from_dict(data)
            assert story.status == StoryStatus(status)


class TestEpic:
    """Tests for Epic dataclass."""

    def test_epic_creation(self) -> None:
        """Test creating an epic from dict."""
        data = {
            "epic": {
                "id": "E1.1",
                "name": "Test Epic",
                "milestone": "M1",
                "status": "in_progress",
                "owner": "orchestrator",
            },
            "stories": [
                {"id": "S1", "name": "Story 1", "status": "completed", "subagent": "coder"},
                {"id": "S2", "name": "Story 2", "status": "planned", "subagent": "sge"},
            ],
            "dependencies": ["E1.0"],
        }

        epic = Epic.from_dict(data)

        assert epic.id == "E1.1"
        assert epic.name == "Test Epic"
        assert epic.milestone == "M1"
        assert epic.owner == "orchestrator"
        assert len(epic.stories) == 2
        assert epic.dependencies == ["E1.0"]

    def test_epic_completion_percentage(self) -> None:
        """Test epic completion calculation."""
        data = {
            "epic": {"id": "E1", "name": "Test", "milestone": "M1", "status": "in_progress", "owner": "test"},
            "stories": [
                {"id": "S1", "name": "S1", "status": "completed", "subagent": "a"},
                {"id": "S2", "name": "S2", "status": "completed", "subagent": "a"},
                {"id": "S3", "name": "S3", "status": "in_progress", "subagent": "a"},
                {"id": "S4", "name": "S4", "status": "planned", "subagent": "a"},
            ],
        }

        epic = Epic.from_dict(data)
        assert epic.completion_percentage() == 50.0

    def test_epic_no_stories_completion(self) -> None:
        """Test epic with no stories returns 0% completion."""
        data = {
            "epic": {"id": "E1", "name": "Empty", "milestone": "M1", "status": "planned", "owner": "test"},
            "stories": [],
        }

        epic = Epic.from_dict(data)
        assert epic.completion_percentage() == 0.0


class TestMilestone:
    """Tests for Milestone dataclass."""

    def test_milestone_creation(self) -> None:
        """Test creating a milestone from dict."""
        data = {
            "id": "M1",
            "name": "Project Init",
            "status": "completed",
            "completion": 95,
            "goal": "Bootstrap the project",
            "epics": [{"$ref": "./epics/epic_1.yaml"}, "./epics/epic_2.yaml"],
            "deliverables": ["README.md", "CLAUDE.md"],
        }

        milestone = Milestone.from_dict(data)

        assert milestone.id == "M1"
        assert milestone.name == "Project Init"
        assert milestone.status == MilestoneStatus.COMPLETED
        assert milestone.completion == 95.0
        assert len(milestone.epics) == 2
        assert len(milestone.deliverables) == 2

    def test_milestone_status_enum(self) -> None:
        """Test all milestone status values."""
        for status in ["planned", "in_progress", "completed"]:
            data = {"id": "M1", "name": "Test", "status": status, "completion": 0, "goal": "Test"}
            milestone = Milestone.from_dict(data)
            assert milestone.status == MilestoneStatus(status)


class TestProjectPlan:
    """Tests for ProjectPlan dataclass."""

    def test_project_plan_creation(self) -> None:
        """Test creating a project plan."""
        plan = ProjectPlan(
            name="Test Project",
            domain="testing",
            goal="Test the loader",
            milestones=[
                Milestone(
                    id="M1",
                    name="Milestone 1",
                    status=MilestoneStatus.COMPLETED,
                    completion=100,
                    goal="Complete M1",
                )
            ],
        )

        assert plan.name == "Test Project"
        assert len(plan.milestones) == 1

    def test_get_milestone(self) -> None:
        """Test getting milestone by ID."""
        m1 = Milestone(id="M1", name="M1", status=MilestoneStatus.COMPLETED, completion=100, goal="M1")
        m2 = Milestone(id="M2", name="M2", status=MilestoneStatus.IN_PROGRESS, completion=50, goal="M2")

        plan = ProjectPlan(name="Test", domain="test", goal="test", milestones=[m1, m2])

        assert plan.get_milestone("M1") == m1
        assert plan.get_milestone("M2") == m2
        assert plan.get_milestone("M3") is None

    def test_get_stories_by_status(self) -> None:
        """Test filtering stories by status."""
        epic = Epic(
            id="E1",
            name="Epic 1",
            milestone="M1",
            status="in_progress",
            owner="test",
            stories=[
                Story(id="S1", name="S1", status=StoryStatus.COMPLETED, subagent="a"),
                Story(id="S2", name="S2", status=StoryStatus.IN_PROGRESS, subagent="b"),
                Story(id="S3", name="S3", status=StoryStatus.COMPLETED, subagent="a"),
            ],
        )

        plan = ProjectPlan(name="Test", domain="test", goal="test", epics={"E1": epic})

        completed = plan.get_stories_by_status(StoryStatus.COMPLETED)
        assert len(completed) == 2

        in_progress = plan.get_stories_by_status(StoryStatus.IN_PROGRESS)
        assert len(in_progress) == 1

    def test_get_stories_by_subagent(self) -> None:
        """Test filtering stories by subagent."""
        epic = Epic(
            id="E1",
            name="Epic 1",
            milestone="M1",
            status="completed",
            owner="test",
            stories=[
                Story(id="S1", name="S1", status=StoryStatus.COMPLETED, subagent="coder"),
                Story(id="S2", name="S2", status=StoryStatus.COMPLETED, subagent="sge"),
                Story(id="S3", name="S3", status=StoryStatus.COMPLETED, subagent="coder"),
            ],
        )

        plan = ProjectPlan(name="Test", domain="test", goal="test", epics={"E1": epic})

        coder_stories = plan.get_stories_by_subagent("coder")
        assert len(coder_stories) == 2

        sge_stories = plan.get_stories_by_subagent("sge")
        assert len(sge_stories) == 1

    def test_overall_completion(self) -> None:
        """Test overall completion calculation."""
        plan = ProjectPlan(
            name="Test",
            domain="test",
            goal="test",
            milestones=[
                Milestone(id="M1", name="M1", status=MilestoneStatus.COMPLETED, completion=100, goal="M1"),
                Milestone(id="M2", name="M2", status=MilestoneStatus.IN_PROGRESS, completion=50, goal="M2"),
            ],
        )

        assert plan.overall_completion() == 75.0

    def test_summary(self) -> None:
        """Test project summary generation."""
        epic = Epic(
            id="E1",
            name="Epic 1",
            milestone="M1",
            status="completed",
            owner="test",
            stories=[
                Story(id="S1", name="S1", status=StoryStatus.COMPLETED, subagent="coder"),
                Story(id="S2", name="S2", status=StoryStatus.IN_PROGRESS, subagent="sge"),
            ],
        )

        plan = ProjectPlan(
            name="Test Project",
            domain="test",
            goal="test",
            milestones=[Milestone(id="M1", name="M1", status=MilestoneStatus.IN_PROGRESS, completion=50, goal="M1")],
            epics={"E1": epic},
        )

        summary = plan.summary()

        assert summary["project"] == "Test Project"
        assert summary["milestones"]["total"] == 1
        assert summary["stories"]["total"] == 2
        assert summary["stories"]["completed"] == 1
        assert summary["stories"]["in_progress"] == 1


class TestPlanningLoader:
    """Tests for PlanningLoader class."""

    def test_loader_initialization(self) -> None:
        """Test loader initialization."""
        loader = PlanningLoader("/tmp/planning")
        assert loader.planning_dir == Path("/tmp/planning")

    def test_load_missing_file(self) -> None:
        """Test loading from non-existent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = PlanningLoader(tmpdir)
            plan = loader.load()

            # Should return empty plan
            assert plan.name == ""
            assert plan.milestones == []

    def test_load_valid_milestones(self) -> None:
        """Test loading valid milestones file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create milestones file
            milestones_data = {
                "project": {
                    "name": "Test Project",
                    "domain": "testing",
                    "goal": "Test the loader",
                },
                "milestones": [
                    {
                        "id": "M1",
                        "name": "Milestone 1",
                        "status": "completed",
                        "completion": 100,
                        "goal": "Complete M1",
                        "epics": [],
                        "deliverables": ["file.md"],
                    }
                ],
            }

            milestones_path = Path(tmpdir) / "milestones.yaml"
            with open(milestones_path, "w") as f:
                yaml.dump(milestones_data, f)

            loader = PlanningLoader(tmpdir)
            plan = loader.load()

            assert plan.name == "Test Project"
            assert len(plan.milestones) == 1
            assert plan.milestones[0].id == "M1"

    def test_load_with_epic_refs(self) -> None:
        """Test loading milestones with epic references."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure
            epics_dir = Path(tmpdir) / "epics"
            epics_dir.mkdir()

            # Create epic file
            epic_data = {
                "epic": {
                    "id": "E1.1",
                    "name": "Test Epic",
                    "milestone": "M1",
                    "status": "completed",
                    "owner": "test",
                },
                "stories": [
                    {
                        "id": "S1.1.1",
                        "name": "Test Story",
                        "status": "completed",
                        "subagent": "coder",
                    }
                ],
            }

            with open(epics_dir / "epic_1.yaml", "w") as f:
                yaml.dump(epic_data, f)

            # Create milestones file
            milestones_data = {
                "project": {"name": "Test", "domain": "test", "goal": "test"},
                "milestones": [
                    {
                        "id": "M1",
                        "name": "M1",
                        "status": "completed",
                        "completion": 100,
                        "goal": "M1",
                        "epics": [{"$ref": "./epics/epic_1.yaml"}],
                    }
                ],
            }

            with open(Path(tmpdir) / "milestones.yaml", "w") as f:
                yaml.dump(milestones_data, f)

            loader = PlanningLoader(tmpdir)
            plan = loader.load()

            assert len(plan.epics) == 1
            assert "E1.1" in plan.epics
            assert len(plan.epics["E1.1"].stories) == 1

    def test_caching(self) -> None:
        """Test that loader caches results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            milestones_data = {
                "project": {"name": "Cached", "domain": "test", "goal": "test"},
                "milestones": [],
            }

            with open(Path(tmpdir) / "milestones.yaml", "w") as f:
                yaml.dump(milestones_data, f)

            loader = PlanningLoader(tmpdir)

            plan1 = loader.load()
            plan2 = loader.load()

            # Should be same object (cached)
            assert plan1 is plan2

            # Force reload should create new object
            plan3 = loader.load(force_reload=True)
            assert plan3 is not plan1

    def test_get_next_tasks(self) -> None:
        """Test getting next tasks to work on."""
        with tempfile.TemporaryDirectory() as tmpdir:
            epic_data = {
                "epic": {"id": "E1", "name": "E1", "milestone": "M1", "status": "in_progress", "owner": "test"},
                "stories": [
                    {"id": "S1", "name": "Done", "status": "completed", "subagent": "coder"},
                    {"id": "S2", "name": "Working", "status": "in_progress", "subagent": "coder"},
                    {"id": "S3", "name": "Next", "status": "planned", "subagent": "sge"},
                ],
            }

            epics_dir = Path(tmpdir) / "epics"
            epics_dir.mkdir()
            with open(epics_dir / "epic_1.yaml", "w") as f:
                yaml.dump(epic_data, f)

            milestones_data = {
                "project": {"name": "Test", "domain": "test", "goal": "test"},
                "milestones": [{"id": "M1", "name": "M1", "status": "in_progress", "completion": 50, "goal": "M1", "epics": [{"$ref": "./epics/epic_1.yaml"}]}],
            }

            with open(Path(tmpdir) / "milestones.yaml", "w") as f:
                yaml.dump(milestones_data, f)

            loader = PlanningLoader(tmpdir)

            # Should return in-progress tasks first
            tasks = loader.get_next_tasks()
            assert len(tasks) == 1
            assert tasks[0].status == StoryStatus.IN_PROGRESS

            # Filter by subagent
            tasks = loader.get_next_tasks(subagent="sge")
            assert len(tasks) == 1
            assert tasks[0].subagent == "sge"


class TestConvenienceFunction:
    """Tests for convenience functions."""

    def test_get_project_plan(self) -> None:
        """Test get_project_plan convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            milestones_data = {
                "project": {"name": "Convenience", "domain": "test", "goal": "test"},
                "milestones": [],
            }

            with open(Path(tmpdir) / "milestones.yaml", "w") as f:
                yaml.dump(milestones_data, f)

            plan = get_project_plan(tmpdir)
            assert plan.name == "Convenience"
