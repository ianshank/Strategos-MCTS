"""
Tactical Data Augmentation Module.

Provides domain-specific data augmentation techniques for:
- Cybersecurity threat scenarios
- Military tactical situations
- Multi-step reasoning problems

These augmentations help increase training data diversity and improve
model robustness for tactical analysis tasks.
"""

import random
from dataclasses import dataclass

from src.observability.logging import get_logger

from .dataset_loader import DatasetSample

logger = get_logger(__name__)


@dataclass
class AugmentationResult:
    """Result of data augmentation."""

    original: DatasetSample
    augmented: list[DatasetSample]
    augmentation_types: list[str]


class TacticalAugmenter:
    """
    Domain-specific data augmentation for tactical analysis.

    Augmentation techniques:
    - Paraphrasing tactical scenarios
    - Varying urgency levels
    - Adding/removing constraints
    - Scenario parameter variation
    - Threat actor substitution
    - Temporal shifting
    """

    # Tactical scenario templates
    URGENCY_MODIFIERS = {
        "high": ["IMMEDIATE", "CRITICAL", "URGENT", "TIME-SENSITIVE"],
        "medium": ["PRIORITY", "IMPORTANT", "ATTENTION REQUIRED"],
        "low": ["ROUTINE", "STANDARD", "WHEN POSSIBLE"],
    }

    THREAT_ACTORS = [
        "APT28",
        "APT29",
        "Lazarus Group",
        "Cozy Bear",
        "Fancy Bear",
        "Unknown Actor",
        "Nation-State Actor",
        "Criminal Organization",
    ]

    ATTACK_VECTORS = [
        "phishing",
        "spear-phishing",
        "watering hole",
        "supply chain compromise",
        "zero-day exploit",
        "credential stuffing",
        "brute force",
        "social engineering",
    ]

    MILITARY_OBJECTIVES = [
        "secure perimeter",
        "establish forward position",
        "conduct reconnaissance",
        "neutralize threat",
        "protect assets",
        "maintain operational security",
        "coordinate with allied forces",
        "execute tactical withdrawal",
    ]

    ENVIRONMENTAL_CONDITIONS = [
        "night operations",
        "adverse weather",
        "limited visibility",
        "urban terrain",
        "mountainous region",
        "coastal area",
        "contested airspace",
        "electronic warfare environment",
    ]

    def __init__(self, seed: int = 42):
        """
        Initialize augmenter.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)
        self._augmentation_count = 0

    def augment_sample(
        self,
        sample: DatasetSample,
        num_augmentations: int = 3,
        techniques: list[str] | None = None,
    ) -> AugmentationResult:
        """
        Augment a single sample.

        Args:
            sample: Original dataset sample
            num_augmentations: Number of augmented versions to create
            techniques: Specific techniques to use (None for random selection)

        Returns:
            AugmentationResult with augmented samples
        """
        available_techniques = [
            "urgency_variation",
            "parameter_substitution",
            "constraint_addition",
            "temporal_shift",
            "perspective_change",
        ]

        if techniques:
            available_techniques = [t for t in techniques if t in available_techniques]

        augmented_samples = []
        used_techniques = []

        for _i in range(num_augmentations):
            technique = self.rng.choice(available_techniques)
            used_techniques.append(technique)

            augmented_text = self._apply_technique(sample.text, sample.domain, technique)

            aug_sample = DatasetSample(
                id=f"{sample.id}_aug_{self._augmentation_count}",
                text=augmented_text,
                metadata={
                    **sample.metadata,
                    "augmentation": technique,
                    "original_id": sample.id,
                },
                labels=sample.labels,
                difficulty=sample.difficulty,
                domain=sample.domain,
                reasoning_steps=sample.reasoning_steps,
            )

            augmented_samples.append(aug_sample)
            self._augmentation_count += 1

        return AugmentationResult(
            original=sample,
            augmented=augmented_samples,
            augmentation_types=used_techniques,
        )

    def _apply_technique(self, text: str, domain: str | None, technique: str) -> str:
        """Apply specific augmentation technique."""
        if technique == "urgency_variation":
            return self._augment_urgency(text)
        elif technique == "parameter_substitution":
            return self._augment_parameters(text, domain)
        elif technique == "constraint_addition":
            return self._augment_constraints(text, domain)
        elif technique == "temporal_shift":
            return self._augment_temporal(text)
        elif technique == "perspective_change":
            return self._augment_perspective(text, domain)
        else:
            return text

    def _augment_urgency(self, text: str) -> str:
        """Vary urgency level in the text."""
        urgency_level = self.rng.choice(list(self.URGENCY_MODIFIERS.keys()))
        modifier = self.rng.choice(self.URGENCY_MODIFIERS[urgency_level])

        # Add urgency prefix
        if urgency_level == "high":
            return f"[{modifier}] {text}"
        elif urgency_level == "medium":
            return f"{modifier}: {text}"
        else:
            return f"({modifier}) {text}"

    def _augment_parameters(self, text: str, domain: str | None) -> str:
        """Substitute domain-specific parameters."""
        if domain == "cybersecurity" or "cyber" in text.lower():
            # Substitute threat actors
            for actor in self.THREAT_ACTORS:
                if actor in text:
                    new_actor = self.rng.choice([a for a in self.THREAT_ACTORS if a != actor])
                    text = text.replace(actor, new_actor)
                    break

            # Substitute attack vectors
            for vector in self.ATTACK_VECTORS:
                if vector in text.lower():
                    new_vector = self.rng.choice([v for v in self.ATTACK_VECTORS if v != vector])
                    text = text.replace(vector, new_vector)
                    break

        elif domain == "military" or any(kw in text.lower() for kw in ["tactical", "military", "reconnaissance"]):
            # Substitute objectives
            for obj in self.MILITARY_OBJECTIVES:
                if obj in text.lower():
                    new_obj = self.rng.choice([o for o in self.MILITARY_OBJECTIVES if o != obj])
                    text = text.replace(obj, new_obj)
                    break

        return text

    def _augment_constraints(self, text: str, domain: str | None) -> str:
        """Add additional constraints to the scenario."""
        constraints = []

        if domain == "cybersecurity":
            constraints = [
                "with limited network visibility",
                "under active attack",
                "with compromised credentials",
                "during maintenance window",
                "with restricted access to logs",
            ]
        elif domain == "military":
            constraints = [
                "with limited ammunition",
                "under communication blackout",
                "with reduced personnel",
                "in contested environment",
                "with time constraint of 2 hours",
            ]
        else:
            constraints = [
                "with incomplete information",
                "under time pressure",
                "with resource constraints",
                "considering multiple stakeholders",
                "with conflicting objectives",
            ]

        if constraints:
            constraint = self.rng.choice(constraints)
            return f"{text} [{constraint}]"

        return text

    def _augment_temporal(self, text: str) -> str:
        """Shift temporal context."""
        temporal_contexts = [
            "In the past 24 hours, ",
            "Over the next week, ",
            "Immediately, ",
            "During the upcoming operation, ",
            "Following initial assessment, ",
        ]

        context = self.rng.choice(temporal_contexts)
        return f"{context}{text.lower()}" if text else text

    def _augment_perspective(self, text: str, domain: str | None) -> str:
        """Change analytical perspective."""
        perspectives = {
            "cybersecurity": [
                "From a threat hunter's perspective: ",
                "Considering the attacker's viewpoint: ",
                "For incident response purposes: ",
                "From a risk management standpoint: ",
            ],
            "military": [
                "From the commander's perspective: ",
                "Considering enemy capabilities: ",
                "For tactical planning purposes: ",
                "From a logistics standpoint: ",
            ],
            "default": [
                "From an analytical perspective: ",
                "Considering all factors: ",
                "For decision-making purposes: ",
                "From a strategic viewpoint: ",
            ],
        }

        domain_perspectives = perspectives.get(domain or "default", perspectives["default"])
        perspective = self.rng.choice(domain_perspectives)

        return f"{perspective}{text}"

    def augment_batch(
        self,
        samples: list[DatasetSample],
        augmentations_per_sample: int = 2,
    ) -> list[DatasetSample]:
        """
        Augment a batch of samples.

        Args:
            samples: List of original samples
            augmentations_per_sample: Number of augmentations per sample

        Returns:
            List of all samples (original + augmented)
        """
        all_samples = list(samples)  # Keep originals

        for sample in samples:
            result = self.augment_sample(sample, num_augmentations=augmentations_per_sample)
            all_samples.extend(result.augmented)

        logger.info(
            f"Augmented {len(samples)} samples to {len(all_samples)} (+{len(all_samples) - len(samples)} augmented)"
        )

        return all_samples

    def create_tactical_scenarios(self, base_samples: list[DatasetSample]) -> list[DatasetSample]:
        """
        Create tactical scenario variations from base samples.

        Combines multiple augmentation techniques to create
        diverse tactical scenarios for training.

        Args:
            base_samples: Base dataset samples

        Returns:
            Extended list with tactical scenario variations
        """
        scenarios = list(base_samples)

        for sample in base_samples:
            # Create high-stakes variant
            high_stakes = self._augment_urgency(sample.text)
            high_stakes = self._augment_constraints(high_stakes, sample.domain)
            scenarios.append(
                DatasetSample(
                    id=f"{sample.id}_highstakes_{self._augmentation_count}",
                    text=high_stakes,
                    metadata={
                        **sample.metadata,
                        "scenario_type": "high_stakes",
                        "original_id": sample.id,
                    },
                    labels=sample.labels,
                    difficulty="hard",  # High stakes scenarios are harder
                    domain=sample.domain,
                    reasoning_steps=sample.reasoning_steps,
                )
            )
            self._augmentation_count += 1

            # Create multi-perspective variant
            if self.rng.random() > 0.5:
                multi_perspective = self._augment_perspective(sample.text, sample.domain)
                scenarios.append(
                    DatasetSample(
                        id=f"{sample.id}_multiperspective_{self._augmentation_count}",
                        text=multi_perspective,
                        metadata={
                            **sample.metadata,
                            "scenario_type": "multi_perspective",
                            "original_id": sample.id,
                        },
                        labels=sample.labels,
                        difficulty=sample.difficulty,
                        domain=sample.domain,
                        reasoning_steps=sample.reasoning_steps,
                    )
                )
                self._augmentation_count += 1

        logger.info(f"Created {len(scenarios) - len(base_samples)} tactical scenarios")
        return scenarios


class CyberSecurityAugmenter(TacticalAugmenter):
    """
    Specialized augmenter for cybersecurity scenarios.

    Focuses on:
    - MITRE ATT&CK technique variations
    - Threat intelligence context
    - Incident response scenarios
    """

    MITRE_TACTICS = [
        "Initial Access",
        "Execution",
        "Persistence",
        "Privilege Escalation",
        "Defense Evasion",
        "Credential Access",
        "Discovery",
        "Lateral Movement",
        "Collection",
        "Exfiltration",
        "Impact",
    ]

    SEVERITY_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def augment_with_mitre_context(self, sample: DatasetSample) -> DatasetSample:
        """
        Add MITRE ATT&CK context to sample.

        Args:
            sample: Original sample

        Returns:
            Augmented sample with MITRE context
        """
        tactic = self.rng.choice(self.MITRE_TACTICS)
        severity = self.rng.choice(self.SEVERITY_LEVELS)

        augmented_text = f"[MITRE ATT&CK: {tactic}] [Severity: {severity}] {sample.text}"

        return DatasetSample(
            id=f"{sample.id}_mitre_{self._augmentation_count}",
            text=augmented_text,
            metadata={
                **sample.metadata,
                "mitre_tactic": tactic,
                "severity": severity,
            },
            labels=sample.labels,
            difficulty=sample.difficulty,
            domain="cybersecurity",
            reasoning_steps=sample.reasoning_steps,
        )


class MilitaryTacticalAugmenter(TacticalAugmenter):
    """
    Specialized augmenter for military tactical scenarios.

    Focuses on:
    - Environmental condition variations
    - Force composition changes
    - Mission objective variations
    """

    FORCE_COMPOSITIONS = [
        "infantry platoon",
        "mechanized company",
        "special operations team",
        "combined arms battalion",
        "air assault element",
    ]

    def augment_with_force_composition(self, sample: DatasetSample) -> DatasetSample:
        """
        Add force composition context to sample.

        Args:
            sample: Original sample

        Returns:
            Augmented sample with force composition
        """
        force = self.rng.choice(self.FORCE_COMPOSITIONS)
        condition = self.rng.choice(self.ENVIRONMENTAL_CONDITIONS)

        augmented_text = f"[Force: {force}] [Conditions: {condition}] {sample.text}"

        return DatasetSample(
            id=f"{sample.id}_tactical_{self._augmentation_count}",
            text=augmented_text,
            metadata={
                **sample.metadata,
                "force_composition": force,
                "environmental_conditions": condition,
            },
            labels=sample.labels,
            difficulty=sample.difficulty,
            domain="military",
            reasoning_steps=sample.reasoning_steps,
        )
