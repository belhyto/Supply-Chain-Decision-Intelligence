from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint, event, inspect as sa_inspect
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .database import Base

def utc_now():
    return datetime.now(timezone.utc)

class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)
    data_source: Mapped[str] = mapped_column(String(40), default="unknown", server_default="unknown", nullable=False)

class DecisionMaker(Base, TimestampMixin):
    __tablename__ = "decision_makers"
    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    role: Mapped[Optional[str]] = mapped_column(String(150))
    description: Mapped[Optional[str]] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    preferences: Mapped[list["Preference"]] = relationship(back_populates="decision_maker", cascade="all, delete-orphan")
    decision_events: Mapped[list["DecisionEvent"]] = relationship(back_populates="decision_maker")
    rule_reviews: Mapped[list["RuleReview"]] = relationship(back_populates="reviewer")

class Situation(Base, TimestampMixin):
    __tablename__ = "situations"
    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    scenario_type: Mapped[Optional[str]] = mapped_column(String(100))
    description: Mapped[Optional[str]] = mapped_column(Text)
    context_data: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    reported_by: Mapped[Optional[str]] = mapped_column(String(150))
    reported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    alternatives: Mapped[list["Alternative"]] = relationship(back_populates="situation", cascade="all, delete-orphan")
    assessments: Mapped[list["Assessment"]] = relationship(back_populates="situation", cascade="all, delete-orphan")
    decision_events: Mapped[list["DecisionEvent"]] = relationship(back_populates="situation", cascade="all, delete-orphan")
    rule_applications: Mapped[list["RuleApplication"]] = relationship(back_populates="situation")

class Alternative(Base, TimestampMixin):
    __tablename__ = "alternatives"
    __table_args__ = (UniqueConstraint("situation_id", "code", name="uq_alternative_situation_code"), Index("ix_alternatives_situation", "situation_id"))
    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    situation_id: Mapped[str] = mapped_column(ForeignKey("situations.id", ondelete="CASCADE"), nullable=False)
    code: Mapped[str] = mapped_column(String(30), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    attributes: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    is_available: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    situation: Mapped["Situation"] = relationship(back_populates="alternatives")
    assessments: Mapped[list["Assessment"]] = relationship(back_populates="alternative")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="alternative")
    decisions: Mapped[list["Decision"]] = relationship(back_populates="chosen_alternative")

class Preference(Base, TimestampMixin):
    __tablename__ = "preferences"
    __table_args__ = (UniqueConstraint("decision_maker_id", "version", name="uq_preference_person_version"), CheckConstraint("version >= 1", name="ck_preference_version_positive"))
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_maker_id: Mapped[str] = mapped_column(ForeignKey("decision_makers.id", ondelete="CASCADE"), nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    weights: Mapped[dict] = mapped_column(JSON, nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    decision_maker: Mapped["DecisionMaker"] = relationship(back_populates="preferences")
    decision_events: Mapped[list["DecisionEvent"]] = relationship(back_populates="preference")

class Assessment(Base, TimestampMixin):
    __tablename__ = "assessments"
    __table_args__ = (UniqueConstraint("situation_id", "alternative_id", "assessment_version", name="uq_assessment_situation_alternative_version"), CheckConstraint("overall_score >= 0 AND overall_score <= 100", name="ck_assessment_score_range"), Index("ix_assessments_situation", "situation_id"))
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    situation_id: Mapped[str] = mapped_column(ForeignKey("situations.id", ondelete="CASCADE"), nullable=False)
    alternative_id: Mapped[str] = mapped_column(ForeignKey("alternatives.id", ondelete="CASCADE"), nullable=False)
    assessment_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    criterion_scores: Mapped[dict] = mapped_column(JSON, nullable=False)
    criterion_weights: Mapped[dict] = mapped_column(JSON, nullable=False)
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    scoring_method: Mapped[str] = mapped_column(String(100), default="weighted_sum", nullable=False)
    explanation: Mapped[Optional[str]] = mapped_column(Text)
    situation: Mapped["Situation"] = relationship(back_populates="assessments")
    alternative: Mapped["Alternative"] = relationship(back_populates="assessments")

class DecisionEvent(Base, TimestampMixin):
    __tablename__ = "decision_events"
    __table_args__ = (UniqueConstraint("situation_id", "decision_maker_id", "event_number", name="uq_decision_event_person_situation_number"), CheckConstraint("event_number >= 1", name="ck_event_number_positive"), Index("ix_decision_events_situation_person", "situation_id", "decision_maker_id"))
    id: Mapped[str] = mapped_column(String(60), primary_key=True)
    situation_id: Mapped[str] = mapped_column(ForeignKey("situations.id", ondelete="CASCADE"), nullable=False)
    decision_maker_id: Mapped[str] = mapped_column(ForeignKey("decision_makers.id"), nullable=False)
    preference_id: Mapped[Optional[int]] = mapped_column(ForeignKey("preferences.id", ondelete="SET NULL"))
    event_number: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="open", nullable=False)
    event_context: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    situation: Mapped["Situation"] = relationship(back_populates="decision_events")
    decision_maker: Mapped["DecisionMaker"] = relationship(back_populates="decision_events")
    preference: Mapped[Optional["Preference"]] = relationship(back_populates="decision_events")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="decision_event", cascade="all, delete-orphan")
    decision: Mapped[Optional["Decision"]] = relationship(back_populates="decision_event", cascade="all, delete-orphan", uselist=False)
    model_explanation: Mapped[Optional["ModelExplanation"]] = relationship(
        back_populates="decision_event",
        cascade="all, delete-orphan",
        uselist=False,
    )


@event.listens_for(DecisionEvent, "before_update")
def _prevent_decision_snapshot_mutation(mapper, connection, target) -> None:
    history = sa_inspect(target).attrs.event_context.history
    if not history.has_changes() or not history.deleted:
        return
    old_context = history.deleted[0] or {}
    new_context = target.event_context or {}
    for key in ("reasoning_snapshot", "human_reasoning"):
        if key in old_context and old_context[key] != new_context.get(key):
            raise ValueError(f"Decision-event {key} is immutable after recording.")


class Prediction(Base, TimestampMixin):
    __tablename__ = "predictions"
    __table_args__ = (UniqueConstraint("decision_event_id", "alternative_id", "model_version", name="uq_prediction_event_alternative_model"), CheckConstraint("probability >= 0 AND probability <= 1", name="ck_prediction_probability_range"), Index("ix_predictions_event", "decision_event_id"))
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_event_id: Mapped[str] = mapped_column(ForeignKey("decision_events.id", ondelete="CASCADE"), nullable=False)
    alternative_id: Mapped[str] = mapped_column(ForeignKey("alternatives.id", ondelete="CASCADE"), nullable=False)
    utility_score: Mapped[Optional[float]] = mapped_column(Float)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), default="softmax_baseline", nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), default="v1", nullable=False)
    explanation: Mapped[Optional[str]] = mapped_column(Text)
    decision_event: Mapped["DecisionEvent"] = relationship(back_populates="predictions")
    alternative: Mapped["Alternative"] = relationship(back_populates="predictions")

class Decision(Base, TimestampMixin):
    __tablename__ = "decisions"
    __table_args__ = (UniqueConstraint("decision_event_id", name="uq_one_actual_decision_per_event"), Index("ix_decisions_chosen_alternative", "chosen_alternative_id"))
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_event_id: Mapped[str] = mapped_column(ForeignKey("decision_events.id", ondelete="CASCADE"), nullable=False)
    chosen_alternative_id: Mapped[str] = mapped_column(ForeignKey("alternatives.id"), nullable=False)
    rationale: Mapped[Optional[str]] = mapped_column(Text)
    label_source: Mapped[str] = mapped_column(String(40), default="observed_human", nullable=False)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    confidence_note: Mapped[Optional[str]] = mapped_column(Text)
    decision_event: Mapped["DecisionEvent"] = relationship(back_populates="decision")
    chosen_alternative: Mapped["Alternative"] = relationship(back_populates="decisions")
    outcome: Mapped[Optional["Outcome"]] = relationship(back_populates="decision", cascade="all, delete-orphan", uselist=False)
    rule_evidence: Mapped[list["RuleEvidence"]] = relationship(back_populates="decision")
    rule_applications: Mapped[list["RuleApplication"]] = relationship(back_populates="decision")

class Outcome(Base, TimestampMixin):
    __tablename__ = "outcomes"
    __table_args__ = (UniqueConstraint("decision_id", name="uq_one_outcome_per_decision"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_id: Mapped[int] = mapped_column(ForeignKey("decisions.id", ondelete="CASCADE"), nullable=False)
    status: Mapped[str] = mapped_column(String(40), default="pending", nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    actual_observed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    observed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    decision: Mapped["Decision"] = relationship(back_populates="outcome")
    rule_feedback: Mapped[list["RuleFeedback"]] = relationship(back_populates="outcome")

class ModelExplanation(Base, TimestampMixin):
    __tablename__ = "model_explanations"
    __table_args__ = (
        UniqueConstraint("prediction_event_id", name="uq_model_explanation_prediction_event"),
        Index("ix_model_explanations_version", "model_version"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_event_id: Mapped[str] = mapped_column(
        ForeignKey("decision_events.id", ondelete="CASCADE"),
        nullable=False,
    )
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    top_features: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    tree_vote_summary: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    rule_paths: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    recurring_rules: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    decision_event: Mapped["DecisionEvent"] = relationship(back_populates="model_explanation")
    extracted_rules: Mapped[list["ExtractedRule"]] = relationship(
        back_populates="model_explanation",
        cascade="all, delete-orphan",
    )

class ExtractedRule(Base, TimestampMixin):
    __tablename__ = "extracted_rules"
    __table_args__ = (
        Index("ix_extracted_rules_event", "prediction_event_id"),
        Index("ix_extracted_rules_scenario", "scenario_id"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    model_explanation_id: Mapped[int] = mapped_column(
        ForeignKey("model_explanations.id", ondelete="CASCADE"),
        nullable=False,
    )
    prediction_event_id: Mapped[str] = mapped_column(
        ForeignKey("decision_events.id", ondelete="CASCADE"),
        nullable=False,
    )
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    tree_id: Mapped[int] = mapped_column(Integer, nullable=False)
    feature_name: Mapped[str] = mapped_column(String(250), nullable=False)
    condition_operator: Mapped[str] = mapped_column(String(10), nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    branch_taken: Mapped[str] = mapped_column(String(10), nullable=False)
    predicted_choice_id: Mapped[str] = mapped_column(
        ForeignKey("alternatives.id", ondelete="CASCADE"),
        nullable=False,
    )
    scenario_id: Mapped[str] = mapped_column(
        ForeignKey("situations.id", ondelete="CASCADE"),
        nullable=False,
    )
    decision_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("decisions.id", ondelete="SET NULL"),
    )
    tree_vote: Mapped[str] = mapped_column(String(30), nullable=False)
    support_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    support_confidence: Mapped[float] = mapped_column(Float, default=0, nullable=False)
    model_explanation: Mapped["ModelExplanation"] = relationship(back_populates="extracted_rules")


class RuleExtraction(Base, TimestampMixin):
    __tablename__ = "rule_extractions"
    __table_args__ = (
        Index("ix_rule_extractions_dataset", "dataset_version"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(150), nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(100), nullable=False)
    extraction_method: Mapped[str] = mapped_column(String(100), nullable=False)
    extraction_parameters: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="completed", nullable=False)
    rules: Mapped[list["DecisionRule"]] = relationship(back_populates="extraction")


class DecisionRule(Base, TimestampMixin):
    __tablename__ = "decision_rules"
    __table_args__ = (
        UniqueConstraint(
            "version_group",
            "version",
            name="uq_decision_rule_version_group",
        ),
        CheckConstraint("version >= 1", name="ck_decision_rule_version_positive"),
        CheckConstraint(
            "status IN ('PENDING_REVIEW', 'APPROVED_FOR_PUBLICATION', 'APPROVED', "
            "'REJECTED', 'CHANGES_REQUESTED', 'DEPRECATED', 'SUPERSEDED')",
            name="ck_decision_rule_status",
        ),
        Index("ix_decision_rules_status", "status"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    rule_name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    conditions_json: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    observed_pattern: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String(40), default="PENDING_REVIEW", nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    version_group: Mapped[str] = mapped_column(String(80), nullable=False)
    extraction_id: Mapped[str] = mapped_column(
        ForeignKey("rule_extractions.id", ondelete="RESTRICT"),
        nullable=False,
    )
    supersedes_rule_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("decision_rules.id", ondelete="SET NULL"),
    )
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    extraction: Mapped["RuleExtraction"] = relationship(back_populates="rules")
    supersedes: Mapped[Optional["DecisionRule"]] = relationship(remote_side=[id])
    evidences: Mapped[list["RuleEvidence"]] = relationship(
        back_populates="rule",
        cascade="all, delete-orphan",
    )
    reviews: Mapped[list["RuleReview"]] = relationship(
        back_populates="rule",
        cascade="all, delete-orphan",
    )
    applications: Mapped[list["RuleApplication"]] = relationship(
        back_populates="rule",
        cascade="all, delete-orphan",
    )
    feedback: Mapped[list["RuleFeedback"]] = relationship(
        back_populates="rule",
        cascade="all, delete-orphan",
    )


@event.listens_for(DecisionRule, "before_update")
def _prevent_published_rule_mutation(mapper, connection, target) -> None:
    state = sa_inspect(target)
    publication_history = state.attrs.published_at.history
    was_published = (
        target.published_at is not None and not publication_history.has_changes()
    ) or bool(
        publication_history.deleted
        and publication_history.deleted[0] is not None
    )
    if not was_published:
        return
    for attribute in (
        "rule_name",
        "description",
        "conditions_json",
        "observed_pattern",
        "version",
        "version_group",
        "extraction_id",
    ):
        if state.attrs[attribute].history.has_changes():
            raise ValueError("Published decision-rule versions are immutable.")


class RuleEvidence(Base, TimestampMixin):
    __tablename__ = "rule_evidence"
    __table_args__ = (
        UniqueConstraint("rule_id", "decision_id", name="uq_rule_evidence_decision"),
        CheckConstraint(
            "evidence_type IN ('supporting', 'contradictory')",
            name="ck_rule_evidence_type",
        ),
        Index("ix_rule_evidence_decision", "decision_id"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    rule_id: Mapped[str] = mapped_column(
        ForeignKey("decision_rules.id", ondelete="CASCADE"),
        nullable=False,
    )
    decision_id: Mapped[int] = mapped_column(
        ForeignKey("decisions.id", ondelete="RESTRICT"),
        nullable=False,
    )
    evidence_type: Mapped[str] = mapped_column(String(20), nullable=False)
    evidence_description: Mapped[str] = mapped_column(Text, nullable=False)
    rule: Mapped["DecisionRule"] = relationship(back_populates="evidences")
    decision: Mapped["Decision"] = relationship(back_populates="rule_evidence")


class RuleReview(Base, TimestampMixin):
    __tablename__ = "rule_reviews"
    __table_args__ = (
        CheckConstraint(
            "review_stage IN ('evidence', 'publication')",
            name="ck_rule_review_stage",
        ),
        CheckConstraint(
            "review_decision IN ('approved', 'rejected', 'changes_requested', "
            "'request_additional_evidence')",
            name="ck_rule_review_decision",
        ),
        Index("ix_rule_reviews_rule", "rule_id"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    rule_id: Mapped[str] = mapped_column(
        ForeignKey("decision_rules.id", ondelete="CASCADE"),
        nullable=False,
    )
    reviewer_id: Mapped[str] = mapped_column(
        ForeignKey("decision_makers.id", ondelete="RESTRICT"),
        nullable=False,
    )
    reviewer_role: Mapped[str] = mapped_column(String(150), nullable=False)
    review_stage: Mapped[str] = mapped_column(String(30), nullable=False)
    review_decision: Mapped[str] = mapped_column(String(40), nullable=False)
    comments: Mapped[str] = mapped_column(Text, default="", nullable=False)
    modified_conditions: Mapped[Optional[list]] = mapped_column(JSON)
    reviewed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    rule: Mapped["DecisionRule"] = relationship(back_populates="reviews")
    reviewer: Mapped["DecisionMaker"] = relationship(back_populates="rule_reviews")


class RuleApplication(Base, TimestampMixin):
    __tablename__ = "rule_applications"
    __table_args__ = (
        CheckConstraint(
            "relevance_status IN ('not_reviewed', 'relevant', 'not_relevant')",
            name="ck_rule_application_relevance",
        ),
        Index("ix_rule_applications_situation", "situation_id"),
        Index("ix_rule_applications_decision", "decision_id"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    rule_id: Mapped[str] = mapped_column(
        ForeignKey("decision_rules.id", ondelete="RESTRICT"),
        nullable=False,
    )
    situation_id: Mapped[str] = mapped_column(
        ForeignKey("situations.id", ondelete="RESTRICT"),
        nullable=False,
    )
    decision_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("decisions.id", ondelete="SET NULL"),
    )
    relevance_status: Mapped[str] = mapped_column(
        String(20),
        default="not_reviewed",
        nullable=False,
    )
    considered_by_human: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    feedback: Mapped[Optional[str]] = mapped_column(Text)
    rule_version: Mapped[int] = mapped_column(Integer, nullable=False)
    rule: Mapped["DecisionRule"] = relationship(back_populates="applications")
    situation: Mapped["Situation"] = relationship(back_populates="rule_applications")
    decision: Mapped[Optional["Decision"]] = relationship(back_populates="rule_applications")


class RuleFeedback(Base, TimestampMixin):
    __tablename__ = "rule_feedback"
    __table_args__ = (
        CheckConstraint(
            "feedback_type IN ('supporting', 'conflicting', 'inconclusive')",
            name="ck_rule_feedback_type",
        ),
        Index("ix_rule_feedback_outcome", "outcome_id"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    rule_id: Mapped[str] = mapped_column(
        ForeignKey("decision_rules.id", ondelete="RESTRICT"),
        nullable=False,
    )
    outcome_id: Mapped[int] = mapped_column(
        ForeignKey("outcomes.id", ondelete="RESTRICT"),
        nullable=False,
    )
    feedback_type: Mapped[str] = mapped_column(String(20), nullable=False)
    comments: Mapped[str] = mapped_column(Text, default="", nullable=False)
    rule: Mapped["DecisionRule"] = relationship(back_populates="feedback")
    outcome: Mapped["Outcome"] = relationship(back_populates="rule_feedback")


class ContextNode(Base, TimestampMixin):
    __tablename__ = "context_graph_nodes"
    __table_args__ = (
        Index("ix_context_nodes_type", "node_type"),
        Index("ix_context_nodes_synthetic", "is_synthetic"),
    )
    node_key: Mapped[str] = mapped_column(String(180), primary_key=True)
    node_type: Mapped[str] = mapped_column(String(50), nullable=False)
    name: Mapped[str] = mapped_column(String(250), nullable=False)
    properties: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    provenance: Mapped[Optional[str]] = mapped_column(String(500))
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    outgoing_relationships: Mapped[list["ContextRelationship"]] = relationship(
        foreign_keys="ContextRelationship.source_node_key",
        back_populates="source",
        cascade="all, delete-orphan",
    )
    incoming_relationships: Mapped[list["ContextRelationship"]] = relationship(
        foreign_keys="ContextRelationship.target_node_key",
        back_populates="target",
        cascade="all, delete-orphan",
    )

class ContextRelationship(Base):
    __tablename__ = "context_graph_relationships"
    __table_args__ = (
        UniqueConstraint("source_node_key", "target_node_key", "relationship_type", name="uq_context_relationship_edge"),
        Index("ix_context_relationship_type", "relationship_type"),
        Index("ix_context_relationship_target", "target_node_key"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source_node_key: Mapped[str] = mapped_column(
        ForeignKey("context_graph_nodes.node_key", ondelete="CASCADE"),
        nullable=False,
    )
    target_node_key: Mapped[str] = mapped_column(
        ForeignKey("context_graph_nodes.node_key", ondelete="CASCADE"),
        nullable=False,
    )
    relationship_type: Mapped[str] = mapped_column(String(60), nullable=False)
    properties: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    provenance: Mapped[Optional[str]] = mapped_column(String(500))
    data_source: Mapped[str] = mapped_column(String(40), default="unknown", server_default="unknown", nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    source: Mapped["ContextNode"] = relationship(
        foreign_keys=[source_node_key],
        back_populates="outgoing_relationships",
    )
    target: Mapped["ContextNode"] = relationship(
        foreign_keys=[target_node_key],
        back_populates="incoming_relationships",
    )
