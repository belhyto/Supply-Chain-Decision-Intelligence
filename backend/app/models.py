from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .database import Base

def utc_now():
    return datetime.now(timezone.utc)

class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

class DecisionMaker(Base, TimestampMixin):
    __tablename__ = "decision_makers"
    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    role: Mapped[Optional[str]] = mapped_column(String(150))
    description: Mapped[Optional[str]] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    preferences: Mapped[list["Preference"]] = relationship(back_populates="decision_maker", cascade="all, delete-orphan")
    decision_events: Mapped[list["DecisionEvent"]] = relationship(back_populates="decision_maker")

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
