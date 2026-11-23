from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.context_graph import add_node, add_relationship, node_key
from app.database import get_db
from app.models import (
    ContextNode,
    Decision,
    DecisionMaker,
    DecisionRule,
    Outcome,
    RuleApplication,
    RuleEvidence,
    RuleExtraction,
    RuleFeedback,
    RuleReview,
    Situation,
)
from app.rule_memory import (
    extract_candidate_rules,
    relevant_rules,
    replace_rule_conditions,
    rule_history,
    serialize_rule,
    serialize_rule_application,
    serialize_rule_evidence,
    serialize_rule_feedback,
    serialize_rule_review,
)

router = APIRouter(prefix="/api/rules", tags=["decision rule memory"])


class RuleExtractionRequest(BaseModel):
    min_matching_decisions: int = Field(default=3, ge=1, le=500)
    min_scenarios: int = Field(default=2, ge=1, le=100)
    min_decision_makers: int = Field(default=2, ge=1, le=100)
    max_rules: int = Field(default=25, ge=1, le=100)


class RuleReviewRequest(BaseModel):
    reviewer_id: str = Field(min_length=1, max_length=50)
    review_stage: Literal["evidence", "publication"]
    review_decision: Literal[
        "approved",
        "rejected",
        "changes_requested",
        "request_additional_evidence",
    ]
    comments: str = Field(default="", max_length=2000)
    modified_conditions: list[dict[str, Any]] | None = Field(default=None, max_length=40)


class RuleFeedbackRequest(BaseModel):
    scenario_id: str = Field(min_length=1, max_length=50)
    decision_id: str | None = Field(default=None, max_length=80)
    outcome_id: int | None = Field(default=None, gt=0)
    relevance_status: Literal["not_reviewed", "relevant", "not_relevant"] = "not_reviewed"
    considered_by_human: bool = False
    feedback: str = Field(default="", max_length=2000)
    feedback_type: Literal["supporting", "conflicting", "inconclusive"] | None = None


def _get_rule(session: Session, rule_id: str) -> DecisionRule:
    rule = session.get(DecisionRule, rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail=f"Unknown decision rule: {rule_id}")
    return rule


def _reviewer(session: Session, reviewer_id: str, stage: str) -> DecisionMaker:
    reviewer = session.get(DecisionMaker, reviewer_id)
    if reviewer is None or not reviewer.is_active or not reviewer.role:
        raise HTTPException(status_code=403, detail="An active reviewer profile is required.")
    role = f"{reviewer.role} {reviewer.name}".casefold()
    if stage == "evidence" and not any(
        marker in role for marker in ("risk", "director", "review")
    ):
        raise HTTPException(
            status_code=403,
            detail="This reviewer profile is not authorized for evidence review.",
        )
    if stage == "publication" and not any(
        marker in role for marker in ("director", "publication approver")
    ):
        raise HTTPException(
            status_code=403,
            detail="This reviewer profile is not authorized for publication approval.",
        )
    return reviewer


def _validate_rule_evidence(rule: DecisionRule) -> None:
    thresholds = rule.extraction.extraction_parameters
    pattern = rule.observed_pattern
    if (
        pattern.get("matching_decisions", 0)
        < thresholds["minimum_matching_decisions"]
        or pattern.get("distinct_scenarios", 0) < thresholds["minimum_scenarios"]
        or pattern.get("distinct_decision_makers", 0)
        < thresholds["minimum_decision_makers"]
    ):
        raise HTTPException(
            status_code=422,
            detail="Rule evidence no longer meets the extraction's configured thresholds.",
        )


def _rule_detail(rule: DecisionRule) -> dict[str, Any]:
    return {
        **serialize_rule(rule),
        "evidence": [
            serialize_rule_evidence(item)
            for item in sorted(rule.evidences, key=lambda evidence: evidence.evidence_type)
        ],
        "reviews": [
            serialize_rule_review(item)
            for item in sorted(rule.reviews, key=lambda review: review.reviewed_at)
        ],
        "applications": [
            serialize_rule_application(item)
            for item in rule.applications
        ],
        "feedback": [
            serialize_rule_feedback(item)
            for item in rule.feedback
        ],
    }


@router.post("/extract", status_code=201)
def start_rule_extraction(
    request: RuleExtractionRequest,
    db: Session = Depends(get_db),
):
    try:
        result = extract_candidate_rules(
            db,
            min_matching_decisions=request.min_matching_decisions,
            min_scenarios=request.min_scenarios,
            min_decision_makers=request.min_decision_makers,
            max_rules=request.max_rules,
        )
        db.commit()
        return result
    except ValueError as error:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception:
        db.rollback()
        raise


@router.get("/candidates")
def get_candidate_rules(db: Session = Depends(get_db)):
    rules = db.scalars(
        select(DecisionRule)
        .where(DecisionRule.status != "APPROVED")
        .order_by(DecisionRule.created_at.desc(), DecisionRule.rule_name)
    ).all()
    return {"rules": [serialize_rule(rule) for rule in rules]}


@router.get("/approved")
def get_approved_rules(db: Session = Depends(get_db)):
    rules = db.scalars(
        select(DecisionRule)
        .where(
            DecisionRule.status == "APPROVED",
            DecisionRule.published_at.is_not(None),
        )
        .order_by(DecisionRule.rule_name, DecisionRule.version.desc())
    ).all()
    return {"rules": [serialize_rule(rule) for rule in rules]}


@router.get("/relevant/{scenario_id}")
def get_relevant_rules(
    scenario_id: str,
    decision_maker_id: str = Query(min_length=1),
    db: Session = Depends(get_db),
):
    if db.get(Situation, scenario_id) is None:
        raise HTTPException(status_code=404, detail="Unknown situation.")
    decision_maker = db.get(DecisionMaker, decision_maker_id)
    if decision_maker is None or not decision_maker.is_active:
        raise HTTPException(status_code=404, detail="Unknown active decision-maker.")
    return {
        "scenario_id": scenario_id,
        "decision_maker_id": decision_maker_id,
        "rules": relevant_rules(db, scenario_id, decision_maker_id),
        "disclaimer": "Retrieved rules are reviewed historical patterns, not recommendations.",
    }


@router.get("/{rule_id}/evidence")
def get_rule_evidence(rule_id: str, db: Session = Depends(get_db)):
    rule = _get_rule(db, rule_id)
    return {
        "rule_id": rule.id,
        "evidence": [
            serialize_rule_evidence(item)
            for item in rule.evidences
        ],
    }


@router.get("/{rule_id}/history")
def get_rule_history(rule_id: str, db: Session = Depends(get_db)):
    rule = _get_rule(db, rule_id)
    return {"rule_id": rule.id, "versions": rule_history(rule)}


@router.get("/{rule_id}")
def get_rule(rule_id: str, db: Session = Depends(get_db)):
    return _rule_detail(_get_rule(db, rule_id))


@router.post("/{rule_id}/review")
def review_rule(
    rule_id: str,
    request: RuleReviewRequest,
    db: Session = Depends(get_db),
):
    rule = _get_rule(db, rule_id)
    reviewer = _reviewer(db, request.reviewer_id, request.review_stage)
    if rule.published_at is not None or rule.status in {
        "APPROVED",
        "REJECTED",
        "DEPRECATED",
        "SUPERSEDED",
    }:
        raise HTTPException(status_code=409, detail="This rule version is immutable.")
    if request.review_stage == "evidence":
        if rule.status not in {"PENDING_REVIEW", "CHANGES_REQUESTED"}:
            raise HTTPException(
                status_code=409,
                detail="Evidence review is not valid for this rule state.",
            )
        if request.modified_conditions is not None:
            try:
                replace_rule_conditions(db, rule, request.modified_conditions)
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
        if request.review_decision == "approved":
            _validate_rule_evidence(rule)
    else:
        if rule.status != "APPROVED_FOR_PUBLICATION":
            raise HTTPException(
                status_code=409,
                detail="The rule must pass evidence review before publication review.",
            )
        if request.modified_conditions is not None:
            raise HTTPException(
                status_code=422,
                detail="Conditions must be edited during evidence review.",
            )

    review = RuleReview(
        id=str(uuid4()),
        rule_id=rule.id,
        reviewer_id=reviewer.id,
        reviewer_role=reviewer.role or "",
        review_stage=request.review_stage,
        review_decision=request.review_decision,
        comments=request.comments.strip(),
        modified_conditions=(
            rule.conditions_json
            if request.modified_conditions is not None
            else None
        ),
        reviewed_at=datetime.now(timezone.utc),
        data_source="observed_human",
    )
    db.add(review)
    if request.review_decision == "rejected":
        rule.status = "REJECTED"
    elif request.review_decision in {"changes_requested", "request_additional_evidence"}:
        rule.status = "CHANGES_REQUESTED"
    elif request.review_stage == "evidence":
        rule.status = "APPROVED_FOR_PUBLICATION"
    else:
        rule.status = "APPROVED"
    db.commit()
    return {
        "rule": serialize_rule(rule),
        "review": serialize_rule_review(review),
    }


@router.post("/{rule_id}/publish")
def publish_rule(rule_id: str, db: Session = Depends(get_db)):
    rule = _get_rule(db, rule_id)
    if rule.status != "APPROVED" or rule.published_at is not None:
        raise HTTPException(
            status_code=409,
            detail="Only publication-approved, unpublished rules can be published.",
        )
    publication_approval = next(
        iter(sorted(
            (
                review for review in rule.reviews
                if review.review_stage == "publication"
                and review.review_decision == "approved"
            ),
            key=lambda review: review.reviewed_at,
            reverse=True,
        )),
        None,
    )
    if publication_approval is None:
        raise HTTPException(
            status_code=409,
            detail="A recorded publication approval is required.",
        )
    extraction = rule.extraction
    rule.published_at = datetime.now(timezone.utc)
    graph_rule_key = node_key("DecisionRule", f"{rule.id}:v{rule.version}")
    add_node(
        db,
        graph_rule_key,
        "DecisionRule",
        rule.rule_name,
        {
            "rule_id": rule.id,
            "rule_name": rule.rule_name,
            "description": rule.description,
            "conditions": rule.conditions_json,
            "observed_pattern": rule.observed_pattern,
            "status": "APPROVED",
            "version": rule.version,
            "model_name": extraction.model_name,
            "model_version": extraction.model_version,
            "dataset_version": extraction.dataset_version,
            "reviewer_id": publication_approval.reviewer_id,
            "reviewed_at": publication_approval.reviewed_at.isoformat(),
            "published_at": rule.published_at.isoformat(),
        },
        f"model:{extraction.model_version}",
    )
    extraction_key = node_key("RuleExtraction", extraction.id)
    add_node(
        db,
        extraction_key,
        "RuleExtraction",
        f"Random Forest extraction {extraction.id[:8]}",
        {
            "model_name": extraction.model_name,
            "model_version": extraction.model_version,
            "dataset_version": extraction.dataset_version,
            "method": extraction.extraction_method,
        },
        f"model:{extraction.model_version}",
    )
    add_relationship(
        db,
        graph_rule_key,
        extraction_key,
        "EXTRACTED_FROM",
        {"extraction_id": extraction.id},
        f"model:{extraction.model_version}",
    )
    reviewer_key = node_key("DecisionMaker", publication_approval.reviewer_id)
    if db.get(ContextNode, reviewer_key) is None:
        reviewer = publication_approval.reviewer
        add_node(
            db,
            reviewer_key,
            "DecisionMaker",
            reviewer.name,
            {"role": reviewer.role, "decision_maker_id": reviewer.id},
            "observed_human:rule-review",
        )
    add_relationship(
        db,
        graph_rule_key,
        reviewer_key,
        "APPROVED_BY",
        {
            "review_id": publication_approval.id,
            "review_stage": publication_approval.review_stage,
        },
        "observed_human:rule-publication",
    )
    for review in rule.reviews:
        review_reviewer_key = node_key("DecisionMaker", review.reviewer_id)
        if db.get(ContextNode, review_reviewer_key) is None:
            add_node(
                db,
                review_reviewer_key,
                "DecisionMaker",
                review.reviewer.name,
                {
                    "role": review.reviewer_role,
                    "decision_maker_id": review.reviewer_id,
                },
                "observed_human:rule-review",
            )
        review_properties = {
            "review_id": review.id,
            "review_stage": review.review_stage,
            "review_decision": review.review_decision,
            "reviewed_at": review.reviewed_at.isoformat(),
            "reviewer_role": review.reviewer_role,
            "comments": review.comments,
            "modified_conditions": review.modified_conditions,
        }
        review_key = node_key("RuleReview", review.id)
        add_node(
            db,
            review_key,
            "RuleReview",
            f"{review.review_stage.title()} review · {review.review_decision}",
            review_properties,
            "observed_human:rule-review",
        )
        add_relationship(
            db,
            graph_rule_key,
            review_key,
            "HAS_REVIEW",
            provenance="observed_human:rule-review",
        )
        add_relationship(
            db,
            review_key,
            review_reviewer_key,
            "REVIEWED_BY",
            review_properties,
            "observed_human:rule-review",
        )
        if (
            review.review_stage == "publication"
            and review.review_decision == "approved"
        ):
            add_relationship(
                db,
                graph_rule_key,
                review_reviewer_key,
                "APPROVED_BY",
                review_properties,
                "observed_human:rule-publication",
            )

    for evidence in rule.evidences:
        decision = evidence.decision
        public_decision_id = decision.confidence_note or f"DEC-{decision.id}"
        decision_key = node_key("Decision", public_decision_id)
        if db.get(ContextNode, decision_key) is None:
            add_node(
                db,
                decision_key,
                "Decision",
                f"{decision.decision_event.decision_maker.name} selected "
                f"{decision.chosen_alternative.name}",
                {
                    "decision_id": public_decision_id,
                    "rationale": decision.rationale,
                    "rationale_source": "human_stated",
                    "source": decision.data_source,
                },
                "observed_human:rule-evidence",
            )
        relationship_type = (
            "SUPPORTED_BY"
            if evidence.evidence_type == "supporting"
            else "CONTRADICTED_BY"
        )
        add_relationship(
            db,
            graph_rule_key,
            decision_key,
            relationship_type,
            {
                "evidence_id": evidence.id,
                "evidence_type": evidence.evidence_type,
                "description": evidence.evidence_description,
            },
            "model:rule-evidence",
        )
        situation_key = node_key("Situation", decision.decision_event.situation_id)
        if db.get(ContextNode, situation_key) is not None:
            add_relationship(
                db,
                graph_rule_key,
                situation_key,
                "APPLIES_TO",
                {"evidence_type": evidence.evidence_type},
                "model:rule-extraction",
            )

    if rule.supersedes_rule_id:
        previous = db.get(DecisionRule, rule.supersedes_rule_id)
        if previous is not None and previous.published_at is not None:
            previous.status = "SUPERSEDED"
            previous_key = node_key(
                "DecisionRule",
                f"{previous.id}:v{previous.version}",
            )
            previous_node = db.get(ContextNode, previous_key)
            if previous_node is not None:
                previous_node.properties = {
                    **previous_node.properties,
                    "status": "SUPERSEDED",
                }
            add_relationship(
                db,
                graph_rule_key,
                previous_key,
                "SUPERSEDES",
                {"previous_version": previous.version},
                "observed_human:rule-publication",
            )
    db.commit()
    return {"rule": serialize_rule(rule), "graph_node_id": graph_rule_key}


@router.post("/{rule_id}/feedback")
def record_rule_feedback(
    rule_id: str,
    request: RuleFeedbackRequest,
    db: Session = Depends(get_db),
):
    rule = _get_rule(db, rule_id)
    if rule.status != "APPROVED" or rule.published_at is None:
        raise HTTPException(
            status_code=409,
            detail="Feedback can only be recorded for a published rule.",
        )
    scenario = db.get(Situation, request.scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail="Unknown situation.")
    decision = None
    if request.decision_id:
        decision = db.scalar(
            select(Decision)
            .where(Decision.confidence_note == request.decision_id)
        )
        if decision is None:
            raise HTTPException(status_code=404, detail="Unknown recorded decision.")
        if decision.decision_event.situation_id != scenario.id:
            raise HTTPException(
                status_code=422,
                detail="The decision does not belong to the supplied situation.",
            )
    application = RuleApplication(
        id=str(uuid4()),
        rule_id=rule.id,
        situation_id=scenario.id,
        decision_id=decision.id if decision else None,
        relevance_status=request.relevance_status,
        considered_by_human=request.considered_by_human,
        feedback=request.feedback.strip() or None,
        rule_version=rule.version,
        data_source="observed_human",
    )
    db.add(application)

    outcome = None
    if request.outcome_id is not None:
        outcome = db.get(Outcome, request.outcome_id)
        if outcome is None:
            raise HTTPException(status_code=404, detail="Unknown decision outcome.")
        if decision is not None and outcome.decision_id != decision.id:
            raise HTTPException(
                status_code=422,
                detail="The outcome does not belong to the supplied decision.",
            )
    if request.feedback_type is not None:
        if outcome is None:
            raise HTTPException(
                status_code=422,
                detail="Outcome-linked feedback requires an outcome_id.",
            )
        rule_feedback = RuleFeedback(
            id=str(uuid4()),
            rule_id=rule.id,
            outcome_id=outcome.id,
            feedback_type=request.feedback_type,
            comments=request.feedback.strip(),
            data_source="observed_human",
        )
        db.add(rule_feedback)
        feedback_key = node_key("RuleFeedback", rule_feedback.id)
        add_node(
            db,
            feedback_key,
            "RuleFeedback",
            f"{request.feedback_type.title()} feedback for {rule.rule_name}",
            {
                "rule_id": rule.id,
                "outcome_id": outcome.id,
                "feedback_type": request.feedback_type,
                "comments": request.feedback.strip(),
            },
            "observed_human:rule-feedback",
        )
        outcome_key = node_key(
            "Outcome",
            outcome.decision.confidence_note or f"DEC-{outcome.decision_id}",
        )
        rule_key = node_key("DecisionRule", f"{rule.id}:v{rule.version}")
        add_relationship(
            db,
            rule_key,
            feedback_key,
            "HAS_RULE_FEEDBACK",
            provenance="observed_human:rule-feedback",
        )
        if db.get(ContextNode, outcome_key):
            add_relationship(
                db,
                feedback_key,
                outcome_key,
                "EVIDENCED_BY",
                provenance="observed_human:rule-feedback",
            )

    rule_key = node_key("DecisionRule", f"{rule.id}:v{rule.version}")
    application_key = node_key("RuleApplication", application.id)
    situation_key = node_key("Situation", scenario.id)
    add_node(
        db,
        application_key,
        "RuleApplication",
        f"{rule.rule_name} applied to {scenario.title}",
        {
            "rule_id": rule.id,
            "rule_version": rule.version,
            "situation_id": scenario.id,
            "decision_id": request.decision_id,
            "relevance_status": request.relevance_status,
            "considered_by_human": request.considered_by_human,
            "feedback": request.feedback.strip(),
        },
        "observed_human:rule-application",
    )
    add_relationship(
        db,
        rule_key,
        application_key,
        "HAS_APPLICATION",
        provenance="observed_human:rule-application",
    )
    add_relationship(
        db,
        application_key,
        situation_key,
        "RETRIEVED_FOR",
        provenance="observed_human:rule-application",
    )
    if decision is not None:
        decision_key = node_key(
            "Decision",
            decision.confidence_note or f"DEC-{decision.id}",
        )
        add_relationship(
            db,
            application_key,
            decision_key,
            "CONSIDERED",
            provenance="observed_human:rule-application",
        )
    db.commit()
    return {
        "application": serialize_rule_application(application),
        "feedback": (
            serialize_rule_feedback(rule_feedback)
            if request.feedback_type is not None
            else None
        ),
    }

