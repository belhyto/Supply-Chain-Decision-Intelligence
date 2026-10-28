import unittest
from copy import deepcopy
from unittest.mock import patch

from fastapi import HTTPException

from app import main


class AssessmentTests(unittest.TestCase):
    def test_default_supply_chain_assessment_preserves_rankings(self):
        result = main.calculate_assessment(main.AssessRequest())

        self.assertEqual(result["stage1"]["top_option"], "RapidRoute Industrial")
        self.assertEqual(len(result["stage2"]), 4)
        for person in result["stage3"]["decision_maker_probabilities"]:
            self.assertAlmostEqual(
                sum(option["probability_percent"] for option in person["probabilities"]),
                100,
                delta=0.1,
            )

    def test_assessment_supports_another_domain_and_criteria_set(self):
        config = {
            "domain_name": "Maritime Operations",
            "scenario": {"id": "MAR-1", "title": "Port congestion", "situation": "A berth is delayed."},
            "criteria": [
                {"key": "safety", "label": "Safety"},
                {"key": "throughput", "label": "Throughput"},
            ],
            "operational_weights": {"safety": 0.8, "throughput": 0.2},
            "options": [
                {"id": "A", "name": "Harbor A", "metrics": {"safety": 90, "throughput": 60}},
                {"id": "B", "name": "Harbor B", "metrics": {"safety": 70, "throughput": 95}},
            ],
            "decision_makers": [
                {
                    "id": "DM-1",
                    "name": "Port Director",
                    "preferences": {"safety": 0.2, "throughput": 0.8},
                }
            ],
            "digital_twin": {
                "state": {
                    "revision": 1,
                    "suppliers": [
                        {
                            "id": "A",
                            "capacity_units": 100,
                            "is_available": True,
                            "metrics": {"safety": 90, "throughput": 60},
                        },
                        {
                            "id": "B",
                            "capacity_units": 100,
                            "is_available": True,
                            "metrics": {"safety": 70, "throughput": 95},
                        },
                    ],
                    "inventory": [],
                    "shipments": [],
                    "orders": [],
                }
            },
        }
        with (
            patch.object(main, "CONFIG", config),
            patch.object(main, "TWIN_STATE", deepcopy(config["digital_twin"]["state"])),
            patch.object(main, "CRITERIA", config["criteria"]),
            patch.object(main, "CRITERION_ORDER", ["safety", "throughput"]),
            patch.object(main, "CRITERION_KEYS", {"safety", "throughput"}),
        ):
            result = main.calculate_assessment(main.AssessRequest())

        self.assertEqual(result["domain_name"], "Maritime Operations")
        self.assertEqual(result["stage1"]["top_option"], "Harbor A")
        self.assertEqual(result["stage2"][0]["top_choice"], "Harbor B")
        self.assertEqual(len(result["stage3"]["decision_maker_probabilities"][0]["probabilities"]), 2)

    def test_invalid_weight_keys_return_a_client_error(self):
        with self.assertRaises(HTTPException) as error:
            main.calculate_assessment(
                main.AssessRequest(operational_weights={"not-a-criterion": 1})
            )

        self.assertEqual(error.exception.status_code, 422)

    def test_synthetic_dataset_has_expected_scenarios_decisions_and_profiles(self):
        self.assertEqual(len(main.TRAINING_DATA["scenarios"]), 8)
        self.assertEqual(len(main.TRAINING_DATA["profiles"]), 4)
        self.assertEqual(len(main.TRAINING_DATA["decisions"]), 32)
        self.assertTrue(all(len(item["options"]) == 3 for item in main.TRAINING_DATA["scenarios"].values()))
        self.assertTrue(all("outcome" in item for item in main.TRAINING_DATA["decisions"]))

    def test_training_evaluation_keeps_model_prediction_and_recorded_choice_distinct(self):
        result = main.evaluate_training_data()

        self.assertEqual(result["sample_count"], 32)
        self.assertEqual(len(result["records"]), 32)
        self.assertIn("predicted_choice", result["records"][0])
        self.assertIn("recorded_choice", result["records"][0])
        self.assertIn("outcome", result["records"][0])
        self.assertIn("brier_score", result["records"][0])
        scenario_one = next(item for item in result["scenario_disagreement"] if item["scenario_id"] == "SCN-001")
        self.assertEqual(scenario_one["decision_count"], 4)
        self.assertGreater(scenario_one["distinct_recorded_choices"], 1)
        self.assertGreater(scenario_one["pairwise_disagreement_count"], 0)

    def test_all_recorded_choices_are_available_in_their_scenario(self):
        for scenario_id, scenario in main.TRAINING_DATA["scenarios"].items():
            assessment = main.calculate_assessment(main.AssessRequest(scenario_id=scenario_id))
            available_ids = {row["option_id"] for row in assessment["stage1"]["rankings"]}
            scenario_decisions = [
                decision for decision in main.TRAINING_DATA["decisions"]
                if decision["scenario_id"] == scenario_id
            ]
            self.assertEqual(len(scenario_decisions), 4, scenario_id)
            for decision in scenario_decisions:
                self.assertIn(
                    decision["recorded_choice_id"],
                    available_ids,
                    f"{decision['decision_id']} records an unavailable alternative.",
                )


    def test_twin_update_changes_later_assessment_and_records_event(self):
        state = deepcopy(main.TWIN_STATE)
        events = []
        with patch.object(main, "TWIN_STATE", state), patch.object(main, "TWIN_EVENTS", events):
            main.update_digital_twin(main.TwinUpdateRequest(
                source="unit test",
                entity_type="supplier",
                entity_id="SUP-C",
                field="capacity_units",
                value=100,
            ))
            result = main.calculate_assessment(main.AssessRequest())

        self.assertEqual(state["revision"], 2)
        self.assertEqual(events[0]["field"], "capacity_units")
        self.assertNotIn("SUP-C", [row["option_id"] for row in result["stage1"]["rankings"]])
        self.assertEqual(result["operational_snapshot"]["twin_revision"], 2)

    def test_decision_outcome_can_apply_operational_twin_update(self):
        with (
            patch.object(main, "TWIN_STATE", deepcopy(main.TWIN_STATE)),
            patch.object(main, "TWIN_EVENTS", []),
            patch.object(main, "DECISION_FEEDBACK", []),
        ):
            decision = main.record_decision(main.DecisionFeedbackRequest(
                decision_maker_id="DM-01",
                chosen_option_id="SUP-A",
                rationale="Selected for demonstration.",
            ))
            updated = main.record_decision_outcome(
                decision["id"],
                main.OutcomeFeedbackRequest(
                    status="delayed",
                    notes="Observed delivery took longer than estimated.",
                    twin_updates=[
                        main.TwinUpdateRequest(
                            source="observed outcome",
                            entity_type="shipment",
                            entity_id="SHP-2026-3402",
                            field="estimated_delivery_days",
                            value=4,
                        )
                    ],
                ),
            )

        self.assertEqual(updated["outcome"]["status"], "delayed")
        self.assertEqual(updated["outcome"]["twin_updates"][0]["value"], 4)


if __name__ == "__main__":
    unittest.main()
