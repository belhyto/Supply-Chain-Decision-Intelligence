"""Append a reproducible synthetic scenario expansion to the training CSVs."""

import csv
import json
import math
from pathlib import Path


DATA_DIR = Path(__file__).parent
SCENARIO_TEMPLATES = [
    (
        "Port congestion",
        "Congestion at the main port threatens inbound stock during rising demand.",
        "Lower freight cost versus route reliability",
        "Hydraulic seal kits",
        850,
        9,
        "High",
        "Harbor Distribution Center",
        {"cost": 0.20, "reliability": 0.30, "speed": 0.30, "risk": 0.20},
    ),
    (
        "Component allocation shortage",
        "A component manufacturer has allocated less capacity than the planned order.",
        "Secure capacity versus protect purchase cost",
        "Industrial control boards",
        1400,
        12,
        "High",
        "Central Assembly Plant",
        {"cost": 0.20, "reliability": 0.30, "speed": 0.20, "risk": 0.30},
    ),
    (
        "Currency volatility",
        "A sharp currency movement has changed import costs before the next replenishment cycle.",
        "Lock in price versus maintain flexible supply",
        "Precision bearings",
        1900,
        18,
        "Medium",
        "East Regional Warehouse",
        {"cost": 0.45, "reliability": 0.25, "speed": 0.10, "risk": 0.20},
    ),
    (
        "Supplier product recall",
        "A supplier recall notice requires the buyer to replace a potentially affected batch.",
        "Fast replacement versus stronger quality assurance",
        "Food-grade pump seals",
        620,
        6,
        "High",
        "Food and Beverage Plant",
        {"cost": 0.10, "reliability": 0.30, "speed": 0.20, "risk": 0.40},
    ),
    (
        "Flooded distribution center",
        "Flooding has restricted access to a distribution center and its normal replenishment route.",
        "Immediate rerouting versus minimize emergency cost",
        "Water-treatment cartridges",
        1100,
        5,
        "High",
        "Riverbend Distribution Center",
        {"cost": 0.15, "reliability": 0.25, "speed": 0.40, "risk": 0.20},
    ),
    (
        "Production labor shortage",
        "A labor shortage at a key supplier may reduce its output during a critical production window.",
        "Supplier capacity versus dependable delivery",
        "Motor drive assemblies",
        760,
        10,
        "Medium",
        "North Assembly Center",
        {"cost": 0.15, "reliability": 0.35, "speed": 0.25, "risk": 0.25},
    ),
    (
        "Cold-chain capacity constraint",
        "Refrigerated transport capacity is limited for a temperature-sensitive replenishment.",
        "Protect product integrity versus control shipping cost",
        "Temperature-sensitive reagents",
        460,
        3,
        "High",
        "Metro Health Warehouse",
        {"cost": 0.10, "reliability": 0.30, "speed": 0.25, "risk": 0.35},
    ),
    (
        "Supplier systems outage",
        "A supplier systems outage has interrupted order visibility and shipment confirmations.",
        "Restore visibility versus diversify supply",
        "Industrial sensor modules",
        980,
        8,
        "Medium",
        "West Technology Depot",
        {"cost": 0.15, "reliability": 0.30, "speed": 0.20, "risk": 0.35},
    ),
    (
        "New product launch",
        "A new product launch requires an initial component build with uncertain demand.",
        "Keep launch cost low versus protect launch availability",
        "Smart valve controllers",
        1250,
        14,
        "Medium",
        "Innovation Fulfillment Hub",
        {"cost": 0.30, "reliability": 0.25, "speed": 0.25, "risk": 0.20},
    ),
    (
        "Supplier consolidation",
        "A proposed consolidation would reduce the supplier base but increase concentration exposure.",
        "Scale economies versus sourcing resilience",
        "Protective packaging film",
        2100,
        16,
        "Medium",
        "South Packaging Plant",
        {"cost": 0.25, "reliability": 0.25, "speed": 0.15, "risk": 0.35},
    ),
    (
        "Customs documentation hold",
        "A documentation issue has held an imported shipment while the receiving site approaches reorder point.",
        "Use local stock versus wait for lower-cost imports",
        "Industrial flow meters",
        540,
        7,
        "High",
        "East Import Center",
        {"cost": 0.15, "reliability": 0.25, "speed": 0.40, "risk": 0.20},
    ),
    (
        "Capacity ramp-up",
        "A planned production ramp requires a larger replenishment than current supplier forecasts.",
        "Scale quickly versus limit supplier execution risk",
        "Power conversion units",
        1750,
        13,
        "Medium",
        "Central Manufacturing Hub",
        {"cost": 0.15, "reliability": 0.30, "speed": 0.25, "risk": 0.30},
    ),
    (
        "Packaging material shortage",
        "A packaging-material shortage threatens finished-goods dispatch during a promotion.",
        "Low unit cost versus protect dispatch schedule",
        "Recyclable shipping cartons",
        3200,
        8,
        "High",
        "South Distribution Center",
        {"cost": 0.35, "reliability": 0.20, "speed": 0.30, "risk": 0.15},
    ),
    (
        "Winter route disruption",
        "Severe winter weather may disrupt the usual transport corridor to a remote depot.",
        "Route dependability versus short lead time",
        "Remote-site service kits",
        680,
        6,
        "High",
        "Mountain Service Depot",
        {"cost": 0.10, "reliability": 0.35, "speed": 0.25, "risk": 0.30},
    ),
    (
        "Perishable inventory expiry",
        "A short remaining shelf life requires replenishment in smaller, more frequent shipments.",
        "Reduce spoilage risk versus capture bulk savings",
        "Perishable test media",
        390,
        4,
        "High",
        "Metro Laboratory Warehouse",
        {"cost": 0.10, "reliability": 0.25, "speed": 0.35, "risk": 0.30},
    ),
    (
        "Supplier emissions requirement",
        "A new procurement policy requires lower-emission transport without missing the delivery window.",
        "Meet sustainability policy versus preserve cost and speed",
        "Low-emission generator parts",
        920,
        11,
        "Low",
        "National Service Center",
        {"cost": 0.20, "reliability": 0.25, "speed": 0.20, "risk": 0.35},
    ),
]

BASE_SCORE_PROFILES = [
    {"cost": 94, "reliability": 70, "speed": 67, "risk": 66},
    {"cost": 72, "reliability": 94, "speed": 82, "risk": 92},
    {"cost": 82, "reliability": 83, "speed": 95, "risk": 78},
]
SUPPLIER_NAMES = ["Northstar Industrial", "ValueLink Supply", "RapidRoute Partners"]
CRITERIA = ("cost", "reliability", "speed", "risk")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def append_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writerows(rows)


def build_rows() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    _, profile_rows = read_csv(DATA_DIR / "decision_maker_profiles.csv")
    profiles = [
        {
            "id": row["decision_maker_id"],
            "name": row["name"],
            "preferences": {key: float(row[key]) for key in CRITERIA},
        }
        for row in profile_rows
    ]
    scenario_rows = []
    option_rows = []
    decision_rows = []
    outcome_rows = []

    for offset, template in enumerate(SCENARIO_TEMPLATES, start=9):
        (
            title,
            situation,
            conflict,
            product,
            required_units,
            delivery_days,
            stockout_risk,
            location,
            operational_weights,
        ) = template
        scenario_id = f"SCN-{offset:03d}"
        demand_increase = (offset * 7) % 41 + 5
        inventory_days = (offset * 3) % 8 + 2
        required_inventory_days = max(delivery_days + 5, inventory_days + 8)
        available_budget = 12 + (offset * 5) % 16
        threat_level = "High" if stockout_risk == "High" else "Medium"
        scenario_rows.append({
            "scenario_id": scenario_id,
            "title": title,
            "situation": situation,
            "decision_conflict": conflict,
            "product": product,
            "required_units": str(required_units),
            "required_within_days": str(delivery_days),
            "stockout_risk": stockout_risk,
            "location": location,
            "operational_weights_json": json.dumps(operational_weights, separators=(",", ":")),
            "demand_increase_percent": str(demand_increase),
            "current_inventory_days": str(inventory_days),
            "required_inventory_days": str(required_inventory_days),
            "available_budget_lakh": str(available_budget),
            "threat_level": threat_level,
        })

        scenario_options: list[dict[str, object]] = []
        profile_rotation = (offset - 9) % 3
        for alternative_index, code in enumerate(("SUP-A", "SUP-B", "SUP-C")):
            family_index = (alternative_index + profile_rotation) % 3
            base_scores = BASE_SCORE_PROFILES[family_index]
            metrics = {
                criterion: max(
                    52,
                    min(
                        98,
                        base_scores[criterion]
                        + ((offset * 11 + family_index * 7 + criterion_index * 5) % 9)
                        - 4,
                    ),
                )
                for criterion_index, criterion in enumerate(CRITERIA)
            }
            lead_time_days = max(1, round(10 - metrics["speed"] / 12))
            capacity_units = int(required_units * (1.2 + alternative_index * 0.2))
            cost_lakh = round(10 + (100 - metrics["cost"]) * 0.18, 2)
            option = {
                "id": code,
                "name": f"{SUPPLIER_NAMES[alternative_index]} · {product}",
                "metrics": metrics,
                "attributes": {
                    "lead_time_days": lead_time_days,
                    "cost_lakh": cost_lakh,
                    "capacity_units": capacity_units,
                },
                "notes": (
                    "Synthetic alternative; criterion scores are illustrative and "
                    "higher scores indicate better suitability."
                ),
                "available": True,
            }
            scenario_options.append(option)
            option_rows.append({
                "scenario_id": scenario_id,
                "option_id": code,
                "name": str(option["name"]),
                "metrics_json": json.dumps(metrics, separators=(",", ":")),
                "attributes_json": json.dumps(option["attributes"], separators=(",", ":")),
                "notes": str(option["notes"]),
                "available": "true",
            })

        for profile in profiles:
            utility_by_code = {
                str(option["id"]): sum(
                    float(option["metrics"][criterion]) * profile["preferences"][criterion]
                    for criterion in CRITERIA
                )
                for option in scenario_options
            }
            chosen_code = max(utility_by_code, key=utility_by_code.get)
            chosen_option = next(
                option for option in scenario_options if option["id"] == chosen_code
            )
            decision_id = f"DEC-SCN{offset:03d}-{profile['id']}"
            decision_rows.append({
                "decision_id": decision_id,
                "scenario_id": scenario_id,
                "decision_maker_id": profile["id"],
                "recorded_choice_id": chosen_code,
                "rationale": (
                    f"Synthetic example: {profile['name']} profile selected the "
                    f"highest weighted-utility alternative ({chosen_code})."
                ),
                "predicted_probabilities_json": "",
            })
            lead_time = int(chosen_option["attributes"]["lead_time_days"])
            status = "fulfilled" if lead_time <= delivery_days else "delayed"
            daily_demand = math.ceil(required_units / required_inventory_days)
            stockout_units = max(0, lead_time - inventory_days) * daily_demand
            outcome_rows.append({
                "decision_id": decision_id,
                "status": status,
                "actual_delivery_days": str(lead_time),
                "cost_variance_percent": str(round((100 - chosen_option["metrics"]["cost"]) / 10, 1)),
                "stockout_units": str(stockout_units),
                "notes": "Synthetic outcome example; not an observed operational result.",
            })

    return scenario_rows, option_rows, decision_rows, outcome_rows


def main() -> None:
    targets = [
        ("scenarios.csv", "scenario_id", 0),
        ("options.csv", "scenario_id", 1),
        ("decisions.csv", "scenario_id", 2),
        ("outcomes.csv", "decision_id", 3),
    ]
    generated = build_rows()
    scenario_ids = {row["scenario_id"] for row in generated[0]}
    current_scenarios = {
        row["scenario_id"]
        for row in read_csv(DATA_DIR / "scenarios.csv")[1]
    }
    if scenario_ids.issubset(current_scenarios):
        print(f"Synthetic scenarios {min(scenario_ids)} through {max(scenario_ids)} already exist.")
        return
    if scenario_ids.intersection(current_scenarios):
        raise RuntimeError("The training CSVs contain a partial generated scenario expansion.")

    for filename, key, generated_index in targets:
        path = DATA_DIR / filename
        fieldnames, current_rows = read_csv(path)
        new_rows = generated[generated_index]
        if any(row[key] in {item[key] for item in current_rows} for row in new_rows):
            raise RuntimeError(f"Generated records already exist in {filename}.")
        append_csv(path, fieldnames, new_rows)
    print(f"Added {len(scenario_ids)} synthetic scenarios and {len(generated[2])} synthetic decision labels.")


if __name__ == "__main__":
    main()
