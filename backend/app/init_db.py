from app import models  # noqa: F401
from sqlalchemy import inspect, text

from app.database import Base, engine


_SNAPSHOT_IMMUTABILITY_MIGRATION = "2026_11_23_decision_snapshot_immutability_v2"
_PUBLISHED_RULE_IMMUTABILITY_MIGRATION = "2026_11_23_published_rule_immutability"


def verify_postgresql_schema(connection) -> None:
    inspector = inspect(connection)
    existing_tables = set(inspector.get_table_names())
    missing_tables = {
        table.name
        for table in Base.metadata.sorted_tables
        if table.name not in existing_tables
    }
    if missing_tables:
        raise RuntimeError(
            "PostgreSQL schema is missing SQLAlchemy tables: "
            + ", ".join(sorted(missing_tables))
        )

    missing_columns = []
    missing_foreign_keys = []
    for table in Base.metadata.sorted_tables:
        reflected_columns = {
            column["name"] for column in inspector.get_columns(table.name)
        }
        for column in table.columns:
            if column.name not in reflected_columns:
                missing_columns.append(f"{table.name}.{column.name}")

        reflected_foreign_keys = {
            (
                tuple(foreign_key["constrained_columns"]),
                foreign_key["referred_table"],
                tuple(foreign_key["referred_columns"]),
                (foreign_key.get("options") or {}).get("ondelete"),
            )
            for foreign_key in inspector.get_foreign_keys(table.name)
        }
        for constraint in table.foreign_key_constraints:
            expected = (
                tuple(column.name for column in constraint.columns),
                next(iter(constraint.elements)).column.table.name,
                tuple(
                    element.column.name
                    for element in constraint.elements
                ),
                next(iter(constraint.elements)).ondelete,
            )
            if not any(
                actual[:3] == expected[:3]
                and (
                    expected[3] is None
                    or (actual[3] or "").upper() == expected[3].upper()
                )
                for actual in reflected_foreign_keys
            ):
                missing_foreign_keys.append(
                    f"{table.name}{expected[0]} -> {expected[1]}{expected[2]}"
                )

    if missing_columns or missing_foreign_keys:
        problems = []
        if missing_columns:
            problems.append("missing columns: " + ", ".join(sorted(missing_columns)))
        if missing_foreign_keys:
            problems.append("missing foreign keys: " + ", ".join(sorted(missing_foreign_keys)))
        raise RuntimeError("PostgreSQL schema validation failed (" + "; ".join(problems) + ").")


def apply_postgresql_migrations(connection) -> None:
    inspector = inspect(connection)
    for table in Base.metadata.sorted_tables:
        if "data_source" not in table.c:
            continue
        existing_columns = {
            column["name"] for column in inspector.get_columns(table.name)
        }
        if "data_source" not in existing_columns:
            connection.execute(text(
                f'ALTER TABLE "{table.name}" '
                "ADD COLUMN data_source VARCHAR(40) NOT NULL DEFAULT 'unknown'"
            ))
    connection.execute(text(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version VARCHAR(120) PRIMARY KEY, "
        "applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    ))
    migration_exists = connection.execute(
        text("SELECT 1 FROM schema_migrations WHERE version = :version"),
        {"version": _SNAPSHOT_IMMUTABILITY_MIGRATION},
    ).scalar_one_or_none()
    if migration_exists is None:
        connection.execute(text("""
            CREATE OR REPLACE FUNCTION prevent_decision_snapshot_mutation()
            RETURNS TRIGGER AS $$
            BEGIN
                IF (
                    OLD.event_context->'reasoning_snapshot' IS NOT NULL
                    AND to_jsonb(OLD.event_context->'reasoning_snapshot')
                        IS DISTINCT FROM to_jsonb(NEW.event_context->'reasoning_snapshot')
                ) OR (
                    OLD.event_context->'human_reasoning' IS NOT NULL
                    AND to_jsonb(OLD.event_context->'human_reasoning')
                        IS DISTINCT FROM to_jsonb(NEW.event_context->'human_reasoning')
                ) THEN
                    RAISE EXCEPTION 'Decision-time snapshots are immutable';
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
        """))
        connection.execute(text(
            "DROP TRIGGER IF EXISTS decision_snapshot_immutable "
            "ON decision_events"
        ))
        connection.execute(text("""
            CREATE TRIGGER decision_snapshot_immutable
            BEFORE UPDATE ON decision_events
            FOR EACH ROW
            EXECUTE FUNCTION prevent_decision_snapshot_mutation()
        """))
        connection.execute(
            text("INSERT INTO schema_migrations (version) VALUES (:version)"),
            {"version": _SNAPSHOT_IMMUTABILITY_MIGRATION},
        )
    rule_migration_exists = connection.execute(
        text("SELECT 1 FROM schema_migrations WHERE version = :version"),
        {"version": _PUBLISHED_RULE_IMMUTABILITY_MIGRATION},
    ).scalar_one_or_none()
    if rule_migration_exists is None:
        connection.execute(text("""
            CREATE OR REPLACE FUNCTION prevent_published_rule_mutation()
            RETURNS TRIGGER AS $$
            BEGIN
                IF OLD.published_at IS NOT NULL AND (
                    OLD.rule_name IS DISTINCT FROM NEW.rule_name
                    OR OLD.description IS DISTINCT FROM NEW.description
                    OR to_jsonb(OLD.conditions_json) IS DISTINCT FROM to_jsonb(NEW.conditions_json)
                    OR to_jsonb(OLD.observed_pattern) IS DISTINCT FROM to_jsonb(NEW.observed_pattern)
                    OR OLD.version IS DISTINCT FROM NEW.version
                    OR OLD.version_group IS DISTINCT FROM NEW.version_group
                    OR OLD.extraction_id IS DISTINCT FROM NEW.extraction_id
                ) THEN
                    RAISE EXCEPTION 'Published decision-rule versions are immutable';
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
        """))
        connection.execute(text(
            "DROP TRIGGER IF EXISTS published_rule_immutable ON decision_rules"
        ))
        connection.execute(text("""
            CREATE TRIGGER published_rule_immutable
            BEFORE UPDATE ON decision_rules
            FOR EACH ROW
            EXECUTE FUNCTION prevent_published_rule_mutation()
        """))
        connection.execute(
            text("INSERT INTO schema_migrations (version) VALUES (:version)"),
            {"version": _PUBLISHED_RULE_IMMUTABILITY_MIGRATION},
        )


def create_tables():
    Base.metadata.create_all(bind=engine)
    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            apply_postgresql_migrations(connection)
        with engine.connect() as connection:
            verify_postgresql_schema(connection)

if __name__ == "__main__":
    create_tables()
    print("Database tables created.")
