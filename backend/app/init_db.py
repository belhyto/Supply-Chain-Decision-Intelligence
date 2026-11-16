from app import models  # noqa: F401
from sqlalchemy import inspect, text

from app.database import Base, engine

def create_tables():
    Base.metadata.create_all(bind=engine)
    if engine.dialect.name == "postgresql":
        inspector = inspect(engine)
        with engine.begin() as connection:
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

if __name__ == "__main__":
    create_tables()
    print("Database tables created.")
