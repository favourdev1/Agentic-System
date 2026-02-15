from agentic_system.database.base import Base
from agentic_system.database.engine import get_engine
from agentic_system.models import SessionRecord  # noqa: F401


def init_database() -> None:
    """Create tables for local/dev usage. Migrations should be preferred in production."""
    Base.metadata.create_all(bind=get_engine())
