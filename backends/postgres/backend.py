from __future__ import annotations

from postbound.postgres import PostgresInterface, connect as connect_postgres


def connect_postgres_database(
    connect_string: str,
    *,
    name: str = "postgres",
    cache_enabled: bool = False,
    private: bool = False,
    refresh: bool = False,
) -> PostgresInterface:
    return connect_postgres(
        connect_string=connect_string,
        name=name,
        cache_enabled=cache_enabled,
        private=private,
        refresh=refresh,
    )
