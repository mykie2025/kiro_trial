"""Neo4j graph persistence tests.

These tests cover basic connection handling and health-check logic for
`Neo4jGraphPersistence`.  They mock the `neo4j.GraphDatabase` driver so no
real Neo4j instance is required.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import Mock, MagicMock, patch

import pytest

from src.config.config_manager import ConfigManager
from src.persistence.neo4j_graph_persistence import (
    Neo4jGraphError,
    Neo4jGraphPersistence,
)


@pytest.fixture()
def mock_config_manager() -> ConfigManager:  # type: ignore[return-value]
    """Return a mocked `ConfigManager` with Neo4j credentials."""
    cfg_mgr = Mock(spec=ConfigManager)

    cfg = Mock()
    cfg.neo4j_uri = "neo4j://localhost:7687"
    cfg.neo4j_username = "neo4j"
    cfg.neo4j_password = "password"
    cfg.neo4j_database = "neo4j"
    cfg_mgr.get_config.return_value = cfg
    cfg_mgr.get_neo4j_driver_config.return_value = {
        "uri": cfg.neo4j_uri,
        "auth": (cfg.neo4j_username, cfg.neo4j_password),
        "database": cfg.neo4j_database,
    }
    return cfg_mgr  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _mock_driver_with_session(mock_session: Mock) -> Mock:
    """Return a mocked Neo4j driver whose `session()` is a CM yielding *mock_session*."""
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_session
    mock_ctx.__exit__.return_value = False

    driver = Mock()
    driver.session.return_value = mock_ctx
    return driver


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@patch("src.persistence.neo4j_graph_persistence.GraphDatabase")
def test_initialization_success(mock_graph_db: Mock, mock_config_manager: ConfigManager) -> None:  # noqa: D401
    """Driver initialises and test query runs."""
    # Arrange: driver returns a session whose test query returns 1.
    mock_session = Mock()
    test_result = Mock()
    test_record = Mock()
    test_record.__getitem__ = Mock(return_value=1)  # RETURN 1 as test
    test_result.single.return_value = test_record
    mock_session.run.return_value = test_result

    driver = _mock_driver_with_session(mock_session)
    mock_graph_db.driver.return_value = driver

    # Act
    persistence = Neo4jGraphPersistence(mock_config_manager)

    # Assert
    assert persistence.driver is driver
    mock_session.run.assert_called_once_with("RETURN 1 as test")


@patch("src.persistence.neo4j_graph_persistence.GraphDatabase")
def test_initialization_connection_error(mock_graph_db: Mock, mock_config_manager: ConfigManager) -> None:  # noqa: D401
    """Exception raised when driver cannot be created."""
    mock_graph_db.driver.side_effect = RuntimeError("boom")

    with pytest.raises(Neo4jGraphError):
        Neo4jGraphPersistence(mock_config_manager)


@patch("src.persistence.neo4j_graph_persistence.GraphDatabase")
def test_health_check_healthy(mock_graph_db: Mock, mock_config_manager: ConfigManager) -> None:  # noqa: D401
    """`health_check` returns expected dict when DB responds."""
    mock_session = Mock()

    # Build different results depending on Cypher query issued.
    def _run(query: str, *args: Any, **kwargs: Any):  # noqa: ANN401
        if "RETURN 1 as test" in query:
            rec = Mock()
            rec.__getitem__ = Mock(return_value=1)
            res = Mock(); res.single.return_value = rec  # type: ignore[assignment]
            return res
        if "dbms.components" in query:
            return [{"name": "Neo4j", "versions": ["5.15"]}]
        if "count(n) as node_count" in query:
            rec = Mock(); rec.__getitem__ = Mock(return_value=10)
            res = Mock(); res.single.return_value = rec
            return res
        if "count(r) as rel_count" in query:
            rec = Mock(); rec.__getitem__ = Mock(return_value=3)
            res = Mock(); res.single.return_value = rec
            return res
        return Mock()

    mock_session.run.side_effect = _run
    driver = _mock_driver_with_session(mock_session)
    mock_graph_db.driver.return_value = driver

    persistence = Neo4jGraphPersistence(mock_config_manager)
    health = persistence.health_check()

    assert health["status"] == "healthy"
    assert health["node_count"] == 10
    assert health["relationship_count"] == 3


@patch("src.persistence.neo4j_graph_persistence.GraphDatabase")
def test_health_check_unhealthy(mock_graph_db: Mock, mock_config_manager: ConfigManager) -> None:  # noqa: D401
    """`health_check` handles errors and marks unhealthy."""
    mock_session = Mock()
    mock_session.run.side_effect = Exception("db down")

    driver = _mock_driver_with_session(mock_session)
    mock_graph_db.driver.return_value = driver

    with patch.object(Neo4jGraphPersistence, "_test_connection"):
        persistence = Neo4jGraphPersistence(mock_config_manager)

    health = persistence.health_check()

    assert health["status"] == "unhealthy"
    assert "error" in health
