"""Boot the plugin package under a synthetic name (mirrors hermes's user-engine
loader, which imports user engines as ``_hermes_user_context_engine.<name>``)
and expose the engine factory fixture."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
HERMES_AGENT = PLUGIN_ROOT.parent / "hermes-agent"
for candidate in (str(HERMES_AGENT),):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

_PACKAGE_NAME = "hce_plugin"


def _load_package() -> types.ModuleType:
    if _PACKAGE_NAME in sys.modules:
        return sys.modules[_PACKAGE_NAME]
    spec = importlib.util.spec_from_file_location(
        _PACKAGE_NAME, PLUGIN_ROOT / "__init__.py",
        submodule_search_locations=[str(PLUGIN_ROOT)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    return module


_load_package()


@pytest.fixture
def make_engine(tmp_path, monkeypatch):
    """Build a bound engine against a temp store; env knobs are per-call."""

    def _make(
        session_id: str = "s1",
        context_length: int = 200_000,
        threshold: float | None = None,
        enabled: bool = True,
    ):
        monkeypatch.setenv("HCE_ENABLED", "1" if enabled else "0")
        if threshold is not None:
            monkeypatch.setenv("HCE_EXECUTE_THRESHOLD", str(threshold))
        from hce_plugin.engine import HermesContextEngine

        engine = HermesContextEngine()
        engine.update_model("test-model", context_length)
        engine.bind_session_state(session_id=session_id)
        engine.on_session_start(session_id, hermes_home=str(tmp_path), model="test-model")
        return engine

    return _make


@pytest.fixture
def store_db(tmp_path):
    return tmp_path / "context-engine" / "context.db"
