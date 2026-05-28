import pytest

from xbitinfo import xbitinfo as xb_module


def test_get_prefect_flow_raises_without_prefect(monkeypatch):
    monkeypatch.setattr(xb_module, "flow", None)
    monkeypatch.setattr(xb_module, "task", None)
    monkeypatch.setattr(xb_module, "unmapped", None)
    monkeypatch.setattr(
        xb_module, "prefect_import_error", ImportError("No module named 'prefect'")
    )

    with pytest.raises(ImportError, match="optional 'prefect' dependency"):
        xb_module.get_prefect_flow(paths=["example.nc"])
