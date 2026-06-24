import pytest

import src.runtime.prolific as prol


def test_load_prolific_config_raises_on_malformed_yaml(tmp_path, monkeypatch):
    proj = tmp_path / "badproj"
    proj.mkdir()
    (proj / "prolific_config.yaml").write_text("reward: [unclosed\n", encoding="utf-8")
    monkeypatch.setattr(prol, "project_dir", lambda pid: tmp_path / pid)

    with pytest.raises(ValueError, match="Malformed Prolific config"):
        prol.load_prolific_config("badproj")


def test_load_prolific_config_overrides_defaults(tmp_path, monkeypatch):
    proj = tmp_path / "goodproj"
    proj.mkdir()
    (proj / "prolific_config.yaml").write_text(
        "reward_per_hour: 1200\nestimated_completion_time: 5\n", encoding="utf-8"
    )
    monkeypatch.setattr(prol, "project_dir", lambda pid: tmp_path / pid)

    cfg = prol.load_prolific_config("goodproj")
    assert cfg["reward_per_hour"] == 1200
    assert cfg["estimated_completion_time"] == 5


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def test_list_submissions_follows_pagination(monkeypatch):
    page1 = {
        "results": [{"id": "s1", "time_taken": 100}],
        "_links": {"next": {"href": "https://api.prolific.com/page2"}},
    }
    page2 = {
        "results": [{"id": "s2", "time_taken": 200}],
        "_links": {"next": None},
    }
    pages = {
        "https://api.prolific.com/api/v1/studies/study1/submissions/": page1,
        "https://api.prolific.com/page2": page2,
    }
    monkeypatch.setattr(prol, "_headers", lambda: {"Authorization": "Token x"})
    monkeypatch.setattr(
        prol.requests, "get", lambda url, **kw: _FakeResponse(200, pages[url])
    )

    submissions, err = prol.list_submissions("study1")
    assert err is None
    assert [s["id"] for s in submissions] == ["s1", "s2"]


def test_list_submissions_terminates_on_degenerate_next_link(monkeypatch):
    # Prolific always returns a non-null `next` href; on the last page it
    # degenerates to ?limit=0&offset=0 and re-serves the same rows. The fetcher
    # must terminate and not double-count.
    page = {
        "results": [
            {"id": "s1", "time_taken": 100},
            {"id": "s2", "time_taken": 200},
        ],
        "_links": {
            "next": {
                "href": "https://api.prolific.com/api/v1/studies/study1/submissions/?limit=0&offset=0"
            }
        },
        "meta": {"count": 2},
    }
    calls = {"n": 0}

    def fake_get(url, **kw):
        calls["n"] += 1
        assert calls["n"] <= 5, "list_submissions followed the degenerate next link in a loop"
        return _FakeResponse(200, page)

    monkeypatch.setattr(prol, "_headers", lambda: {"Authorization": "Token x"})
    monkeypatch.setattr(prol.requests, "get", fake_get)

    submissions, err = prol.list_submissions("study1")
    assert err is None
    assert [s["id"] for s in submissions] == ["s1", "s2"]  # deduped, not repeated


def test_list_submissions_returns_error_on_non_200(monkeypatch):
    monkeypatch.setattr(prol, "_headers", lambda: {"Authorization": "Token x"})
    monkeypatch.setattr(
        prol.requests, "get", lambda url, **kw: _FakeResponse(403, "forbidden")
    )

    submissions, err = prol.list_submissions("study1")
    assert submissions is None
    assert "403" in err
