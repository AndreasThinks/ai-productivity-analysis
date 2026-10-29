import json
import random
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from scripts.analyse_v4 import account_row, build_features, did, window_features
from scripts.pseudonymise import pseudonym
from scripts.rescrape_v4 import (
    TOOL_LAUNCH,
    WINDOW_DAYS,
    classify_adopter,
    date_ranges,
    detect_agent,
    draw_placebo_target,
    event_windows,
    strip_agent_markers,
)

UTC = timezone.utc


def _commit(ts, message="fix bug", author_name="Dev", repo="o/r", parents=1):
    return {"sha": f"{ts.isoformat()}-{message}", "authored_at": ts.isoformat(), "committed_at": ts.isoformat(),
            "message": message, "author_name": author_name, "author_email": "dev@example.com",
            "committer_name": author_name, "committer_email": "dev@example.com",
            "repo": repo, "n_parents": parents}


def test_windows_are_equal_length_and_half_open():
    w = event_windows(datetime(2025, 6, 15, 13, 30, tzinfo=UTC))
    assert w.event == datetime(2025, 6, 15, tzinfo=UTC)
    assert w.event - w.pre_start == w.post_end - w.event == timedelta(days=WINDOW_DAYS)
    assert w.label(w.pre_start) == "pre"
    assert w.label(w.event - timedelta(seconds=1)) == "pre"
    assert w.label(w.event) == "post"
    assert w.label(w.post_end) is None


def test_classify_adopter_rejects_pre_launch_and_incomplete_windows():
    scrape = datetime(2026, 11, 1, tzinfo=UTC)
    assert classify_adopter(None, "claude", scrape) == "no_marker_found"
    assert classify_adopter(datetime(2023, 12, 1, tzinfo=UTC), "claude", scrape) == "pre_launch"
    assert classify_adopter(TOOL_LAUNCH["claude"], "claude", scrape) == "eligible"
    assert classify_adopter(datetime(2026, 9, 1, tzinfo=UTC), "claude", scrape) == "post_window_incomplete"
    # Aider launched earlier, so a 2024 marker is fine for Aider
    assert classify_adopter(datetime(2024, 3, 1, tzinfo=UTC), "aider", scrape) == "eligible"


@pytest.mark.parametrize("commit,tool", [
    ({"message": "feat: x\n\nCo-Authored-By: Claude <noreply@anthropic.com>"}, "claude"),
    ({"message": "feat: x\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"}, "claude"),
    ({"message": "x\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)"}, "claude"),
    ({"message": "x", "author_name": "Jane Doe (aider)"}, "aider"),
    ({"message": "x\n\nCo-authored-by: aider (gpt-4o) <aider@aider.chat>"}, "aider"),
    ({"message": "x\n\nCo-authored-by: Cursor Agent <cursoragent@cursor.com>"}, "cursor"),
    ({"message": "fix typo", "author_name": "Jane"}, None),
    ({"message": "Mention Claude Shannon in the docs"}, None),
])
def test_detect_agent(commit, tool):
    assert detect_agent(commit) == tool


def test_strip_agent_markers_removes_trailers_only():
    msg = "feat: add parser\n\nLonger body.\n\n🤖 Generated with [Claude Code](https://claude.ai/code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>"
    assert strip_agent_markers(msg) == "feat: add parser\n\nLonger body."


def test_date_ranges_cover_interval_without_gaps():
    a, b = datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 7, 1, tzinfo=UTC)
    parts = date_ranges(a, b, 4)
    assert parts[0][0] == a and parts[-1][1] == b
    assert all(x[1] == y[0] for x, y in zip(parts, parts[1:]))


def test_placebo_target_comes_from_adopter_dates():
    dates = [datetime(2025, 3, 1, tzinfo=UTC), datetime(2025, 9, 1, tzinfo=UTC)]
    rng = random.Random(0)
    assert {draw_placebo_target(dates, rng) for _ in range(50)} == set(dates)


def test_window_features_skip_merges_and_use_calendar_weeks():
    t0 = datetime(2025, 1, 6, 9, tzinfo=UTC)
    commits = [_commit(t0), _commit(t0 + timedelta(minutes=30), "feat: y"),
               _commit(t0 + timedelta(days=7)), _commit(t0 + timedelta(days=7, hours=1), "Merge", parents=2)]
    f = window_features(commits)
    assert f["n_commits"] == 3
    assert f["commits_per_week"] == pytest.approx(3 / (WINDOW_DAYS / 7))
    assert f["active_weeks"] == 2 and f["active_days"] == 2
    assert f["frac_conventional"] == pytest.approx(1 / 3)
    assert f["frac_burst"] == pytest.approx(0.5)


def test_account_row_separates_agent_commits_and_pseudonymises(tmp_path):
    event = datetime(2025, 6, 2, tzinfo=UTC)
    pre = [_commit(event - timedelta(days=d)) for d in (10, 20, 30, 40, 50, 60)]
    post = [_commit(event + timedelta(days=d)) for d in (1, 2, 3)] + [
        _commit(event + timedelta(days=d, minutes=m), "x\n\nCo-Authored-By: Claude <noreply@anthropic.com>")
        for d in (1, 2, 3) for m in (5, 10)]
    cache = {"login": "SomeUser", "group": "adopter", "tool": "claude", "event_date": event.isoformat(),
             "commits": pre + post, "commits_truncated": False,
             "prs_pre": {"prs_opened": 2, "prs_merged": 1, "prs_merged_external": 0},
             "prs_post": {"prs_opened": 5, "prs_merged": 4, "prs_merged_external": 1}}
    row = account_row(cache, salt="s")
    assert row["account_id"] == pseudonym("someuser", "s") and "SomeUser" not in json.dumps(row)
    assert row["pre_n_commits"] == 6 and row["post_n_commits"] == 9
    assert row["post_human_n_commits"] == 3
    assert row["post_agent_share"] == pytest.approx(6 / 9)
    assert row["delta_prs_merged"] == 3
    # agent trailers never reach the style features
    assert row["post_mean_msg_len"] < 10

    (tmp_path / "SomeUser.json").write_text(json.dumps(cache))
    assert len(build_features(tmp_path, salt="s", min_pre_commits=5)) == 1
    assert len(build_features(tmp_path, salt="s", min_pre_commits=7)) == 0


def test_did_recovers_a_planted_effect():
    rows = []
    rng = random.Random(1)
    for i in range(60):
        treated = int(i < 20)
        pre = rng.gauss(10, 2)
        rows.append({"treated": treated, "pre_y": pre, "delta_y": 5 * treated + rng.gauss(0, 1)})
    res = did(pd.DataFrame(rows), ["y"], "test")
    assert res.loc[0, "coef"] == pytest.approx(5, abs=1)
    assert res.loc[0, "p"] < 0.001
