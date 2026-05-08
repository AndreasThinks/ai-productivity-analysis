from datetime import datetime, timezone

import pandas as pd

from scripts.pr_outcome_metrics import (
    atomic_write_json,
    build_did_row,
    compute_did_results,
    record_status,
    select_accounts_for_scrape,
    summarize_pr_outcomes,
    window_for_timestamp,
)


def test_window_for_timestamp_uses_global_pre_and_post_cutoffs():
    assert window_for_timestamp("2023-12-31T23:59:59Z") == "pre"
    assert window_for_timestamp("2024-01-01T00:00:00Z") == "post"
    assert window_for_timestamp("2021-12-31T23:59:59Z") is None


def test_summarize_pr_outcomes_counts_merge_rate_size_and_latency():
    prs = [
        {
            "number": 1,
            "created_at": "2023-01-10T00:00:00Z",
            "closed_at": "2023-01-12T00:00:00Z",
            "merged_at": "2023-01-11T12:00:00Z",
            "additions": 100,
            "deletions": 20,
            "changed_files": 4,
            "comments": 3,
            "review_comments": 2,
            "commits": 5,
        },
        {
            "number": 2,
            "created_at": "2023-03-01T00:00:00Z",
            "closed_at": "2023-03-03T00:00:00Z",
            "merged_at": None,
            "additions": 10,
            "deletions": 5,
            "changed_files": 1,
            "comments": 1,
            "review_comments": 0,
            "commits": 1,
        },
        {
            "number": 3,
            "created_at": "2024-02-01T00:00:00Z",
            "closed_at": "2024-02-02T00:00:00Z",
            "merged_at": "2024-02-02T00:00:00Z",
            "additions": 40,
            "deletions": 10,
            "changed_files": 2,
            "comments": 4,
            "review_comments": 6,
            "commits": 3,
        },
    ]

    metrics = summarize_pr_outcomes(
        prs,
        post_window_end=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert metrics["pre_prs_opened"] == 2
    assert metrics["pre_prs_merged"] == 1
    assert metrics["pre_merge_rate"] == 0.5
    assert metrics["pre_prs_closed_unmerged"] == 1
    assert metrics["pre_median_hours_to_merge"] == 36.0
    assert metrics["pre_mean_lines_changed"] == 67.5
    assert metrics["pre_mean_review_comments"] == 1.0
    assert metrics["post_prs_opened"] == 1
    assert metrics["post_prs_merged"] == 1
    assert metrics["post_merge_rate"] == 1.0
    assert abs(metrics["post_opened_prs_per_month"] - (1 / 24)) < 1e-6


def test_build_did_row_adds_delta_columns_and_preserves_label():
    feature_row = {"login": "dev", "label": 1, "marker_confidence": "high"}
    metrics = {
        "pre_prs_merged": 1,
        "post_prs_merged": 3,
        "pre_merge_rate": 0.25,
        "post_merge_rate": 0.75,
    }

    row = build_did_row(feature_row, metrics)

    assert row["login"] == "dev"
    assert row["label"] == 1
    assert row["delta_prs_merged"] == 2
    assert row["delta_merge_rate"] == 0.5


def test_compute_did_results_recovers_treatment_effect_with_baseline_control():
    df = pd.DataFrame(
        [
            {"label": 0, "pre_prs_merged": 1, "delta_prs_merged": 0},
            {"label": 0, "pre_prs_merged": 2, "delta_prs_merged": 0},
            {"label": 1, "pre_prs_merged": 1, "delta_prs_merged": 2},
            {"label": 1, "pre_prs_merged": 2, "delta_prs_merged": 2},
        ]
    )

    results = compute_did_results(df, ["prs_merged"])

    assert len(results) == 1
    assert results.iloc[0]["metric"] == "prs_merged"
    assert abs(results.iloc[0]["treatment_coef"] - 2.0) < 1e-9


def test_select_accounts_for_scrape_balances_limited_smoke_runs():
    accounts = pd.DataFrame(
        [
            {"login": "pos1", "label": 1},
            {"login": "pos2", "label": 1},
            {"login": "pos3", "label": 1},
            {"login": "neg1", "label": 0},
            {"login": "neg2", "label": 0},
            {"login": "neg3", "label": 0},
        ]
    )

    selected = select_accounts_for_scrape(accounts, 4)

    assert selected["login"].tolist() == ["pos1", "pos2", "neg1", "neg2"]
    assert selected["label"].tolist() == [1, 1, 0, 0]


def test_atomic_write_json_does_not_leave_tmp_file(tmp_path):
    target = tmp_path / "dev.json"

    atomic_write_json(target, {"login": "dev", "prs": [{"number": 1}]})

    assert target.exists()
    assert not target.with_suffix(".json.tmp").exists()
    assert '"login": "dev"' in target.read_text()


def test_record_status_appends_csv_rows_with_header(tmp_path):
    status_path = tmp_path / "status.csv"

    record_status(status_path, "dev1", "done", 3)
    record_status(status_path, "dev2", "error", 0, "boom")

    rows = status_path.read_text().splitlines()
    assert rows[0] == "timestamp,login,status,n_prs,error"
    assert ",dev1,done,3," in rows[1]
    assert ",dev2,error,0,boom" in rows[2]
