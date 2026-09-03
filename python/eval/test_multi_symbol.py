from python.eval.multi_symbol import discover_lobster_samples


def test_discover_lobster_samples_matches_ticker_date_level_convention(tmp_path):
    good = tmp_path / "AMZN_2012-06-21_10"
    good.mkdir()
    (good / "message_10.csv").write_text("")
    (good / "orderbook_10.csv").write_text("")

    missing_orderbook = tmp_path / "MSFT_2012-06-21_10"
    missing_orderbook.mkdir()
    (missing_orderbook / "message_10.csv").write_text("")

    not_a_dataset_dir = tmp_path / "not_a_dataset"
    not_a_dataset_dir.mkdir()

    datasets = discover_lobster_samples(tmp_path)

    assert [d.ticker for d in datasets] == ["AMZN"]
    assert datasets[0].date == "2012-06-21"
    assert datasets[0].level == 10


def test_discover_lobster_samples_sorted_by_ticker(tmp_path):
    for ticker in ["MSFT", "AMZN", "GOOG"]:
        d = tmp_path / f"{ticker}_2012-06-21_10"
        d.mkdir()
        (d / "message_10.csv").write_text("")
        (d / "orderbook_10.csv").write_text("")

    datasets = discover_lobster_samples(tmp_path)
    assert [d.ticker for d in datasets] == ["AMZN", "GOOG", "MSFT"]
