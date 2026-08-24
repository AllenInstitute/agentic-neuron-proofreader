import json
from types import SimpleNamespace
from unittest import mock

from agentic_neuron_proofreader.utils import img_util


def _future(*, value=None, error=None):
    future = mock.Mock()
    if error is not None:
        future.result.side_effect = error
    else:
        future.result.return_value = value
    return future


def _successful_open():
    raw = SimpleNamespace(
        state="value",
        value=json.dumps({
            "type": "segmentation",
            "scales": [{}],
        }).encode("utf8"),
    )
    store = mock.Mock()
    store.read.return_value = _future(value=raw)
    return _future(value=store)


def test_is_precomputed_retries_transient_failures():
    transient = _future(error=RuntimeError("credentials still initializing"))
    with mock.patch.object(
        img_util.ts.KvStore,
        "open",
        side_effect=[transient, transient, _successful_open()],
    ) as open_mock, mock.patch.object(img_util.time, "sleep") as sleep_mock:
        assert img_util.is_precomputed(
            "gs://bucket/path/", max_attempts=3,
            initial_backoff_seconds=0.25,
        )

    assert open_mock.call_count == 3
    assert [call.args[0] for call in sleep_mock.call_args_list] == [0.25, 0.5]


def test_is_precomputed_stops_after_max_attempts():
    attempts = [
        _future(error=RuntimeError("temporary failure")) for _ in range(3)
    ]
    with mock.patch.object(
        img_util.ts.KvStore, "open", side_effect=attempts,
    ) as open_mock, mock.patch.object(img_util.time, "sleep") as sleep_mock:
        assert not img_util.is_precomputed(
            "gs://bucket/path/", max_attempts=3,
            initial_backoff_seconds=0.1,
        )

    assert open_mock.call_count == 3
    assert [call.args[0] for call in sleep_mock.call_args_list] == [0.1, 0.2]


def test_is_precomputed_does_not_retry_missing_metadata():
    raw = SimpleNamespace(state="missing", value=b"")
    store = mock.Mock()
    store.read.return_value = _future(value=raw)
    with mock.patch.object(
        img_util.ts.KvStore, "open", return_value=_future(value=store),
    ) as open_mock, mock.patch.object(img_util.time, "sleep") as sleep_mock:
        assert not img_util.is_precomputed(
            "gs://bucket/path/", max_attempts=4,
            initial_backoff_seconds=0.5,
        )

    open_mock.assert_called_once()
    sleep_mock.assert_not_called()


def test_is_precomputed_does_not_retry_invalid_metadata():
    raw = SimpleNamespace(state="value", value=b"not-json")
    store = mock.Mock()
    store.read.return_value = _future(value=raw)
    with mock.patch.object(
        img_util.ts.KvStore, "open", return_value=_future(value=store),
    ) as open_mock, mock.patch.object(img_util.time, "sleep") as sleep_mock:
        assert not img_util.is_precomputed(
            "gs://bucket/path/", max_attempts=4,
            initial_backoff_seconds=0.5,
        )

    open_mock.assert_called_once()
    sleep_mock.assert_not_called()
