from vla_sim.provenance import immutable_mismatches


def test_provenance_rejects_dataset_or_inference_argument_changes():
    previous = {"dataset_sha256": "a", "arguments_sha256": "b", "checkpoint": "c"}
    current = {"dataset_sha256": "changed", "arguments_sha256": "b", "checkpoint": "c"}
    assert immutable_mismatches(
        previous, current, ("dataset_sha256", "arguments_sha256", "checkpoint")
    ) == ["dataset_sha256"]
