from rogii import schema


def test_schema_is_not_silently_invented() -> None:
    assert schema.TARGET_COLUMN == "TBD_after_data_inspection"
    assert schema.ID_COLUMN == "TBD_after_data_inspection"
