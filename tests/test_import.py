def test_import():
    import src

    assert src.__file__ is not None
