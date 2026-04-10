def test_wgdos_extension_importable():
    from ppfive import wgdos

    assert wgdos._wgdos is not None
    assert hasattr(wgdos._wgdos, "unwgdos")
