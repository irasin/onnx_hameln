from onnx_hameln import HPM, HamelnPatternManager


class TestHamelnPatternManager:

    def test_singleton(self):
        instance = HamelnPatternManager._instance
        assert HPM is instance

        new_hpm = HamelnPatternManager()
        assert HPM is new_hpm

    def test_get_available_pattern(self):
        all_pattern = HPM.get_available_pattern()

        assert len(all_pattern) == 1
