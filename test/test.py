from onnx_hameln import HPM, HamelnPatternManager

if __name__ == "__main__":
    
    print(HamelnPatternManager._instance is None)
    print(HPM.get_available_pattern())

    
    manager = HamelnPatternManager()
    print(manager.get_available_pattern())
