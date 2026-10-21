class SimpleClassifier:
    def __init__(self, input_size:int, num_classes:int):
        self.input_size = input_size,
        self.num_classes = num_classes
        
    def describe(self) -> str:
        return(
            f"Classifier: input_size={self.input_size}, "
            f"num_classes = {self.num_classes}"
        )
        
    