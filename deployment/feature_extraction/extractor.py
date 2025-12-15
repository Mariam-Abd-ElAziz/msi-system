# wrapper/adapter code for the feature extraction module

# to be rewritten correctly
from teammate_code.extractor import MyFeatureExtractor

class FeatureExtractorWrapper:
    def __init__(self):
        self.extractor = MyFeatureExtractor()

    def extract(self, frame):
        # returns a 1D array of features
        return self.extractor.process(frame)