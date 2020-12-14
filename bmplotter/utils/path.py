import os
import sys
import tempfile

class TempFolder:
    def __init__(self):
        self.dir = tempfile.TemporaryDirectory(prefix=sys.argv[0])

    def __del__(self):
        self.dir.cleanup()

    def getPath(self, rel=""):
        assert os.path.exists(self.dir.name) # make sure the temp dir exists
        if len(rel) > 0:
            path = os.path.join(self.dir.name, rel)
        else:
            path = self.dir.name
        return path #, os.path.exists(path)