"""Utils
"""

class Deprecated(Exception):
    """Deprecated function
    """

    def __init__(self):
        super().__init__("FUNCTION HAS BEEN DEPRECATED")
