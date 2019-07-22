class ShapeError(Exception):
    """Exception raised for errors in the input shape.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message='Wrong shape!'):
        self.message = message


class DimensionError(Exception):
    """Exception raised for errors concerning the dimension of the used spaces.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message='Wrong dimensions!'):
        self.message = message
