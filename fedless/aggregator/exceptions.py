class AggregationError(Exception):
    pass


class InsufficientClientResults(AggregationError):
    pass


class UnknownCardinalityError(AggregationError):
    pass


class InvalidParameterShapeError(AggregationError):
    pass
