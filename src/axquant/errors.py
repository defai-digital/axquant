class AxquantError(Exception):
    pass


class ArtifactError(AxquantError):
    pass


class BackendUnavailableError(AxquantError):
    pass


class PlanningError(AxquantError):
    pass


class PublishingError(AxquantError):
    pass


class ValidationGateError(AxquantError):
    pass


class BenchmarkError(AxquantError):
    pass


class InvariantViolationError(BenchmarkError):
    pass


class ProbeError(AxquantError):
    pass


class CacheError(AxquantError):
    pass


class CaptureError(AxquantError):
    pass


class QuantizerError(AxquantError):
    pass


class RefinementError(AxquantError):
    pass
