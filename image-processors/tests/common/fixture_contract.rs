use serde::Deserialize;

const MAX_ABSOLUTE_TOLERANCE: f64 = 0.06;
const MAX_RELATIVE_TOLERANCE: f64 = 0.0;

#[derive(Clone, Copy, Debug, Deserialize)]
pub(crate) struct FloatTolerance {
    pub(crate) absolute_tolerance: f64,
    pub(crate) relative_tolerance: f64,
}

impl FloatTolerance {
    pub(crate) fn allowed_delta(self, expected: f64) -> f64 {
        self.absolute_tolerance + self.relative_tolerance * expected.abs()
    }

    pub(crate) fn matches(self, actual: f64, expected: f64) -> bool {
        (actual - expected).abs() <= self.allowed_delta(expected)
    }

    fn validate(self, label: &str) {
        assert!(
            self.absolute_tolerance.is_finite() && self.absolute_tolerance >= 0.0,
            "{label} absolute tolerance must be finite and non-negative"
        );
        assert!(
            self.absolute_tolerance <= MAX_ABSOLUTE_TOLERANCE,
            "{label} absolute tolerance {} exceeds reviewed maximum {MAX_ABSOLUTE_TOLERANCE}",
            self.absolute_tolerance
        );
        assert!(
            self.relative_tolerance.is_finite() && self.relative_tolerance >= 0.0,
            "{label} relative tolerance must be finite and non-negative"
        );
        assert!(
            self.relative_tolerance <= MAX_RELATIVE_TOLERANCE,
            "{label} relative tolerance {} exceeds reviewed maximum {MAX_RELATIVE_TOLERANCE}",
            self.relative_tolerance
        );
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub(crate) struct ExactComparison {
    shape: bool,
    dtype: bool,
    integer_payload: bool,
    metadata: bool,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub(crate) struct ComparisonPolicy {
    pub(crate) full_payload: FloatTolerance,
    pub(crate) statistics: FloatTolerance,
    exact: ExactComparison,
}

impl ComparisonPolicy {
    pub(crate) fn validate(self) {
        self.full_payload.validate("full-payload float");
        self.statistics.validate("statistics float");
        assert!(
            self.exact.shape
                && self.exact.dtype
                && self.exact.integer_payload
                && self.exact.metadata,
            "shape, dtype, integer payloads, and metadata must use exact comparison"
        );
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct FixtureLocations {
    pub(crate) locations: Vec<String>,
    pub(crate) description: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ProcessorConfigContract {
    pub(crate) locations: Vec<String>,
    pub(crate) explicit: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct AcceptanceContract {
    pub(crate) deterministic_input: FixtureLocations,
    pub(crate) processor_config: ProcessorConfigContract,
    pub(crate) comparison: ComparisonPolicy,
}

impl AcceptanceContract {
    pub(crate) fn validate(&self) {
        assert!(
            !self.deterministic_input.locations.is_empty(),
            "deterministic input locations must be recorded"
        );
        assert!(
            self.deterministic_input
                .locations
                .iter()
                .all(|location| !location.is_empty()),
            "deterministic input locations must not be empty"
        );
        assert!(
            !self.deterministic_input.description.is_empty(),
            "deterministic input generation must be described"
        );
        assert!(
            self.processor_config.explicit,
            "processor configuration must be explicit"
        );
        assert!(
            !self.processor_config.locations.is_empty()
                && self
                    .processor_config
                    .locations
                    .iter()
                    .all(|location| !location.is_empty()),
            "processor configuration locations must be recorded"
        );
        self.comparison.validate();
    }
}
