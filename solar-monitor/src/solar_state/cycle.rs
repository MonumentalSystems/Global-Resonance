//! Solar cycle context — where are we in the 11-year cycle?

use serde::Serialize;

/// Solar cycle state.
#[derive(Debug, Clone, Serialize)]
pub struct CycleState {
    /// Current sunspot number (monthly).
    pub ssn: f64,
    /// Smoothed sunspot number (13-month running mean).
    pub smoothed_ssn: f64,
    /// F10.7 cm solar radio flux (sfu).
    pub f10_7: f64,
    /// Smoothed F10.7.
    pub smoothed_f10_7: f64,
    /// Current cycle number (25 as of 2019-2030+).
    pub cycle_number: u8,
    /// Estimated phase within current cycle.
    pub phase: CyclePhase,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CyclePhase {
    /// Near solar minimum, very low activity.
    Minimum,
    /// Activity rising toward maximum.
    Rising,
    /// Near solar maximum, peak activity.
    Maximum,
    /// Activity declining from maximum.
    Declining,
}

impl CycleState {
    pub fn new() -> Self {
        Self {
            ssn: 0.0,
            smoothed_ssn: 0.0,
            f10_7: 70.0,
            smoothed_f10_7: 70.0,
            cycle_number: 25,
            phase: CyclePhase::Rising,
        }
    }

    /// Activity level (0..1) based on cycle position.
    /// Used as a prior for overall threat assessment.
    pub fn activity_level(&self) -> f64 {
        // F10.7 ranges: minimum ~65, maximum ~200+
        // Normalize to 0..1
        ((self.f10_7 - 65.0) / 150.0).min(1.0).max(0.0)
    }

    pub fn phase_label(&self) -> &'static str {
        match self.phase {
            CyclePhase::Minimum => "minimum",
            CyclePhase::Rising => "rising",
            CyclePhase::Maximum => "maximum",
            CyclePhase::Declining => "declining",
        }
    }

    /// Update from SWPC observed solar cycle indices JSON.
    pub fn update_from_indices(
        &mut self,
        ssn: f64,
        smoothed_ssn: f64,
        f10_7: f64,
        smoothed_f10_7: f64,
    ) {
        self.ssn = ssn;
        self.smoothed_ssn = if smoothed_ssn > 0.0 {
            smoothed_ssn
        } else {
            self.smoothed_ssn
        };
        self.f10_7 = f10_7;
        self.smoothed_f10_7 = if smoothed_f10_7 > 0.0 {
            smoothed_f10_7
        } else {
            self.smoothed_f10_7
        };

        // Estimate phase from SSN trend
        self.phase = if self.ssn < 30.0 {
            CyclePhase::Minimum
        } else if self.ssn > 150.0 {
            CyclePhase::Maximum
        } else if self.smoothed_ssn > 0.0 && self.ssn > self.smoothed_ssn {
            CyclePhase::Rising
        } else {
            CyclePhase::Declining
        };
    }
}
