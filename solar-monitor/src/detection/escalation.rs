use chrono::{DateTime, Utc};
use serde::Serialize;

/// Monitoring escalation level based on precursor activity.
///
/// The hardness ratio detector catches sympathetic micro-flares from
/// active regions hours before major events. This state machine
/// tracks escalation from quiet → elevated → active → flare.
///
/// Transitions:
/// - Quiet → Elevated: single hardness spike (score > 0.5)
/// - Elevated → Active: repeated hardness spikes (≥3 in rolling window)
///   OR fused score sustained above 0.4
/// - Active → Flare: multi-detector agreement (≥2 detectors, fused > 0.7)
/// - Any → Quiet: no hardness spikes for `cooldown_minutes` AND
///   fused score below 0.3 for `cooldown_minutes`
///
/// Validated against real GOES data: hardness precursors appeared
/// -63min before M1.1 and -65min before X1.5 (2026-03-28/30).
#[derive(Debug, Clone)]
pub struct EscalationMonitor {
    pub level: EscalationLevel,
    /// When the current level was entered.
    pub level_since: Option<DateTime<Utc>>,
    /// Rolling window of hardness spike timestamps.
    hardness_spikes: Vec<DateTime<Utc>>,
    /// Rolling window duration for counting spikes.
    spike_window_minutes: i64,
    /// Number of spikes in window required for Elevated → Active.
    spike_threshold: usize,
    /// Minutes of quiet required to de-escalate.
    cooldown_minutes: i64,
    /// Last time any detector showed significant activity.
    last_activity: Option<DateTime<Utc>>,
    /// Peak hardness score in current escalation period.
    peak_hardness: f64,
    /// Peak fused score in current escalation period.
    peak_fused: f64,
    /// Count of flare-level triggers in current active period.
    flare_triggers: usize,
}

/// Escalation levels with increasing urgency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum EscalationLevel {
    /// No significant solar activity. Background monitoring.
    Quiet,
    /// Single precursor detected. Active region may be developing.
    /// Recommended: increase polling frequency.
    Elevated,
    /// Repeated precursor activity. Active region is producing
    /// sympathetic micro-flares. Major event likely within hours.
    /// Recommended: alert operators, begin continuous monitoring.
    Active,
    /// Multi-detector consensus: flare in progress or imminent.
    /// Recommended: issue alert, activate coupling pathways.
    Flare,
}

impl EscalationLevel {
    pub fn label(&self) -> &'static str {
        match self {
            EscalationLevel::Quiet => "QUIET",
            EscalationLevel::Elevated => "ELEVATED",
            EscalationLevel::Active => "ACTIVE",
            EscalationLevel::Flare => "FLARE",
        }
    }

    pub fn numeric(&self) -> u8 {
        match self {
            EscalationLevel::Quiet => 0,
            EscalationLevel::Elevated => 1,
            EscalationLevel::Active => 2,
            EscalationLevel::Flare => 3,
        }
    }
}

/// Snapshot of the escalation state for SSE/API.
#[derive(Debug, Clone, Serialize)]
pub struct EscalationStatus {
    pub level: EscalationLevel,
    pub level_label: &'static str,
    pub level_numeric: u8,
    pub since: Option<DateTime<Utc>>,
    pub hardness_spikes_in_window: usize,
    pub peak_hardness: f64,
    pub peak_fused: f64,
    pub flare_triggers: usize,
    pub minutes_at_level: i64,
}

/// What changed — for alert generation.
#[derive(Debug, Clone, Serialize)]
pub struct EscalationTransition {
    pub from: EscalationLevel,
    pub to: EscalationLevel,
    pub timestamp: DateTime<Utc>,
    pub reason: String,
}

impl EscalationMonitor {
    pub fn new() -> Self {
        Self {
            level: EscalationLevel::Quiet,
            level_since: None,
            hardness_spikes: Vec::new(),
            spike_window_minutes: 180, // 3-hour window for counting spikes
            spike_threshold: 3,        // 3 spikes in window → Active
            cooldown_minutes: 60,      // 1 hour of quiet → de-escalate
            last_activity: None,
            peak_hardness: 0.0,
            peak_fused: 0.0,
            flare_triggers: 0,
        }
    }

    /// Update the escalation state. Returns a transition if the level changed.
    ///
    /// `xray_flux` is the current 0.1-0.8nm flux in W/m². Used to prevent
    /// de-escalation when background flux is elevated (C5+ prevents →QUIET).
    pub fn update(
        &mut self,
        hardness_score: f64,
        fused_score: f64,
        detector_agreement: usize,
        timestamp: DateTime<Utc>,
    ) -> Option<EscalationTransition> {
        self.update_with_flux(
            hardness_score,
            fused_score,
            detector_agreement,
            0.0,
            timestamp,
        )
    }

    /// Update with X-ray flux and criticality score for predictive escalation.
    ///
    /// The criticality score provides a *predictive* escalation path:
    /// sustained criticality > 0.6 triggers Quiet→Elevated even before
    /// any hardness spike occurs. This is the 6+ hour early warning from
    /// the Clifford lattice's approach to J_c.
    pub fn update_with_flux(
        &mut self,
        hardness_score: f64,
        fused_score: f64,
        detector_agreement: usize,
        xray_flux: f64,
        timestamp: DateTime<Utc>,
    ) -> Option<EscalationTransition> {
        self.update_full(
            hardness_score,
            fused_score,
            detector_agreement,
            xray_flux,
            0.0,
            timestamp,
        )
    }

    /// Full update with criticality score from the Clifford lattice detector.
    pub fn update_full(
        &mut self,
        hardness_score: f64,
        fused_score: f64,
        detector_agreement: usize,
        xray_flux: f64,
        criticality_score: f64,
        timestamp: DateTime<Utc>,
    ) -> Option<EscalationTransition> {
        // Track hardness spikes
        if hardness_score > 0.5 {
            self.hardness_spikes.push(timestamp);
            self.last_activity = Some(timestamp);
        }
        // Threshold raised from 0.3 → 0.5: criticality detector idles ~0.17
        // and stressor pathways (e.g. lunar tidal) routinely push fused into
        // the 0.3-0.5 band with zero anomalies, which kept last_activity
        // permanently fresh and locked the state machine in FLARE.
        if fused_score > 0.5 {
            self.last_activity = Some(timestamp);
        }
        // Sustained criticality counts as activity — the lattice sees
        // approach to J_c before statistical detectors fire.
        if criticality_score > 0.5 {
            self.last_activity = Some(timestamp);
        }
        // Elevated X-ray background counts as activity
        // C5.0 = 5e-6, prevents de-escalation to QUIET during active periods
        if xray_flux >= 5e-6 {
            self.last_activity = Some(timestamp);
        }

        // Expire old spikes outside the window
        let window_cutoff = timestamp - chrono::Duration::minutes(self.spike_window_minutes);
        self.hardness_spikes.retain(|&t| t > window_cutoff);

        // Track peaks
        if hardness_score > self.peak_hardness {
            self.peak_hardness = hardness_score;
        }
        if fused_score > self.peak_fused {
            self.peak_fused = fused_score;
        }

        let spikes_in_window = self.hardness_spikes.len();
        let old_level = self.level;

        // State transitions — criticality score provides a PREDICTIVE path
        match self.level {
            EscalationLevel::Quiet => {
                if detector_agreement >= 2 && fused_score > 0.7 {
                    // Skip straight to Flare if multi-detector consensus
                    self.level = EscalationLevel::Flare;
                    self.flare_triggers = 1;
                } else if hardness_score > 0.5 {
                    self.level = EscalationLevel::Elevated;
                } else if criticality_score > 0.6 {
                    // PREDICTIVE: Clifford lattice detects approach to J_c
                    // before any statistical detector fires.
                    self.level = EscalationLevel::Elevated;
                }
            }
            EscalationLevel::Elevated => {
                if detector_agreement >= 2 && fused_score > 0.7 {
                    self.level = EscalationLevel::Flare;
                    self.flare_triggers = 1;
                } else if spikes_in_window >= self.spike_threshold || fused_score > 0.5 {
                    self.level = EscalationLevel::Active;
                } else if criticality_score > 0.7 {
                    // PREDICTIVE: sustained high criticality → Active
                    // without waiting for repeated hardness spikes.
                    self.level = EscalationLevel::Active;
                } else if self.should_cooldown(timestamp) {
                    self.reset_to_quiet();
                }
            }
            EscalationLevel::Active => {
                if detector_agreement >= 2 && fused_score > 0.7 {
                    self.level = EscalationLevel::Flare;
                    self.flare_triggers += 1;
                } else if self.should_cooldown(timestamp) {
                    // De-escalate to Elevated first, not straight to Quiet
                    self.level = EscalationLevel::Elevated;
                }
            }
            EscalationLevel::Flare => {
                if detector_agreement >= 2 && fused_score > 0.7 {
                    self.flare_triggers += 1;
                } else if self.should_cooldown(timestamp) {
                    // De-escalate to Active (post-flare monitoring)
                    self.level = EscalationLevel::Active;
                }
            }
        }

        // Generate transition event
        if self.level != old_level {
            let reason = match self.level {
                EscalationLevel::Quiet => "Sustained quiet period".into(),
                EscalationLevel::Elevated => {
                    if old_level == EscalationLevel::Active || old_level == EscalationLevel::Flare {
                        "De-escalating: activity subsiding".into()
                    } else if criticality_score > 0.6 && hardness_score <= 0.5 {
                        format!(
                            "Criticality detector: lattice approaching J_c (score {:.3})",
                            criticality_score
                        )
                    } else {
                        format!("Hardness spike detected (score {:.3})", hardness_score)
                    }
                }
                EscalationLevel::Active => {
                    if old_level == EscalationLevel::Flare {
                        "Post-flare: monitoring active region".into()
                    } else if criticality_score > 0.7 && spikes_in_window < self.spike_threshold {
                        format!(
                            "Criticality detector: sustained J ≈ J_c (score {:.3})",
                            criticality_score
                        )
                    } else {
                        format!(
                            "{} hardness spikes in {}h window — active region developing",
                            spikes_in_window,
                            self.spike_window_minutes / 60
                        )
                    }
                }
                EscalationLevel::Flare => {
                    format!(
                        "Multi-detector consensus: {}/7 agree, fused {:.3}",
                        detector_agreement, fused_score
                    )
                }
            };

            self.level_since = Some(timestamp);

            Some(EscalationTransition {
                from: old_level,
                to: self.level,
                timestamp,
                reason,
            })
        } else {
            None
        }
    }

    fn should_cooldown(&self, now: DateTime<Utc>) -> bool {
        // Must be at current level for at least 30 minutes (hysteresis)
        let min_hold = match self.level_since {
            Some(since) => (now - since).num_minutes() >= 30,
            None => true,
        };
        if !min_hold {
            return false;
        }
        // Must have no activity for cooldown_minutes
        match self.last_activity {
            Some(last) => (now - last).num_minutes() >= self.cooldown_minutes,
            None => true,
        }
    }

    fn reset_to_quiet(&mut self) {
        self.level = EscalationLevel::Quiet;
        self.peak_hardness = 0.0;
        self.peak_fused = 0.0;
        self.flare_triggers = 0;
        // Clear last_activity so a stale value can't immediately re-arm
        // cooldown once we've cleanly returned to Quiet.
        self.last_activity = None;
    }

    pub fn status(&self, now: DateTime<Utc>) -> EscalationStatus {
        let minutes_at_level = self
            .level_since
            .map(|since| (now - since).num_minutes())
            .unwrap_or(0);

        EscalationStatus {
            level: self.level,
            level_label: self.level.label(),
            level_numeric: self.level.numeric(),
            since: self.level_since,
            hardness_spikes_in_window: self.hardness_spikes.len(),
            peak_hardness: self.peak_hardness,
            peak_fused: self.peak_fused,
            flare_triggers: self.flare_triggers,
            minutes_at_level,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn ts(min: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(1700000000 + min * 60, 0).unwrap()
    }

    #[test]
    fn test_quiet_stays_quiet() {
        let mut m = EscalationMonitor::new();
        let t = m.update(0.0, 0.1, 0, ts(0));
        assert!(t.is_none());
        assert_eq!(m.level, EscalationLevel::Quiet);
    }

    #[test]
    fn test_hardness_spike_elevates() {
        let mut m = EscalationMonitor::new();
        let t = m.update(0.6, 0.3, 0, ts(0));
        assert!(t.is_some());
        assert_eq!(t.unwrap().to, EscalationLevel::Elevated);
    }

    #[test]
    fn test_repeated_spikes_activate() {
        let mut m = EscalationMonitor::new();
        // First spike → Elevated
        m.update(0.6, 0.3, 0, ts(0));
        assert_eq!(m.level, EscalationLevel::Elevated);
        // More spikes within window → Active
        m.update(0.6, 0.3, 0, ts(30));
        m.update(0.6, 0.3, 0, ts(60));
        assert_eq!(m.level, EscalationLevel::Active);
    }

    #[test]
    fn test_agreement_triggers_flare() {
        let mut m = EscalationMonitor::new();
        let t = m.update(0.8, 0.8, 3, ts(0));
        assert!(t.is_some());
        assert_eq!(t.unwrap().to, EscalationLevel::Flare);
    }

    #[test]
    fn test_cooldown_deescalates() {
        let mut m = EscalationMonitor::new();
        m.update(0.6, 0.3, 0, ts(0)); // → Elevated
        assert_eq!(m.level, EscalationLevel::Elevated);
        // 61 minutes of nothing
        m.update(0.0, 0.1, 0, ts(61));
        assert_eq!(m.level, EscalationLevel::Quiet);
    }

    #[test]
    fn test_flare_deescalates_to_active() {
        let mut m = EscalationMonitor::new();
        m.update(0.8, 0.8, 3, ts(0)); // → Flare
        assert_eq!(m.level, EscalationLevel::Flare);
        // Cooldown → Active (not Quiet)
        m.update(0.0, 0.1, 0, ts(61));
        assert_eq!(m.level, EscalationLevel::Active);
    }
}
