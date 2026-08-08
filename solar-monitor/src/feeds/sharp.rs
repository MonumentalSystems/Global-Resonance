use chrono::{DateTime, NaiveDateTime, Utc};
use serde::{Deserialize, Serialize};

/// SHARP (Space-weather HMI Active Region Patch) parameters from JSOC.
///
/// These are the key magnetic field parameters that predict flare activity.
/// Available at 12-minute cadence from SDO/HMI vector magnetograms.
/// Uses hmi.sharp_cea_720s (cylindrical equal area projection) for
/// accurate parameters away from disk center.
///
/// 9 parameters following SolarFlareNet (Abduallah et al. 2023, Sci Rep):
/// TOTUSJH, TOTUSJZ, USFLUX, MEANALP, R_VALUE, TOTPOT, SAVNCPP, AREA_ACR, ABSNJZH
/// Plus MEANGBZ and SHRGT45 from Bobra & Couvidat (2015).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharpRecord {
    pub time_tag: DateTime<Utc>,
    /// HARP number (tracks a specific active region across the disk).
    pub harpnum: u32,
    /// Total unsigned magnetic flux (Mx). Size proxy.
    /// Higher = more energy available. X-producers typically > 1e22 Mx.
    pub usflux: f64,
    /// Mean gradient of vertical field (G/Mm). Shear/PIL proxy.
    /// Higher = stronger polarity inversion line = more flare-prone.
    pub meangbz: f64,
    /// Mean current helicity (G²/m). Twist proxy.
    /// Measures how twisted/sheared the magnetic field is.
    pub meanjzh: f64,
    /// Total unsigned current helicity (G²/m * Mm²). Energy proxy.
    /// Total amount of non-potential (free) magnetic energy.
    pub totusjh: f64,
    /// Fraction of area with shear angle > 45° (%). Complexity proxy.
    /// High SHRGT45 = complex, non-potential field configuration.
    pub shrgt45: f64,
    /// Active region area (μHem). Size.
    pub area_acr: f64,
    /// Sum of flux near polarity inversion line (Mx).
    /// WHERE reconnection happens — PIL is where [F, ∇F] is maximal.
    /// Highest-ranked individual predictor in several studies.
    pub r_value: f64,
    /// Total photospheric magnetic free energy density (ergs/cm³).
    /// (B_obs - B_pot)² — energy ABOVE potential field = available for flare.
    /// Directly measures free energy for the KT transition.
    pub totpot: f64,
    /// Total unsigned vertical current (A). Direct measure of ||J_z||.
    pub totusjz: f64,
    /// Sum of modulus of net current per polarity (A).
    /// Current imbalance = non-neutralized currents = twist asymmetry.
    pub savncpp: f64,
    /// Absolute value of net current helicity (G²/m).
    pub absnjzh: f64,
    /// Mean characteristic twist parameter α.
    pub meanalp: f64,
}

/// Rate of change of SHARP parameters over 1 hour.
/// The *trend* in magnetic complexity is often more predictive than
/// the absolute value — rapid increase in helicity or shear indicates
/// energy injection that precedes flares.
#[derive(Debug, Clone, Serialize)]
pub struct SharpTrend {
    pub harpnum: u32,
    /// dUSFLUX/dt (Mx/hour). Positive = flux emergence.
    pub d_usflux: f64,
    /// dMEANGBZ/dt (G/Mm/hour). Positive = gradient strengthening.
    pub d_meangbz: f64,
    /// dMEANJZH/dt. Positive = helicity injection.
    pub d_meanjzh: f64,
    /// dSHRGT45/dt (%/hour). Positive = increasing complexity.
    pub d_shrgt45: f64,
}

/// Experimental diagnostic motivated by Kim et al. (2026), ApJL 1005 L26.
///
/// Their model used 24-hour time series and a nonlinear interaction between
/// total and absolute current helicity.  This instantaneous proxy is therefore
/// exposed for validation only; it is not a calibrated flare probability and
/// is not included in `sharp_flare_risk`.
#[derive(Debug, Clone, Serialize)]
pub struct HelicityInteractionDiagnostic {
    pub interaction_proxy: f64,
    pub total_unsigned_current_helicity: f64,
    pub absolute_net_current_helicity: f64,
    pub source_doi: &'static str,
    pub operational_score_modified: bool,
}

pub fn helicity_interaction_diagnostic(record: &SharpRecord) -> HelicityInteractionDiagnostic {
    let total = if record.totusjh.is_finite() {
        record.totusjh.abs()
    } else {
        0.0
    };
    let absolute = if record.absnjzh.is_finite() {
        record.absnjzh.abs()
    } else {
        0.0
    };
    HelicityInteractionDiagnostic {
        interaction_proxy: total * absolute,
        total_unsigned_current_helicity: total,
        absolute_net_current_helicity: absolute,
        source_doi: "10.3847/2041-8213/ae6cf8",
        operational_score_modified: false,
    }
}

/// Fetch SHARP parameters for all active HARPs from JSOC.
///
/// Uses the JSOC JSON API. Returns the latest 12-minute record for each HARP.
/// Note: JSOC data has ~2-3 hour latency compared to real-time GOES.
pub async fn fetch_latest(client: &reqwest::Client) -> Result<Vec<SharpRecord>, super::FeedError> {
    // Query: all HARPs, latest available record
    // Uses hmi.sharp_cea_720s (cylindrical equal area) for projection-corrected
    // parameters. CEA is more accurate for ARs away from disk center.
    let now = Utc::now();
    // JSOC processing lag is variable (3-48h+). Try progressively older windows.
    let query_time = (now - chrono::Duration::hours(6)).format("%Y.%m.%d_%H:%M:%S_TAI");
    let url = format!(
        "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info?\
        ds=hmi.sharp_cea_720s[][{}/2h@720s]&\
        key=T_REC,HARPNUM,USFLUX,MEANGBZ,MEANJZH,TOTUSJH,SHRGT45,AREA_ACR,\
        R_VALUE,TOTPOT,TOTUSJZ,SAVNCPP,ABSNJZH,MEANALP&\
        op=rs_list",
        query_time
    );

    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| super::FeedError {
            feed: "sharp".into(),
            message: format!("JSOC HTTP error: {e}"),
        })?;

    let data: JsocResponse = resp.json().await.map_err(|e| super::FeedError {
        feed: "sharp".into(),
        message: format!("JSOC JSON error: {e}"),
    })?;

    if data.status != 0 || data.count == 0 {
        return Ok(Vec::new());
    }

    let kw = &data.keywords;
    let names: Vec<&str> = kw.iter().map(|k| k.name.as_str()).collect();
    let n_records = kw[0].values.len();

    let mut records: Vec<SharpRecord> = Vec::new();
    // Group by HARP, keep only the latest record per HARP
    let mut latest_by_harp: std::collections::HashMap<u32, SharpRecord> =
        std::collections::HashMap::new();

    for i in 0..n_records {
        let get_str = |name: &str| -> &str {
            let idx = names.iter().position(|&n| n == name).unwrap_or(0);
            kw[idx].values[i].as_str().unwrap_or("")
        };
        let get_f64 = |name: &str| -> f64 {
            let idx = names.iter().position(|&n| n == name).unwrap_or(0);
            match &kw[idx].values[i] {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
                serde_json::Value::String(s) => s.parse().unwrap_or(0.0),
                _ => 0.0,
            }
        };
        let get_u32 = |name: &str| -> u32 { get_f64(name) as u32 };

        let time_str = get_str("T_REC");
        let time_tag = parse_jsoc_time(time_str).unwrap_or(now);
        let harpnum = get_u32("HARPNUM");
        let usflux = get_f64("USFLUX");

        // Skip invalid records
        if usflux <= 0.0 || harpnum == 0 {
            continue;
        }

        let record = SharpRecord {
            time_tag,
            harpnum,
            usflux,
            meangbz: get_f64("MEANGBZ"),
            meanjzh: get_f64("MEANJZH"),
            totusjh: get_f64("TOTUSJH"),
            shrgt45: get_f64("SHRGT45"),
            area_acr: get_f64("AREA_ACR"),
            r_value: get_f64("R_VALUE"),
            totpot: get_f64("TOTPOT"),
            totusjz: get_f64("TOTUSJZ"),
            savncpp: get_f64("SAVNCPP"),
            absnjzh: get_f64("ABSNJZH"),
            meanalp: get_f64("MEANALP"),
        };

        // Keep latest per HARP
        let entry = latest_by_harp.entry(harpnum).or_insert(record.clone());
        if record.time_tag > entry.time_tag {
            *entry = record;
        }
    }

    Ok(latest_by_harp.into_values().collect())
}

/// Compute a flare risk score from SHARP parameters.
///
/// Uses all 9 SolarFlareNet parameters plus SHRGT45 and MEANGBZ.
/// Weights reflect Bobra & Couvidat (2015) feature importance ranking
/// updated with the SolarFlareNet findings: R_VALUE and TOTPOT are
/// the strongest individual predictors of flare productivity.
///
/// Returns 0..1 flare risk for this active region.
pub fn sharp_flare_risk(record: &SharpRecord) -> f64 {
    // R_VALUE: flux near PIL (Mx). Most predictive single parameter.
    // Quiet AR: ~1e3, flare-prone: ~1e4-1e5, X-producing: >1e5
    let r_value_score = if record.r_value > 100.0 {
        ((record.r_value.log10() - 2.0) / 3.0).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // TOTPOT: free magnetic energy (ergs/cm³).
    // Quiet: ~1e3, active: ~1e4, X-producing: >5e4
    let totpot_score = if record.totpot > 100.0 {
        ((record.totpot.log10() - 2.0) / 3.0).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // USFLUX: total unsigned flux (Mx).
    let flux_score = if record.usflux > 1e18 {
        ((record.usflux.log10() - 21.0) / 2.0).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // SHRGT45: shear fraction (%). Non-planarity of the field.
    let shear_score = (record.shrgt45 / 40.0).clamp(0.0, 1.0);

    // MEANGBZ: field gradient at PIL (G/Mm).
    let grad_score = (record.meangbz / 150.0).clamp(0.0, 1.0);

    // TOTUSJH: total unsigned current helicity.
    let helicity_score = if record.totusjh.abs() > 100.0 {
        (record.totusjh.abs().log10() / 5.0).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // TOTUSJZ: total unsigned vertical current.
    let current_score = if record.totusjz > 1e10 {
        ((record.totusjz.log10() - 10.0) / 3.0).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Weighted combination:
    // R_VALUE and TOTPOT are the strongest per SolarFlareNet ablation.
    // USFLUX is Bobra & Couvidat's top-ranked.
    let risk = r_value_score * 0.20
        + totpot_score * 0.20
        + flux_score * 0.15
        + shear_score * 0.15
        + grad_score * 0.10
        + helicity_score * 0.10
        + current_score * 0.10;
    risk.clamp(0.0, 1.0)
}

// JSOC JSON response format
#[derive(Deserialize)]
struct JsocResponse {
    status: i32,
    count: usize,
    #[serde(default)]
    keywords: Vec<JsocKeyword>,
}

#[derive(Deserialize)]
struct JsocKeyword {
    name: String,
    values: Vec<serde_json::Value>,
}

fn parse_jsoc_time(s: &str) -> Option<DateTime<Utc>> {
    // JSOC format: "2024.05.10_00:00:00_TAI"
    let s = s.trim_end_matches("_TAI").trim_end_matches("_UTC");
    NaiveDateTime::parse_from_str(s, "%Y.%m.%d_%H:%M:%S")
        .ok()
        .map(|ndt| ndt.and_utc())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_jsoc_time() {
        let t = parse_jsoc_time("2024.05.10_00:00:00_TAI").unwrap();
        assert_eq!(t.year(), 2024);
    }

    use chrono::Datelike;

    fn make_record(
        usflux: f64,
        meangbz: f64,
        shrgt45: f64,
        totusjh: f64,
        r_value: f64,
        totpot: f64,
        totusjz: f64,
    ) -> SharpRecord {
        SharpRecord {
            time_tag: Utc::now(),
            harpnum: 1,
            usflux,
            meangbz,
            meanjzh: 0.01,
            totusjh,
            shrgt45,
            area_acr: 100.0,
            r_value,
            totpot,
            totusjz,
            savncpp: 0.0,
            absnjzh: 0.0,
            meanalp: 0.0,
        }
    }

    #[test]
    fn test_flare_risk_quiet_ar() {
        let rec = make_record(1e20, 10.0, 5.0, 10.0, 100.0, 100.0, 1e9);
        assert!(sharp_flare_risk(&rec) < 0.3);
    }

    #[test]
    fn test_flare_risk_complex_ar() {
        let rec = make_record(5e22, 150.0, 40.0, 1e5, 1e5, 5e4, 1e13);
        assert!(sharp_flare_risk(&rec) > 0.6);
    }

    #[test]
    fn test_helicity_interaction_is_diagnostic_only() {
        let mut rec = make_record(5e22, 150.0, 40.0, 2e4, 1e5, 5e4, 1e13);
        rec.absnjzh = 30.0;
        let diagnostic = helicity_interaction_diagnostic(&rec);
        assert_eq!(diagnostic.interaction_proxy, 6e5);
        assert!(!diagnostic.operational_score_modified);
        let json = serde_json::to_value(&diagnostic).unwrap();
        assert_eq!(json["source_doi"], "10.3847/2041-8213/ae6cf8");
        assert_eq!(json["operational_score_modified"], false);
    }
}
