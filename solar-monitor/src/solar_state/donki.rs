//! DONKI (NASA Space Weather Database) feed ingestion.
//!
//! Fetches CMEs, flares, SEPs, geomagnetic storms, HSS, and IP shocks
//! from the DONKI REST API and populates the solar state layers.

use chrono::{DateTime, Duration, NaiveDateTime, Utc};
use serde::Deserialize;

use super::activity::{ActivityState, FlareEvent, XrayBackground};
use super::cycle::CycleState;
use super::disk::DiskState;
use super::geospace::{GeomagStorm, GeospaceState};
use super::heliosphere::{CmeInTransit, HeliosphereState, HighSpeedStream, IpShock};
use super::SolarState;

const DONKI_BASE: &str = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get";
const SWPC_BASE: &str = "https://services.swpc.noaa.gov/json";

/// Fetch all DONKI + SWPC data and build a SolarState snapshot.
pub async fn fetch_solar_state(client: &reqwest::Client) -> Result<SolarState, String> {
    let now = Utc::now();
    let start_30d = (now - Duration::days(30)).format("%Y-%m-%d").to_string();
    let end = now.format("%Y-%m-%d").to_string();
    let start_7d = (now - Duration::days(7)).format("%Y-%m-%d").to_string();

    // Pre-compute URLs (tokio::join! borrows them, so they must outlive the join)
    let url_cme = format!("{}/CME?startDate={}&endDate={}", DONKI_BASE, start_30d, end);
    let url_flr = format!("{}/FLR?startDate={}&endDate={}", DONKI_BASE, start_7d, end);
    let url_gst = format!("{}/GST?startDate={}&endDate={}", DONKI_BASE, start_30d, end);
    let url_hss = format!("{}/HSS?startDate={}&endDate={}", DONKI_BASE, start_30d, end);
    let url_ips = format!("{}/IPS?startDate={}&endDate={}", DONKI_BASE, start_30d, end);
    let url_regions = format!("{}/solar_regions.json", SWPC_BASE);
    let url_cycle = format!(
        "{}/solar-cycle/observed-solar-cycle-indices.json",
        SWPC_BASE
    );
    let url_dst = format!("{}/geospace/geospace_dst_1_hour.json", SWPC_BASE);
    let url_probs = format!("{}/solar_probabilities.json", SWPC_BASE);
    // Real-time feeds for solar wind + X-ray
    let url_sw_mag =
        "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json".to_string();
    let url_sw_plasma =
        "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json".to_string();
    let url_xray = format!("{}/goes/primary/xrays-1-day.json", SWPC_BASE);
    let url_kp_rt = format!("{}/planetary_k_index_1m.json", SWPC_BASE);

    // Fetch all sources concurrently
    let (
        cme_res,
        flr_res,
        gst_res,
        hss_res,
        ips_res,
        regions_res,
        cycle_res,
        dst_res,
        probs_res,
        sw_mag_res,
        sw_plasma_res,
        xray_res,
        kp_rt_res,
    ) = tokio::join!(
        fetch_json::<Vec<DonkiCme>>(client, &url_cme),
        fetch_json::<Vec<DonkiFlare>>(client, &url_flr),
        fetch_json::<Vec<DonkiGst>>(client, &url_gst),
        fetch_json::<Vec<DonkiHss>>(client, &url_hss),
        fetch_json::<Vec<DonkiIps>>(client, &url_ips),
        fetch_json::<Vec<SwpcRegion>>(client, &url_regions),
        fetch_json::<Vec<SwpcCycleIndex>>(client, &url_cycle),
        fetch_json::<Vec<SwpcDst>>(client, &url_dst),
        fetch_json::<Vec<SwpcFlareProb>>(client, &url_probs),
        fetch_json::<Vec<Vec<serde_json::Value>>>(client, &url_sw_mag),
        fetch_json::<Vec<Vec<serde_json::Value>>>(client, &url_sw_plasma),
        fetch_json::<Vec<SwpcXray>>(client, &url_xray),
        fetch_json::<Vec<Vec<serde_json::Value>>>(client, &url_kp_rt),
    );

    // Build cycle state
    let mut cycle = CycleState::new();
    if let Ok(indices) = &cycle_res {
        if let Some(latest) = indices.last() {
            cycle.update_from_indices(
                latest.ssn.unwrap_or(0.0),
                latest.smoothed_ssn.unwrap_or(-1.0),
                latest.f10_7.unwrap_or(70.0),
                latest.smoothed_f10_7.unwrap_or(-1.0),
            );
        }
    }

    // Build disk state
    let mut disk = DiskState::new();
    if let Ok(regions) = &regions_res {
        // Filter to today's regions only
        let today = now.format("%Y-%m-%d").to_string();
        for r in regions {
            if r.observed_date == today {
                disk.active_regions.push(super::disk::ActiveRegion {
                    region: r.region,
                    latitude: r.latitude,
                    longitude: r.longitude,
                    location: r.location.clone().unwrap_or_default(),
                    area: r.area.unwrap_or(0),
                    spot_class: r.spot_class.clone().unwrap_or_default(),
                    number_spots: r.number_spots.unwrap_or(0),
                    mag_class: r.mag_class.clone().unwrap_or_default(),
                    c_flare_probability: r.c_flare_probability.unwrap_or(0.0),
                    m_flare_probability: r.m_flare_probability.unwrap_or(0.0),
                    x_flare_probability: r.x_flare_probability.unwrap_or(0.0),
                    proton_probability: r.proton_probability.unwrap_or(0.0),
                    c_xray_events: r.c_xray_events.unwrap_or(0),
                    m_xray_events: r.m_xray_events.unwrap_or(0),
                    x_xray_events: r.x_xray_events.unwrap_or(0),
                });
            }
        }
        disk.total_spot_area = disk.active_regions.iter().map(|ar| ar.area).sum();
        disk.last_update = Some(now);
    }
    if let Ok(probs) = &probs_res {
        if let Some(latest) = probs.last() {
            disk.flare_probabilities = super::disk::FlareProbs {
                c_class_1_day: latest.c_class_1_day.unwrap_or(0.0),
                m_class_1_day: latest.m_class_1_day.unwrap_or(0.0),
                x_class_1_day: latest.x_class_1_day.unwrap_or(0.0),
                proton_1_day: latest.proton_1_day.unwrap_or(0.0),
            };
        }
    }

    // Build activity state
    let mut activity = ActivityState::new();
    if let Ok(flares) = &flr_res {
        let cutoff_24h = now - Duration::hours(24);
        for f in flares {
            if let (Some(begin), Some(peak)) = (
                parse_donki_time(&f.begin_time),
                parse_donki_time(&f.peak_time),
            ) {
                let event = FlareEvent {
                    begin_time: begin,
                    peak_time: peak,
                    end_time: f.end_time.as_ref().and_then(|t| parse_donki_time(t)),
                    class_type: f.class_type.clone().unwrap_or_default(),
                    source_location: f.source_location.clone().unwrap_or_default(),
                    active_region: f.active_region_num,
                    linked_cme_ids: f
                        .linked_events
                        .as_ref()
                        .map(|events| {
                            events
                                .iter()
                                .filter(|e| e.activity_id.contains("CME"))
                                .map(|e| e.activity_id.clone())
                                .collect()
                        })
                        .unwrap_or_default(),
                    linked_sep_ids: f
                        .linked_events
                        .as_ref()
                        .map(|events| {
                            events
                                .iter()
                                .filter(|e| e.activity_id.contains("SEP"))
                                .map(|e| e.activity_id.clone())
                                .collect()
                        })
                        .unwrap_or_default(),
                };
                if begin > cutoff_24h {
                    activity.flares_24h.push(event.clone());
                }
                activity.latest_flare = Some(event);
            }
        }
    }
    if let Ok(cmes) = &cme_res {
        let cutoff_24h = now - Duration::hours(24);
        let cutoff_7d = now - Duration::days(7);
        activity.cme_count_24h = cmes
            .iter()
            .filter(|c| parse_donki_time(&c.start_time).map_or(false, |t| t > cutoff_24h))
            .count();
        activity.cme_count_7d = cmes
            .iter()
            .filter(|c| parse_donki_time(&c.start_time).map_or(false, |t| t > cutoff_7d))
            .count();
    }

    // Build heliosphere state
    let mut heliosphere = HeliosphereState::new();
    if let Ok(cmes) = &cme_res {
        for cme in cmes {
            if let Some(analyses) = &cme.cme_analyses {
                for analysis in analyses {
                    if analysis.is_earth_gb.unwrap_or(false) {
                        let launch = parse_donki_time(&cme.start_time);
                        let speed = analysis.speed.unwrap_or(0.0);
                        // Estimate arrival: distance / speed (1 AU ≈ 1.5e8 km)
                        let est_arrival: Option<DateTime<Utc>> = if speed > 0.0 {
                            launch.map(|t| t + Duration::hours((1.5e8 / speed / 3600.0) as i64))
                        } else {
                            None
                        };
                        let hours_to = est_arrival
                            .map(|a| (a - now).num_minutes() as f64 / 60.0)
                            .filter(|&h| h > 0.0);

                        if let Some(t) = launch {
                            heliosphere.earth_directed_cmes.push(CmeInTransit {
                                launch_time: t,
                                speed,
                                half_angle: analysis.half_angle.unwrap_or(0.0),
                                estimated_arrival: est_arrival,
                                hours_to_arrival: hours_to,
                                source_region: cme.active_region_num,
                                associated_flare: None,
                                activity_id: cme.activity_id.clone(),
                            });
                        }
                    }
                }
            }
        }
    }
    if let Ok(hss_list) = &hss_res {
        for hss in hss_list {
            if let Some(t) = parse_donki_time(&hss.event_time) {
                heliosphere.active_hss.push(HighSpeedStream {
                    event_time: t,
                    activity_id: hss.hss_id.clone(),
                });
            }
        }
    }
    if let Ok(shocks) = &ips_res {
        for s in shocks {
            if let Some(t) = parse_donki_time(&s.event_time) {
                heliosphere.recent_shocks.push(IpShock {
                    event_time: t,
                    location: s.location.clone().unwrap_or_default(),
                    activity_id: s.activity_id.clone(),
                });
            }
        }
    }

    // Populate solar wind from real-time DSCOVR/ACE data
    if let (Ok(mag_rows), Ok(plasma_rows)) = (&sw_mag_res, &sw_plasma_res) {
        // mag: [time_tag, bx_gsm, by_gsm, bz_gsm, ...]
        // plasma: [time_tag, density, speed, temperature]
        if let Some(last_mag) = mag_rows.last() {
            if last_mag.len() >= 4 {
                heliosphere.solar_wind.bz = parse_json_f64(&last_mag[3]).unwrap_or(0.0);
                heliosphere.solar_wind.by = parse_json_f64(&last_mag[2]).unwrap_or(0.0);
            }
        }
        if let Some(last_plasma) = plasma_rows.last() {
            if last_plasma.len() >= 3 {
                heliosphere.solar_wind.density = parse_json_f64(&last_plasma[1]).unwrap_or(5.0);
                heliosphere.solar_wind.speed = parse_json_f64(&last_plasma[2]).unwrap_or(400.0);
                // Dynamic pressure: P_dyn = n * m_p * v^2 (in nPa)
                let n = heliosphere.solar_wind.density;
                let v = heliosphere.solar_wind.speed;
                heliosphere.solar_wind.dynamic_pressure = 1.672e-6 * n * v * v; // nPa
            }
        }
    }

    // Populate X-ray flux from GOES
    if let Ok(xray_data) = &xray_res {
        // Filter to 0.1-0.8nm (long channel), take latest
        if let Some(latest) = xray_data
            .iter()
            .filter(|x| x.energy.as_deref() == Some("0.1-0.8nm"))
            .last()
        {
            activity.xray_flux = latest.flux.unwrap_or(0.0);
            activity.xray_background = if activity.xray_flux >= 1e-5 {
                XrayBackground::M
            } else if activity.xray_flux >= 1e-6 {
                XrayBackground::C
            } else if activity.xray_flux >= 1e-7 {
                XrayBackground::B
            } else {
                XrayBackground::A
            };
        }
    }

    // Build geospace state
    let mut geospace = GeospaceState::new();
    if let Ok(dst_data) = &dst_res {
        if let Some(latest) = dst_data.last() {
            geospace.dst = latest.dst.unwrap_or(0.0);
        }
    }
    // Real-time Kp (1-min cadence, more current than DONKI storms)
    if let Ok(kp_rows) = &kp_rt_res {
        // Format: [time_tag, Kp, Kp_fraction, ...]
        if let Some(last_kp) = kp_rows.last() {
            if last_kp.len() >= 2 {
                geospace.kp = parse_json_f64(&last_kp[1]).unwrap_or(0.0);
            }
        }
    }
    if let Ok(storms) = &gst_res {
        for s in storms {
            if let Some(t) = parse_donki_time(&s.start_time) {
                let kp_vals: Vec<f64> = s
                    .all_kp_index
                    .as_ref()
                    .map(|indices| indices.iter().filter_map(|i| i.kp_index).collect())
                    .unwrap_or_default();
                let peak_kp = kp_vals.iter().cloned().fold(0.0f64, f64::max);
                geospace.kp = geospace.kp.max(peak_kp);
                geospace.active_storms.push(GeomagStorm {
                    start_time: t,
                    kp_indices: kp_vals,
                    peak_kp,
                });
            }
        }
    }
    geospace.update_storm_level();

    // Assemble full state
    let mut state = SolarState {
        timestamp: now,
        cycle,
        disk,
        activity,
        heliosphere,
        geospace,
        threat_level: 0.0,
        summary: String::new(),
    };

    state.threat_level = state.compute_threat();
    state.summary = state.generate_summary();

    Ok(state)
}

// ----- DONKI JSON structs -----

#[derive(Deserialize)]
struct DonkiCme {
    #[serde(rename = "activityID")]
    activity_id: String,
    #[serde(rename = "startTime")]
    start_time: String,
    #[serde(rename = "activeRegionNum")]
    active_region_num: Option<u32>,
    #[serde(rename = "cmeAnalyses")]
    cme_analyses: Option<Vec<CmeAnalysis>>,
}

#[derive(Deserialize)]
struct CmeAnalysis {
    speed: Option<f64>,
    #[serde(rename = "halfAngle")]
    half_angle: Option<f64>,
    #[serde(rename = "isEarthGB")]
    is_earth_gb: Option<bool>,
}

#[derive(Deserialize)]
struct DonkiFlare {
    #[serde(rename = "beginTime")]
    begin_time: String,
    #[serde(rename = "peakTime")]
    peak_time: String,
    #[serde(rename = "endTime")]
    end_time: Option<String>,
    #[serde(rename = "classType")]
    class_type: Option<String>,
    #[serde(rename = "sourceLocation")]
    source_location: Option<String>,
    #[serde(rename = "activeRegionNum")]
    active_region_num: Option<u32>,
    #[serde(rename = "linkedEvents")]
    linked_events: Option<Vec<LinkedEvent>>,
}

#[derive(Deserialize)]
struct LinkedEvent {
    #[serde(rename = "activityID")]
    activity_id: String,
}

#[derive(Deserialize)]
struct DonkiGst {
    #[serde(rename = "startTime")]
    start_time: String,
    #[serde(rename = "allKpIndex")]
    all_kp_index: Option<Vec<KpEntry>>,
}

#[derive(Deserialize)]
struct KpEntry {
    #[serde(rename = "kpIndex")]
    kp_index: Option<f64>,
}

#[derive(Deserialize)]
struct DonkiHss {
    #[serde(rename = "hssID")]
    hss_id: String,
    #[serde(rename = "eventTime")]
    event_time: String,
}

#[derive(Deserialize)]
struct DonkiIps {
    #[serde(rename = "activityID")]
    activity_id: String,
    #[serde(rename = "eventTime")]
    event_time: String,
    location: Option<String>,
}

// ----- SWPC JSON structs -----

#[derive(Deserialize)]
struct SwpcRegion {
    observed_date: String,
    region: u32,
    latitude: i32,
    longitude: i32,
    location: Option<String>,
    area: Option<u32>,
    spot_class: Option<String>,
    number_spots: Option<u32>,
    mag_class: Option<String>,
    c_flare_probability: Option<f64>,
    m_flare_probability: Option<f64>,
    x_flare_probability: Option<f64>,
    proton_probability: Option<f64>,
    c_xray_events: Option<u32>,
    m_xray_events: Option<u32>,
    x_xray_events: Option<u32>,
}

#[derive(Deserialize)]
struct SwpcCycleIndex {
    #[serde(rename = "time-tag")]
    time_tag: String,
    ssn: Option<f64>,
    smoothed_ssn: Option<f64>,
    #[serde(rename = "f10.7")]
    f10_7: Option<f64>,
    #[serde(rename = "smoothed_f10.7")]
    smoothed_f10_7: Option<f64>,
}

#[derive(Deserialize)]
struct SwpcDst {
    time_tag: String,
    dst: Option<f64>,
}

#[derive(Deserialize)]
struct SwpcFlareProb {
    c_class_1_day: Option<f64>,
    m_class_1_day: Option<f64>,
    x_class_1_day: Option<f64>,
    #[serde(rename = "10mev_protons_1_day")]
    proton_1_day: Option<f64>,
}

#[derive(Deserialize)]
struct SwpcXray {
    flux: Option<f64>,
    energy: Option<String>,
}

// ----- Helpers -----

fn parse_json_f64(v: &serde_json::Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}

async fn fetch_json<T: serde::de::DeserializeOwned>(
    client: &reqwest::Client,
    url: &str,
) -> Result<T, String> {
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("HTTP error fetching {}: {}", url, e))?;
    resp.json::<T>()
        .await
        .map_err(|e| format!("JSON parse error from {}: {}", url, e))
}

fn parse_donki_time(s: &str) -> Option<DateTime<Utc>> {
    // DONKI format: "2026-03-30T03:19Z" or "2026-03-30T03:19:00Z"
    NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%MZ")
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%SZ"))
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.fZ"))
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M"))
        .ok()
        .map(|ndt| ndt.and_utc())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_donki_time() {
        assert!(parse_donki_time("2026-03-30T03:19Z").is_some());
        assert!(parse_donki_time("2026-03-30T03:19:00Z").is_some());
    }
}
