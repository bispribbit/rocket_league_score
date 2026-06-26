//! Calibration scan: measure how common mid-ladder big-delta (party) lobbies are
//! on ballchasing, WITHOUT downloading any replay files.
//!
//! The list endpoint already returns each replay's per-player ranks, so we can
//! compute the within-lobby spread (top player MMR minus lobby median) straight
//! from list pages. This tells us the yield rate and how many summaries we must
//! scan to collect a target number of training lobbies.
//!
//! Usage:
//!   cargo run --release --example scan_mixed_rank -- [min_rank] [max_rank] [max_scan]
//!   cargo run --release --example scan_mixed_rank -- gold-1 champion-3 20000
//!
//! Requires `BALLCHASING_API_KEY` (and `DATABASE_URL`, read by the shared config).

#![allow(clippy::indexing_slicing)]

use std::collections::HashSet;

use anyhow::Result;
use ballchasing_downloader::BallchasingClient;
use ballchasing_downloader::api::client::next_replay_list_url_with_max_rank;
use replay_structs::{GameMode, ReplaySummary};
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

/// Per-request page size (API max is 200).
const PAGE_SIZE: usize = 200;

/// Lobby-median MMR window we consider "mid-ladder" (≈ Gold-1 … Champion-3).
const MID_LADDER_MMR_LOW: f64 = 450.0;
const MID_LADDER_MMR_HIGH: f64 = 1450.0;

/// Canonical coarse-rank MMR (tier midpoints), matching the SQL mapping used for
/// the existing dataset audit. Index order = ascending rank.
const TIER_MMR: &[(&str, f64)] = &[
    ("bronze-1", 130.0),
    ("bronze-2", 194.0),
    ("bronze-3", 257.0),
    ("silver-1", 321.0),
    ("silver-2", 386.0),
    ("silver-3", 451.0),
    ("gold-1", 516.0),
    ("gold-2", 580.0),
    ("gold-3", 644.0),
    ("platinum-1", 709.0),
    ("platinum-2", 773.0),
    ("platinum-3", 837.0),
    ("diamond-1", 902.0),
    ("diamond-2", 966.0),
    ("diamond-3", 1030.0),
    ("champion-1", 1127.0),
    ("champion-2", 1258.0),
    ("champion-3", 1388.0),
    ("grand-champion-1", 1520.0),
    ("grand-champion-2", 1651.0),
    ("grand-champion-3", 1782.0),
    ("supersonic-legend", 2200.0),
];

fn rank_id_to_mmr(rank_id: &str) -> Option<f64> {
    TIER_MMR
        .iter()
        .find(|(id, _)| *id == rank_id)
        .map(|(_, mmr)| *mmr)
}

fn median(sorted: &[f64]) -> f64 {
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        f64::midpoint(sorted[mid - 1], sorted[mid])
    } else {
        sorted[mid]
    }
}

/// Known-rank player MMRs in a lobby, pulled from both team rosters.
fn lobby_player_mmrs(summary: &ReplaySummary) -> Vec<f64> {
    let mut mmrs = Vec::new();
    for team in [summary.blue.as_ref(), summary.orange.as_ref()]
        .into_iter()
        .flatten()
    {
        for player in team.players.iter().flatten() {
            if let Some(rank) = &player.rank
                && let Some(mmr) = rank_id_to_mmr(&rank.id)
            {
                mmrs.push(mmr);
            }
        }
    }
    mmrs
}

#[derive(Default)]
struct Tally {
    scanned: usize,
    with_3plus_known: usize,
    gap_ge_100: usize,
    gap_ge_150: usize,
    gap_ge_200: usize,
    mid_gap_ge_100: usize,
    mid_gap_ge_150: usize,
    mid_gap_ge_200: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("warn"))
        .init();

    let args: Vec<String> = std::env::args().collect();
    let min_rank = args.get(1).cloned().unwrap_or_else(|| "gold-1".to_string());
    let max_rank = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "champion-3".to_string());
    let max_scan: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20_000);

    println!(
        "Scanning ranked-standard band [{min_rank} .. {max_rank}], up to {max_scan} summaries\n"
    );

    let client = BallchasingClient::new()?;
    let mut tally = Tally::default();
    let mut seen_ids: HashSet<String> = HashSet::new();

    let first = client
        .list_replays_in_band(GameMode::RankedStandard, &min_rank, &max_rank, PAGE_SIZE)
        .await?;
    let mut next_url = first.next.clone();
    process_page(&first.list, &mut tally, &mut seen_ids);

    while tally.scanned < max_scan {
        let Some(raw_next) = next_url.as_deref() else {
            println!("Reached end of results (no more pages).");
            break;
        };
        let page_url = next_replay_list_url_with_max_rank(raw_next, &max_rank)?;
        let page = match client.fetch_replay_list_page(&page_url).await {
            Ok(page) => page,
            Err(error) => {
                warn!(%error, "Page fetch failed; stopping scan early");
                break;
            }
        };
        if page.list.is_empty() {
            break;
        }
        process_page(&page.list, &mut tally, &mut seen_ids);
        next_url.clone_from(&page.next);
        if tally.scanned % 2000 < PAGE_SIZE {
            info!(
                scanned = tally.scanned,
                mid_gap150 = tally.mid_gap_ge_150,
                "progress"
            );
        }
    }

    report(&tally, &min_rank, &max_rank);
    Ok(())
}

fn process_page(summaries: &[ReplaySummary], tally: &mut Tally, seen: &mut HashSet<String>) {
    for summary in summaries {
        if !seen.insert(summary.id.clone()) {
            continue;
        }
        tally.scanned += 1;

        let mut mmrs = lobby_player_mmrs(summary);
        if mmrs.len() < 3 {
            continue;
        }
        tally.with_3plus_known += 1;
        mmrs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = median(&mmrs);
        let top_gap = mmrs.last().copied().unwrap_or(med) - med;
        let is_mid = (MID_LADDER_MMR_LOW..MID_LADDER_MMR_HIGH).contains(&med);

        if top_gap >= 100.0 {
            tally.gap_ge_100 += 1;
            if is_mid {
                tally.mid_gap_ge_100 += 1;
            }
        }
        if top_gap >= 150.0 {
            tally.gap_ge_150 += 1;
            if is_mid {
                tally.mid_gap_ge_150 += 1;
            }
        }
        if top_gap >= 200.0 {
            tally.gap_ge_200 += 1;
            if is_mid {
                tally.mid_gap_ge_200 += 1;
            }
        }
    }
}

fn report(tally: &Tally, min_rank: &str, max_rank: &str) {
    println!("\n==================== CALIBRATION RESULT ====================");
    println!("band                : ranked-standard [{min_rank} .. {max_rank}]");
    println!("summaries scanned   : {}", tally.scanned);
    println!("  with >=3 known    : {}", tally.with_3plus_known);
    println!(
        "any-tier  gap>=100/150/200 : {} / {} / {}",
        tally.gap_ge_100, tally.gap_ge_150, tally.gap_ge_200
    );
    println!(
        "mid-ladder gap>=100/150/200: {} / {} / {}",
        tally.mid_gap_ge_100, tally.mid_gap_ge_150, tally.mid_gap_ge_200
    );

    let scanned = tally.scanned.max(1) as f64;
    let yield_150 = tally.mid_gap_ge_150 as f64 / scanned;
    let yield_100 = tally.mid_gap_ge_100 as f64 / scanned;
    println!(
        "\nmid-ladder yield    : {:.3}% (gap>=150), {:.3}% (gap>=100)",
        100.0 * yield_150,
        100.0 * yield_100
    );

    if yield_150 > 0.0 {
        for target in [1_000usize, 2_500] {
            let needed = (target as f64 / yield_150).ceil();
            let requests = (needed / PAGE_SIZE as f64).ceil();
            let minutes = requests / 60.0; // list endpoint capped at ~1 req/s
            println!(
                "to collect {target} (gap>=150): scan ~{needed:.0} summaries (~{requests:.0} requests, ~{minutes:.0} min at 1 req/s)"
            );
        }
    } else {
        println!("(no gap>=150 mid-ladder lobbies found in this sample — widen band or rethink)");
    }
    println!("===========================================================");
}
