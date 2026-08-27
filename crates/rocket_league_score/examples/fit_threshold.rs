//! Fit the smurf flag threshold from data and publish the precision/recall curve.
//!
//! Step 2.5 of `docs/smurf-detection-handoff.md`. The shipped rule flags a player at
//! `+200 MMR` over their lobby median; step 1 measured the achievable margin at `+10` to
//! `+18 MMR`, so the rule almost never fires and detection reads 0–3 %. That is a
//! threshold problem, not an ordering problem — `concordance = 0.619` on mixed lobbies
//! says the ordering carries real signal.
//!
//! This binary does three things the in-training `mixed_detection_rate` cannot:
//!
//! 1. **Scores every player, not just the top-labelled one in a mixed lobby.** False
//!    positives only exist when innocent players can be flagged, and without them there is
//!    no precision to report.
//! 2. **Fits on one set of lobbies and reports on another.** The fit/held-out split is by
//!    replay (see `threshold::split_by_replay`), because margins are lobby-relative.
//! 3. **Publishes the whole curve.** At `conc = 0.619` no single operating point is good;
//!    printing the trade-off is more honest than choosing the threshold that flatters the
//!    detection rate.
//!
//! Input is the CSV written by `revalidate --dump-predictions`, so re-fitting costs
//! milliseconds instead of the ~1 h a scoring pass takes.
//!
//! Usage:
//!   cargo run --release --example revalidate -- \
//!       --model models/lstm_v20/checkpoint_best --split evaluation \
//!       --dump-predictions data/lstm_v20_eval_predictions.csv
//!   cargo run --release --example fit_threshold -- \
//!       --predictions data/lstm_v20_eval_predictions.csv
//!
//! Needs no database and no GPU.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use ml_model_training::threshold::{
    self, MarginSample, OperatingPoint, PlayerPrediction, SMURF_LABEL_MARGIN_MMR,
};
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(name = "fit_threshold")]
#[command(about = "Fit the smurf flag threshold and print its precision/recall curve", long_about = None)]
struct Args {
    /// CSV written by `revalidate --dump-predictions`.
    #[arg(short = 'p', long)]
    predictions: PathBuf,

    /// Share of lobbies used to fit the threshold. The rest is held out for reporting.
    #[arg(long, default_value_t = 0.5)]
    fit_fraction: f64,

    /// Number of curve rows to print, sampled evenly across recall.
    #[arg(long, default_value_t = 20)]
    curve_rows: usize,

    /// Also write the full held-out curve to this CSV.
    #[arg(long)]
    dump_curve: Option<PathBuf>,
}

fn read_predictions(path: &PathBuf) -> Result<Vec<PlayerPrediction>> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let mut rows = Vec::new();
    for (index, record) in reader.records().enumerate() {
        let record = record.with_context(|| format!("bad CSV record at row {}", index + 2))?;
        let field = |column: usize| -> Result<&str> {
            record
                .get(column)
                .with_context(|| format!("missing column {column} at row {}", index + 2))
        };
        rows.push(PlayerPrediction {
            replay_id: Uuid::parse_str(field(0)?)?,
            slot: field(1)?.parse()?,
            prediction_mmr: field(2)?.parse()?,
            target_mmr: field(3)?.parse()?,
            segments: field(4)?.parse()?,
        });
    }
    anyhow::ensure!(!rows.is_empty(), "{} contained no rows", path.display());
    Ok(rows)
}

/// Number of distinct lobbies represented in `samples`.
fn lobby_count(samples: &[MarginSample]) -> usize {
    let ids: std::collections::HashSet<Uuid> = samples.iter().map(|s| s.replay_id).collect();
    ids.len()
}

fn describe(name: &str, samples: &[MarginSample]) {
    let positives = samples.iter().filter(|s| s.is_positive()).count();
    println!(
        "{name}: {} players in {} lobbies, {positives} smurf proxies ({:.2} % base rate)",
        samples.len(),
        lobby_count(samples),
        100.0 * threshold::positive_base_rate(samples),
    );
}

/// Percentile of `values` by nearest rank. `values` need not be sorted.
fn percentile(values: &mut [f32], fraction: f64) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((fraction * values.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1);
    values.get(index).copied().unwrap_or(f32::NAN)
}

/// The margin distribution, split by class.
///
/// This is the distribution step 2.5 asks the threshold to be a percentile *of*, and it is
/// the fastest read on whether any threshold can work: if the positive and negative
/// quantiles sit on top of each other, no cut separates them and the curve will say so.
fn print_margin_distribution(samples: &[MarginSample]) {
    let mut positives: Vec<f32> = samples
        .iter()
        .filter(|s| s.is_positive())
        .map(|s| s.margin_mmr)
        .collect();
    let mut negatives: Vec<f32> = samples
        .iter()
        .filter(|s| !s.is_positive())
        .map(|s| s.margin_mmr)
        .collect();

    println!(
        "\n--- Held-out margin distribution (prediction − lobby median prediction, MMR) ---"
    );
    println!(
        "  {:<12} {:>6} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "class", "n", "p05", "p25", "median", "p75", "p95", "p99"
    );
    for (label, values) in [
        ("smurf proxy", &mut positives),
        ("everyone else", &mut negatives),
    ] {
        if values.is_empty() {
            println!("  {label:<12} {:>6}        --", 0);
            continue;
        }
        println!(
            "  {label:<12} {:>6} {:>+9.1} {:>+9.1} {:>+9.1} {:>+9.1} {:>+9.1} {:>+9.1}",
            values.len(),
            percentile(values, 0.05),
            percentile(values, 0.25),
            percentile(values, 0.50),
            percentile(values, 0.75),
            percentile(values, 0.95),
            percentile(values, 0.99),
        );
    }
}

fn print_point(label: &str, point: &OperatingPoint) {
    println!(
        "  {label:<34} threshold={:>+8.1} MMR  flagged={:>5} ({:>5.2} %)  \
         TP={:>4}  precision={:>5.1} %  recall={:>5.1} %  F1={:.3}",
        point.threshold_mmr,
        point.flagged,
        100.0 * point.flag_rate,
        point.true_positives,
        100.0 * point.precision,
        100.0 * point.recall,
        point.f1,
    );
}

/// Evenly spaced rows from the curve, so a 10 000-point sweep prints as a readable table.
///
/// Sampled by index rather than by recall: the curve is already ordered by loosening
/// threshold, and index-spacing keeps the strictest and loosest points in view.
fn sample_curve(curve: &[OperatingPoint], rows: usize) -> Vec<OperatingPoint> {
    if curve.is_empty() || rows == 0 {
        return Vec::new();
    }
    if curve.len() <= rows {
        return curve.to_vec();
    }
    (0..rows)
        .filter_map(|row| {
            let index = row * (curve.len() - 1) / (rows - 1);
            curve.get(index).copied()
        })
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();

    let predictions = read_predictions(&args.predictions)?;

    // Before fitting anything, confirm the dump reproduces the metrics the scoring pass
    // logged. If these disagree with the `revalidate` output, the dump misrepresents what
    // was measured and every number below is fitted on the wrong thing.
    let cross_check = threshold::mixed_lobby_cross_check(&predictions);
    println!(
        "\n=== Cross-check against the in-training metric (whole dump, not split) ===\n\
         mixed lobbies={}  scored={}  top1={:.1}% (chance 16.7%)  mean margin={:+.0} MMR\n\
         These must match the `revalidate` log for the same checkpoint.",
        cross_check.mixed_lobbies,
        cross_check.scored_lobbies,
        100.0 * cross_check.top1_rate,
        cross_check.mean_margin_mmr,
    );

    let (fit_rows, held_out_rows) = threshold::split_by_replay(&predictions, args.fit_fraction);
    let fit = threshold::margin_samples(&fit_rows);
    let held_out = threshold::margin_samples(&held_out_rows);

    anyhow::ensure!(
        !fit.is_empty() && !held_out.is_empty(),
        "fit-fraction {} left one side empty ({} fit / {} held out)",
        args.fit_fraction,
        fit.len(),
        held_out.len()
    );

    println!(
        "\n=== Threshold fit ===\nsource: {}\npositive = label at least {:.0} MMR above the lobby median label\n",
        args.predictions.display(),
        SMURF_LABEL_MARGIN_MMR,
    );
    describe("fit     ", &fit);
    describe("held-out", &held_out);

    // Without positives on either side every rate below is 0 by construction, which looks
    // like a measured result and is not one. Say so rather than printing a table of zeros.
    let fit_positives = fit.iter().filter(|s| s.is_positive()).count();
    let held_out_positives = held_out.iter().filter(|s| s.is_positive()).count();
    anyhow::ensure!(
        fit_positives > 0 && held_out_positives > 0,
        "no smurf proxies to fit against ({fit_positives} in fit, {held_out_positives} held out) \
         — this dump has no lobby whose top label sits {SMURF_LABEL_MARGIN_MMR:.0} MMR above its \
         median. Score more replays, or a split with mixed lobbies in it."
    );

    print_margin_distribution(&held_out);

    // The shipped rule, on held-out data. This is the number step 2.5 exists to replace.
    println!("\n--- Shipped rule, held-out ---");
    let shipped =
        threshold::evaluate_threshold(&held_out, ml_model::SMURF_MARGIN_OVER_LOBBY_MEDIAN_MMR);
    print_point("+200 MMR (as shipped)", &shipped);

    // Two ways of fitting, both chosen on `fit` and reported on `held_out`.
    println!("\n--- Fitted thresholds (chosen on fit, scored on held-out) ---");

    let base_rate = threshold::positive_base_rate(&fit);
    let percentile_threshold = threshold::threshold_at_flag_rate(&fit, base_rate);
    print_point(
        "percentile @ base rate",
        &threshold::evaluate_threshold(&held_out, percentile_threshold),
    );

    let fit_curve = threshold::precision_recall_curve(&fit);
    if let Some(best) = threshold::best_f1(&fit_curve) {
        print_point(
            "max-F1 on fit",
            &threshold::evaluate_threshold(&held_out, best.threshold_mmr),
        );
        println!(
            "  (that threshold scored F1={:.3} on the fit set itself)",
            best.f1
        );
    }

    // A shortlist is the realistic product: flag the top N % and hand them to review.
    println!("\n--- Fixed flag rates (chosen on fit, scored on held-out) ---");
    for rate in [0.01, 0.02, 0.05, 0.10] {
        let threshold_mmr = threshold::threshold_at_flag_rate(&fit, rate);
        print_point(
            &format!("top {:.0} % of players", 100.0 * rate),
            &threshold::evaluate_threshold(&held_out, threshold_mmr),
        );
    }

    let held_out_curve = threshold::precision_recall_curve(&held_out);
    println!(
        "\n--- Held-out precision/recall curve ({} distinct operating points) ---",
        held_out_curve.len()
    );
    for point in sample_curve(&held_out_curve, args.curve_rows) {
        print_point("", &point);
    }

    let average_precision = threshold::average_precision(&held_out_curve);
    let held_out_base_rate = threshold::positive_base_rate(&held_out);
    println!(
        "\naverage precision = {:.4}  (a ranker with no signal scores the base rate, {:.4}; \
         lift = {:.2}x)",
        average_precision,
        held_out_base_rate,
        if held_out_base_rate > 0.0 {
            f64::from(average_precision) / held_out_base_rate
        } else {
            0.0
        },
    );

    if let Some(path) = args.dump_curve.as_ref() {
        let mut writer = csv::Writer::from_path(path)?;
        writer.write_record([
            "threshold_mmr",
            "flagged",
            "flag_rate",
            "true_positives",
            "precision",
            "recall",
            "f1",
        ])?;
        for point in &held_out_curve {
            writer.write_record([
                format!("{:.4}", point.threshold_mmr),
                point.flagged.to_string(),
                format!("{:.6}", point.flag_rate),
                point.true_positives.to_string(),
                format!("{:.6}", point.precision),
                format!("{:.6}", point.recall),
                format!("{:.6}", point.f1),
            ])?;
        }
        writer.flush()?;
        println!("held-out curve written to {}", path.display());
    }

    Ok(())
}
