//! End to end: the shipped bundle scores real replays the way the app does.
//!
//! Runs `SkillModelBundle::analyze` — the exact call the app makes — on the repository's
//! sample replays and checks the output is complete and plausible. Run with `--nocapture`
//! to see what the app would show.

use std::path::{Path, PathBuf};

use skill_model::SkillModelBundle;

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn sample_replays() -> Vec<PathBuf> {
    let root = repository_root();
    let mut replays = vec![root.join("test_data/2af51380-05b5-44ac-8b31-94b8b0f8da84.replay")];
    if let Ok(entries) = std::fs::read_dir(root.join("data/smurf")) {
        let mut smurfs: Vec<PathBuf> = entries.flatten().map(|entry| entry.path()).collect();
        smurfs.sort();
        replays.extend(smurfs);
    }
    replays
}

#[test]
fn bundle_scores_sample_replays() {
    let bytes = std::fs::read(repository_root().join("data/skill_model.bin"))
        .expect("data/skill_model.bin exists (run the `train` binary)");
    let bundle = SkillModelBundle::from_bytes(&bytes).expect("bundle decodes");
    assert!(
        bundle.match_model.matches_current_stats(),
        "bundle is stale: retrain"
    );

    for path in sample_replays() {
        let replay = std::fs::read(&path).expect("sample replay is readable");
        let parsed = replay_parser::parse_replay_from_bytes(&replay).expect("sample replay parses");
        let analysis = bundle.analyze(&parsed);

        println!(
            "\n{}",
            path.file_name()
                .map(|n| n.to_string_lossy())
                .unwrap_or_default()
        );
        let mut scored = 0;
        for slot in 0..feature_extractor::TOTAL_PLAYERS {
            let Some(name) = analysis.names.get(slot).filter(|name| !name.is_empty()) else {
                continue;
            };
            let mmr = analysis.match_mmr.get(slot).copied().flatten();
            let form: Vec<String> = analysis
                .timeline
                .iter()
                .map(|window| {
                    window
                        .player_mmr
                        .get(slot)
                        .copied()
                        .flatten()
                        .map_or_else(|| "  -".to_string(), |value| format!("{value:>5.0}"))
                })
                .collect();
            let roast = analysis
                .coaching
                .get(slot)
                .and_then(|advice| advice.as_ref())
                .map(|advice| advice.line(0))
                .unwrap_or_default();
            println!(
                "  {name:<20} {:>6} [{}]  {roast}",
                mmr.map_or_else(|| "-".to_string(), |value| format!("{value:.0}")),
                form.join(" ")
            );
            if let Some(mmr) = mmr {
                scored += 1;
                assert!(
                    (0.0..=2500.0).contains(&mmr),
                    "{name}: implausible MMR {mmr}"
                );
                assert!(!roast.is_empty(), "{name}: no roast");
            }
        }
        assert!(
            scored >= 5,
            "{}: only {scored} players scored",
            path.display()
        );
        // Short clips get fewer windows; every replay gets at least one.
        assert!(
            !analysis.timeline.is_empty(),
            "{}: no timeline windows",
            path.display()
        );
    }
}
