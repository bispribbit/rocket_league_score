//! One roast per player: what the next rank up does that you don't.
//!
//! Each player is compared with the **median player [`TARGET_TIER_STEP`] tiers above their
//! predicted tier** (Bronze I → Silver I, Gold III → Platinum III) on a curated list of
//! stats that a person can act on. The stat with the largest shortfall, measured in that
//! tier's own spread, picks the line. Because the reference is always the rank just above,
//! a bronze is told to hit the ball forward and a champ to fast-aerial — never the reverse.
//!
//! Only stats that actually improve from the player's tier to the target tier are eligible
//! (see [`MINIMUM_PROGRESSION`]); otherwise everyone would be told to do whatever varies
//! most between individual players, whether or not it is what separates the ranks.
//!
//! The reference table ([`CoachingTable`]) is built from training data by the exporter and
//! shipped inside the model bundle.

use feature_extractor::MATCH_STAT_NAMES;
use serde::{Deserialize, Serialize};

/// How many tiers above the player's predicted tier the comparison targets. Three tiers is
/// one named rank (Silver I → Gold I).
pub const TARGET_TIER_STEP: usize = 3;

/// A stat must move by at least this many target-tier spreads between the player's tier
/// and the target tier to count as something that separates the two.
pub const MINIMUM_PROGRESSION: f32 = 0.15;

/// The player must trail the target tier's median by at least this many spreads.
pub const MINIMUM_SHORTFALL: f32 = 0.25;

/// Which way is better for a stat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Better {
    Higher,
    Lower,
}

/// How a stat value is written in a line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    /// A 0–1 fraction shown as a percentage.
    Percent,
    /// A rate shown as `x.x/min`.
    PerMinute,
    /// A whole-match count.
    Count,
    /// Ball speed normalised by 6,000 uu/s, shown in km/h.
    BallSpeed,
}

impl Unit {
    /// Smallest gap worth roasting someone over, in this unit's raw value. Below it the two
    /// numbers in the line look the same to a reader ("0% vs 1%"), whatever the spread says.
    #[must_use]
    pub const fn minimum_gap(self) -> f32 {
        match self {
            Self::Percent => 0.05,
            Self::PerMinute => 0.8,
            Self::Count => 1.0,
            // ≈ 6 km/h.
            Self::BallSpeed => 0.03,
        }
    }

    /// Formats one value for a line.
    #[must_use]
    pub fn format(self, value: f32) -> String {
        match self {
            Self::Percent => format!("{:.0}%", value * 100.0),
            Self::PerMinute => format!("{value:.1}/min"),
            Self::Count => format!("{value:.0}"),
            // 6,000 uu/s × 0.036 km/h per uu/s.
            Self::BallSpeed => format!("{:.0} km/h", value * 216.0),
        }
    }
}

/// One coachable stat and its lines.
///
/// Lines use `{yours}`, `{theirs}` and `{next}` (plural rank name, e.g. "Golds").
#[derive(Debug, Clone, Copy)]
pub struct CoachingTip {
    /// Stat name in [`MATCH_STAT_NAMES`].
    pub stat: &'static str,
    pub better: Better,
    pub unit: Unit,
    pub lines: &'static [&'static str],
}

/// The curated, actionable stats. Order is the table's column order.
pub const COACHING_TIPS: &[CoachingTip] = &[
    CoachingTip {
        stat: "touches_per_minute",
        better: Better::Higher,
        unit: Unit::PerMinute,
        lines: &[
            "The ball isn't lava. {next} touch it {theirs}, you managed {yours}.",
            "Spotted near the ball occasionally. {next} hit it {theirs}; you: {yours}.",
        ],
    },
    CoachingTip {
        stat: "forward_touch_fraction",
        better: Better::Higher,
        unit: Unit::Percent,
        lines: &[
            "Only {yours} of your touches went toward their net. {next}: {theirs}. Aim for the big net, bro.",
            "Hitting the ball is step one. Step two is the right direction: {next} send {theirs} forward, you {yours}.",
        ],
    },
    CoachingTip {
        stat: "mean_ball_speed_after_touch",
        better: Better::Higher,
        unit: Unit::BallSpeed,
        lines: &[
            "Your touches leave the ball at {yours}. {next} launch it at {theirs}. Hit it like it owes you money.",
            "Gentle little taps at {yours}. {next} send it at {theirs}.",
        ],
    },
    CoachingTip {
        stat: "boost_empty_fraction",
        better: Better::Lower,
        unit: Unit::Percent,
        lines: &[
            "On zero boost {yours} of the game. {next}: {theirs}. The pads are free, grab some.",
            "Running on fumes {yours} of the time ({next}: {theirs}). Boost is not a collectible.",
        ],
    },
    CoachingTip {
        stat: "small_pads_per_minute",
        better: Better::Higher,
        unit: Unit::PerMinute,
        lines: &[
            "Small pads exist. {next} grab {theirs}, you grab {yours}. Stop driving past the snacks.",
        ],
    },
    CoachingTip {
        stat: "supersonic_fraction",
        better: Better::Higher,
        unit: Unit::Percent,
        lines: &[
            "Supersonic {yours} of the time. {next}: {theirs}. The fast button is right there.",
            "{next} hit top speed {theirs} of the game. You: {yours}. Sunday drive?",
        ],
    },
    CoachingTip {
        stat: "slow_fraction",
        better: Better::Lower,
        unit: Unit::Percent,
        lines: &["Basically parked {yours} of the game. {next} only crawl {theirs}."],
    },
    CoachingTip {
        stat: "overcommit_fraction",
        better: Better::Lower,
        unit: Unit::Percent,
        lines: &[
            "Upfield while a teammate had the ball {yours} of the time ({next}: {theirs}). Your net misses you.",
            "Ahead of the play {yours} of the game. {next}: {theirs}. Someone has to rotate back. It's you.",
        ],
    },
    CoachingTip {
        stat: "goal_side_when_defending_fraction",
        better: Better::Higher,
        unit: Unit::Percent,
        lines: &[
            "When they attacked, you were on the right side of the ball {yours} of the time. {next}: {theirs}. Defend from goal-side.",
        ],
    },
    CoachingTip {
        stat: "double_commit_fraction",
        better: Better::Lower,
        unit: Unit::Percent,
        lines: &[
            "Double-committing {yours} of the time ({next}: {theirs}). Your teammate had it. Probably.",
        ],
    },
    CoachingTip {
        stat: "goal_side_after_touch_fraction",
        better: Better::Higher,
        unit: Unit::Percent,
        lines: &[
            "After touching the ball you got back goal-side {yours} of the time ({next}: {theirs}). Stop admiring your own touch.",
        ],
    },
    CoachingTip {
        stat: "kickoff_win_fraction",
        better: Better::Higher,
        unit: Unit::Percent,
        lines: &[
            "Won {yours} of your kickoffs. {next}: {theirs}. Practise one kickoff. Any kickoff.",
        ],
    },
    CoachingTip {
        stat: "demos_received",
        better: Better::Lower,
        unit: Unit::Count,
        lines: &["Demolished {yours} times. {next}: {theirs}. Check your mirrors."],
    },
    CoachingTip {
        stat: "dodges_per_minute",
        better: Better::Higher,
        unit: Unit::PerMinute,
        lines: &["Flips: {yours}. {next}: {theirs}. Flipping is free speed."],
    },
    CoachingTip {
        stat: "high_air_fraction",
        better: Better::Higher,
        unit: Unit::Percent,
        lines: &[
            "{next} spend {theirs} of the game airborne. You: {yours}. Gravity is optional.",
            "Grounded {yours} of the way, metaphorically. {next} fly {theirs} of the game.",
        ],
    },
    CoachingTip {
        stat: "aerial_touch_fraction",
        better: Better::Higher,
        unit: Unit::Percent,
        lines: &["{yours} of your touches were in the air. {next}: {theirs}. Free play is free."],
    },
    CoachingTip {
        stat: "fast_aerial_fraction",
        better: Better::Higher,
        unit: Unit::Percent,
        lines: &[
            "Your aerials take the scenic route: {yours} fast aerials vs {theirs} for {next}. Jump, jump, boost.",
        ],
    },
    CoachingTip {
        stat: "wavedash_like_dodges_per_minute",
        better: Better::Higher,
        unit: Unit::PerMinute,
        lines: &["{next} wavedash {theirs}. You: {yours}. Land with some style."],
    },
    CoachingTip {
        stat: "powerslide_fraction",
        better: Better::Higher,
        unit: Unit::Percent,
        lines: &[
            "Powersliding {yours} of the time, {next} {theirs}. Turning is a skill and there's a button for it.",
        ],
    },
];

/// Lines for a player with nothing left to learn from the rank above (or already SSL).
pub const NOTHING_TO_FIX_LINES: &[&str] = &[
    "Honestly? Nothing to fix. Suspicious.",
    "No notes. Go touch grass instead.",
];

/// Reference values of one tier for every tip, aligned with [`COACHING_TIPS`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TierReference {
    /// Median of each tip's stat among players labelled with this tier.
    pub medians: Vec<f32>,
    /// Robust spread (inter-quartile range / 1.349) of each tip's stat in this tier.
    pub spreads: Vec<f32>,
}

/// Reference stats for every tier, Bronze I (index 0) to Supersonic Legend (index 21).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CoachingTable {
    pub tiers: Vec<TierReference>,
    /// Per tip: whether the data agrees with its declared direction (medians move the
    /// declared way from Bronze to SSL). Tips the data contradicts are never used.
    pub enabled: Vec<bool>,
}

/// Plural display name per tier index (0 = Bronze I … 21 = SSL), as used in `{next}`.
const TIER_GROUP_PLURALS: [&str; 22] = [
    "Bronzes",
    "Bronzes",
    "Bronzes",
    "Silvers",
    "Silvers",
    "Silvers",
    "Golds",
    "Golds",
    "Golds",
    "Platinums",
    "Platinums",
    "Platinums",
    "Diamonds",
    "Diamonds",
    "Diamonds",
    "Champs",
    "Champs",
    "Champs",
    "GCs",
    "GCs",
    "GCs",
    "SSLs",
];

/// Tier index (0 = Bronze I … 21 = SSL) for an MMR value.
#[must_use]
pub fn tier_index(mmr: f32) -> usize {
    let rank: replay_structs::Rank = replay_structs::RankDivision::from(mmr).into();
    usize::try_from(rank.as_numeric_index() - 1)
        .unwrap_or(0)
        .min(21)
}

/// The chosen advice for one player.
#[derive(Debug, Clone)]
pub struct CoachingAdvice {
    /// The tip, or `None` when there is nothing to fix.
    pub tip: Option<&'static CoachingTip>,
    /// The player's value of the tip's stat.
    pub yours: f32,
    /// The target tier's median.
    pub theirs: f32,
    /// Plural name of the target tier's rank.
    pub next: &'static str,
    /// Shortfall in target-tier spreads (0 when there is nothing to fix).
    pub shortfall: f32,
}

impl CoachingAdvice {
    /// Renders the advice with line variant `variant` (wrapped to the available lines).
    #[must_use]
    pub fn line(&self, variant: usize) -> String {
        let Some(tip) = self.tip else {
            return NOTHING_TO_FIX_LINES
                .get(variant % NOTHING_TO_FIX_LINES.len())
                .copied()
                .unwrap_or_default()
                .to_string();
        };
        tip.lines
            .get(variant % tip.lines.len().max(1))
            .copied()
            .unwrap_or_default()
            .replace("{yours}", &tip.unit.format(self.yours))
            .replace("{theirs}", &tip.unit.format(self.theirs))
            .replace("{next}", self.next)
    }
}

impl CoachingTable {
    /// Picks the advice for a player predicted at `predicted_mmr` with match stats `stats`.
    ///
    /// Returns `None` only when the table is empty (no reference data shipped).
    #[must_use]
    pub fn advise(&self, predicted_mmr: f32, stats: &[f32]) -> Option<CoachingAdvice> {
        let last = self.tiers.len().checked_sub(1)?;
        let current = tier_index(predicted_mmr).min(last);
        let target = (current + TARGET_TIER_STEP).min(last);
        let next = TIER_GROUP_PLURALS.get(target).copied().unwrap_or("SSLs");
        let nothing = CoachingAdvice {
            tip: None,
            yours: 0.0,
            theirs: 0.0,
            next,
            shortfall: 0.0,
        };
        if target == current {
            return Some(nothing);
        }
        let current_reference = self.tiers.get(current)?;
        let target_reference = self.tiers.get(target)?;

        let mut best = nothing;
        for (tip_index, tip) in COACHING_TIPS.iter().enumerate() {
            if !self.enabled.get(tip_index).copied().unwrap_or(false) {
                continue;
            }
            let Some(stat_index) = MATCH_STAT_NAMES.iter().position(|name| *name == tip.stat)
            else {
                continue;
            };
            let Some(&yours) = stats.get(stat_index) else {
                continue;
            };
            let Some(&theirs) = target_reference.medians.get(tip_index) else {
                continue;
            };
            let Some(&spread) = target_reference.spreads.get(tip_index) else {
                continue;
            };
            let Some(&current_median) = current_reference.medians.get(tip_index) else {
                continue;
            };
            if spread <= f32::EPSILON {
                continue;
            }
            let sign = match tip.better {
                Better::Higher => 1.0,
                Better::Lower => -1.0,
            };
            let progression = sign * (theirs - current_median) / spread;
            let shortfall = sign * (theirs - yours) / spread;
            let visible = (theirs - yours).abs() >= tip.unit.minimum_gap()
                && tip.unit.format(theirs) != tip.unit.format(yours);
            if visible
                && progression >= MINIMUM_PROGRESSION
                && shortfall >= MINIMUM_SHORTFALL
                && shortfall > best.shortfall
            {
                best = CoachingAdvice {
                    tip: Some(tip),
                    yours,
                    theirs,
                    next,
                    shortfall,
                };
            }
        }
        Some(best)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stat_column(name: &str) -> usize {
        MATCH_STAT_NAMES
            .iter()
            .position(|stat| *stat == name)
            .unwrap_or(usize::MAX)
    }

    fn tip_column(name: &str) -> usize {
        COACHING_TIPS
            .iter()
            .position(|tip| tip.stat == name)
            .unwrap_or(usize::MAX)
    }

    /// A table where only touches/min and air time improve with rank; air time only
    /// starts mattering from Gold up.
    fn table() -> CoachingTable {
        let tiers = (0..22)
            .map(|tier| {
                let mut medians = vec![0.5; COACHING_TIPS.len()];
                let spreads = vec![0.1; COACHING_TIPS.len()];
                medians[tip_column("touches_per_minute")] = 1.0f32.mul_add(tier as f32, 2.0);
                medians[tip_column("high_air_fraction")] = if tier < 6 {
                    0.5
                } else {
                    0.1f32.mul_add((tier - 6) as f32, 0.5)
                };
                TierReference { medians, spreads }
            })
            .collect();
        CoachingTable {
            tiers,
            enabled: vec![true; COACHING_TIPS.len()],
        }
    }

    #[test]
    fn every_tip_names_a_real_stat() {
        for tip in COACHING_TIPS {
            assert!(
                MATCH_STAT_NAMES.contains(&tip.stat),
                "{} is not a match stat",
                tip.stat
            );
            assert!(!tip.lines.is_empty(), "{} has no lines", tip.stat);
        }
    }

    /// A low-ranked player who never jumps is still told to touch the ball: air time does
    /// not separate their tier from the target tier, however far behind they are.
    #[test]
    fn advice_targets_what_separates_the_next_rank() {
        let mut stats = vec![0.5; MATCH_STAT_NAMES.len()];
        stats[stat_column("touches_per_minute")] = 1.0;
        stats[stat_column("high_air_fraction")] = 0.0;
        let bronze = table().advise(100.0, &stats).expect("table is not empty");
        assert_eq!(bronze.tip.map(|tip| tip.stat), Some("touches_per_minute"));
        assert_eq!(bronze.next, "Silvers");
        assert!(bronze.line(0).contains("Silvers"));
    }

    #[test]
    fn nothing_to_fix_when_ahead_of_the_next_rank() {
        let stats = vec![10.0; MATCH_STAT_NAMES.len()];
        let advice = table().advise(100.0, &stats).expect("table is not empty");
        assert!(advice.tip.is_none());
        assert!(NOTHING_TO_FIX_LINES.contains(&advice.line(0).as_str()));
    }

    #[test]
    fn units_format_for_humans() {
        assert_eq!(Unit::Percent.format(0.123), "12%");
        assert_eq!(Unit::PerMinute.format(7.04), "7.0/min");
        assert_eq!(Unit::BallSpeed.format(0.5), "108 km/h");
    }
}
