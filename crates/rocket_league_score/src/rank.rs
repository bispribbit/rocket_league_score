//! Rocket League rank definitions and MMR mappings.
//!
//! This module provides a comprehensive enum for all Rocket League ranks
//! with their corresponding MMR (Match Making Rating) values.

use core::fmt;
use core::str::FromStr;

/// Represents a Rocket League competitive rank with division.
///
/// Each rank (except Supersonic Legend) has 4 divisions.
/// MMR values are based on the 3v3 ranked playlist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Rank {
    // Bronze I
    BronzeIDivision1,
    BronzeIDivision2,
    BronzeIDivision3,
    BronzeIDivision4,
    // Bronze II
    BronzeIIDivision1,
    BronzeIIDivision2,
    BronzeIIDivision3,
    BronzeIIDivision4,
    // Bronze III
    BronzeIIIDivision1,
    BronzeIIIDivision2,
    BronzeIIIDivision3,
    BronzeIIIDivision4,
    // Silver I
    SilverIDivision1,
    SilverIDivision2,
    SilverIDivision3,
    SilverIDivision4,
    // Silver II
    SilverIIDivision1,
    SilverIIDivision2,
    SilverIIDivision3,
    SilverIIDivision4,
    // Silver III
    SilverIIIDivision1,
    SilverIIIDivision2,
    SilverIIIDivision3,
    SilverIIIDivision4,
    // Gold I
    GoldIDivision1,
    GoldIDivision2,
    GoldIDivision3,
    GoldIDivision4,
    // Gold II
    GoldIIDivision1,
    GoldIIDivision2,
    GoldIIDivision3,
    GoldIIDivision4,
    // Gold III
    GoldIIIDivision1,
    GoldIIIDivision2,
    GoldIIIDivision3,
    GoldIIIDivision4,
    // Platinum I
    PlatinumIDivision1,
    PlatinumIDivision2,
    PlatinumIDivision3,
    PlatinumIDivision4,
    // Platinum II
    PlatinumIIDivision1,
    PlatinumIIDivision2,
    PlatinumIIDivision3,
    PlatinumIIDivision4,
    // Platinum III
    PlatinumIIIDivision1,
    PlatinumIIIDivision2,
    PlatinumIIIDivision3,
    PlatinumIIIDivision4,
    // Diamond I
    DiamondIDivision1,
    DiamondIDivision2,
    DiamondIDivision3,
    DiamondIDivision4,
    // Diamond II
    DiamondIIDivision1,
    DiamondIIDivision2,
    DiamondIIDivision3,
    DiamondIIDivision4,
    // Diamond III
    DiamondIIIDivision1,
    DiamondIIIDivision2,
    DiamondIIIDivision3,
    DiamondIIIDivision4,
    // Champion I
    ChampionIDivision1,
    ChampionIDivision2,
    ChampionIDivision3,
    ChampionIDivision4,
    // Champion II
    ChampionIIDivision1,
    ChampionIIDivision2,
    ChampionIIDivision3,
    ChampionIIDivision4,
    // Champion III
    ChampionIIIDivision1,
    ChampionIIIDivision2,
    ChampionIIIDivision3,
    ChampionIIIDivision4,
    // Grand Champion I
    GrandChampionIDivision1,
    GrandChampionIDivision2,
    GrandChampionIDivision3,
    GrandChampionIDivision4,
    // Grand Champion II
    GrandChampionIIDivision1,
    GrandChampionIIDivision2,
    GrandChampionIIDivision3,
    GrandChampionIIDivision4,
    // Grand Champion III
    GrandChampionIIIDivision1,
    GrandChampionIIIDivision2,
    GrandChampionIIIDivision3,
    GrandChampionIIIDivision4,
    // Supersonic Legend
    SupersonicLegend,
}

impl Rank {
    /// Returns the minimum MMR value for this rank.
    #[must_use]
    #[expect(
        clippy::match_same_arms,
        reason = "Intentional: values from official rank data"
    )]
    pub const fn mmr_min(self) -> i32 {
        match self {
            // Bronze I
            Self::BronzeIDivision1 => 0,
            Self::BronzeIDivision2 => 121,
            Self::BronzeIDivision3 => 138,
            Self::BronzeIDivision4 => 157,
            // Bronze II
            Self::BronzeIIDivision1 => 175,
            Self::BronzeIIDivision2 => 180,
            Self::BronzeIIDivision3 => 203,
            Self::BronzeIIDivision4 => 217,
            // Bronze III
            Self::BronzeIIIDivision1 => 234,
            Self::BronzeIIIDivision2 => 240,
            Self::BronzeIIIDivision3 => 261,
            Self::BronzeIIIDivision4 => 277,
            // Silver I
            Self::SilverIDivision1 => 280,
            Self::SilverIDivision2 => 300,
            Self::SilverIDivision3 => 318,
            Self::SilverIDivision4 => 337,
            // Silver II
            Self::SilverIIDivision1 => 340,
            Self::SilverIIDivision2 => 360,
            Self::SilverIIDivision3 => 378,
            Self::SilverIIDivision4 => 397,
            // Silver III
            Self::SilverIIIDivision1 => 400,
            Self::SilverIIIDivision2 => 419,
            Self::SilverIIIDivision3 => 438,
            Self::SilverIIIDivision4 => 457,
            // Gold I
            Self::GoldIDivision1 => 460,
            Self::GoldIDivision2 => 479,
            Self::GoldIDivision3 => 498,
            Self::GoldIDivision4 => 517,
            // Gold II
            Self::GoldIIDivision1 => 520,
            Self::GoldIIDivision2 => 539,
            Self::GoldIIDivision3 => 558,
            Self::GoldIIDivision4 => 577,
            // Gold III
            Self::GoldIIIDivision1 => 580,
            Self::GoldIIIDivision2 => 580,
            Self::GoldIIIDivision3 => 580,
            Self::GoldIIIDivision4 => 637,
            // Platinum I
            Self::PlatinumIDivision1 => 640,
            Self::PlatinumIDivision2 => 659,
            Self::PlatinumIDivision3 => 640,
            Self::PlatinumIDivision4 => 640,
            // Platinum II
            Self::PlatinumIIDivision1 => 715,
            Self::PlatinumIIDivision2 => 719,
            Self::PlatinumIIDivision3 => 738,
            Self::PlatinumIIDivision4 => 757,
            // Platinum III
            Self::PlatinumIIIDivision1 => 760,
            Self::PlatinumIIIDivision2 => 760,
            Self::PlatinumIIIDivision3 => 760,
            Self::PlatinumIIIDivision4 => 760,
            // Diamond I
            Self::DiamondIDivision1 => 835,
            Self::DiamondIDivision2 => 820,
            Self::DiamondIDivision3 => 820,
            Self::DiamondIDivision4 => 892,
            // Diamond II
            Self::DiamondIIDivision1 => 915,
            Self::DiamondIIDivision2 => 924,
            Self::DiamondIIDivision3 => 900,
            Self::DiamondIIDivision4 => 972,
            // Diamond III
            Self::DiamondIIIDivision1 => 980,
            Self::DiamondIIIDivision2 => 980,
            Self::DiamondIIIDivision3 => 980,
            Self::DiamondIIIDivision4 => 980,
            // Champion I
            Self::ChampionIDivision1 => 1075,
            Self::ChampionIDivision2 => 1095,
            Self::ChampionIDivision3 => 1128,
            Self::ChampionIDivision4 => 1162,
            // Champion II
            Self::ChampionIIDivision1 => 1195,
            Self::ChampionIIDivision2 => 1215,
            Self::ChampionIIDivision3 => 1248,
            Self::ChampionIIDivision4 => 1282,
            // Champion III
            Self::ChampionIIIDivision1 => 1315,
            Self::ChampionIIIDivision2 => 1335,
            Self::ChampionIIIDivision3 => 1369,
            Self::ChampionIIIDivision4 => 1402,
            // Grand Champion I
            Self::GrandChampionIDivision1 => 1436,
            Self::GrandChampionIDivision2 => 1460,
            Self::GrandChampionIDivision3 => 1498,
            Self::GrandChampionIDivision4 => 1537,
            // Grand Champion II
            Self::GrandChampionIIDivision1 => 1575,
            Self::GrandChampionIIDivision2 => 1600,
            Self::GrandChampionIIDivision3 => 1645,
            Self::GrandChampionIIDivision4 => 1677,
            // Grand Champion III
            Self::GrandChampionIIIDivision1 => 1706,
            Self::GrandChampionIIIDivision2 => 1746,
            Self::GrandChampionIIIDivision3 => 1794,
            Self::GrandChampionIIIDivision4 => 1810,
            // Supersonic Legend
            Self::SupersonicLegend => 1883,
        }
    }

    /// Returns the maximum MMR value for this rank.
    #[must_use]
    pub const fn mmr_max(self) -> i32 {
        match self {
            // Bronze I
            Self::BronzeIDivision1 => 117,
            Self::BronzeIDivision2 => 130,
            Self::BronzeIDivision3 => 156,
            Self::BronzeIDivision4 => 174,
            // Bronze II
            Self::BronzeIIDivision1 => 178,
            Self::BronzeIIDivision2 => 194,
            Self::BronzeIIDivision3 => 215,
            Self::BronzeIIDivision4 => 234,
            // Bronze III
            Self::BronzeIIIDivision1 => 238,
            Self::BronzeIIIDivision2 => 257,
            Self::BronzeIIIDivision3 => 275,
            Self::BronzeIIIDivision4 => 294,
            // Silver I
            Self::SilverIDivision1 => 298,
            Self::SilverIDivision2 => 317,
            Self::SilverIDivision3 => 336,
            Self::SilverIDivision4 => 354,
            // Silver II
            Self::SilverIIDivision1 => 358,
            Self::SilverIIDivision2 => 377,
            Self::SilverIIDivision3 => 396,
            Self::SilverIIDivision4 => 415,
            // Silver III
            Self::SilverIIIDivision1 => 418,
            Self::SilverIIIDivision2 => 437,
            Self::SilverIIIDivision3 => 456,
            Self::SilverIIIDivision4 => 474,
            // Gold I
            Self::GoldIDivision1 => 478,
            Self::GoldIDivision2 => 497,
            Self::GoldIDivision3 => 516,
            Self::GoldIDivision4 => 534,
            // Gold II
            Self::GoldIIDivision1 => 538,
            Self::GoldIIDivision2 => 557,
            Self::GoldIIDivision3 => 576,
            Self::GoldIIDivision4 => 590,
            // Gold III
            Self::GoldIIIDivision1 => 598,
            Self::GoldIIIDivision2 => 617,
            Self::GoldIIIDivision3 => 636,
            Self::GoldIIIDivision4 => 654,
            // Platinum I
            Self::PlatinumIDivision1 => 658,
            Self::PlatinumIDivision2 => 677,
            Self::PlatinumIDivision3 => 696,
            Self::PlatinumIDivision4 => 714,
            // Platinum II
            Self::PlatinumIIDivision1 => 718,
            Self::PlatinumIIDivision2 => 737,
            Self::PlatinumIIDivision3 => 756,
            Self::PlatinumIIDivision4 => 774,
            // Platinum III
            Self::PlatinumIIIDivision1 => 778,
            Self::PlatinumIIIDivision2 => 797,
            Self::PlatinumIIIDivision3 => 816,
            Self::PlatinumIIIDivision4 => 834,
            // Diamond I
            Self::DiamondIDivision1 => 843,
            Self::DiamondIDivision2 => 867,
            Self::DiamondIDivision3 => 891,
            Self::DiamondIDivision4 => 914,
            // Diamond II
            Self::DiamondIIDivision1 => 923,
            Self::DiamondIIDivision2 => 947,
            Self::DiamondIIDivision3 => 971,
            Self::DiamondIIDivision4 => 994,
            // Diamond III
            Self::DiamondIIIDivision1 => 1003,
            Self::DiamondIIIDivision2 => 1027,
            Self::DiamondIIIDivision3 => 1051,
            Self::DiamondIIIDivision4 => 1074,
            // Champion I
            Self::ChampionIDivision1 => 1093,
            Self::ChampionIDivision2 => 1127,
            Self::ChampionIDivision3 => 1160,
            Self::ChampionIDivision4 => 1180,
            // Champion II
            Self::ChampionIIDivision1 => 1213,
            Self::ChampionIIDivision2 => 1247,
            Self::ChampionIIDivision3 => 1271,
            Self::ChampionIIDivision4 => 1300,
            // Champion III
            Self::ChampionIIIDivision1 => 1333,
            Self::ChampionIIIDivision2 => 1366,
            Self::ChampionIIIDivision3 => 1393,
            Self::ChampionIIIDivision4 => 1419,
            // Grand Champion I
            Self::GrandChampionIDivision1 => 1458,
            Self::GrandChampionIDivision2 => 1485,
            Self::GrandChampionIDivision3 => 1535,
            Self::GrandChampionIDivision4 => 1561,
            // Grand Champion II
            Self::GrandChampionIIDivision1 => 1598,
            Self::GrandChampionIIDivision2 => 1636,
            Self::GrandChampionIIDivision3 => 1660,
            Self::GrandChampionIIDivision4 => 1708,
            // Grand Champion III
            Self::GrandChampionIIIDivision1 => 1739,
            Self::GrandChampionIIIDivision2 => 1780,
            Self::GrandChampionIIIDivision3 => 1809,
            Self::GrandChampionIIIDivision4 => 1882,
            // Supersonic Legend (unbounded, using 2000 as a practical max)
            Self::SupersonicLegend => 2000,
        }
    }

    /// Returns the middle MMR value for this rank.
    #[must_use]
    pub const fn mmr_middle(self) -> i32 {
        i32::midpoint(self.mmr_min(), self.mmr_max())
    }

    /// Creates a `Rank` from a rank ID string (e.g., "grand-champion-3") and optional division.
    ///
    /// This is useful for parsing metadata from ballchasing.com API.
    #[must_use]
    pub fn from_rank_id(rank_id: &str, division: Option<i32>) -> Option<Self> {
        let division = division.unwrap_or(1).clamp(1, 4);

        match (rank_id, division) {
            ("supersonic-legend", _) => Some(Self::SupersonicLegend),
            // Grand Champion III
            ("grand-champion-3", 1) => Some(Self::GrandChampionIIIDivision1),
            ("grand-champion-3", 2) => Some(Self::GrandChampionIIIDivision2),
            ("grand-champion-3", 3) => Some(Self::GrandChampionIIIDivision3),
            ("grand-champion-3", 4) => Some(Self::GrandChampionIIIDivision4),
            // Grand Champion II
            ("grand-champion-2", 1) => Some(Self::GrandChampionIIDivision1),
            ("grand-champion-2", 2) => Some(Self::GrandChampionIIDivision2),
            ("grand-champion-2", 3) => Some(Self::GrandChampionIIDivision3),
            ("grand-champion-2", 4) => Some(Self::GrandChampionIIDivision4),
            // Grand Champion I
            ("grand-champion-1", 1) => Some(Self::GrandChampionIDivision1),
            ("grand-champion-1", 2) => Some(Self::GrandChampionIDivision2),
            ("grand-champion-1", 3) => Some(Self::GrandChampionIDivision3),
            ("grand-champion-1", 4) => Some(Self::GrandChampionIDivision4),
            // Champion III
            ("champion-3", 1) => Some(Self::ChampionIIIDivision1),
            ("champion-3", 2) => Some(Self::ChampionIIIDivision2),
            ("champion-3", 3) => Some(Self::ChampionIIIDivision3),
            ("champion-3", 4) => Some(Self::ChampionIIIDivision4),
            // Champion II
            ("champion-2", 1) => Some(Self::ChampionIIDivision1),
            ("champion-2", 2) => Some(Self::ChampionIIDivision2),
            ("champion-2", 3) => Some(Self::ChampionIIDivision3),
            ("champion-2", 4) => Some(Self::ChampionIIDivision4),
            // Champion I
            ("champion-1", 1) => Some(Self::ChampionIDivision1),
            ("champion-1", 2) => Some(Self::ChampionIDivision2),
            ("champion-1", 3) => Some(Self::ChampionIDivision3),
            ("champion-1", 4) => Some(Self::ChampionIDivision4),
            // Diamond III
            ("diamond-3", 1) => Some(Self::DiamondIIIDivision1),
            ("diamond-3", 2) => Some(Self::DiamondIIIDivision2),
            ("diamond-3", 3) => Some(Self::DiamondIIIDivision3),
            ("diamond-3", 4) => Some(Self::DiamondIIIDivision4),
            // Diamond II
            ("diamond-2", 1) => Some(Self::DiamondIIDivision1),
            ("diamond-2", 2) => Some(Self::DiamondIIDivision2),
            ("diamond-2", 3) => Some(Self::DiamondIIDivision3),
            ("diamond-2", 4) => Some(Self::DiamondIIDivision4),
            // Diamond I
            ("diamond-1", 1) => Some(Self::DiamondIDivision1),
            ("diamond-1", 2) => Some(Self::DiamondIDivision2),
            ("diamond-1", 3) => Some(Self::DiamondIDivision3),
            ("diamond-1", 4) => Some(Self::DiamondIDivision4),
            // Platinum III
            ("platinum-3", 1) => Some(Self::PlatinumIIIDivision1),
            ("platinum-3", 2) => Some(Self::PlatinumIIIDivision2),
            ("platinum-3", 3) => Some(Self::PlatinumIIIDivision3),
            ("platinum-3", 4) => Some(Self::PlatinumIIIDivision4),
            // Platinum II
            ("platinum-2", 1) => Some(Self::PlatinumIIDivision1),
            ("platinum-2", 2) => Some(Self::PlatinumIIDivision2),
            ("platinum-2", 3) => Some(Self::PlatinumIIDivision3),
            ("platinum-2", 4) => Some(Self::PlatinumIIDivision4),
            // Platinum I
            ("platinum-1", 1) => Some(Self::PlatinumIDivision1),
            ("platinum-1", 2) => Some(Self::PlatinumIDivision2),
            ("platinum-1", 3) => Some(Self::PlatinumIDivision3),
            ("platinum-1", 4) => Some(Self::PlatinumIDivision4),
            // Gold III
            ("gold-3", 1) => Some(Self::GoldIIIDivision1),
            ("gold-3", 2) => Some(Self::GoldIIIDivision2),
            ("gold-3", 3) => Some(Self::GoldIIIDivision3),
            ("gold-3", 4) => Some(Self::GoldIIIDivision4),
            // Gold II
            ("gold-2", 1) => Some(Self::GoldIIDivision1),
            ("gold-2", 2) => Some(Self::GoldIIDivision2),
            ("gold-2", 3) => Some(Self::GoldIIDivision3),
            ("gold-2", 4) => Some(Self::GoldIIDivision4),
            // Gold I
            ("gold-1", 1) => Some(Self::GoldIDivision1),
            ("gold-1", 2) => Some(Self::GoldIDivision2),
            ("gold-1", 3) => Some(Self::GoldIDivision3),
            ("gold-1", 4) => Some(Self::GoldIDivision4),
            // Silver III
            ("silver-3", 1) => Some(Self::SilverIIIDivision1),
            ("silver-3", 2) => Some(Self::SilverIIIDivision2),
            ("silver-3", 3) => Some(Self::SilverIIIDivision3),
            ("silver-3", 4) => Some(Self::SilverIIIDivision4),
            // Silver II
            ("silver-2", 1) => Some(Self::SilverIIDivision1),
            ("silver-2", 2) => Some(Self::SilverIIDivision2),
            ("silver-2", 3) => Some(Self::SilverIIDivision3),
            ("silver-2", 4) => Some(Self::SilverIIDivision4),
            // Silver I
            ("silver-1", 1) => Some(Self::SilverIDivision1),
            ("silver-1", 2) => Some(Self::SilverIDivision2),
            ("silver-1", 3) => Some(Self::SilverIDivision3),
            ("silver-1", 4) => Some(Self::SilverIDivision4),
            // Bronze III
            ("bronze-3", 1) => Some(Self::BronzeIIIDivision1),
            ("bronze-3", 2) => Some(Self::BronzeIIIDivision2),
            ("bronze-3", 3) => Some(Self::BronzeIIIDivision3),
            ("bronze-3", 4) => Some(Self::BronzeIIIDivision4),
            // Bronze II
            ("bronze-2", 1) => Some(Self::BronzeIIDivision1),
            ("bronze-2", 2) => Some(Self::BronzeIIDivision2),
            ("bronze-2", 3) => Some(Self::BronzeIIDivision3),
            ("bronze-2", 4) => Some(Self::BronzeIIDivision4),
            // Bronze I
            ("bronze-1", 1) => Some(Self::BronzeIDivision1),
            ("bronze-1", 2) => Some(Self::BronzeIDivision2),
            ("bronze-1", 3) => Some(Self::BronzeIDivision3),
            ("bronze-1", 4) => Some(Self::BronzeIDivision4),
            _ => None,
        }
    }

    /// Returns an iterator over all possible ranks in order from lowest to highest.
    pub fn all() -> impl Iterator<Item = Self> {
        [
            Self::BronzeIDivision1,
            Self::BronzeIDivision2,
            Self::BronzeIDivision3,
            Self::BronzeIDivision4,
            Self::BronzeIIDivision1,
            Self::BronzeIIDivision2,
            Self::BronzeIIDivision3,
            Self::BronzeIIDivision4,
            Self::BronzeIIIDivision1,
            Self::BronzeIIIDivision2,
            Self::BronzeIIIDivision3,
            Self::BronzeIIIDivision4,
            Self::SilverIDivision1,
            Self::SilverIDivision2,
            Self::SilverIDivision3,
            Self::SilverIDivision4,
            Self::SilverIIDivision1,
            Self::SilverIIDivision2,
            Self::SilverIIDivision3,
            Self::SilverIIDivision4,
            Self::SilverIIIDivision1,
            Self::SilverIIIDivision2,
            Self::SilverIIIDivision3,
            Self::SilverIIIDivision4,
            Self::GoldIDivision1,
            Self::GoldIDivision2,
            Self::GoldIDivision3,
            Self::GoldIDivision4,
            Self::GoldIIDivision1,
            Self::GoldIIDivision2,
            Self::GoldIIDivision3,
            Self::GoldIIDivision4,
            Self::GoldIIIDivision1,
            Self::GoldIIIDivision2,
            Self::GoldIIIDivision3,
            Self::GoldIIIDivision4,
            Self::PlatinumIDivision1,
            Self::PlatinumIDivision2,
            Self::PlatinumIDivision3,
            Self::PlatinumIDivision4,
            Self::PlatinumIIDivision1,
            Self::PlatinumIIDivision2,
            Self::PlatinumIIDivision3,
            Self::PlatinumIIDivision4,
            Self::PlatinumIIIDivision1,
            Self::PlatinumIIIDivision2,
            Self::PlatinumIIIDivision3,
            Self::PlatinumIIIDivision4,
            Self::DiamondIDivision1,
            Self::DiamondIDivision2,
            Self::DiamondIDivision3,
            Self::DiamondIDivision4,
            Self::DiamondIIDivision1,
            Self::DiamondIIDivision2,
            Self::DiamondIIDivision3,
            Self::DiamondIIDivision4,
            Self::DiamondIIIDivision1,
            Self::DiamondIIIDivision2,
            Self::DiamondIIIDivision3,
            Self::DiamondIIIDivision4,
            Self::ChampionIDivision1,
            Self::ChampionIDivision2,
            Self::ChampionIDivision3,
            Self::ChampionIDivision4,
            Self::ChampionIIDivision1,
            Self::ChampionIIDivision2,
            Self::ChampionIIDivision3,
            Self::ChampionIIDivision4,
            Self::ChampionIIIDivision1,
            Self::ChampionIIIDivision2,
            Self::ChampionIIIDivision3,
            Self::ChampionIIIDivision4,
            Self::GrandChampionIDivision1,
            Self::GrandChampionIDivision2,
            Self::GrandChampionIDivision3,
            Self::GrandChampionIDivision4,
            Self::GrandChampionIIDivision1,
            Self::GrandChampionIIDivision2,
            Self::GrandChampionIIDivision3,
            Self::GrandChampionIIDivision4,
            Self::GrandChampionIIIDivision1,
            Self::GrandChampionIIIDivision2,
            Self::GrandChampionIIIDivision3,
            Self::GrandChampionIIIDivision4,
            Self::SupersonicLegend,
        ]
        .into_iter()
    }
}

impl fmt::Display for Rank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            // Bronze I
            Self::BronzeIDivision1 => "Bronze I Division 1",
            Self::BronzeIDivision2 => "Bronze I Division 2",
            Self::BronzeIDivision3 => "Bronze I Division 3",
            Self::BronzeIDivision4 => "Bronze I Division 4",
            // Bronze II
            Self::BronzeIIDivision1 => "Bronze II Division 1",
            Self::BronzeIIDivision2 => "Bronze II Division 2",
            Self::BronzeIIDivision3 => "Bronze II Division 3",
            Self::BronzeIIDivision4 => "Bronze II Division 4",
            // Bronze III
            Self::BronzeIIIDivision1 => "Bronze III Division 1",
            Self::BronzeIIIDivision2 => "Bronze III Division 2",
            Self::BronzeIIIDivision3 => "Bronze III Division 3",
            Self::BronzeIIIDivision4 => "Bronze III Division 4",
            // Silver I
            Self::SilverIDivision1 => "Silver I Division 1",
            Self::SilverIDivision2 => "Silver I Division 2",
            Self::SilverIDivision3 => "Silver I Division 3",
            Self::SilverIDivision4 => "Silver I Division 4",
            // Silver II
            Self::SilverIIDivision1 => "Silver II Division 1",
            Self::SilverIIDivision2 => "Silver II Division 2",
            Self::SilverIIDivision3 => "Silver II Division 3",
            Self::SilverIIDivision4 => "Silver II Division 4",
            // Silver III
            Self::SilverIIIDivision1 => "Silver III Division 1",
            Self::SilverIIIDivision2 => "Silver III Division 2",
            Self::SilverIIIDivision3 => "Silver III Division 3",
            Self::SilverIIIDivision4 => "Silver III Division 4",
            // Gold I
            Self::GoldIDivision1 => "Gold I Division 1",
            Self::GoldIDivision2 => "Gold I Division 2",
            Self::GoldIDivision3 => "Gold I Division 3",
            Self::GoldIDivision4 => "Gold I Division 4",
            // Gold II
            Self::GoldIIDivision1 => "Gold II Division 1",
            Self::GoldIIDivision2 => "Gold II Division 2",
            Self::GoldIIDivision3 => "Gold II Division 3",
            Self::GoldIIDivision4 => "Gold II Division 4",
            // Gold III
            Self::GoldIIIDivision1 => "Gold III Division 1",
            Self::GoldIIIDivision2 => "Gold III Division 2",
            Self::GoldIIIDivision3 => "Gold III Division 3",
            Self::GoldIIIDivision4 => "Gold III Division 4",
            // Platinum I
            Self::PlatinumIDivision1 => "Platinum I Division 1",
            Self::PlatinumIDivision2 => "Platinum I Division 2",
            Self::PlatinumIDivision3 => "Platinum I Division 3",
            Self::PlatinumIDivision4 => "Platinum I Division 4",
            // Platinum II
            Self::PlatinumIIDivision1 => "Platinum II Division 1",
            Self::PlatinumIIDivision2 => "Platinum II Division 2",
            Self::PlatinumIIDivision3 => "Platinum II Division 3",
            Self::PlatinumIIDivision4 => "Platinum II Division 4",
            // Platinum III
            Self::PlatinumIIIDivision1 => "Platinum III Division 1",
            Self::PlatinumIIIDivision2 => "Platinum III Division 2",
            Self::PlatinumIIIDivision3 => "Platinum III Division 3",
            Self::PlatinumIIIDivision4 => "Platinum III Division 4",
            // Diamond I
            Self::DiamondIDivision1 => "Diamond I Division 1",
            Self::DiamondIDivision2 => "Diamond I Division 2",
            Self::DiamondIDivision3 => "Diamond I Division 3",
            Self::DiamondIDivision4 => "Diamond I Division 4",
            // Diamond II
            Self::DiamondIIDivision1 => "Diamond II Division 1",
            Self::DiamondIIDivision2 => "Diamond II Division 2",
            Self::DiamondIIDivision3 => "Diamond II Division 3",
            Self::DiamondIIDivision4 => "Diamond II Division 4",
            // Diamond III
            Self::DiamondIIIDivision1 => "Diamond III Division 1",
            Self::DiamondIIIDivision2 => "Diamond III Division 2",
            Self::DiamondIIIDivision3 => "Diamond III Division 3",
            Self::DiamondIIIDivision4 => "Diamond III Division 4",
            // Champion I
            Self::ChampionIDivision1 => "Champion I Division 1",
            Self::ChampionIDivision2 => "Champion I Division 2",
            Self::ChampionIDivision3 => "Champion I Division 3",
            Self::ChampionIDivision4 => "Champion I Division 4",
            // Champion II
            Self::ChampionIIDivision1 => "Champion II Division 1",
            Self::ChampionIIDivision2 => "Champion II Division 2",
            Self::ChampionIIDivision3 => "Champion II Division 3",
            Self::ChampionIIDivision4 => "Champion II Division 4",
            // Champion III
            Self::ChampionIIIDivision1 => "Champion III Division 1",
            Self::ChampionIIIDivision2 => "Champion III Division 2",
            Self::ChampionIIIDivision3 => "Champion III Division 3",
            Self::ChampionIIIDivision4 => "Champion III Division 4",
            // Grand Champion I
            Self::GrandChampionIDivision1 => "Grand Champion I Division 1",
            Self::GrandChampionIDivision2 => "Grand Champion I Division 2",
            Self::GrandChampionIDivision3 => "Grand Champion I Division 3",
            Self::GrandChampionIDivision4 => "Grand Champion I Division 4",
            // Grand Champion II
            Self::GrandChampionIIDivision1 => "Grand Champion II Division 1",
            Self::GrandChampionIIDivision2 => "Grand Champion II Division 2",
            Self::GrandChampionIIDivision3 => "Grand Champion II Division 3",
            Self::GrandChampionIIDivision4 => "Grand Champion II Division 4",
            // Grand Champion III
            Self::GrandChampionIIIDivision1 => "Grand Champion III Division 1",
            Self::GrandChampionIIIDivision2 => "Grand Champion III Division 2",
            Self::GrandChampionIIIDivision3 => "Grand Champion III Division 3",
            Self::GrandChampionIIIDivision4 => "Grand Champion III Division 4",
            // Supersonic Legend
            Self::SupersonicLegend => "Supersonic Legend",
        };
        write!(f, "{name}")
    }
}

/// Error type for rank parsing failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseRankError {
    /// The invalid input string.
    pub input: String,
}

impl fmt::Display for ParseRankError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid rank: '{}'", self.input)
    }
}

impl core::error::Error for ParseRankError {}

impl FromStr for Rank {
    type Err = ParseRankError;

    /// Parses a rank from a string.
    ///
    /// Accepts formats like:
    /// - `SupersonicLegend` or `Supersonic Legend`
    /// - `GrandChampionIIIDivision1` or `Grand Champion III Division 1`
    /// - `gc3d1` (short form)
    /// - `bronze-1` with division number (ballchasing format)
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Normalize: lowercase and remove spaces/dashes/underscores
        let normalized = s.to_lowercase().replace([' ', '-', '_'], "");

        // Try short forms first (e.g., "gc3d1", "c2d3", "ssld1")
        if let Some(rank) = parse_short_form(&normalized) {
            return Ok(rank);
        }

        // Try full forms
        let rank = match normalized.as_str() {
            // Supersonic Legend
            "supersoniclegend" | "ssl" => Self::SupersonicLegend,

            // Grand Champion III
            "grandchampioniiidivision1" | "grandchampion3division1" => {
                Self::GrandChampionIIIDivision1
            }
            "grandchampioniiidivision2" | "grandchampion3division2" => {
                Self::GrandChampionIIIDivision2
            }
            "grandchampioniiidivision3" | "grandchampion3division3" => {
                Self::GrandChampionIIIDivision3
            }
            "grandchampioniiidivision4" | "grandchampion3division4" => {
                Self::GrandChampionIIIDivision4
            }
            // Grand Champion II
            "grandchampioniidivision1" | "grandchampion2division1" => {
                Self::GrandChampionIIDivision1
            }
            "grandchampioniidivision2" | "grandchampion2division2" => {
                Self::GrandChampionIIDivision2
            }
            "grandchampioniidivision3" | "grandchampion2division3" => {
                Self::GrandChampionIIDivision3
            }
            "grandchampioniidivision4" | "grandchampion2division4" => {
                Self::GrandChampionIIDivision4
            }
            // Grand Champion I
            "grandchampionidivision1" | "grandchampion1division1" => Self::GrandChampionIDivision1,
            "grandchampionidivision2" | "grandchampion1division2" => Self::GrandChampionIDivision2,
            "grandchampionidivision3" | "grandchampion1division3" => Self::GrandChampionIDivision3,
            "grandchampionidivision4" | "grandchampion1division4" => Self::GrandChampionIDivision4,
            // Champion III
            "championiiidivision1" | "champion3division1" => Self::ChampionIIIDivision1,
            "championiiidivision2" | "champion3division2" => Self::ChampionIIIDivision2,
            "championiiidivision3" | "champion3division3" => Self::ChampionIIIDivision3,
            "championiiidivision4" | "champion3division4" => Self::ChampionIIIDivision4,
            // Champion II
            "championiidivision1" | "champion2division1" => Self::ChampionIIDivision1,
            "championiidivision2" | "champion2division2" => Self::ChampionIIDivision2,
            "championiidivision3" | "champion2division3" => Self::ChampionIIDivision3,
            "championiidivision4" | "champion2division4" => Self::ChampionIIDivision4,
            // Champion I
            "championidivision1" | "champion1division1" => Self::ChampionIDivision1,
            "championidivision2" | "champion1division2" => Self::ChampionIDivision2,
            "championidivision3" | "champion1division3" => Self::ChampionIDivision3,
            "championidivision4" | "champion1division4" => Self::ChampionIDivision4,
            // Diamond III
            "diamondiiidivision1" | "diamond3division1" => Self::DiamondIIIDivision1,
            "diamondiiidivision2" | "diamond3division2" => Self::DiamondIIIDivision2,
            "diamondiiidivision3" | "diamond3division3" => Self::DiamondIIIDivision3,
            "diamondiiidivision4" | "diamond3division4" => Self::DiamondIIIDivision4,
            // Diamond II
            "diamondiidivision1" | "diamond2division1" => Self::DiamondIIDivision1,
            "diamondiidivision2" | "diamond2division2" => Self::DiamondIIDivision2,
            "diamondiidivision3" | "diamond2division3" => Self::DiamondIIDivision3,
            "diamondiidivision4" | "diamond2division4" => Self::DiamondIIDivision4,
            // Diamond I
            "diamondidivision1" | "diamond1division1" => Self::DiamondIDivision1,
            "diamondidivision2" | "diamond1division2" => Self::DiamondIDivision2,
            "diamondidivision3" | "diamond1division3" => Self::DiamondIDivision3,
            "diamondidivision4" | "diamond1division4" => Self::DiamondIDivision4,
            // Platinum III
            "platinumiiidivision1" | "platinum3division1" => Self::PlatinumIIIDivision1,
            "platinumiiidivision2" | "platinum3division2" => Self::PlatinumIIIDivision2,
            "platinumiiidivision3" | "platinum3division3" => Self::PlatinumIIIDivision3,
            "platinumiiidivision4" | "platinum3division4" => Self::PlatinumIIIDivision4,
            // Platinum II
            "platinumiidivision1" | "platinum2division1" => Self::PlatinumIIDivision1,
            "platinumiidivision2" | "platinum2division2" => Self::PlatinumIIDivision2,
            "platinumiidivision3" | "platinum2division3" => Self::PlatinumIIDivision3,
            "platinumiidivision4" | "platinum2division4" => Self::PlatinumIIDivision4,
            // Platinum I
            "platinumidivision1" | "platinum1division1" => Self::PlatinumIDivision1,
            "platinumidivision2" | "platinum1division2" => Self::PlatinumIDivision2,
            "platinumidivision3" | "platinum1division3" => Self::PlatinumIDivision3,
            "platinumidivision4" | "platinum1division4" => Self::PlatinumIDivision4,
            // Gold III
            "goldiiidivision1" | "gold3division1" => Self::GoldIIIDivision1,
            "goldiiidivision2" | "gold3division2" => Self::GoldIIIDivision2,
            "goldiiidivision3" | "gold3division3" => Self::GoldIIIDivision3,
            "goldiiidivision4" | "gold3division4" => Self::GoldIIIDivision4,
            // Gold II
            "goldiidivision1" | "gold2division1" => Self::GoldIIDivision1,
            "goldiidivision2" | "gold2division2" => Self::GoldIIDivision2,
            "goldiidivision3" | "gold2division3" => Self::GoldIIDivision3,
            "goldiidivision4" | "gold2division4" => Self::GoldIIDivision4,
            // Gold I
            "goldidivision1" | "gold1division1" => Self::GoldIDivision1,
            "goldidivision2" | "gold1division2" => Self::GoldIDivision2,
            "goldidivision3" | "gold1division3" => Self::GoldIDivision3,
            "goldidivision4" | "gold1division4" => Self::GoldIDivision4,
            // Silver III
            "silveriiidivision1" | "silver3division1" => Self::SilverIIIDivision1,
            "silveriiidivision2" | "silver3division2" => Self::SilverIIIDivision2,
            "silveriiidivision3" | "silver3division3" => Self::SilverIIIDivision3,
            "silveriiidivision4" | "silver3division4" => Self::SilverIIIDivision4,
            // Silver II
            "silveriidivision1" | "silver2division1" => Self::SilverIIDivision1,
            "silveriidivision2" | "silver2division2" => Self::SilverIIDivision2,
            "silveriidivision3" | "silver2division3" => Self::SilverIIDivision3,
            "silveriidivision4" | "silver2division4" => Self::SilverIIDivision4,
            // Silver I
            "silveridivision1" | "silver1division1" => Self::SilverIDivision1,
            "silveridivision2" | "silver1division2" => Self::SilverIDivision2,
            "silveridivision3" | "silver1division3" => Self::SilverIDivision3,
            "silveridivision4" | "silver1division4" => Self::SilverIDivision4,
            // Bronze III
            "bronzeiiidivision1" | "bronze3division1" => Self::BronzeIIIDivision1,
            "bronzeiiidivision2" | "bronze3division2" => Self::BronzeIIIDivision2,
            "bronzeiiidivision3" | "bronze3division3" => Self::BronzeIIIDivision3,
            "bronzeiiidivision4" | "bronze3division4" => Self::BronzeIIIDivision4,
            // Bronze II
            "bronzeiidivision1" | "bronze2division1" => Self::BronzeIIDivision1,
            "bronzeiidivision2" | "bronze2division2" => Self::BronzeIIDivision2,
            "bronzeiidivision3" | "bronze2division3" => Self::BronzeIIDivision3,
            "bronzeiidivision4" | "bronze2division4" => Self::BronzeIIDivision4,
            // Bronze I
            "bronzeidivision1" | "bronze1division1" => Self::BronzeIDivision1,
            "bronzeidivision2" | "bronze1division2" => Self::BronzeIDivision2,
            "bronzeidivision3" | "bronze1division3" => Self::BronzeIDivision3,
            "bronzeidivision4" | "bronze1division4" => Self::BronzeIDivision4,

            _ => {
                return Err(ParseRankError {
                    input: s.to_string(),
                });
            }
        };

        Ok(rank)
    }
}

/// Parses short form ranks like `gc3d1`, `c2d3`, `d1d4`, `ssl`.
#[expect(clippy::too_many_lines)]
fn parse_short_form(s: &str) -> Option<Rank> {
    use Rank::{
        BronzeIDivision1, BronzeIDivision2, BronzeIDivision3, BronzeIDivision4, BronzeIIDivision1,
        BronzeIIDivision2, BronzeIIDivision3, BronzeIIDivision4, BronzeIIIDivision1,
        BronzeIIIDivision2, BronzeIIIDivision3, BronzeIIIDivision4, ChampionIDivision1,
        ChampionIDivision2, ChampionIDivision3, ChampionIDivision4, ChampionIIDivision1,
        ChampionIIDivision2, ChampionIIDivision3, ChampionIIDivision4, ChampionIIIDivision1,
        ChampionIIIDivision2, ChampionIIIDivision3, ChampionIIIDivision4, DiamondIDivision1,
        DiamondIDivision2, DiamondIDivision3, DiamondIDivision4, DiamondIIDivision1,
        DiamondIIDivision2, DiamondIIDivision3, DiamondIIDivision4, DiamondIIIDivision1,
        DiamondIIIDivision2, DiamondIIIDivision3, DiamondIIIDivision4, GoldIDivision1,
        GoldIDivision2, GoldIDivision3, GoldIDivision4, GoldIIDivision1, GoldIIDivision2,
        GoldIIDivision3, GoldIIDivision4, GoldIIIDivision1, GoldIIIDivision2, GoldIIIDivision3,
        GoldIIIDivision4, GrandChampionIDivision1, GrandChampionIDivision2,
        GrandChampionIDivision3, GrandChampionIDivision4, GrandChampionIIDivision1,
        GrandChampionIIDivision2, GrandChampionIIDivision3, GrandChampionIIDivision4,
        GrandChampionIIIDivision1, GrandChampionIIIDivision2, GrandChampionIIIDivision3,
        GrandChampionIIIDivision4, PlatinumIDivision1, PlatinumIDivision2, PlatinumIDivision3,
        PlatinumIDivision4, PlatinumIIDivision1, PlatinumIIDivision2, PlatinumIIDivision3,
        PlatinumIIDivision4, PlatinumIIIDivision1, PlatinumIIIDivision2, PlatinumIIIDivision3,
        PlatinumIIIDivision4, SilverIDivision1, SilverIDivision2, SilverIDivision3,
        SilverIDivision4, SilverIIDivision1, SilverIIDivision2, SilverIIDivision3,
        SilverIIDivision4, SilverIIIDivision1, SilverIIIDivision2, SilverIIIDivision3,
        SilverIIIDivision4, SupersonicLegend,
    };

    // Handle "ssl" specially
    if s == "ssl" {
        return Some(SupersonicLegend);
    }

    // Pattern: <rank_prefix><tier>d<division>
    // gc = Grand Champion, c = Champion, d = Diamond, p = Platinum,
    // g = Gold, s = Silver, b = Bronze
    let patterns: &[(&str, fn(i32, i32) -> Option<Rank>)] = &[
        ("gc", |tier, div| match (tier, div) {
            (3, 1) => Some(GrandChampionIIIDivision1),
            (3, 2) => Some(GrandChampionIIIDivision2),
            (3, 3) => Some(GrandChampionIIIDivision3),
            (3, 4) => Some(GrandChampionIIIDivision4),
            (2, 1) => Some(GrandChampionIIDivision1),
            (2, 2) => Some(GrandChampionIIDivision2),
            (2, 3) => Some(GrandChampionIIDivision3),
            (2, 4) => Some(GrandChampionIIDivision4),
            (1, 1) => Some(GrandChampionIDivision1),
            (1, 2) => Some(GrandChampionIDivision2),
            (1, 3) => Some(GrandChampionIDivision3),
            (1, 4) => Some(GrandChampionIDivision4),
            _ => None,
        }),
        ("c", |tier, div| match (tier, div) {
            (3, 1) => Some(ChampionIIIDivision1),
            (3, 2) => Some(ChampionIIIDivision2),
            (3, 3) => Some(ChampionIIIDivision3),
            (3, 4) => Some(ChampionIIIDivision4),
            (2, 1) => Some(ChampionIIDivision1),
            (2, 2) => Some(ChampionIIDivision2),
            (2, 3) => Some(ChampionIIDivision3),
            (2, 4) => Some(ChampionIIDivision4),
            (1, 1) => Some(ChampionIDivision1),
            (1, 2) => Some(ChampionIDivision2),
            (1, 3) => Some(ChampionIDivision3),
            (1, 4) => Some(ChampionIDivision4),
            _ => None,
        }),
        ("d", |tier, div| match (tier, div) {
            (3, 1) => Some(DiamondIIIDivision1),
            (3, 2) => Some(DiamondIIIDivision2),
            (3, 3) => Some(DiamondIIIDivision3),
            (3, 4) => Some(DiamondIIIDivision4),
            (2, 1) => Some(DiamondIIDivision1),
            (2, 2) => Some(DiamondIIDivision2),
            (2, 3) => Some(DiamondIIDivision3),
            (2, 4) => Some(DiamondIIDivision4),
            (1, 1) => Some(DiamondIDivision1),
            (1, 2) => Some(DiamondIDivision2),
            (1, 3) => Some(DiamondIDivision3),
            (1, 4) => Some(DiamondIDivision4),
            _ => None,
        }),
        ("p", |tier, div| match (tier, div) {
            (3, 1) => Some(PlatinumIIIDivision1),
            (3, 2) => Some(PlatinumIIIDivision2),
            (3, 3) => Some(PlatinumIIIDivision3),
            (3, 4) => Some(PlatinumIIIDivision4),
            (2, 1) => Some(PlatinumIIDivision1),
            (2, 2) => Some(PlatinumIIDivision2),
            (2, 3) => Some(PlatinumIIDivision3),
            (2, 4) => Some(PlatinumIIDivision4),
            (1, 1) => Some(PlatinumIDivision1),
            (1, 2) => Some(PlatinumIDivision2),
            (1, 3) => Some(PlatinumIDivision3),
            (1, 4) => Some(PlatinumIDivision4),
            _ => None,
        }),
        ("g", |tier, div| match (tier, div) {
            (3, 1) => Some(GoldIIIDivision1),
            (3, 2) => Some(GoldIIIDivision2),
            (3, 3) => Some(GoldIIIDivision3),
            (3, 4) => Some(GoldIIIDivision4),
            (2, 1) => Some(GoldIIDivision1),
            (2, 2) => Some(GoldIIDivision2),
            (2, 3) => Some(GoldIIDivision3),
            (2, 4) => Some(GoldIIDivision4),
            (1, 1) => Some(GoldIDivision1),
            (1, 2) => Some(GoldIDivision2),
            (1, 3) => Some(GoldIDivision3),
            (1, 4) => Some(GoldIDivision4),
            _ => None,
        }),
        ("s", |tier, div| match (tier, div) {
            (3, 1) => Some(SilverIIIDivision1),
            (3, 2) => Some(SilverIIIDivision2),
            (3, 3) => Some(SilverIIIDivision3),
            (3, 4) => Some(SilverIIIDivision4),
            (2, 1) => Some(SilverIIDivision1),
            (2, 2) => Some(SilverIIDivision2),
            (2, 3) => Some(SilverIIDivision3),
            (2, 4) => Some(SilverIIDivision4),
            (1, 1) => Some(SilverIDivision1),
            (1, 2) => Some(SilverIDivision2),
            (1, 3) => Some(SilverIDivision3),
            (1, 4) => Some(SilverIDivision4),
            _ => None,
        }),
        ("b", |tier, div| match (tier, div) {
            (3, 1) => Some(BronzeIIIDivision1),
            (3, 2) => Some(BronzeIIIDivision2),
            (3, 3) => Some(BronzeIIIDivision3),
            (3, 4) => Some(BronzeIIIDivision4),
            (2, 1) => Some(BronzeIIDivision1),
            (2, 2) => Some(BronzeIIDivision2),
            (2, 3) => Some(BronzeIIDivision3),
            (2, 4) => Some(BronzeIIDivision4),
            (1, 1) => Some(BronzeIDivision1),
            (1, 2) => Some(BronzeIDivision2),
            (1, 3) => Some(BronzeIDivision3),
            (1, 4) => Some(BronzeIDivision4),
            _ => None,
        }),
    ];

    for (prefix, handler) in patterns {
        if let Some(rest) = s.strip_prefix(prefix)
            && let Some((tier_str, div_str)) = rest.split_once('d')
            && let (Ok(tier), Ok(div)) = (tier_str.parse::<i32>(), div_str.parse::<i32>())
        {
            return handler(tier, div);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmr_values() {
        // Check some known MMR values
        assert_eq!(Rank::SupersonicLegend.mmr_min(), 1883);
        assert_eq!(Rank::SupersonicLegend.mmr_max(), 2000);

        assert_eq!(Rank::GrandChampionIIIDivision1.mmr_min(), 1706);
        assert_eq!(Rank::GrandChampionIIIDivision1.mmr_max(), 1739);

        assert_eq!(Rank::BronzeIDivision1.mmr_min(), 0);
        assert_eq!(Rank::BronzeIDivision1.mmr_max(), 117);
    }

    #[test]
    fn test_mmr_middle() {
        // BronzeIDivision1: min=0, max=117, middle=58
        assert_eq!(Rank::BronzeIDivision1.mmr_middle(), 58);
        // SupersonicLegend: min=1883, max=2000, middle=1941
        assert_eq!(Rank::SupersonicLegend.mmr_middle(), 1941);
    }

    #[test]
    fn test_from_rank_id() {
        assert_eq!(
            Rank::from_rank_id("supersonic-legend", None),
            Some(Rank::SupersonicLegend)
        );
        assert_eq!(
            Rank::from_rank_id("grand-champion-3", Some(1)),
            Some(Rank::GrandChampionIIIDivision1)
        );
        assert_eq!(
            Rank::from_rank_id("grand-champion-3", Some(2)),
            Some(Rank::GrandChampionIIIDivision2)
        );
        assert_eq!(
            Rank::from_rank_id("bronze-1", Some(1)),
            Some(Rank::BronzeIDivision1)
        );
        assert_eq!(Rank::from_rank_id("unknown-rank", Some(1)), None);
    }

    #[test]
    fn test_from_str() {
        // Full form
        assert_eq!(
            "SupersonicLegend".parse::<Rank>().unwrap(),
            Rank::SupersonicLegend
        );
        assert_eq!(
            "Supersonic Legend".parse::<Rank>().unwrap(),
            Rank::SupersonicLegend
        );
        assert_eq!(
            "Grand Champion III Division 1".parse::<Rank>().unwrap(),
            Rank::GrandChampionIIIDivision1
        );

        // Short form
        assert_eq!("ssl".parse::<Rank>().unwrap(), Rank::SupersonicLegend);
        assert_eq!(
            "gc3d1".parse::<Rank>().unwrap(),
            Rank::GrandChampionIIIDivision1
        );
        assert_eq!("c2d3".parse::<Rank>().unwrap(), Rank::ChampionIIDivision3);
        assert_eq!("d1d4".parse::<Rank>().unwrap(), Rank::DiamondIDivision4);
        assert_eq!("b1d1".parse::<Rank>().unwrap(), Rank::BronzeIDivision1);
    }

    #[test]
    fn test_from_str_error() {
        assert!("invalid_rank".parse::<Rank>().is_err());
        assert!("".parse::<Rank>().is_err());
    }

    #[test]
    fn test_display() {
        assert_eq!(Rank::SupersonicLegend.to_string(), "Supersonic Legend");
        assert_eq!(
            Rank::GrandChampionIIIDivision1.to_string(),
            "Grand Champion III Division 1"
        );
        assert_eq!(Rank::BronzeIDivision1.to_string(), "Bronze I Division 1");
    }

    #[test]
    fn test_all_ranks_ordered() {
        let ranks: Vec<_> = Rank::all().collect();
        assert_eq!(ranks.len(), 85); // 21 ranks * 4 divisions + 1 SSL

        // First should be Bronze I Div 1
        assert_eq!(ranks[0], Rank::BronzeIDivision1);
        // Last should be SSL
        assert_eq!(ranks[84], Rank::SupersonicLegend);
    }

    #[test]
    fn test_ord() {
        assert!(Rank::BronzeIDivision1 < Rank::BronzeIDivision2);
        assert!(Rank::BronzeIDivision4 < Rank::BronzeIIDivision1);
        assert!(Rank::GrandChampionIIIDivision4 < Rank::SupersonicLegend);
    }
}
