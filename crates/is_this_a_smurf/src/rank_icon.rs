//! Static rank tier icons under `assets/ranks` (one image per Roman tier, not per division).
//!
//! Icons must be registered with [`asset!`] so the Dioxus CLI copies them into the build output.
//! Sources are PNG on disk; [`manganis`] resizes at build time and keeps PNG for fast decode in the browser.

#![expect(clippy::volatile_composites)] // from manganis `asset!` macro

use manganis::{Asset, AssetOptions, ImageFormat, ImageSize, asset};
use replay_structs::RankDivision;

/// Maximum on-screen size is about 48 CSS pixels; 128 covers roughly 2× density.
const RANK_ICON_WIDTH_PX: u32 = 128;
const RANK_ICON_HEIGHT_PX: u32 = 128;

macro_rules! rank_icon_asset {
    ($path:literal) => {
        asset!(
            $path,
            AssetOptions::image()
                .with_size(ImageSize::Manual {
                    width: RANK_ICON_WIDTH_PX,
                    height: RANK_ICON_HEIGHT_PX,
                })
                .with_format(ImageFormat::Png)
        )
    };
}

const RANK_BRONZE_1: Asset = rank_icon_asset!("/assets/ranks/bronze-1.png");
const RANK_BRONZE_2: Asset = rank_icon_asset!("/assets/ranks/bronze-2.png");
const RANK_BRONZE_3: Asset = rank_icon_asset!("/assets/ranks/bronze-3.png");
const RANK_SILVER_1: Asset = rank_icon_asset!("/assets/ranks/silver-1.png");
const RANK_SILVER_2: Asset = rank_icon_asset!("/assets/ranks/silver-2.png");
const RANK_SILVER_3: Asset = rank_icon_asset!("/assets/ranks/silver-3.png");
const RANK_GOLD_1: Asset = rank_icon_asset!("/assets/ranks/gold-1.png");
const RANK_GOLD_2: Asset = rank_icon_asset!("/assets/ranks/gold-2.png");
const RANK_GOLD_3: Asset = rank_icon_asset!("/assets/ranks/gold-3.png");
const RANK_PLATINIUM_1: Asset = rank_icon_asset!("/assets/ranks/platinium-1.png");
const RANK_PLATINIUM_2: Asset = rank_icon_asset!("/assets/ranks/platinium-2.png");
const RANK_PLATINIUM_3: Asset = rank_icon_asset!("/assets/ranks/platinium-3.png");
const RANK_DIAMOND_1: Asset = rank_icon_asset!("/assets/ranks/diamond-1.png");
const RANK_DIAMOND_2: Asset = rank_icon_asset!("/assets/ranks/diamond-2.png");
const RANK_DIAMOND_3: Asset = rank_icon_asset!("/assets/ranks/diamond-3.png");
const RANK_CHAMPION_1: Asset = rank_icon_asset!("/assets/ranks/champion-1.png");
const RANK_CHAMPION_2: Asset = rank_icon_asset!("/assets/ranks/champion-2.png");
const RANK_CHAMPION_3: Asset = rank_icon_asset!("/assets/ranks/champion-3.png");
const RANK_GRAND_CHAMPION_1: Asset = rank_icon_asset!("/assets/ranks/grand-champion-1.png");
const RANK_GRAND_CHAMPION_2: Asset = rank_icon_asset!("/assets/ranks/grand-champion-2.png");
const RANK_GRAND_CHAMPION_3: Asset = rank_icon_asset!("/assets/ranks/grand-champion-3.png");
const RANK_SUPER_SONIC_LEGEND: Asset = rank_icon_asset!("/assets/ranks/super-sonic-legend.png");

/// Bundled asset for the rank tier icon matching `division`.
///
/// Asset file names follow the on-disk pack (including the `platinium` spelling).
#[must_use]
pub(crate) const fn rank_division_icon_asset(division: RankDivision) -> Asset {
    match division {
        RankDivision::BronzeIDivision1
        | RankDivision::BronzeIDivision2
        | RankDivision::BronzeIDivision3
        | RankDivision::BronzeIDivision4 => RANK_BRONZE_1,
        RankDivision::BronzeIIDivision1
        | RankDivision::BronzeIIDivision2
        | RankDivision::BronzeIIDivision3
        | RankDivision::BronzeIIDivision4 => RANK_BRONZE_2,
        RankDivision::BronzeIIIDivision1
        | RankDivision::BronzeIIIDivision2
        | RankDivision::BronzeIIIDivision3
        | RankDivision::BronzeIIIDivision4 => RANK_BRONZE_3,
        RankDivision::SilverIDivision1
        | RankDivision::SilverIDivision2
        | RankDivision::SilverIDivision3
        | RankDivision::SilverIDivision4 => RANK_SILVER_1,
        RankDivision::SilverIIDivision1
        | RankDivision::SilverIIDivision2
        | RankDivision::SilverIIDivision3
        | RankDivision::SilverIIDivision4 => RANK_SILVER_2,
        RankDivision::SilverIIIDivision1
        | RankDivision::SilverIIIDivision2
        | RankDivision::SilverIIIDivision3
        | RankDivision::SilverIIIDivision4 => RANK_SILVER_3,
        RankDivision::GoldIDivision1
        | RankDivision::GoldIDivision2
        | RankDivision::GoldIDivision3
        | RankDivision::GoldIDivision4 => RANK_GOLD_1,
        RankDivision::GoldIIDivision1
        | RankDivision::GoldIIDivision2
        | RankDivision::GoldIIDivision3
        | RankDivision::GoldIIDivision4 => RANK_GOLD_2,
        RankDivision::GoldIIIDivision1
        | RankDivision::GoldIIIDivision2
        | RankDivision::GoldIIIDivision3
        | RankDivision::GoldIIIDivision4 => RANK_GOLD_3,
        RankDivision::PlatinumIDivision1
        | RankDivision::PlatinumIDivision2
        | RankDivision::PlatinumIDivision3
        | RankDivision::PlatinumIDivision4 => RANK_PLATINIUM_1,
        RankDivision::PlatinumIIDivision1
        | RankDivision::PlatinumIIDivision2
        | RankDivision::PlatinumIIDivision3
        | RankDivision::PlatinumIIDivision4 => RANK_PLATINIUM_2,
        RankDivision::PlatinumIIIDivision1
        | RankDivision::PlatinumIIIDivision2
        | RankDivision::PlatinumIIIDivision3
        | RankDivision::PlatinumIIIDivision4 => RANK_PLATINIUM_3,
        RankDivision::DiamondIDivision1
        | RankDivision::DiamondIDivision2
        | RankDivision::DiamondIDivision3
        | RankDivision::DiamondIDivision4 => RANK_DIAMOND_1,
        RankDivision::DiamondIIDivision1
        | RankDivision::DiamondIIDivision2
        | RankDivision::DiamondIIDivision3
        | RankDivision::DiamondIIDivision4 => RANK_DIAMOND_2,
        RankDivision::DiamondIIIDivision1
        | RankDivision::DiamondIIIDivision2
        | RankDivision::DiamondIIIDivision3
        | RankDivision::DiamondIIIDivision4 => RANK_DIAMOND_3,
        RankDivision::ChampionIDivision1
        | RankDivision::ChampionIDivision2
        | RankDivision::ChampionIDivision3
        | RankDivision::ChampionIDivision4 => RANK_CHAMPION_1,
        RankDivision::ChampionIIDivision1
        | RankDivision::ChampionIIDivision2
        | RankDivision::ChampionIIDivision3
        | RankDivision::ChampionIIDivision4 => RANK_CHAMPION_2,
        RankDivision::ChampionIIIDivision1
        | RankDivision::ChampionIIIDivision2
        | RankDivision::ChampionIIIDivision3
        | RankDivision::ChampionIIIDivision4 => RANK_CHAMPION_3,
        RankDivision::GrandChampionIDivision1
        | RankDivision::GrandChampionIDivision2
        | RankDivision::GrandChampionIDivision3
        | RankDivision::GrandChampionIDivision4 => RANK_GRAND_CHAMPION_1,
        RankDivision::GrandChampionIIDivision1
        | RankDivision::GrandChampionIIDivision2
        | RankDivision::GrandChampionIIDivision3
        | RankDivision::GrandChampionIIDivision4 => RANK_GRAND_CHAMPION_2,
        RankDivision::GrandChampionIIIDivision1
        | RankDivision::GrandChampionIIIDivision2
        | RankDivision::GrandChampionIIIDivision3
        | RankDivision::GrandChampionIIIDivision4 => RANK_GRAND_CHAMPION_3,
        RankDivision::SupersonicLegend => RANK_SUPER_SONIC_LEGEND,
    }
}
