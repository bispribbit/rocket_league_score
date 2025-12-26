use core::str::FromStr;

/// Game mode enum matching the `PostgreSQL` type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, sqlx::Type)]
#[sqlx(type_name = "game_mode", rename_all = "snake_case")]
pub enum GameMode {
    UnrankedDuels,
    UnrankedDoubles,
    UnrankedStandard,
    UnrankedChaos,
    Private,
    Season,
    Offline,
    RankedDuels,
    RankedDoubles,
    RankedSoloStandard,
    RankedStandard,
    Snowday,
    Rocketlabs,
    Hoops,
    Rumble,
    Tournament,
    Dropshot,
    RankedHoops,
    RankedRumble,
    RankedDropshot,
    RankedSnowday,
    DropshotRumble,
    Heatseeker,
}

impl GameMode {
    /// Returns the API string representation for this game mode.
    #[must_use]
    pub const fn as_api_string(self) -> &'static str {
        match self {
            Self::UnrankedDuels => "unranked-duels",
            Self::UnrankedDoubles => "unranked-doubles",
            Self::UnrankedStandard => "unranked-standard",
            Self::UnrankedChaos => "unranked-chaos",
            Self::Private => "private",
            Self::Season => "season",
            Self::Offline => "offline",
            Self::RankedDuels => "ranked-duels",
            Self::RankedDoubles => "ranked-doubles",
            Self::RankedSoloStandard => "ranked-solo-standard",
            Self::RankedStandard => "ranked-standard",
            Self::Snowday => "snowday",
            Self::Rocketlabs => "rocketlabs",
            Self::Hoops => "hoops",
            Self::Rumble => "rumble",
            Self::Tournament => "tournament",
            Self::Dropshot => "dropshot",
            Self::RankedHoops => "ranked-hoops",
            Self::RankedRumble => "ranked-rumble",
            Self::RankedDropshot => "ranked-dropshot",
            Self::RankedSnowday => "ranked-snowday",
            Self::DropshotRumble => "dropshot-rumble",
            Self::Heatseeker => "heatseeker",
        }
    }
}

impl FromStr for GameMode {
    type Err = anyhow::Error;

    /// Returns the game mode from a string representation.
    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s.to_lowercase().replace('_', "-").as_str() {
            "unranked-duels" => Ok(Self::UnrankedDuels),
            "unranked-doubles" => Ok(Self::UnrankedDoubles),
            "unranked-standard" => Ok(Self::UnrankedStandard),
            "unranked-chaos" => Ok(Self::UnrankedChaos),
            "private" => Ok(Self::Private),
            "season" => Ok(Self::Season),
            "offline" => Ok(Self::Offline),
            "ranked-duels" => Ok(Self::RankedDuels),
            "ranked-doubles" => Ok(Self::RankedDoubles),
            "ranked-solo-standard" => Ok(Self::RankedSoloStandard),
            "ranked-standard" => Ok(Self::RankedStandard),
            "snowday" => Ok(Self::Snowday),
            "rocketlabs" => Ok(Self::Rocketlabs),
            "hoops" => Ok(Self::Hoops),
            "rumble" => Ok(Self::Rumble),
            "tournament" => Ok(Self::Tournament),
            "dropshot" => Ok(Self::Dropshot),
            "ranked-hoops" => Ok(Self::RankedHoops),
            "ranked-rumble" => Ok(Self::RankedRumble),
            "ranked-dropshot" => Ok(Self::RankedDropshot),
            "ranked-snowday" => Ok(Self::RankedSnowday),
            "dropshot-rumble" => Ok(Self::DropshotRumble),
            "heatseeker" => Ok(Self::Heatseeker),
            _ => Err(anyhow::anyhow!("Invalid game mode: {s}")),
        }
    }
}
