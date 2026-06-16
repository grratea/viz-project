import pandas as pd
import json

YEAR_FROM = 2000
YEAR_TO   = 2024

# Minimalna dob za F1 vozača (FIA propis: 18 godina)
MIN_AGE = 18

races        = pd.read_csv("races.csv")
results      = pd.read_csv("results.csv")
drivers      = pd.read_csv("drivers.csv")
constructors = pd.read_csv("constructors.csv")
standings    = pd.read_csv("driver_standings.csv")
con_stand    = pd.read_csv("constructor_standings.csv")

# ── čišćenje \N vrijednosti ───────────────────────────────
for df in [results, standings, con_stand, drivers]:
    df.replace(r"\\N", pd.NA, regex=True, inplace=True)

# ── Izračunaj godinu rođenja vozača ───────────────────────
# drivers.csv sadrži kolonu 'dob' u formatu YYYY-MM-DD
drivers["birth_year"] = pd.to_datetime(
    drivers["dob"], errors="coerce"
).dt.year

# ── filtriraj godine utrka ────────────────────────────────
races = races[(races["year"] >= YEAR_FROM) & (races["year"] <= YEAR_TO)].copy()
race_ids = set(races["raceId"])

results   = results[results["raceId"].isin(race_ids)].copy()
standings = standings[standings["raceId"].isin(race_ids)].copy()
con_stand = con_stand[con_stand["raceId"].isin(race_ids)].copy()

# ── numeričke konverzije ──────────────────────────────────
results["points"]     = pd.to_numeric(results["points"],     errors="coerce").fillna(0)
results["position"]   = pd.to_numeric(results["position"],   errors="coerce")
standings["points"]   = pd.to_numeric(standings["points"],   errors="coerce").fillna(0)
standings["wins"]     = pd.to_numeric(standings["wins"],     errors="coerce").fillna(0)
con_stand["points"]   = pd.to_numeric(con_stand["points"],   errors="coerce").fillna(0)
con_stand["position"] = pd.to_numeric(con_stand["position"], errors="coerce")

# ── join rezultata ────────────────────────────────────────
results = results.merge(races[["raceId","year","name","round"]], on="raceId", how="left")
results = results.merge(
    drivers[["driverId","forename","surname","nationality","birth_year"]],
    on="driverId", how="left"
)
results = results.merge(
    constructors[["constructorId","name","nationality"]].rename(
        columns={"name":"constructor","nationality":"constructor_nat"}),
    on="constructorId", how="left"
)
results["driver"] = results["forename"] + " " + results["surname"]


before = len(results)
results = results[
    results["birth_year"].isna() |  
    (results["year"] - results["birth_year"] >= MIN_AGE)
].copy()
after = len(results)


# ── join standings ────────────────────────────────────────
standings = standings.merge(races[["raceId","year"]], on="raceId", how="left")
standings = standings.merge(
    drivers[["driverId","forename","surname","birth_year"]],
    on="driverId", how="left"
)
standings["driver"] = standings["forename"] + " " + standings["surname"]

# Primijeni isti filter i na standings
standings = standings[
    standings["birth_year"].isna() |
    (standings["year"] - standings["birth_year"] >= MIN_AGE)
].copy()

# ── join constructor standings ────────────────────────────
con_stand = con_stand.merge(races[["raceId","year","round"]], on="raceId", how="left")
con_stand = con_stand.merge(
    constructors[["constructorId","name"]].rename(columns={"name":"constructor"}),
    on="constructorId", how="left"
)

years_list = sorted(races["year"].unique().tolist())

# ============================================================
# 1. DRIVER POINTS PER SEASON
# ============================================================

last_round = standings.groupby(["year","driverId"])["raceId"].transform("max")
final_standings = standings[standings["raceId"] == last_round].copy()
final_standings = final_standings.drop_duplicates(subset=["year","driverId"])

top_drivers_ids = (
    final_standings.groupby("driverId")["points"].sum()
    .nlargest(10).index.tolist()
)
top_names = (
    final_standings[final_standings["driverId"].isin(top_drivers_ids)]
    .drop_duplicates("driverId")[["driverId","driver"]]
    .set_index("driverId")["driver"].to_dict()
)

driver_points_season = {}
for did in top_drivers_ids:
    name = top_names[did]
    rows = final_standings[final_standings["driverId"] == did][["year","points"]].sort_values("year")
    driver_points_season[name] = {int(r["year"]): float(r["points"]) for _, r in rows.iterrows()}

# ============================================================
# 2. CONSTRUCTOR WINS PER SEASON
# ============================================================

wins = results[results["position"] == 1].copy()
constr_wins = wins.groupby(["year","constructor"]).size().reset_index(name="wins")
top_constr = (
    constr_wins.groupby("constructor")["wins"].sum()
    .nlargest(10).index.tolist()
)
constr_wins_dict = {}
for _, row in constr_wins[constr_wins["constructor"].isin(top_constr)].iterrows():
    c, y, w = row["constructor"], int(row["year"]), int(row["wins"])
    constr_wins_dict.setdefault(c, {})[y] = w

# ============================================================
# 3. CHOROPLETH — statistike po državama
# ============================================================

NAT_TO_ISO3 = {
    "British":"GBR","German":"DEU","Finnish":"FIN","French":"FRA",
    "Australian":"AUS","Brazilian":"BRA","Spanish":"ESP","Austrian":"AUT",
    "Italian":"ITA","Dutch":"NLD","Canadian":"CAN","American":"USA",
    "Mexican":"MEX","Polish":"POL","Hungarian":"HUN","Danish":"DNK",
    "Swedish":"SWE","Swiss":"CHE","Belgian":"BEL","Japanese":"JPN",
    "Russian":"RUS","Chinese":"CHN","Indian":"IND","Colombian":"COL",
    "Venezuelan":"VEN","Monegasque":"MCO","New Zealander":"NZL",
    "Argentine":"ARG","Portuguese":"PRT","Czech":"CZE","Thai":"THA",
    "South African":"ZAF","Malaysian":"MYS","Indonesian":"IDN",
    "Irish":"IRL","Chilean":"CHL","Uruguayan":"URY","Korean":"KOR",
    "Singaporean":"SGP","American-Italian":"USA","Rhodesian":"ZWE",
    "East German":"DEU","Zimbabwean":"ZWE","Liechtensteiner":"LIE",
}

choropleth_data = {}

for yr in years_list:
    yr_results = results[results["year"] == yr].copy()
    yr_results["iso3"] = yr_results["nationality"].map(NAT_TO_ISO3)
    yr_results = yr_results.dropna(subset=["iso3"])

    choropleth_data[yr] = {
        "points":  yr_results.groupby("iso3")["points"].sum().round(1).to_dict(),
        "wins":    yr_results[yr_results["position"]==1].groupby("iso3").size().to_dict(),
        "drivers": yr_results.groupby("iso3")["driverId"].nunique().to_dict(),
        "podiums": yr_results[yr_results["position"]<=3].groupby("iso3").size().to_dict(),
    }

all_results = results.copy()
all_results["iso3"] = all_results["nationality"].map(NAT_TO_ISO3)
all_results = all_results.dropna(subset=["iso3"])

choropleth_total = {
    "points":  all_results.groupby("iso3")["points"].sum().round(1).to_dict(),
    "wins":    all_results[all_results["position"]==1].groupby("iso3").size().to_dict(),
    "drivers": all_results.groupby("iso3")["driverId"].nunique().to_dict(),
    "podiums": all_results[all_results["position"]<=3].groupby("iso3").size().to_dict(),
}

driver_by_country = (
    all_results.groupby(["iso3","driver"])["points"].sum()
    .reset_index().sort_values(["iso3","points"], ascending=[True,False])
)
choropleth_top_driver = (
    driver_by_country.groupby("iso3").first().reset_index()
    [["iso3","driver","points"]].set_index("iso3").to_dict(orient="index")
)

# ============================================================
# 4. BUMP CHART
# ============================================================

bump_data = {}

for yr in years_list:
    yr_races = races[races["year"] == yr].sort_values("round")
    rounds_in_yr = sorted(yr_races["round"].unique())
    yr_cs = con_stand[con_stand["year"] == yr].copy()

    last_r = yr_cs.groupby("constructorId")["round"].transform("max")
    final_cs = yr_cs[yr_cs["round"] == last_r].drop_duplicates("constructorId")
    top10_ids = (
        final_cs.nsmallest(10, "position")["constructorId"].tolist()
        if "position" in final_cs.columns and final_cs["position"].notna().any()
        else final_cs.nlargest(10, "points")["constructorId"].tolist()
    )
    top10_names = (
        constructors[constructors["constructorId"].isin(top10_ids)]
        .set_index("constructorId")["name"].to_dict()
    )

    season_rows = []
    for rnd in rounds_in_yr:
        snap = (
            yr_cs[yr_cs["round"] <= rnd]
            .sort_values("round", ascending=False)
            .drop_duplicates("constructorId")
        )
        snap = snap[snap["constructorId"].isin(top10_ids)].copy()
        snap["constructor"] = snap["constructorId"].map(top10_names)

        if snap["position"].notna().any():
            snap = snap.sort_values(
                ["position", "points"], ascending=[True, False]
            ).reset_index(drop=True)
        else:
            snap = snap.sort_values("points", ascending=False).reset_index(drop=True)
        snap["rank"] = snap.index + 1

        for _, row in snap.iterrows():
            if pd.notna(row["constructor"]):
                season_rows.append({
                    "constructor": row["constructor"],
                    "round": int(rnd),
                    "rank": int(row["rank"]),
                    "points": float(row["points"]),
                })
    bump_data[int(yr)] = season_rows

# ============================================================
# 5. ANIMATION DATA
# ============================================================

anim_years = sorted(races["year"].unique())[-5:]
anim_data = {}

for yr in anim_years:
    yr_races = races[races["year"] == yr].sort_values("round")
    rounds = []
    for _, race_row in yr_races.iterrows():
        rid = race_row["raceId"]
        st  = standings[standings["raceId"] == rid][["driver","points","wins"]].copy()
        st  = st.sort_values("points", ascending=False).head(10)
        rounds.append({
            "round": int(race_row["round"]),
            "race":  race_row["name"],
            "standings": st.to_dict(orient="records"),
        })
    anim_data[int(yr)] = rounds

# ============================================================
# OUTPUT
# ============================================================
output = {
    "years":                 years_list,
    "top_drivers":           list(driver_points_season.keys()),
    "driver_points_season":  driver_points_season,
    "top_constructors":      top_constr,
    "constructor_wins":      constr_wins_dict,
    "choropleth_data":       {str(k): v for k, v in choropleth_data.items()},
    "choropleth_total":      choropleth_total,
    "choropleth_top_driver": choropleth_top_driver,
    "bump_data":             {str(k): v for k, v in bump_data.items()},
    "bump_years":            years_list,
    "animation_data":        {str(k): v for k, v in anim_data.items()},
    "anim_years":            [int(y) for y in anim_years],
}

with open("f1_data.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False)

print("Generiran f1_data.json")
