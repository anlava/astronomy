"""
Enrich a list of ZTF OIDs with Gaia DR3 and Hunt+ 2024 open-cluster data.

Pipeline per OID:
  1. RA/Dec from the SNAD ZTF DR API (db.ztf.snad.space).
  2. Cone search in Hunt+ 2024 cluster members (VizieR J/A+A/686/A42/members).
  3. Cluster-level parameters from Hunt+ 2024 clusters table
     (J/A+A/686/A42/clusters: distance, type, N members, RV, ...).
  4. Cone search in Gaia DR3 (VizieR I/355/gaiadr3).

Usage:
    python enrich_flares.py flares_list.csv -o flares_enriched.csv
    python enrich_flares.py flares_list.csv --radius 2.0 --verbose

Input CSV must contain an "oid" column (one ZTF OID per row).
Output CSV contains one row per OID; unmatched objects have empty fields.
"""

import argparse
import logging
import time
from pathlib import Path

import astropy.units as u
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

logger = logging.getLogger(__name__)

SNAD_META_URL = "https://db.ztf.snad.space/api/v3/data/latest/oid/meta/json"

VIZIER_HUNT_MEMBERS = "J/A+A/686/A42/members"
VIZIER_HUNT_CLUSTERS = "J/A+A/686/A42/clusters"
VIZIER_GAIA_DR3 = "I/355/gaiadr3"

# Columns kept from each catalog (renamed on output)
HUNT_MEMBER_COLS = {
    "Name": "hunt_cluster_name",
    "GaiaDR3": "hunt_gaia_dr3",
    "Prob": "hunt_membership_prob",
    "pmRA": "hunt_pmra",
    "pmDE": "hunt_pmde",
    "Plx": "hunt_plx",
    "Gmag": "hunt_gmag",
}
HUNT_CLUSTER_COLS = {
    "Name": "cluster_name",
    "Type": "cluster_type",
    "N": "cluster_n_members",
    "dist50": "cluster_dist50_pc",
    "r50": "cluster_r50_deg",
    "pmRA": "cluster_pmra",
    "pmDE": "cluster_pmde",
    "Plx": "cluster_plx",
    "RV": "cluster_rv_kms",
}
GAIA_COLS = {
    "Source": "gaia_source_id",
    "Plx": "gaia_plx_mas",
    "e_Plx": "gaia_plx_err",
    "pmRA": "gaia_pmra",
    "e_pmRA": "gaia_pmra_err",
    "pmDE": "gaia_pmde",
    "e_pmDE": "gaia_pmde_err",
    "Gmag": "gaia_gmag",
    "BP-RP": "gaia_bp_rp",
    "RUWE": "gaia_ruwe",
    "RV": "gaia_rv_kms",
    "Teff": "gaia_teff_k",
    "Dist": "gaia_dist_pc",
}

ALL_OUT_COLS = (
    ["oid", "ra_deg", "dec_deg"]
    + list(GAIA_COLS.values())
    + ["gaia_sep_arcsec"]
    + list(HUNT_MEMBER_COLS.values())
    + ["hunt_sep_arcsec"]
    + [v for k, v in HUNT_CLUSTER_COLS.items() if k != "Name"]
)


def get_ztf_coords(oid: int, session: requests.Session, retries: int = 3) -> tuple[float, float] | None:
    """Fetch (ra, dec) of a ZTF OID from the SNAD ZTF DR API."""
    for attempt in range(retries):
        try:
            r = session.get(SNAD_META_URL, params={"oid": oid}, timeout=30)
            if r.status_code == 404:
                logger.warning(f"oid={oid}: not found in SNAD ZTF DR API (404)")
                return None
            r.raise_for_status()
            meta = r.json()[str(oid)]["coord"]
            return float(meta["ra"]), float(meta["dec"])
        except (requests.RequestException, KeyError, ValueError) as e:
            logger.warning(f"oid={oid}: SNAD API error (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(2**attempt)
    return None


def cone_search(vizier: Vizier, coord: SkyCoord, catalog: str, radius_arcsec: float):
    """Nearest-neighbour cone search in a VizieR catalog. Returns (table_row, sep_arcsec) or None."""
    try:
        res = vizier.query_region(coord, radius=radius_arcsec * u.arcsec, catalog=catalog)
    except Exception as e:
        logger.warning(f"VizieR query failed for {catalog}: {e}")
        return None
    if not res or len(res[0]) == 0:
        return None
    table = res[0]
    # Coordinate column names differ across catalogs
    for ra_col, dec_col in (("_RA.icrs", "_DE.icrs"), ("RAJ2000", "DEJ2000"), ("RA_ICRS", "DE_ICRS")):
        if ra_col in table.colnames and dec_col in table.colnames:
            cat_coords = SkyCoord(ra=table[ra_col], dec=table[dec_col], unit="deg")
            break
    else:
        logger.warning(f"No coordinate columns found in {catalog}")
        return None
    seps = coord.separation(cat_coords).arcsec
    best = seps.argmin()
    return table[best], float(seps[best])


def fetch_hunt_cluster(vizier: Vizier, cluster_name: str):
    """Fetch cluster-level parameters from the Hunt+ clusters table by name."""
    try:
        res = vizier.query_constraints(catalog=VIZIER_HUNT_CLUSTERS, Name=cluster_name)
    except Exception as e:
        logger.warning(f"VizieR cluster query failed for {cluster_name}: {e}")
        return None
    if not res or len(res[0]) == 0:
        return None
    return res[0][0]


def enrich_oid(
    oid: int,
    session: requests.Session,
    vizier: Vizier,
    radius_arcsec: float,
    cluster_cache: dict,
) -> dict:
    row = {c: None for c in ALL_OUT_COLS}
    row["oid"] = oid

    coords = get_ztf_coords(oid, session)
    if coords is None:
        return row
    ra, dec = coords
    row["ra_deg"], row["dec_deg"] = ra, dec
    coord = SkyCoord(ra=ra, dec=dec, unit="deg")

    # Gaia DR3
    gaia = cone_search(vizier, coord, VIZIER_GAIA_DR3, radius_arcsec)
    if gaia is not None:
        g_row, g_sep = gaia
        for src, dst in GAIA_COLS.items():
            if src not in g_row.colnames:
                continue
            val = g_row[src]
            row[dst] = None if getattr(val, "masked", False) else val
        row["gaia_sep_arcsec"] = round(g_sep, 3)

    # Hunt+ 2024 members
    hunt = cone_search(vizier, coord, VIZIER_HUNT_MEMBERS, radius_arcsec)
    if hunt is not None:
        h_row, h_sep = hunt
        for src, dst in HUNT_MEMBER_COLS.items():
            val = h_row[src]
            row[dst] = None if getattr(val, "masked", False) else val
        row["hunt_sep_arcsec"] = round(h_sep, 3)

        # Cluster-level parameters (cached by cluster name)
        cluster_name = row["hunt_cluster_name"]
        if cluster_name:
            if cluster_name not in cluster_cache:
                cluster_cache[cluster_name] = fetch_hunt_cluster(vizier, cluster_name)
            c_row = cluster_cache[cluster_name]
            if c_row is not None:
                for src, dst in HUNT_CLUSTER_COLS.items():
                    if src == "Name":
                        continue
                    val = c_row[src]
                    row[dst] = None if getattr(val, "masked", False) else val

    return row


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Enrich ZTF OIDs with Gaia DR3 and Hunt+ 2024 cluster data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input", type=Path, help="CSV with an 'oid' column")
    ap.add_argument("-o", "--output", type=Path, default=Path("flares_enriched.csv"))
    ap.add_argument("--radius", type=float, default=1.5, help="Cone-search radius, arcsec")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    oids = pd.read_csv(args.input)["oid"].astype("int64").tolist()
    logger.info(f"Loaded {len(oids)} oids from {args.input}")

    vizier = Vizier(columns=["all"])  # all columns, we filter later
    vizier.ROW_LIMIT = 50

    session = requests.Session()
    cluster_cache: dict = {}
    rows = []

    for i, oid in enumerate(oids, 1):
        row = enrich_oid(oid, session, vizier, args.radius, cluster_cache)
        rows.append(row)
        status = []
        if row["gaia_source_id"]:
            status.append(f"Gaia={row['gaia_source_id']}")
        if row["hunt_cluster_name"]:
            status.append(f"cluster={row['hunt_cluster_name']}(P={row['hunt_membership_prob']:.2f})")
        logger.info(f"[{i}/{len(oids)}] oid={oid} " + ("; ".join(status) if status else "no matches"))

    out = pd.DataFrame(rows, columns=ALL_OUT_COLS)
    out.to_csv(args.output, index=False)
    n_gaia = out["gaia_source_id"].notna().sum()
    n_hunt = out["hunt_cluster_name"].notna().sum()
    logger.info(
        f"Saved {len(out)} rows to {args.output}: "
        f"Gaia matches={n_gaia}, Hunt+ cluster members={n_hunt}"
    )


if __name__ == "__main__":
    main()
