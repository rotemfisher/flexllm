"""
Seeds the vdot_paces table with Daniels' training paces.

The VDOT tables in Daniels' Running Formula are stored as raster images —
they can't be extracted by PDF parsers. This seeds the data directly.

Paces are stored as seconds-per-km. The table covers VDOT 30–85.
Zones: E (Easy), M (Marathon), T (Threshold/Tempo), I (Interval), R (Repetition)

Note: Daniels' table does not include every VDOT value; 81–84 are absent.
get_vdot_paces() will return the nearest available value for missing entries.

Run once:
    python etl/seed_vdot.py
    DATABASE_URL=postgresql://user:pass@host/db python etl/seed_vdot.py
"""

import os

import psycopg2

_DEFAULT_DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://localhost:5432/flexllm"
)

# fmt: off
# Source: Daniels' Running Formula 4th ed., Table 3.1
# Paces in sec/km.  i_pace and r_pace are per-km equivalents.
# Note: not every VDOT integer is in the source table (e.g. 81–84 are absent).
VDOT_DATA = [
    # vdot  E_slow  E_fast   M      T      I      R
    (30,    531,    504,    490,    472,    450,    430),
    (32,    510,    484,    471,    453,    432,    413),
    (34,    490,    465,    452,    435,    415,    396),
    (36,    471,    447,    435,    418,    399,    381),
    (38,    454,    431,    419,    403,    384,    366),
    (40,    438,    415,    404,    388,    370,    353),
    (42,    423,    401,    390,    375,    357,    341),
    (44,    409,    388,    377,    363,    345,    329),
    (46,    396,    375,    365,    351,    334,    319),
    (48,    383,    364,    354,    340,    323,    309),
    (50,    372,    353,    343,    330,    313,    299),
    (52,    361,    343,    333,    320,    304,    290),
    (54,    351,    333,    324,    311,    295,    282),
    (56,    341,    324,    315,    302,    287,    274),
    (58,    332,    315,    306,    294,    279,    266),
    (60,    323,    307,    298,    286,    271,    259),
    (62,    315,    299,    290,    279,    264,    252),
    (64,    307,    292,    283,    272,    257,    245),
    (66,    300,    285,    276,    265,    251,    239),
    (68,    293,    278,    270,    259,    245,    233),
    (70,    286,    272,    264,    253,    239,    228),
    (72,    280,    266,    258,    247,    234,    223),
    (74,    274,    260,    252,    242,    229,    218),
    (76,    268,    255,    247,    237,    224,    213),
    (78,    263,    250,    242,    232,    219,    209),
    (80,    258,    245,    237,    227,    215,    204),
    (82,    253,    240,    232,    223,    210,    200),
    (85,    246,    233,    226,    216,    204,    194),
]
# fmt: on


def sec_to_pace(sec: int) -> str:
    """Convert seconds/km to mm:ss string for display."""
    return f"{sec // 60}:{sec % 60:02d}"


def seed(database_url: str = _DEFAULT_DATABASE_URL) -> None:
    con = psycopg2.connect(database_url)
    try:
        cur = con.cursor()
        cur.executemany(
            """
            INSERT INTO vdot_paces
                (vdot, e_pace_slow_sec, e_pace_fast_sec, m_pace_sec,
                 t_pace_sec, i_pace_sec, r_pace_sec)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (vdot) DO NOTHING
            """,
            VDOT_DATA,
        )
        con.commit()

        cur.execute("SELECT COUNT(*) FROM vdot_paces")
        count = cur.fetchone()[0]
        print(f"Seeded {count} VDOT rows")

        cur.execute("SELECT * FROM vdot_paces WHERE vdot = 50")
        row = cur.fetchone()
        if row:
            vdot, e_slow, e_fast, m, t, i, r = row
            print(f"\nVDOT 50 sample:")
            print(f"  Easy:       {sec_to_pace(e_slow)} – {sec_to_pace(e_fast)} /km")
            print(f"  Marathon:   {sec_to_pace(m)} /km")
            print(f"  Threshold:  {sec_to_pace(t)} /km")
            print(f"  Interval:   {sec_to_pace(i)} /km")
            print(f"  Rep:        {sec_to_pace(r)} /km")
    finally:
        con.close()


if __name__ == "__main__":
    seed()
