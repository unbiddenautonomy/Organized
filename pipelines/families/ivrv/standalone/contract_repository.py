"""Legacy spot/IV engine only: contracts from the master file, or else from PostgreSQL.

Everything file-based lives in voltr.instruments.master; this module adds the
older engine's database fallback.
"""

from __future__ import annotations

from decimal import Decimal

import psycopg

import _repo_path  # noqa: F401  (temporary, until step 10)
from voltr.instruments.master import OptionContract, load_nearest_expiry_options as _from_file

CONTRACT_SQL = """
WITH nearest_expiry AS (
    SELECT MIN(xprydt::date) AS expiry
    FROM public.nse_fo_contract
    WHERE UPPER(tckrsymb) = %(symbol)s
      AND UPPER(optntp) IN ('CE', 'PE')
      AND xprydt::date >= CURRENT_DATE
)
SELECT
    fininstrmid,
    undrlygfininstrmid,
    fininstrmnm,
    tckrsymb,
    xprydt::date,
    strkpric::numeric,
    UPPER(optntp)
FROM public.nse_fo_contract
CROSS JOIN nearest_expiry
WHERE UPPER(tckrsymb) = %(symbol)s
  AND UPPER(optntp) IN ('CE', 'PE')
  AND xprydt::date = nearest_expiry.expiry
ORDER BY strkpric::numeric, UPPER(optntp), fininstrmid::bigint
"""


def load_nearest_expiry_options(config: dict) -> dict[int, OptionContract]:
    """Return nearest non-expired CE/PE contracts keyed by integer feed token."""
    if config.get("contract_file"):
        return _from_file(config)
    symbol = str(config.get("contract_symbol", "BANKNIFTY")).strip().upper()
    connection_options = {
        "host": config["db_host"],
        "port": int(config["db_port"]),
        "user": config["db_user"],
        "password": config["db_password"],
        "dbname": config["db_name"],
        "sslmode": config.get("db_sslmode", "disable"),
        "connect_timeout": 5,
    }
    with psycopg.connect(**connection_options) as connection:
        rows = connection.execute(CONTRACT_SQL, {"symbol": symbol}).fetchall()

    contracts: dict[int, OptionContract] = {}
    for row in rows:
        contract = OptionContract(
            token=int(row[0]),
            underlying_token=int(row[1]),
            instrument_name=str(row[2]).strip(),
            symbol=str(row[3]).strip().upper(),
            expiry=row[4],
            strike_price=Decimal(row[5]),
            option_type=str(row[6]).strip().upper(),
        )
        contracts[contract.token] = contract
    if not contracts:
        raise RuntimeError(f"no non-expired CE/PE contracts found for {symbol}")
    return contracts
