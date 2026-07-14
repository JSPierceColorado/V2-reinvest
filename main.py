import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import gspread
import requests
from fastapi import FastAPI, Header, HTTPException
from google.oauth2.service_account import Credentials


APP_VERSION = "0.3.0-health-aware-rsi-fix"

STATE_HEADERS = ["key", "value", "updated_at"]
LOTS_HEADERS = [
    "lot_id",
    "symbol",
    "remaining_qty",
    "cost_per_share",
    "acquired_at",
    "source_activity_id",
    "source",
]
ACTIVITY_HEADERS = [
    "activity_id",
    "activity_time",
    "symbol",
    "side",
    "qty",
    "price",
    "realized_pl",
    "profitable",
    "pending_added",
    "note",
    "processed_at",
]
ORDER_HEADERS = [
    "order_id",
    "client_order_id",
    "submitted_at",
    "symbol",
    "notional",
    "status",
    "dry_run",
    "pending_spent",
    "note",
]

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("profit-reinvestor")

app = FastAPI(title="Profit Reinvestor", version=APP_VERSION)

RUN_LOCK = threading.Lock()
STOP_EVENT = threading.Event()
WORKER_THREAD: Optional[threading.Thread] = None
LAST_STATUS: Dict[str, Any] = {"state": "starting", "version": APP_VERSION}


@dataclass(frozen=True)
class Config:
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_paper: bool
    trading_base_url: str
    data_base_url: str
    alpaca_data_feed: str

    google_sheet_id: str
    google_service_account_json: str
    state_tab: str
    lots_tab: str
    activity_tab: str
    orders_tab: str

    invest_target_symbols: Tuple[str, ...]
    ignored_profit_symbols: Set[str]
    min_child_notional: Decimal
    dry_run: bool
    rsi_enabled: bool
    rsi_period: int
    rsi_threshold: Decimal
    rsi_bars_limit: int
    rsi_timeframe: str
    rsi_adjustment: str
    rsi_lookback_days: int

    # Reinvestment portfolio-health gate. RED pauses all reinvestment, YELLOW
    # reduces order size, and GREEN invests the full pending amount.
    health_gate_enabled: bool
    health_max_position_count: int
    health_yellow_exposure_pct: Decimal
    health_red_exposure_pct: Decimal
    health_yellow_min_cash_pct: Decimal
    health_red_min_cash_pct: Decimal
    health_yellow_drawdown_pct: Decimal
    health_red_drawdown_pct: Decimal
    health_yellow_red_position_pct: Decimal
    health_red_red_position_pct: Decimal
    health_require_positive_total_realized: bool
    health_yellow_investment_multiplier: Decimal
    health_equity_high_watermark: Decimal

    seed_positions_on_first_run: bool
    page_size: int
    max_pages_per_cycle: int
    request_timeout_seconds: float
    request_retries: int
    request_sleep_seconds: float
    rate_limit_sleep_seconds: float
    error_body_max_chars: int

    poll_seconds: int
    error_backoff_seconds: int
    bot_auto_start: bool
    bot_run_token: str


class HttpStatusError(RuntimeError):
    def __init__(self, method: str, url: str, status_code: int, body: str) -> None:
        self.method = method
        self.url = url
        self.status_code = status_code
        self.body = body
        suffix = f" body={body}" if body else ""
        super().__init__(f"{method} {url} failed status={status_code}{suffix}")


def env_required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def env_decimal(name: str, default: str) -> Decimal:
    raw = os.getenv(name, default).strip()
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise RuntimeError(f"Invalid decimal env var {name}={raw!r}") from exc
    if not value.is_finite():
        raise RuntimeError(f"Invalid decimal env var {name}={raw!r}")
    return value


def env_symbols(name: str, default: str) -> Tuple[str, ...]:
    raw = os.getenv(name, default)
    result: List[str] = []
    seen: Set[str] = set()
    for item in raw.replace("\n", ",").split(","):
        symbol = clean_symbol(item)
        if symbol and symbol not in seen:
            seen.add(symbol)
            result.append(symbol)
    return tuple(result)


def load_config() -> Config:
    alpaca_paper = env_bool("ALPACA_PAPER", True)
    default_base_url = "https://paper-api.alpaca.markets" if alpaca_paper else "https://api.alpaca.markets"
    targets = env_symbols("INVEST_TARGET_SYMBOLS", "SPY,BND")
    if len(targets) < 2:
        raise RuntimeError("INVEST_TARGET_SYMBOLS must include at least two symbols")

    ignored_default = ",".join(targets)
    ignored = set(env_symbols("IGNORED_PROFIT_SYMBOLS", ignored_default))

    cfg = Config(
        alpaca_api_key=env_required("ALPACA_API_KEY"),
        alpaca_secret_key=env_required("ALPACA_SECRET_KEY"),
        alpaca_paper=alpaca_paper,
        trading_base_url=os.getenv("ALPACA_TRADING_BASE_URL", default_base_url).strip() or default_base_url,
        data_base_url=os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets").strip() or "https://data.alpaca.markets",
        alpaca_data_feed=os.getenv("ALPACA_DATA_FEED", "iex").strip().lower(),
        google_sheet_id=env_required("GOOGLE_SHEET_ID"),
        google_service_account_json=env_required("GOOGLE_SERVICE_ACCOUNT_JSON"),
        state_tab=os.getenv("PROFIT_STATE_TAB", "ProfitReinvestState").strip() or "ProfitReinvestState",
        lots_tab=os.getenv("PROFIT_LOTS_TAB", "ProfitReinvestLots").strip() or "ProfitReinvestLots",
        activity_tab=os.getenv("PROFIT_ACTIVITY_TAB", "ProfitReinvestActivity").strip() or "ProfitReinvestActivity",
        orders_tab=os.getenv("PROFIT_ORDERS_TAB", "ProfitReinvestOrders").strip() or "ProfitReinvestOrders",
        invest_target_symbols=targets,
        ignored_profit_symbols=ignored,
        min_child_notional=env_decimal("MIN_CHILD_NOTIONAL", "1.00"),
        dry_run=env_bool("DRY_RUN", True),
        rsi_enabled=env_bool("RSI_ENABLED", True),
        rsi_period=max(2, env_int("RSI_PERIOD", 14)),
        rsi_threshold=env_decimal("RSI_THRESHOLD", "30"),
        rsi_bars_limit=max(20, env_int("RSI_BARS_LIMIT", 100)),
        rsi_timeframe=os.getenv("RSI_TIMEFRAME", "1Day").strip() or "1Day",
        rsi_adjustment=os.getenv("RSI_ADJUSTMENT", "split").strip().lower() or "split",
        rsi_lookback_days=max(30, env_int("RSI_LOOKBACK_DAYS", 365)),
        health_gate_enabled=env_bool("REINVEST_HEALTH_GATE_ENABLED", True),
        health_max_position_count=max(1, env_int("REINVEST_MAX_POSITION_COUNT", 60)),
        health_yellow_exposure_pct=env_decimal("REINVEST_YELLOW_EXPOSURE_PCT", "0.75"),
        health_red_exposure_pct=env_decimal("REINVEST_RED_EXPOSURE_PCT", "0.85"),
        health_yellow_min_cash_pct=env_decimal("REINVEST_YELLOW_MIN_CASH_PCT", "0.10"),
        health_red_min_cash_pct=env_decimal("REINVEST_RED_MIN_CASH_PCT", "0.05"),
        health_yellow_drawdown_pct=env_decimal("REINVEST_YELLOW_DRAWDOWN_PCT", "0.05"),
        health_red_drawdown_pct=env_decimal("REINVEST_RED_DRAWDOWN_PCT", "0.10"),
        health_yellow_red_position_pct=env_decimal("REINVEST_YELLOW_RED_POSITION_PCT", "0.55"),
        health_red_red_position_pct=env_decimal("REINVEST_RED_RED_POSITION_PCT", "0.70"),
        health_require_positive_total_realized=env_bool(
            "REINVEST_REQUIRE_POSITIVE_TOTAL_REALIZED", True
        ),
        health_yellow_investment_multiplier=env_decimal("REINVEST_YELLOW_MULTIPLIER", "0.25"),
        health_equity_high_watermark=env_decimal("REINVEST_EQUITY_HIGH_WATERMARK", "0"),
        seed_positions_on_first_run=env_bool("SEED_POSITIONS_ON_FIRST_RUN", True),
        page_size=max(1, min(100, env_int("ACTIVITY_PAGE_SIZE", 100))),
        max_pages_per_cycle=max(1, env_int("MAX_ACTIVITY_PAGES_PER_CYCLE", 10)),
        request_timeout_seconds=env_float("REQUEST_TIMEOUT_SECONDS", 10.0),
        request_retries=max(1, env_int("REQUEST_RETRIES", 3)),
        request_sleep_seconds=max(0.0, env_float("REQUEST_SLEEP_SECONDS", 0.25)),
        rate_limit_sleep_seconds=max(0.0, env_float("RATE_LIMIT_SLEEP_SECONDS", 10.0)),
        error_body_max_chars=max(0, env_int("ERROR_BODY_MAX_CHARS", 800)),
        poll_seconds=max(1, env_int("POLL_SECONDS", 60)),
        error_backoff_seconds=max(1, env_int("ERROR_BACKOFF_SECONDS", 300)),
        bot_auto_start=env_bool("BOT_AUTO_START", True),
        bot_run_token=os.getenv("BOT_RUN_TOKEN", "").strip(),
    )

    if cfg.min_child_notional <= 0:
        raise RuntimeError("MIN_CHILD_NOTIONAL must be greater than 0")
    if cfg.rsi_threshold <= 0 or cfg.rsi_threshold >= 100:
        raise RuntimeError("RSI_THRESHOLD must be greater than 0 and less than 100")
    if cfg.rsi_bars_limit <= cfg.rsi_period:
        raise RuntimeError("RSI_BARS_LIMIT must be greater than RSI_PERIOD")
    if cfg.rsi_lookback_days <= 0:
        raise RuntimeError("RSI_LOOKBACK_DAYS must be greater than 0")
    if cfg.health_yellow_exposure_pct <= 0 or cfg.health_red_exposure_pct <= 0:
        raise RuntimeError("Reinvestment exposure thresholds must be greater than 0")
    if cfg.health_red_exposure_pct < cfg.health_yellow_exposure_pct:
        raise RuntimeError("REINVEST_RED_EXPOSURE_PCT must be >= REINVEST_YELLOW_EXPOSURE_PCT")
    if not (Decimal("0") <= cfg.health_red_min_cash_pct <= cfg.health_yellow_min_cash_pct <= Decimal("1")):
        raise RuntimeError(
            "Cash thresholds must satisfy 0 <= RED_MIN_CASH <= YELLOW_MIN_CASH <= 1"
        )
    if not (Decimal("0") <= cfg.health_yellow_drawdown_pct <= cfg.health_red_drawdown_pct <= Decimal("1")):
        raise RuntimeError(
            "Drawdown thresholds must satisfy 0 <= YELLOW_DRAWDOWN <= RED_DRAWDOWN <= 1"
        )
    if not (
        Decimal("0")
        <= cfg.health_yellow_red_position_pct
        <= cfg.health_red_red_position_pct
        <= Decimal("1")
    ):
        raise RuntimeError(
            "Red-position thresholds must satisfy 0 <= YELLOW_RED_POSITION <= RED_RED_POSITION <= 1"
        )
    if not (Decimal("0") < cfg.health_yellow_investment_multiplier <= Decimal("1")):
        raise RuntimeError("REINVEST_YELLOW_MULTIPLIER must be > 0 and <= 1")
    if cfg.health_equity_high_watermark < 0:
        raise RuntimeError("REINVEST_EQUITY_HIGH_WATERMARK cannot be negative")
    return cfg


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clean_symbol(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[^A-Z0-9./-]", "", str(value).strip().upper())


def to_decimal(value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]:
    if value is None:
        return default
    try:
        text = str(value).strip().replace(",", "")
        if not text:
            return default
        result = Decimal(text)
        return result if result.is_finite() else default
    except (InvalidOperation, ValueError):
        return default


def decimal_text(value: Decimal, places: str = "0.0001") -> str:
    return str(value.quantize(Decimal(places), rounding=ROUND_DOWN))


def cents_down(value: Decimal) -> Decimal:
    if value <= 0:
        return Decimal("0.00")
    return value.quantize(Decimal("0.01"), rounding=ROUND_DOWN)


def safe_ratio(numerator: Decimal, denominator: Decimal) -> Decimal:
    if denominator <= 0:
        return Decimal("0")
    return numerator / denominator


def google_service_info(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    try:
        value = json.loads(raw)
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, dict):
            raise ValueError("service account JSON is not an object")
        return value
    except Exception as exc:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON") from exc


def gspread_client(cfg: Config) -> gspread.Client:
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(google_service_info(cfg.google_service_account_json), scopes=scopes)
    return gspread.authorize(creds)


def ensure_worksheet(spreadsheet: Any, title: str, headers: Sequence[str]) -> gspread.Worksheet:
    try:
        worksheet = spreadsheet.worksheet(title)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=title, rows=100, cols=max(1, len(headers)))

    values = worksheet.get("1:1") or []
    current_headers = values[0] if values else []
    if current_headers[: len(headers)] != list(headers):
        worksheet.update(values=[list(headers)], range_name="A1", value_input_option="RAW")
    return worksheet


def open_store(cfg: Config) -> Dict[str, gspread.Worksheet]:
    spreadsheet = gspread_client(cfg).open_by_key(cfg.google_sheet_id)
    return {
        "state": ensure_worksheet(spreadsheet, cfg.state_tab, STATE_HEADERS),
        "lots": ensure_worksheet(spreadsheet, cfg.lots_tab, LOTS_HEADERS),
        "activity": ensure_worksheet(spreadsheet, cfg.activity_tab, ACTIVITY_HEADERS),
        "orders": ensure_worksheet(spreadsheet, cfg.orders_tab, ORDER_HEADERS),
    }


def rows_from_sheet(worksheet: gspread.Worksheet, headers: Sequence[str]) -> List[Dict[str, str]]:
    values = worksheet.get_all_values()
    if len(values) < 2:
        return []
    result: List[Dict[str, str]] = []
    for raw in values[1:]:
        if not any(str(cell).strip() for cell in raw):
            continue
        item = {headers[i]: raw[i] if i < len(raw) else "" for i in range(len(headers))}
        result.append(item)
    return result


def replace_sheet_rows(worksheet: gspread.Worksheet, headers: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    values = [list(headers)]
    for row in rows:
        values.append([str(row.get(header, "")) for header in headers])
    worksheet.clear()
    worksheet.update(values=values, range_name="A1", value_input_option="RAW")


def state_from_rows(rows: Sequence[Dict[str, str]]) -> Dict[str, str]:
    return {row.get("key", ""): row.get("value", "") for row in rows if row.get("key", "")}


def state_rows(state: Dict[str, Any]) -> List[Dict[str, str]]:
    now = utc_now_iso()
    rows = []
    for key in sorted(state):
        rows.append({"key": key, "value": str(state[key]), "updated_at": now})
    return rows


def pending_key(symbol: str) -> str:
    return f"pending_{clean_symbol(symbol)}"


def state_symbol_key(prefix: str, symbol: str) -> str:
    return f"{prefix}_{clean_symbol(symbol)}"


def get_pending(state: Dict[str, str], symbol: str) -> Decimal:
    return to_decimal(state.get(pending_key(symbol)), Decimal("0")) or Decimal("0")


def set_pending(state: Dict[str, Any], symbol: str, value: Decimal) -> None:
    state[pending_key(symbol)] = decimal_text(max(value, Decimal("0")))


def total_pending(state: Dict[str, str], symbols: Sequence[str]) -> Decimal:
    return sum((get_pending(state, symbol) for symbol in symbols), Decimal("0"))


def alpaca_headers(cfg: Config) -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": cfg.alpaca_api_key,
        "APCA-API-SECRET-KEY": cfg.alpaca_secret_key,
        "Content-Type": "application/json",
    }


def response_body_for_log(resp: requests.Response, cfg: Config) -> str:
    text = (resp.text or "").strip().replace("\n", " ")
    if cfg.error_body_max_chars <= 0:
        return ""
    if len(text) > cfg.error_body_max_chars:
        return text[: cfg.error_body_max_chars] + "...[truncated]"
    return text


def raise_for_status_with_body(resp: requests.Response, cfg: Config, method: str, url: str) -> None:
    if resp.status_code >= 400:
        raise HttpStatusError(method, url, resp.status_code, response_body_for_log(resp, cfg))


def http_get(session: requests.Session, cfg: Config, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = f"{cfg.trading_base_url}{path}"
    return http_get_url(session, cfg, url, params)


def http_get_url(session: requests.Session, cfg: Config, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
    last_exc: Optional[BaseException] = None
    for attempt in range(1, cfg.request_retries + 1):
        try:
            resp = session.get(url, headers=alpaca_headers(cfg), params=params, timeout=cfg.request_timeout_seconds)
            if resp.status_code == 429 and attempt < cfg.request_retries:
                time.sleep(cfg.rate_limit_sleep_seconds)
                continue
            if 500 <= resp.status_code < 600 and attempt < cfg.request_retries:
                time.sleep(min(2**attempt, 30))
                continue
            raise_for_status_with_body(resp, cfg, "GET", url)
            if cfg.request_sleep_seconds:
                time.sleep(cfg.request_sleep_seconds)
            return resp.json() if resp.text else None
        except Exception as exc:
            last_exc = exc
            if attempt < cfg.request_retries:
                time.sleep(min(2**attempt, 30))
                continue
    raise RuntimeError(f"GET failed after {cfg.request_retries} attempts: {url}: {last_exc}")


def http_post_once(session: requests.Session, cfg: Config, path: str, payload: Dict[str, Any]) -> Any:
    url = f"{cfg.trading_base_url}{path}"
    resp = session.post(
        url,
        headers=alpaca_headers(cfg),
        data=json.dumps(payload),
        timeout=cfg.request_timeout_seconds,
    )
    raise_for_status_with_body(resp, cfg, "POST", url)
    if cfg.request_sleep_seconds:
        time.sleep(cfg.request_sleep_seconds)
    return resp.json() if resp.text else None


def list_positions(session: requests.Session, cfg: Config) -> List[Dict[str, Any]]:
    positions = http_get(session, cfg, "/v2/positions")
    return positions if isinstance(positions, list) else []


def get_account(session: requests.Session, cfg: Config) -> Dict[str, Any]:
    account = http_get(session, cfg, "/v2/account")
    return account if isinstance(account, dict) else {}


def account_buying_power(session: requests.Session, cfg: Config) -> Decimal:
    account = get_account(session, cfg)
    return to_decimal(account.get("buying_power"), Decimal("0")) or Decimal("0")


def get_order_by_client_order_id(session: requests.Session, cfg: Config, client_order_id: str) -> Optional[Dict[str, Any]]:
    try:
        order = http_get(session, cfg, "/v2/orders:by_client_order_id", {"client_order_id": client_order_id})
    except HttpStatusError as exc:
        if exc.status_code == 404:
            return None
        raise
    return order if isinstance(order, dict) else None


def extract_bar_close(bar: Dict[str, Any]) -> Optional[Decimal]:
    value = bar.get("c", bar.get("close"))
    close = to_decimal(value)
    if close is None or close <= 0:
        return None
    return close


def fetch_daily_closes(
    session: requests.Session,
    cfg: Config,
    symbol: str,
) -> Tuple[List[Decimal], Dict[str, Any]]:
    """Fetch the newest daily closes needed for RSI.

    Alpaca defaults the historical-bars start time to the beginning of the
    current day when `start` is omitted. With a 1Day timeframe that commonly
    returns zero or one bar, which caused the old `not_enough_rsi_bars` result.
    We now provide an explicit lookback window and request descending order so
    the response contains the newest bars rather than the oldest bars in that
    window. The closes are sorted back into chronological order for RSI.
    """
    end_dt = utc_now()
    start_dt = end_dt - timedelta(days=cfg.rsi_lookback_days)
    start_text = start_dt.date().isoformat()
    end_text = end_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    params: Dict[str, Any] = {
        "timeframe": cfg.rsi_timeframe,
        "start": start_text,
        "end": end_text,
        "limit": cfg.rsi_bars_limit,
        "adjustment": cfg.rsi_adjustment,
        "sort": "desc",
    }
    if cfg.alpaca_data_feed:
        params["feed"] = cfg.alpaca_data_feed

    url = f"{cfg.data_base_url}/v2/stocks/{symbol}/bars"
    payload = http_get_url(session, cfg, url, params)
    bars = payload.get("bars") if isinstance(payload, dict) else None
    if not isinstance(bars, list):
        bars = []

    timestamped_closes: List[Tuple[str, Decimal]] = []
    for bar in bars:
        if not isinstance(bar, dict):
            continue
        close = extract_bar_close(bar)
        if close is None:
            continue
        timestamp = str(bar.get("t") or bar.get("timestamp") or "")
        timestamped_closes.append((timestamp, close))

    # RSI must be calculated oldest -> newest even though the API request is
    # descending so the limit selects the latest observations.
    timestamped_closes.sort(key=lambda item: item[0])
    closes = [close for _, close in timestamped_closes]
    diagnostics = {
        "start": start_text,
        "end": end_text,
        "timeframe": cfg.rsi_timeframe,
        "feed": cfg.alpaca_data_feed,
        "adjustment": cfg.rsi_adjustment,
        "bars_requested": cfg.rsi_bars_limit,
        "bars_returned": len(closes),
        "next_page_token_present": bool(payload.get("next_page_token")) if isinstance(payload, dict) else False,
    }
    return closes, diagnostics


def calculate_rsi(closes: Sequence[Decimal], period: int) -> Optional[Decimal]:
    if len(closes) <= period:
        return None

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    seed = deltas[:period]
    avg_gain = sum((max(delta, Decimal("0")) for delta in seed), Decimal("0")) / Decimal(period)
    avg_loss = sum((abs(min(delta, Decimal("0"))) for delta in seed), Decimal("0")) / Decimal(period)

    for delta in deltas[period:]:
        gain = max(delta, Decimal("0"))
        loss = abs(min(delta, Decimal("0")))
        avg_gain = ((avg_gain * Decimal(period - 1)) + gain) / Decimal(period)
        avg_loss = ((avg_loss * Decimal(period - 1)) + loss) / Decimal(period)

    if avg_loss == 0:
        return Decimal("100")
    relative_strength = avg_gain / avg_loss
    return Decimal("100") - (Decimal("100") / (Decimal("1") + relative_strength))


def rsi_signal_for_symbol(session: requests.Session, cfg: Config, symbol: str) -> Dict[str, Any]:
    if not cfg.rsi_enabled:
        return {"symbol": symbol, "enabled": False, "ready": True, "rsi": None, "reason": "rsi_disabled"}

    closes, diagnostics = fetch_daily_closes(session, cfg, symbol)
    rsi = calculate_rsi(closes, cfg.rsi_period)
    if rsi is None:
        return {
            "symbol": symbol,
            "enabled": True,
            "ready": False,
            "rsi": None,
            "bars": len(closes),
            "required_bars": cfg.rsi_period + 1,
            "reason": "not_enough_rsi_bars",
            "data": diagnostics,
        }

    ready = rsi <= cfg.rsi_threshold
    return {
        "symbol": symbol,
        "enabled": True,
        "ready": ready,
        "rsi": decimal_text(rsi, "0.01"),
        "threshold": decimal_text(cfg.rsi_threshold, "0.01"),
        "bars": len(closes),
        "required_bars": cfg.rsi_period + 1,
        "reason": "rsi_at_or_below_threshold" if ready else "rsi_above_threshold",
        "data": diagnostics,
    }


def seed_lots_from_positions(
    session: requests.Session,
    cfg: Config,
    lots: List[Dict[str, str]],
) -> Tuple[int, List[str]]:
    seeded = 0
    skipped: List[str] = []
    for pos in list_positions(session, cfg):
        symbol = clean_symbol(pos.get("symbol"))
        side = str(pos.get("side", "")).strip().lower()
        if not symbol or symbol in cfg.ignored_profit_symbols:
            continue
        if side and side != "long":
            skipped.append(symbol)
            continue
        qty = to_decimal(pos.get("qty"), Decimal("0")) or Decimal("0")
        avg_entry = to_decimal(pos.get("avg_entry_price"), Decimal("0")) or Decimal("0")
        if qty <= 0 or avg_entry <= 0:
            continue
        lots.append(
            {
                "lot_id": f"seed-{symbol}-{utc_now().strftime('%Y%m%d%H%M%S')}",
                "symbol": symbol,
                "remaining_qty": decimal_text(qty, "0.000000001"),
                "cost_per_share": decimal_text(avg_entry, "0.000001"),
                "acquired_at": utc_now_iso(),
                "source_activity_id": "",
                "source": "current_position_seed",
            }
        )
        seeded += 1
    return seeded, skipped


def ensure_initialized(
    session: requests.Session,
    cfg: Config,
    state: Dict[str, Any],
    lots: List[Dict[str, str]],
) -> Dict[str, Any]:
    if str(state.get("initialized", "")).lower() == "true":
        return {"seeded_lots": 0, "skipped_seed_symbols": []}

    seeded = 0
    skipped: List[str] = []
    if cfg.seed_positions_on_first_run:
        seeded, skipped = seed_lots_from_positions(session, cfg, lots)

    state["initialized"] = "true"
    state["initialized_at"] = utc_now_iso()
    state["last_activity_time"] = utc_now_iso()
    state.setdefault("total_realized_profit", "0.0000")
    state.setdefault("total_positive_profit", "0.0000")
    state.setdefault("total_reinvested", "0.0000")
    state.setdefault("available_to_invest", "0.0000")
    state.setdefault("next_order_seq", "0")
    for symbol in cfg.invest_target_symbols:
        state.setdefault(pending_key(symbol), "0.0000")

    return {"seeded_lots": seeded, "skipped_seed_symbols": skipped}


def fetch_fill_activities(session: requests.Session, cfg: Config, after: str) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"after": after, "direction": "asc", "page_size": cfg.page_size}
    activities: List[Dict[str, Any]] = []
    for page in range(cfg.max_pages_per_cycle):
        page_items = http_get(session, cfg, "/v2/account/activities/FILL", params)
        if not isinstance(page_items, list) or not page_items:
            break
        activities.extend(item for item in page_items if isinstance(item, dict))
        if len(page_items) < cfg.page_size:
            break
        last_id = str(page_items[-1].get("id", "")).strip()
        if not last_id:
            break
        params["page_token"] = last_id
        logger.info("Fetched full activity page %s; continuing with page_token", page + 1)
    activities.sort(key=lambda x: (str(x.get("transaction_time", "")), str(x.get("id", ""))))
    return activities


def lot_sort_key(row: Dict[str, str]) -> Tuple[str, str]:
    return (row.get("acquired_at", ""), row.get("lot_id", ""))


def add_buy_lot(activity: Dict[str, Any], lots: List[Dict[str, str]], symbol: str, qty: Decimal, price: Decimal) -> None:
    activity_id = str(activity.get("id", "")).strip()
    acquired_at = str(activity.get("transaction_time", "")).strip() or utc_now_iso()
    lots.append(
        {
            "lot_id": activity_id or f"buy-{symbol}-{acquired_at}",
            "symbol": symbol,
            "remaining_qty": decimal_text(qty, "0.000000001"),
            "cost_per_share": decimal_text(price, "0.000001"),
            "acquired_at": acquired_at,
            "source_activity_id": activity_id,
            "source": "fill",
        }
    )


def realize_sell_fifo(
    lots: List[Dict[str, str]],
    symbol: str,
    qty: Decimal,
    sell_price: Decimal,
) -> Tuple[Decimal, Decimal]:
    remaining = qty
    matched = Decimal("0")
    realized = Decimal("0")
    symbol_lots = sorted((lot for lot in lots if lot.get("symbol") == symbol), key=lot_sort_key)

    for lot in symbol_lots:
        if remaining <= 0:
            break
        lot_qty = to_decimal(lot.get("remaining_qty"), Decimal("0")) or Decimal("0")
        lot_cost = to_decimal(lot.get("cost_per_share"), Decimal("0")) or Decimal("0")
        if lot_qty <= 0 or lot_cost <= 0:
            continue
        take = min(remaining, lot_qty)
        realized += take * (sell_price - lot_cost)
        matched += take
        remaining -= take
        lot["remaining_qty"] = decimal_text(lot_qty - take, "0.000000001")

    lots[:] = [
        lot
        for lot in lots
        if (to_decimal(lot.get("remaining_qty"), Decimal("0")) or Decimal("0")) > Decimal("0.000000000")
    ]
    return realized, qty - matched


def allocate_profit_to_targets(state: Dict[str, Any], targets: Sequence[str], profit: Decimal) -> Decimal:
    if profit <= 0:
        return Decimal("0")
    share = profit / Decimal(len(targets))
    for symbol in targets:
        current = to_decimal(state.get(pending_key(symbol)), Decimal("0")) or Decimal("0")
        set_pending(state, symbol, current + share)
    return profit


def process_activities(
    cfg: Config,
    state: Dict[str, Any],
    lots: List[Dict[str, str]],
    activity_rows: List[Dict[str, str]],
    activities: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    processed_ids = {row.get("activity_id", "") for row in activity_rows if row.get("activity_id", "")}
    new_rows: List[Dict[str, str]] = []
    max_time = str(state.get("last_activity_time", ""))
    summary = {
        "fills_seen": 0,
        "fills_processed": 0,
        "buy_lots_added": 0,
        "sell_fills": 0,
        "profitable_sells": 0,
        "positive_profit_added": "0.0000",
        "ignored_symbol_fills": 0,
        "unmatched_sell_fills": 0,
    }

    total_realized = to_decimal(state.get("total_realized_profit"), Decimal("0")) or Decimal("0")
    total_positive = to_decimal(state.get("total_positive_profit"), Decimal("0")) or Decimal("0")
    positive_added = Decimal("0")

    for activity in activities:
        summary["fills_seen"] += 1
        activity_id = str(activity.get("id", "")).strip()
        if not activity_id or activity_id in processed_ids:
            continue

        symbol = clean_symbol(activity.get("symbol"))
        side = str(activity.get("side", "")).strip().lower()
        qty = to_decimal(activity.get("qty"), Decimal("0")) or Decimal("0")
        price = to_decimal(activity.get("price"), Decimal("0")) or Decimal("0")
        activity_time = str(activity.get("transaction_time", "")).strip() or utc_now_iso()
        note = ""
        realized = Decimal("0")
        pending_added = Decimal("0")

        if activity_time > max_time:
            max_time = activity_time

        if not symbol or qty <= 0 or price <= 0 or side not in {"buy", "sell"}:
            note = "invalid_or_unsupported_fill"
        elif symbol in cfg.ignored_profit_symbols:
            summary["ignored_symbol_fills"] += 1
            note = "ignored_profit_symbol"
        elif side == "buy":
            add_buy_lot(activity, lots, symbol, qty, price)
            summary["buy_lots_added"] += 1
            note = "buy_lot_added"
        elif side == "sell":
            summary["sell_fills"] += 1
            realized, unmatched_qty = realize_sell_fifo(lots, symbol, qty, price)
            total_realized += realized
            if unmatched_qty > 0:
                summary["unmatched_sell_fills"] += 1
                note = f"unmatched_sell_qty={decimal_text(unmatched_qty, '0.000000001')}"
            else:
                note = "sell_matched_fifo"
            if realized > 0:
                summary["profitable_sells"] += 1
                pending_added = allocate_profit_to_targets(state, cfg.invest_target_symbols, realized)
                total_positive += realized
                positive_added += realized

        new_rows.append(
            {
                "activity_id": activity_id,
                "activity_time": activity_time,
                "symbol": symbol,
                "side": side,
                "qty": decimal_text(qty, "0.000000001") if qty else "",
                "price": decimal_text(price, "0.000001") if price else "",
                "realized_pl": decimal_text(realized),
                "profitable": "true" if realized > 0 else "false",
                "pending_added": decimal_text(pending_added),
                "note": note,
                "processed_at": utc_now_iso(),
            }
        )
        processed_ids.add(activity_id)
        summary["fills_processed"] += 1

    state["last_activity_time"] = max_time or utc_now_iso()
    state["total_realized_profit"] = decimal_text(total_realized)
    state["total_positive_profit"] = decimal_text(total_positive)
    summary["positive_profit_added"] = decimal_text(positive_added)
    activity_rows.extend(new_rows)
    return summary


def activity_profit_factor(activity_rows: Sequence[Dict[str, str]]) -> Optional[Decimal]:
    gross_profit = Decimal("0")
    gross_loss = Decimal("0")
    for row in activity_rows:
        realized = to_decimal(row.get("realized_pl"), Decimal("0")) or Decimal("0")
        if realized > 0:
            gross_profit += realized
        elif realized < 0:
            gross_loss += abs(realized)
    if gross_loss == 0:
        return None if gross_profit == 0 else Decimal("999999")
    return gross_profit / gross_loss


def evaluate_reinvestment_health(
    cfg: Config,
    state: Dict[str, Any],
    account: Dict[str, Any],
    positions: Sequence[Dict[str, Any]],
    activity_rows: Sequence[Dict[str, str]],
) -> Dict[str, Any]:
    equity = to_decimal(account.get("equity"), Decimal("0")) or Decimal("0")
    cash = to_decimal(account.get("cash"), Decimal("0")) or Decimal("0")
    long_market_value = to_decimal(account.get("long_market_value"), Decimal("0")) or Decimal("0")
    buying_power = to_decimal(account.get("buying_power"), Decimal("0")) or Decimal("0")

    valid_positions = [
        position
        for position in positions
        if clean_symbol(position.get("symbol"))
        and (to_decimal(position.get("qty"), Decimal("0")) or Decimal("0")) != 0
    ]
    position_count = len(valid_positions)
    red_position_count = sum(
        1
        for position in valid_positions
        if (to_decimal(position.get("unrealized_pl"), Decimal("0")) or Decimal("0")) < 0
    )

    exposure_pct = safe_ratio(max(long_market_value, Decimal("0")), equity)
    cash_pct = safe_ratio(cash, equity)
    red_position_pct = safe_ratio(Decimal(red_position_count), Decimal(position_count))

    state_hwm = to_decimal(state.get("equity_high_watermark"), Decimal("0")) or Decimal("0")
    high_watermark = max(cfg.health_equity_high_watermark, state_hwm, equity)
    drawdown_pct = Decimal("0")
    if high_watermark > 0 and equity > 0:
        drawdown_pct = max(Decimal("0"), Decimal("1") - (equity / high_watermark))

    total_realized = to_decimal(state.get("total_realized_profit"), Decimal("0")) or Decimal("0")
    profit_factor = activity_profit_factor(activity_rows)

    red_reasons: List[str] = []
    yellow_reasons: List[str] = []

    if equity <= 0:
        red_reasons.append("equity_not_positive")
    if position_count > cfg.health_max_position_count:
        red_reasons.append(
            f"position_count={position_count}>max={cfg.health_max_position_count}"
        )
    if exposure_pct >= cfg.health_red_exposure_pct:
        red_reasons.append(
            f"exposure={decimal_text(exposure_pct, '0.0001')}>=red={decimal_text(cfg.health_red_exposure_pct, '0.0001')}"
        )
    elif exposure_pct >= cfg.health_yellow_exposure_pct:
        yellow_reasons.append(
            f"exposure={decimal_text(exposure_pct, '0.0001')}>=yellow={decimal_text(cfg.health_yellow_exposure_pct, '0.0001')}"
        )

    if cash_pct <= cfg.health_red_min_cash_pct:
        red_reasons.append(
            f"cash_pct={decimal_text(cash_pct, '0.0001')}<=red_min={decimal_text(cfg.health_red_min_cash_pct, '0.0001')}"
        )
    elif cash_pct <= cfg.health_yellow_min_cash_pct:
        yellow_reasons.append(
            f"cash_pct={decimal_text(cash_pct, '0.0001')}<=yellow_min={decimal_text(cfg.health_yellow_min_cash_pct, '0.0001')}"
        )

    if drawdown_pct >= cfg.health_red_drawdown_pct:
        red_reasons.append(
            f"drawdown={decimal_text(drawdown_pct, '0.0001')}>=red={decimal_text(cfg.health_red_drawdown_pct, '0.0001')}"
        )
    elif drawdown_pct >= cfg.health_yellow_drawdown_pct:
        yellow_reasons.append(
            f"drawdown={decimal_text(drawdown_pct, '0.0001')}>=yellow={decimal_text(cfg.health_yellow_drawdown_pct, '0.0001')}"
        )

    if red_position_pct >= cfg.health_red_red_position_pct:
        red_reasons.append(
            f"red_position_pct={decimal_text(red_position_pct, '0.0001')}>=red={decimal_text(cfg.health_red_red_position_pct, '0.0001')}"
        )
    elif red_position_pct >= cfg.health_yellow_red_position_pct:
        yellow_reasons.append(
            f"red_position_pct={decimal_text(red_position_pct, '0.0001')}>=yellow={decimal_text(cfg.health_yellow_red_position_pct, '0.0001')}"
        )

    if cfg.health_require_positive_total_realized and total_realized <= 0:
        red_reasons.append(
            f"total_realized_profit={decimal_text(total_realized)}<=0"
        )

    if not cfg.health_gate_enabled:
        mode = "GREEN"
        multiplier = Decimal("1")
        reasons = ["health_gate_disabled"]
    elif red_reasons:
        mode = "RED"
        multiplier = Decimal("0")
        reasons = red_reasons + yellow_reasons
    elif yellow_reasons:
        mode = "YELLOW"
        multiplier = cfg.health_yellow_investment_multiplier
        reasons = yellow_reasons
    else:
        mode = "GREEN"
        multiplier = Decimal("1")
        reasons = ["all_health_checks_passed"]

    state["equity_high_watermark"] = decimal_text(high_watermark)
    state["last_health_checked_at"] = utc_now_iso()
    state["last_health_mode"] = mode
    state["last_health_reasons"] = json.dumps(reasons, separators=(",", ":"))
    state["last_health_multiplier"] = decimal_text(multiplier)
    state["last_health_equity"] = decimal_text(equity)
    state["last_health_cash"] = decimal_text(cash)
    state["last_health_cash_pct"] = decimal_text(cash_pct)
    state["last_health_exposure_pct"] = decimal_text(exposure_pct)
    state["last_health_drawdown_pct"] = decimal_text(drawdown_pct)
    state["last_health_position_count"] = str(position_count)
    state["last_health_red_position_count"] = str(red_position_count)
    state["last_health_red_position_pct"] = decimal_text(red_position_pct)
    state["last_health_total_realized_profit"] = decimal_text(total_realized)
    state["last_health_profit_factor"] = (
        decimal_text(profit_factor) if profit_factor is not None else ""
    )

    return {
        "enabled": cfg.health_gate_enabled,
        "mode": mode,
        "multiplier": decimal_text(multiplier),
        "reasons": reasons,
        "equity": decimal_text(equity),
        "cash": decimal_text(cash),
        "cash_pct": decimal_text(cash_pct),
        "long_market_value": decimal_text(long_market_value),
        "exposure_pct": decimal_text(exposure_pct),
        "buying_power": decimal_text(buying_power),
        "position_count": position_count,
        "max_position_count": cfg.health_max_position_count,
        "red_position_count": red_position_count,
        "red_position_pct": decimal_text(red_position_pct),
        "equity_high_watermark": decimal_text(high_watermark),
        "drawdown_pct": decimal_text(drawdown_pct),
        "total_realized_profit": decimal_text(total_realized),
        "profit_factor": decimal_text(profit_factor) if profit_factor is not None else None,
        "require_positive_total_realized": cfg.health_require_positive_total_realized,
    }


def safe_client_order_id(seq: int, symbol: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]", "", symbol.upper())[:12]
    return f"profit-reinvest-{seq}-{clean.lower()}"[:48]


def submit_notional_buy(
    session: requests.Session,
    cfg: Config,
    symbol: str,
    notional: Decimal,
    client_order_id: str,
) -> Dict[str, Any]:
    payload = {
        "symbol": symbol,
        "notional": f"{notional:.2f}",
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
        "client_order_id": client_order_id,
    }
    return http_post_once(session, cfg, "/v2/orders", payload)


def record_order_row(
    order_rows: List[Dict[str, str]],
    order: Dict[str, Any],
    symbol: str,
    notional: Decimal,
    client_order_id: str,
    dry_run: bool,
    note: str,
) -> None:
    order_rows.append(
        {
            "order_id": str(order.get("id", "")) if order else "",
            "client_order_id": client_order_id,
            "submitted_at": utc_now_iso(),
            "symbol": symbol,
            "notional": f"{notional:.2f}",
            "status": str(order.get("status", "dry_run" if dry_run else "")) if order else "dry_run",
            "dry_run": "true" if dry_run else "false",
            "pending_spent": f"{notional:.2f}" if not dry_run else "0.00",
            "note": note,
        }
    )


def try_submit_or_recover_existing(
    session: requests.Session,
    cfg: Config,
    symbol: str,
    notional: Decimal,
    client_order_id: str,
) -> Tuple[Dict[str, Any], str]:
    try:
        return submit_notional_buy(session, cfg, symbol, notional, client_order_id), "submitted"
    except HttpStatusError as exc:
        body = exc.body.lower()
        duplicateish = exc.status_code in {400, 422} and (
            "client_order_id" in body or "duplicate" in body or "already" in body
        )
        if not duplicateish:
            raise
        existing = get_order_by_client_order_id(session, cfg, client_order_id)
        if existing is None:
            raise
        return existing, "recovered_existing_client_order_id"


def invest_pending(
    session: requests.Session,
    cfg: Config,
    state: Dict[str, Any],
    order_rows: List[Dict[str, str]],
    worksheets: Dict[str, gspread.Worksheet],
    health: Dict[str, Any],
) -> Dict[str, Any]:
    summary = {
        "orders_submitted": 0,
        "dry_run_orders": 0,
        "skipped_below_min": [],
        "skipped_rsi": [],
        "skipped_health": [],
        "skipped_buying_power": [],
        "errors": [],
        "rsi": {},
        "health": health,
    }
    buying_power: Optional[Decimal] = None
    if not cfg.dry_run:
        buying_power = to_decimal(health.get("buying_power"), Decimal("0")) or Decimal("0")
    total_reinvested = to_decimal(state.get("total_reinvested"), Decimal("0")) or Decimal("0")
    seq = int(to_decimal(state.get("next_order_seq"), Decimal("0")) or Decimal("0"))
    rsi_cache: Dict[str, Dict[str, Any]] = {}
    state_changed = False
    health_mode = str(health.get("mode", "RED")).upper()
    health_multiplier = to_decimal(health.get("multiplier"), Decimal("0")) or Decimal("0")

    for symbol in cfg.invest_target_symbols:
        pending = get_pending(state, symbol)
        full_notional = cents_down(pending)
        if full_notional < cfg.min_child_notional:
            summary["skipped_below_min"].append({"symbol": symbol, "pending": decimal_text(pending)})
            continue

        # Continue evaluating/logging RSI even while the health gate is RED so
        # a portfolio-health pause cannot hide a market-data problem.
        if symbol not in rsi_cache:
            try:
                rsi_cache[symbol] = rsi_signal_for_symbol(session, cfg, symbol)
            except Exception as exc:
                logger.exception("Could not evaluate RSI for %s", symbol)
                rsi_cache[symbol] = {
                    "symbol": symbol,
                    "enabled": cfg.rsi_enabled,
                    "ready": False,
                    "rsi": None,
                    "reason": f"rsi_error: {exc}",
                }
        signal = rsi_cache[symbol]
        summary["rsi"][symbol] = signal
        state[state_symbol_key("last_rsi", symbol)] = signal.get("rsi") or ""
        state[state_symbol_key("last_rsi_reason", symbol)] = signal.get("reason") or ""
        state[state_symbol_key("last_rsi_bars", symbol)] = str(signal.get("bars", ""))
        state[state_symbol_key("last_rsi_checked_at", symbol)] = utc_now_iso()
        state_changed = True
        if not signal.get("ready"):
            summary["skipped_rsi"].append(
                {
                    "symbol": symbol,
                    "pending": decimal_text(pending),
                    "notional": f"{full_notional:.2f}",
                    "rsi": signal.get("rsi"),
                    "bars": signal.get("bars"),
                    "reason": signal.get("reason"),
                }
            )
            continue

        if health_mode == "RED" or health_multiplier <= 0:
            summary["skipped_health"].append(
                {
                    "symbol": symbol,
                    "pending": decimal_text(pending),
                    "full_notional": f"{full_notional:.2f}",
                    "mode": health_mode,
                    "reasons": health.get("reasons", []),
                }
            )
            continue

        notional = cents_down(pending * health_multiplier)
        if notional < cfg.min_child_notional:
            summary["skipped_health"].append(
                {
                    "symbol": symbol,
                    "pending": decimal_text(pending),
                    "full_notional": f"{full_notional:.2f}",
                    "adjusted_notional": f"{notional:.2f}",
                    "mode": health_mode,
                    "multiplier": decimal_text(health_multiplier),
                    "reason": "health_adjusted_notional_below_min",
                }
            )
            continue

        if buying_power is not None and buying_power < notional:
            summary["skipped_buying_power"].append(
                {"symbol": symbol, "notional": f"{notional:.2f}", "buying_power": decimal_text(buying_power)}
            )
            continue

        seq += 1
        client_order_id = safe_client_order_id(seq, symbol)
        if cfg.dry_run:
            summary["dry_run_orders"] += 1
            continue

        try:
            order, note = try_submit_or_recover_existing(session, cfg, symbol, notional, client_order_id)
            note = f"{note};health_mode={health_mode};health_multiplier={decimal_text(health_multiplier)}"
            record_order_row(order_rows, order, symbol, notional, client_order_id, False, note)
            set_pending(state, symbol, pending - notional)
            state["next_order_seq"] = str(seq)
            total_reinvested += notional
            state["total_reinvested"] = decimal_text(total_reinvested)
            state_changed = True
            replace_sheet_rows(worksheets["state"], STATE_HEADERS, state_rows(state))
            replace_sheet_rows(worksheets["orders"], ORDER_HEADERS, order_rows)
            summary["orders_submitted"] += 1
            if buying_power is not None:
                buying_power -= notional
        except Exception as exc:
            logger.exception("Could not submit reinvestment order for %s", symbol)
            summary["errors"].append({"symbol": symbol, "error": str(exc)})

    state["available_to_invest"] = decimal_text(total_pending(state, cfg.invest_target_symbols))
    state_changed = True

    if state_changed:
        replace_sheet_rows(worksheets["state"], STATE_HEADERS, state_rows(state))
    return summary


def run_cycle(source: str = "manual") -> Dict[str, Any]:
    if not RUN_LOCK.acquire(blocking=False):
        logger.info("Profit reinvestor cycle skipped source=%s reason=already_running", source)
        return {"status": "busy", "source": source, "version": APP_VERSION}

    started = utc_now_iso()
    logger.info("Profit reinvestor cycle started source=%s", source)
    summary: Dict[str, Any] = {
        "status": "ok",
        "version": APP_VERSION,
        "source": source,
        "started_at": started,
        "finished_at": None,
    }

    try:
        cfg = load_config()
        with requests.Session() as session:
            worksheets = open_store(cfg)
            state = state_from_rows(rows_from_sheet(worksheets["state"], STATE_HEADERS))
            lots = rows_from_sheet(worksheets["lots"], LOTS_HEADERS)
            activity_rows = rows_from_sheet(worksheets["activity"], ACTIVITY_HEADERS)
            order_rows = rows_from_sheet(worksheets["orders"], ORDER_HEADERS)

            init_summary = ensure_initialized(session, cfg, state, lots)
            after = str(state.get("last_activity_time", "") or utc_now_iso())
            activities = fetch_fill_activities(session, cfg, after)
            activity_summary = process_activities(cfg, state, lots, activity_rows, activities)

            account = get_account(session, cfg)
            positions = list_positions(session, cfg)
            health_summary = evaluate_reinvestment_health(
                cfg,
                state,
                account,
                positions,
                activity_rows,
            )

            state["last_cycle_started_at"] = started
            state["last_cycle_finished_at"] = utc_now_iso()
            state["last_cycle_source"] = source
            for symbol in cfg.invest_target_symbols:
                state.setdefault(pending_key(symbol), "0.0000")
            state["available_to_invest"] = decimal_text(total_pending(state, cfg.invest_target_symbols))

            replace_sheet_rows(worksheets["lots"], LOTS_HEADERS, lots)
            replace_sheet_rows(worksheets["activity"], ACTIVITY_HEADERS, activity_rows)
            replace_sheet_rows(worksheets["state"], STATE_HEADERS, state_rows(state))

            invest_summary = invest_pending(session, cfg, state, order_rows, worksheets, health_summary)
            summary.update(
                {
                    "dry_run": cfg.dry_run,
                    "targets": list(cfg.invest_target_symbols),
                    "ignored_profit_symbols": sorted(cfg.ignored_profit_symbols),
                    "initialization": init_summary,
                    "activity": activity_summary,
                    "health": health_summary,
                    "investing": invest_summary,
                    "pending": {symbol: decimal_text(get_pending(state, symbol)) for symbol in cfg.invest_target_symbols},
                    "available_to_invest": decimal_text(total_pending(state, cfg.invest_target_symbols)),
                    "total_positive_profit": state.get("total_positive_profit", "0.0000"),
                    "total_reinvested": state.get("total_reinvested", "0.0000"),
                }
            )
    except Exception as exc:
        logger.exception("Profit reinvestor cycle failed")
        summary["status"] = "error"
        summary["error"] = str(exc)
    finally:
        summary["finished_at"] = utc_now_iso()
        LAST_STATUS.clear()
        LAST_STATUS.update(summary)
        RUN_LOCK.release()
        if summary.get("status") == "ok":
            activity = summary.get("activity", {})
            investing = summary.get("investing", {})
            logger.info(
                "Profit reinvestor cycle finished source=%s fills_processed=%s available_to_invest=%s "
                "health_mode=%s orders_submitted=%s dry_run_orders=%s skipped_below_min=%s "
                "skipped_rsi=%s skipped_health=%s errors=%s",
                source,
                activity.get("fills_processed"),
                summary.get("available_to_invest"),
                summary.get("health", {}).get("mode"),
                investing.get("orders_submitted"),
                investing.get("dry_run_orders"),
                len(investing.get("skipped_below_min", [])),
                len(investing.get("skipped_rsi", [])),
                len(investing.get("skipped_health", [])),
                len(investing.get("errors", [])),
            )
        else:
            logger.info(
                "Profit reinvestor cycle finished source=%s status=%s error=%s",
                source,
                summary.get("status"),
                summary.get("error"),
            )

    return summary


def worker_loop() -> None:
    while not STOP_EVENT.is_set():
        result = run_cycle("loop")
        sleep_seconds = load_sleep_seconds(result)
        logger.info("Profit reinvestor sleeping seconds=%s last_status=%s", sleep_seconds, result.get("status"))
        STOP_EVENT.wait(sleep_seconds)


def load_sleep_seconds(last_result: Dict[str, Any]) -> int:
    try:
        cfg = load_config()
        return cfg.error_backoff_seconds if last_result.get("status") == "error" else cfg.poll_seconds
    except Exception:
        return 300


def require_token(cfg: Config, token_header: Optional[str], authorization: Optional[str]) -> None:
    if not cfg.bot_run_token:
        return
    bearer = ""
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization[7:].strip()
    if token_header == cfg.bot_run_token or bearer == cfg.bot_run_token:
        return
    raise HTTPException(status_code=401, detail="Invalid or missing BOT_RUN_TOKEN")


@app.on_event("startup")
def startup_event() -> None:
    global WORKER_THREAD
    try:
        cfg = load_config()
    except Exception as exc:
        LAST_STATUS.update({"state": "config_error", "error": str(exc), "version": APP_VERSION})
        logger.error("Config error on startup: %s", exc)
        return
    if cfg.bot_auto_start:
        WORKER_THREAD = threading.Thread(target=worker_loop, name="profit-reinvestor-loop", daemon=True)
        WORKER_THREAD.start()
        LAST_STATUS.update({"state": "running", "version": APP_VERSION})
        logger.info("Profit reinvestor background loop started poll_seconds=%s", cfg.poll_seconds)
    else:
        LAST_STATUS.update({"state": "idle", "version": APP_VERSION})
        logger.info("Profit reinvestor background loop disabled BOT_AUTO_START=false")


@app.on_event("shutdown")
def shutdown_event() -> None:
    STOP_EVENT.set()
    if WORKER_THREAD and WORKER_THREAD.is_alive():
        WORKER_THREAD.join(timeout=10)


@app.get("/")
def root() -> Dict[str, Any]:
    return {"service": "profit-reinvestor", "version": APP_VERSION, "status": LAST_STATUS.get("status", LAST_STATUS.get("state"))}


@app.get("/status")
def status(
    x_bot_run_token: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    cfg = load_config()
    require_token(cfg, x_bot_run_token, authorization)
    return dict(LAST_STATUS)


@app.post("/run")
def run_now(
    x_bot_run_token: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    cfg = load_config()
    require_token(cfg, x_bot_run_token, authorization)
    result = run_cycle("manual")
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result)
    return result