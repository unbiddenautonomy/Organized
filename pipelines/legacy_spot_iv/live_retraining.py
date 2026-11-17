"""Non-blocking intraday retraining driven by matured UDP/LZO forecast outcomes."""
from __future__ import annotations

import json
import math
import os
import shutil
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, time
from pathlib import Path
from tempfile import mkdtemp

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


FEATURES = ("current", "base_prediction", "base_delta", "time_sin", "time_cos")


class LiveRetrainingManager:
    """Learns causal intraday residual corrections while base models stay immutable."""

    def __init__(self, config: dict, base_dir: Path):
        self.enabled = bool(config.get("intraday_retraining_enabled", False))
        self.interval = int(config.get("intraday_retraining_minutes", 120))
        self.minimum_rows = int(config.get("intraday_retraining_minimum_rows", 100))
        self.validation_fraction = float(config.get("intraday_retraining_validation_fraction", .2))
        self.minimum_improvement = float(config.get("intraday_retraining_minimum_improvement", .0))
        self.root = Path(config.get("intraday_artifact_dir", "artifacts/intraday_live"))
        self.eod_root = Path(config.get("eod_artifact_dir", "artifacts/eod_live"))
        self.eod_enabled = bool(config.get("eod_retraining_enabled", True))
        self.eod_after = time.fromisoformat(str(config.get("eod_retraining_after", "15:45")))
        self.eod_minimum_rows = int(config.get("eod_retraining_minimum_rows", 100))
        self.database = Path(config.get("intraday_training_database", "artifacts/intraday_live/training.db"))
        self.shadow_path = Path(config.get("intraday_shadow_prediction_file", "logs/intraday_shadow_predictions.jsonl"))
        self.live_output_path = Path(config.get("retraining_live_output_file", "logs/retraining_live_outputs.jsonl"))
        for name in ("root", "eod_root", "database", "shadow_path", "live_output_path"):
            value = getattr(self, name)
            if not value.is_absolute(): setattr(self, name, base_dir / value)
        self.root.mkdir(parents=True, exist_ok=True); self.eod_root.mkdir(parents=True, exist_ok=True); self.database.parent.mkdir(parents=True, exist_ok=True)
        self.shadow_path.parent.mkdir(parents=True, exist_ok=True)
        self.live_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.shadow = self.shadow_path.open("a", encoding="utf-8", buffering=1) if self.enabled else None
        self.live_output = self.live_output_path.open("a", encoding="utf-8", buffering=1) if self.enabled else None
        self.lock = threading.RLock(); self.worker: threading.Thread | None = None
        self.last_schedule_minute: int | None = None; self.latest_exchange_minute: int | None = None
        self.bundles: dict[tuple[str, int], dict] = {}
        self.eod_bundles: dict[tuple[str, int], dict] = {}
        self.latest_shadow: dict[str, dict] = {}
        self.latest_outputs: dict[str, dict] = {}
        self.last_eod_date: str | None = None
        with closing(sqlite3.connect(self.database)) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS forecasts(
                target TEXT NOT NULL, horizon INTEGER NOT NULL, prediction_time TEXT NOT NULL,
                source_minute INTEGER NOT NULL, due_minute INTEGER NOT NULL, current REAL NOT NULL,
                base_prediction REAL NOT NULL, actual REAL, settled_minute INTEGER,
                PRIMARY KEY(target,horizon,source_minute))""")
            db.execute("CREATE TABLE IF NOT EXISTS training_runs(cadence TEXT NOT NULL, session_date TEXT NOT NULL, status TEXT NOT NULL, PRIMARY KEY(cadence,session_date))")
            db.commit()
            row=db.execute("SELECT session_date FROM training_runs WHERE cadence='eod' AND status='published' ORDER BY session_date DESC LIMIT 1").fetchone()
            self.last_eod_date=row[0] if row else None
        self._load_latest(self.root, self.bundles); self._load_latest(self.eod_root, self.eod_bundles)

    @staticmethod
    def _minute(event: dict) -> int:
        return int(datetime.fromisoformat(event["exchange_minute"]).timestamp() // 60)

    @staticmethod
    def _vector(current: float, prediction: float, minute: int) -> list[float]:
        minute_of_day = minute % 1440; angle = 2 * math.pi * minute_of_day / 1440
        return [current, prediction, prediction-current, math.sin(angle), math.cos(angle)]

    def on_spot_prediction(self, event: dict) -> None:
        self._on_prediction("spot", "current_spot", "predicted_spot", event)

    def on_iv_prediction(self, event: dict) -> None:
        self._on_prediction("iv", "current_iv", "predicted_iv", event)

    def _on_prediction(self, target: str, current_field: str, prediction_field: str, event: dict) -> None:
        if not self.enabled: return
        minute = self._minute(event); current = float(event[current_field])
        with self.lock, closing(sqlite3.connect(self.database, timeout=10)) as db:
            db.execute("UPDATE forecasts SET actual=?,settled_minute=? WHERE target=? AND actual IS NULL AND due_minute<=?",
                       (current, minute, target, minute))
            for horizon_text, forecast in event.get("predictions", {}).items():
                prediction = forecast.get(prediction_field)
                if prediction is None or not math.isfinite(float(prediction)): continue
                horizon = int(horizon_text)
                db.execute("INSERT OR IGNORE INTO forecasts VALUES(?,?,?,?,?,?,?,NULL,NULL)",
                           (target, horizon, event["exchange_minute"], minute, minute+horizon, current, float(prediction)))
                self._write_shadow(target, horizon, event["exchange_minute"], minute, current, float(prediction))
                self._write_live_output(target, horizon, event["exchange_minute"], minute, current, float(prediction))
            db.commit()
            self.latest_exchange_minute = minute
            due = self.last_schedule_minute is None or minute-self.last_schedule_minute >= self.interval
            if due and (self.worker is None or not self.worker.is_alive()):
                self.last_schedule_minute = minute
                self.worker = threading.Thread(target=self._train_all, args=(minute,"intraday"), name="intraday-retrainer", daemon=True)
                self.worker.start()

    def tick(self, now: datetime | None = None) -> None:
        """Schedule EOD training from the main loop without blocking it."""
        if not self.enabled or not self.eod_enabled or self.latest_exchange_minute is None: return
        now=now or datetime.now(); session_date=now.date().isoformat()
        if now.time() < self.eod_after or self.last_eod_date == session_date: return
        with self.lock:
            if self.worker is not None and self.worker.is_alive(): return
            self.last_eod_date=session_date
            self.worker=threading.Thread(target=self._train_eod,args=(self.latest_exchange_minute,session_date),name="eod-retrainer",daemon=True)
            self.worker.start()

    def _train_eod(self, cutoff: int, session_date: str) -> None:
        published=self._train_all(cutoff,"eod")
        with closing(sqlite3.connect(self.database)) as db:
            db.execute("INSERT OR REPLACE INTO training_runs VALUES('eod',?,?)",(session_date,"published" if published else "skipped")); db.commit()
        if not published: self.last_eod_date=None

    def _write_shadow(self, target: str, horizon: int, stamp: str, minute: int, current: float, base: float) -> None:
        bundle = self.bundles.get((target, horizon))
        if bundle is None or self.shadow is None: return
        x = np.asarray([self._vector(current, base, minute)], dtype=float)
        correction = float(bundle["model"].predict(bundle["scaler"].transform(x))[0])
        event = {"event_type":"intraday_shadow_prediction","exchange_minute":stamp,"target":target,
                 "horizon":horizon,"base_prediction":base,"residual_correction":correction,
                 "intraday_prediction":base+correction,"model_id":bundle["model_id"],"trained_through":bundle["trained_through"]}
        self.latest_shadow[f"{target}:{horizon}"] = event
        self.shadow.write(json.dumps(event, separators=(",", ":")) + "\n")

    def _correct(self, bundle: dict | None, current: float, base: float, minute: int) -> tuple[float | None,float | None,str | None]:
        if bundle is None: return None,None,None
        x=np.asarray([self._vector(current,base,minute)],dtype=float)
        correction=float(bundle["model"].predict(bundle["scaler"].transform(x))[0])
        return base+correction,correction,bundle["model_id"]

    def _write_live_output(self,target:str,horizon:int,stamp:str,minute:int,current:float,base:float)->None:
        if self.live_output is None:return
        intra,intra_delta,intra_id=self._correct(self.bundles.get((target,horizon)),current,base,minute)
        eod,eod_delta,eod_id=self._correct(self.eod_bundles.get((target,horizon)),current,base,minute)
        event={"event_type":"retraining_live_output","exchange_minute":stamp,"target":target,"horizon":horizon,
               "current":current,"base_prediction":base,
               "intraday_prediction":intra,"intraday_correction":intra_delta,"intraday_model_id":intra_id,
               "eod_prediction":eod,"eod_correction":eod_delta,"eod_model_id":eod_id}
        self.latest_outputs[f"{target}:{horizon}"]=event
        self.live_output.write(json.dumps(event,separators=(",", ":"))+"\n")

    def _rows(self, target: str, horizon: int, cutoff: int) -> list[sqlite3.Row]:
        with closing(sqlite3.connect(self.database, timeout=10)) as db:
            db.row_factory = sqlite3.Row
            return db.execute("SELECT * FROM forecasts WHERE target=? AND horizon=? AND actual IS NOT NULL AND settled_minute<=? ORDER BY source_minute",
                              (target, horizon, cutoff)).fetchall()

    def _train_all(self, cutoff: int, cadence: str = "intraday") -> bool:
        published = []
        for target in ("spot", "iv"):
            for horizon in (5, 15):
                result = self._train_one(target, horizon, cutoff, cadence)
                if result: published.append(result)
        if published: self._publish(published, cutoff, cadence)
        return bool(published)

    def _train_one(self, target: str, horizon: int, cutoff: int, cadence: str) -> dict | None:
        rows = self._rows(target, horizon, cutoff)
        minimum=self.eod_minimum_rows if cadence=="eod" else self.minimum_rows
        if len(rows) < minimum: return None
        x = np.asarray([self._vector(r["current"], r["base_prediction"], r["source_minute"]) for r in rows], dtype=float)
        y = np.asarray([r["actual"]-r["base_prediction"] for r in rows], dtype=float)
        split = max(1, min(len(rows)-1, int(len(rows)*(1-self.validation_fraction))))
        scaler = StandardScaler().fit(x[:split]); model = Ridge(alpha=10.0).fit(scaler.transform(x[:split]), y[:split])
        corrected = np.asarray([r["base_prediction"] for r in rows[split:]]) + model.predict(scaler.transform(x[split:]))
        actual = np.asarray([r["actual"] for r in rows[split:]])
        base = np.asarray([r["base_prediction"] for r in rows[split:]])
        base_mae=float(np.mean(np.abs(base-actual))); corrected_mae=float(np.mean(np.abs(corrected-actual)))
        improvement=(base_mae-corrected_mae)/max(base_mae,1e-12)
        if improvement < self.minimum_improvement: return None
        trained_through=datetime.fromtimestamp(cutoff*60).isoformat(sep=" ")
        model_id=f"{cadence}-{target}-{horizon}m-{cutoff}"
        return {"target":target,"horizon":horizon,"model_id":model_id,"features":FEATURES,"scaler":scaler,"model":model,
                "cadence":cadence,"trained_through":trained_through,"training_rows":split,"validation_rows":len(rows)-split,
                "base_validation_mae":base_mae,"candidate_validation_mae":corrected_mae,"relative_improvement":improvement}

    def _publish(self, bundles: list[dict], cutoff: int, cadence: str) -> None:
        destination=self.eod_root if cadence=="eod" else self.root
        temporary=Path(mkdtemp(prefix="training-",dir=destination)); final=destination/f"version-{cutoff}"
        try:
            manifest=[]
            for bundle in bundles:
                joblib.dump(bundle,temporary/f"{bundle['target']}_{bundle['horizon']}m.joblib")
                manifest.append({k:v for k,v in bundle.items() if k not in {"model","scaler"}})
            (temporary/"manifest.json").write_text(json.dumps({"status":"validated","cutoff_minute":cutoff,"models":manifest},indent=2),encoding="utf-8")
            os.replace(temporary,final)
            (destination/"CURRENT").write_text(final.name,encoding="utf-8")
            with self.lock:
                target_bundles=self.eod_bundles if cadence=="eod" else self.bundles
                for bundle in bundles: target_bundles[(bundle["target"],bundle["horizon"])]=bundle
        finally:
            if temporary.exists(): shutil.rmtree(temporary,ignore_errors=True)

    @staticmethod
    def _load_latest(root: Path, target_bundles: dict) -> None:
        pointer=root/"CURRENT"
        if not pointer.exists(): return
        folder=root/pointer.read_text(encoding="utf-8").strip()
        if not folder.is_dir(): return
        for path in folder.glob("*_?m.joblib"):
            bundle=joblib.load(path); target_bundles[(bundle["target"],int(bundle["horizon"]))]=bundle
        for path in folder.glob("*_??m.joblib"):
            bundle=joblib.load(path); target_bundles[(bundle["target"],int(bundle["horizon"]))]=bundle

    def snapshot(self) -> dict:
        with closing(sqlite3.connect(self.database)) as db:
            settled=db.execute("SELECT COUNT(*) FROM forecasts WHERE actual IS NOT NULL").fetchone()[0]
            pending=db.execute("SELECT COUNT(*) FROM forecasts WHERE actual IS NULL").fetchone()[0]
            groups=db.execute("""SELECT target,horizon,
                SUM(CASE WHEN actual IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN actual IS NULL THEN 1 ELSE 0 END)
                FROM forecasts GROUP BY target,horizon ORDER BY target,horizon""").fetchall()
        candidates=[]
        for cadence,collection in (("intraday",self.bundles),("eod",self.eod_bundles)):
          for (target,horizon),bundle in sorted(collection.items()):
            candidates.append({key:bundle.get(key) for key in (
                "cadence","target","horizon","model_id","trained_through","training_rows","validation_rows",
                "base_validation_mae","candidate_validation_mae","relative_improvement")})
        remaining = None if self.latest_exchange_minute is None or self.last_schedule_minute is None else max(
            0, self.interval-(self.latest_exchange_minute-self.last_schedule_minute))
        return {"enabled":self.enabled,"interval_minutes":self.interval,"settled_rows":settled,"pending_rows":pending,
                "training_running":bool(self.worker and self.worker.is_alive()),
                "latest_exchange_minute":self.latest_exchange_minute,"minutes_to_next_retraining":remaining,
                "ledger":[{"target":row[0],"horizon":row[1],"settled":row[2],"pending":row[3]} for row in groups],
                "eod_enabled":self.eod_enabled,"eod_after":self.eod_after.isoformat(timespec="minutes"),"last_eod_date":self.last_eod_date,
                "candidates":candidates,"latest_shadow":self.latest_shadow,"latest_outputs":self.latest_outputs}

    def close(self) -> None:
        if self.worker and self.worker.is_alive(): self.worker.join(timeout=30)
        if self.shadow: self.shadow.close()
        if self.live_output: self.live_output.close()
