"""Statistical validation suite for the promoted 120-minute RV model."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from enhanced_rv_model import (
    ENHANCED_FEATURES, EPS, HAR_FEATURES, OUT, SEED, TRAIN_DAYS,
    bounds, clipped, fit_model, qlike, smear_tables,
)


VALIDATION_OUT = OUT / "validation_120"
ALPHAS = (0.1, 1.0, 10.0, 100.0, 300.0)


def hac_lrv(values: np.ndarray, lag: int | None = None) -> float:
    values = np.asarray(values, float)
    n = len(values); lag = lag if lag is not None else max(1, int(n ** (1 / 3)))
    centered = values - values.mean()
    result = float(centered @ centered / n)
    for j in range(1, min(lag, n - 1) + 1):
        gamma = float(centered[j:] @ centered[:-j] / n)
        result += 2 * (1 - j / (lag + 1)) * gamma
    return max(result, EPS)


def mean_hac_test(values: np.ndarray, null: float = 0.0, alternative: str = "greater") -> dict:
    values = np.asarray(values, float); n = len(values)
    lag = max(1, int(n ** (1 / 3)))
    se = np.sqrt(hac_lrv(values, lag) / n)
    statistic = (values.mean() - null) / se
    if alternative == "greater": p = 1 - stats.t.cdf(statistic, n - 1)
    elif alternative == "less": p = stats.t.cdf(statistic, n - 1)
    else: p = 2 * stats.t.sf(abs(statistic), n - 1)
    return {"mean": float(values.mean()), "se": float(se), "statistic": float(statistic), "pvalue": float(p), "hac_lag": lag}


def dm_hac(daily: np.ndarray) -> dict:
    result = mean_hac_test(daily)
    n = len(daily)
    correction = np.sqrt((n - 1) / n)
    statistic = result["statistic"] * correction
    result.update({"statistic": float(statistic), "pvalue": float(1 - stats.t.cdf(statistic, n - 1)), "harvey_correction": float(correction)})
    return result


def hac_regression(y: np.ndarray, x: np.ndarray, lag: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    y=np.asarray(y,float); x=np.asarray(x,float); n=len(y); lag=lag if lag is not None else max(1,int(n**(1/3)))
    inv=np.linalg.pinv(x.T@x); beta=inv@x.T@y; residual=y-x@beta; scores=x*residual[:,None]
    meat=scores.T@scores
    for j in range(1,min(lag,n-1)+1):
        cross=scores[j:].T@scores[:-j]; meat+=(1-j/(lag+1))*(cross+cross.T)
    return beta, inv@meat@inv


def tune(data: pd.DataFrame, features: list[str], target: str) -> float:
    dev, val = data[data["year"].le(2022)].copy(), data[data["year"].eq(2023)].copy()
    low, high = bounds(dev, features)
    dev.loc[:,features]=clipped(dev,features,low,high); val_x=clipped(val,features,low,high)
    best=(np.inf,None)
    for alpha in ALPHAS:
        model,residuals=fit_model(dev,features,target,alpha); _,smear,_=smear_tables(dev,residuals)
        base=np.exp(model.predict(val_x)); factors=val["time_bucket"].map(lambda b:smear.get(int(b),smear[-1])).to_numpy()
        loss=float(qlike(val[target].to_numpy(),base*factors).mean())
        if loss<best[0]: best=(loss,alpha)
    return float(best[1])


def rolling_predictions(panel: pd.DataFrame, horizon: int, mode: str = "normal") -> tuple[pd.DataFrame, dict]:
    target=f"future_var_{horizon}"; new_features=[f for f in ENHANCED_FEATURES if f not in HAR_FEATURES]
    data=panel[np.isfinite(panel[ENHANCED_FEATURES+[target]]).all(axis=1)&panel[target].gt(0)].copy()
    if mode=="placebo":
        order=data.sort_values(["datetime"]); key=order["datetime"].dt.strftime("%H:%M")
        order["test_target"]=order.groupby(key)[target].shift(-1); data=order.dropna(subset=["test_target"]); target="test_target"
    alphas={"har":tune(data,HAR_FEATURES,target),"enhanced":tune(data,ENHANCED_FEATURES,target)}
    evaluation=data[data["year"].ge(2024)].copy(); month_key=evaluation["datetime"].dt.to_period("M"); frames=[]; embargo_ok=True
    rng=np.random.default_rng(SEED+{"normal":0,"placebo":1,"permutation":2}[mode])
    for month in month_key.drop_duplicates().sort_values():
        test=evaluation[month_key.eq(month)].copy(); prior=data.loc[data["date"].lt(test["date"].min()),"date"].drop_duplicates().sort_values(); train_days=prior.iloc[-TRAIN_DAYS:]
        train=data[data["date"].isin(train_days)].copy()
        if len(train)<2000: continue
        embargo_ok &= bool(train["datetime"].max()+pd.Timedelta(minutes=120)<test["datetime"].min())
        if mode=="permutation":
            for frame in (train,test):
                for feature in new_features:
                    frame[feature]=frame.groupby("time_bucket")[feature].transform(lambda s:s.iloc[rng.permutation(len(s))].to_numpy())
        result=test[["datetime","date","time_bucket","log_rv_past_5",target]].rename(columns={target:"actual"}).copy()
        minute=test["datetime"].dt.hour*60+test["datetime"].dt.minute; result["tod"]=(minute-555)/375.0
        for name,features in {"har":HAR_FEATURES,"enhanced":ENHANCED_FEATURES}.items():
            low,high=bounds(train,features); fitted=train.copy(); fitted.loc[:,features]=clipped(fitted,features,low,high); test_x=clipped(test,features,low,high)
            model,residuals=fit_model(fitted,features,target,alphas[name]); _,smear,q90=smear_tables(fitted,residuals); raw_log=model.predict(test_x)
            factors=test["time_bucket"].map(lambda b:smear.get(int(b),smear[-1])).to_numpy(float)
            result[name]=np.exp(raw_log)*factors; result[f"log_{name}"]=np.log(result[name])
            tail=test["time_bucket"].map(lambda b:q90.get(int(b),q90[-1])).to_numpy(float)
            result[f"{name}_q90"]=np.exp(raw_log+tail)
        frames.append(result)
    return pd.concat(frames,ignore_index=True),{"alphas":alphas,"embargo_ok":embargo_ok}


def fixed_grid(pred: pd.DataFrame) -> pd.DataFrame:
    minute=pred["datetime"].dt.minute
    return pred[minute.mod(15).eq(0)].copy()


def daily_losses(pred: pd.DataFrame) -> pd.DataFrame:
    z=fixed_grid(pred); z["har_loss"]=qlike(z["actual"].to_numpy(),z["har"].to_numpy()); z["enhanced_loss"]=qlike(z["actual"].to_numpy(),z["enhanced"].to_numpy()); z["d"]=z["har_loss"]-z["enhanced_loss"]
    return z.groupby("date",as_index=False).agg(D=("d","mean"),har_loss=("har_loss","mean"),enhanced_loss=("enhanced_loss","mean"),tod=("tod","mean"),log_rv_past_5=("log_rv_past_5","mean"))


def stationary_indices(n: int, reps: int, mean_block: int, seed: int) -> np.ndarray:
    rng=np.random.default_rng(seed); p=1/mean_block; out=np.empty((reps,n),dtype=np.int32)
    for b in range(reps):
        i=int(rng.integers(n))
        for t in range(n):
            out[b,t]=i; i=int(rng.integers(n)) if rng.random()<p else (i+1)%n
    return out


def stationary_bootstrap(daily: pd.DataFrame) -> dict:
    d=daily["D"].to_numpy(); n=len(d); rho=float(pd.Series(d).autocorr(1)); mean_block=max(2,min(20,int(round((max(2*rho,EPS)/(max(1-rho,0.05)**2))**(1/3)*n**(1/3))))) if rho>0 else 2
    idx=stationary_indices(n,10000,mean_block,SEED); samples=d[idx]; means=samples.mean(axis=1); centered=means-d.mean(); p=float(np.mean(centered>=d.mean()))
    ses=np.maximum(samples.std(axis=1,ddof=1)/np.sqrt(n),EPS); tstar=centered/ses; se=np.sqrt(hac_lrv(d)/n); ci=[float(d.mean()-np.quantile(tstar,.975)*se),float(d.mean()-np.quantile(tstar,.025)*se)]
    lh=daily["har_loss"].to_numpy(); le=daily["enhanced_loss"].to_numpy(); ratio=lh.mean()/le.mean()-1; ratios=lh[idx].mean(axis=1)/le[idx].mean(axis=1)-1
    return {"mean_block":mean_block,"pvalue":p,"studentized_ci":ci,"ratio":float(ratio),"ratio_ci":[float(np.quantile(ratios,.025)),float(np.quantile(ratios,.975))],"indices":idx}


def sign_tests(d: np.ndarray) -> dict:
    rng=np.random.default_rng(SEED); exceed=0; total=100000
    for start in range(0,total,5000):
        count=min(5000,total-start); signs=rng.choice((-1,1),size=(count,len(d))); exceed+=int(np.sum((signs*d).mean(axis=1)>=d.mean()))
    wins=int(np.sum(d>0)); sign=stats.binomtest(wins,len(d),.5,alternative="greater")
    return {"sign_flip_pvalue":exceed/total,"winning_days":wins,"days":len(d),"win_rate":wins/len(d),"sign_test_pvalue":float(sign.pvalue)}


def gw_test(daily: pd.DataFrame) -> dict:
    z=daily.copy(); z["lag_D"]=z["D"].shift(1); z=z.dropna(); instruments=np.column_stack([np.ones(len(z)),z["tod"],z["lag_D"],z["log_rv_past_5"]]); moments=instruments*z["D"].to_numpy()[:,None]; n=len(z); centered=moments-moments.mean(axis=0); lag=max(1,int(n**(1/3))); omega=centered.T@centered/n
    for j in range(1,lag+1):
        cross=centered[j:].T@centered[:-j]/n; omega+=(1-j/(lag+1))*(cross+cross.T)
    rank=int(np.linalg.matrix_rank(instruments)); stat=float(n*moments.mean(axis=0)@np.linalg.pinv(omega)@moments.mean(axis=0)); return {"statistic":stat,"df":rank,"pvalue":float(stats.chi2.sf(stat,rank)),"note":"Degrees of freedom use the numerical rank of the daily instrument matrix."}


def calibration(pred: pd.DataFrame) -> dict:
    z=pred[pred["datetime"].dt.strftime("%H:%M").eq("11:15")].copy(); logy=np.log(z["actual"].to_numpy()); logp=np.log(z["enhanced"].to_numpy()); x=np.column_stack([np.ones(len(z)),logp]); beta,cov=hac_regression(logy,x); delta=beta-np.array([0.,1.]); wald=float(delta@np.linalg.pinv(cov)@delta)
    ratio=z["actual"].to_numpy()/z["enhanced"].to_numpy(); level=mean_hac_test(ratio,null=1,alternative="two-sided")
    diff=np.log(z["enhanced"].to_numpy())-np.log(z["har"].to_numpy()); y=logy-np.log(z["har"].to_numpy()); b,c=hac_regression(y,np.column_stack([np.ones(len(z)),diff])); se=np.sqrt(c[1,1])
    return {"timestamp":"11:15","n":len(z),"mz_intercept":float(beta[0]),"mz_slope":float(beta[1]),"mz_wald":wald,"mz_pvalue":float(stats.chi2.sf(wald,2)),"level_ratio_mean":level["mean"],"level_ratio_pvalue":level["pvalue"],"encompassing_lambda":float(b[1]),"lambda_eq_0_pvalue":float(1-stats.t.cdf(b[1]/se,len(z)-2)),"lambda_eq_1_pvalue":float(stats.t.cdf((b[1]-1)/se,len(z)-2))}


def coverage(pred: pd.DataFrame) -> dict:
    z=pred[pred["datetime"].dt.strftime("%H:%M").eq("11:15")].copy(); hits=(z["actual"]>z["enhanced_q90"]).astype(int).to_numpy(); n=len(hits); x=int(hits.sum()); pi=x/n; p=.1
    ll0=x*np.log(p)+(n-x)*np.log(1-p); ll1=x*np.log(max(pi,EPS))+(n-x)*np.log(max(1-pi,EPS)); lr_uc=float(-2*(ll0-ll1)); counts=np.zeros((2,2),int)
    for a,b in zip(hits[:-1],hits[1:]): counts[a,b]+=1
    p01=counts[0,1]/max(counts[0].sum(),1); p11=counts[1,1]/max(counts[1].sum(),1); pall=(counts[:,1].sum()/max(counts.sum(),1))
    def term(c,prob): return c*np.log(max(prob,EPS))
    ll_ind0=term(counts[:,1].sum(),pall)+term(counts[:,0].sum(),1-pall); ll_ind1=term(counts[0,1],p01)+term(counts[0,0],1-p01)+term(counts[1,1],p11)+term(counts[1,0],1-p11); lr_ind=float(-2*(ll_ind0-ll_ind1)); lr_cc=lr_uc+lr_ind
    return {"timestamp":"11:15","n":n,"exceedances":x,"hit_rate":pi,"kupiec_lr":lr_uc,"kupiec_pvalue":float(stats.chi2.sf(lr_uc,1)),"christoffersen_lr":lr_ind,"christoffersen_pvalue":float(stats.chi2.sf(lr_ind,1)),"conditional_coverage_lr":lr_cc,"conditional_coverage_pvalue":float(stats.chi2.sf(lr_cc,2))}


def stability(daily: pd.DataFrame) -> dict:
    d=daily["D"].to_numpy(); n=len(d); window=max(10,int(.3*n)); omega=np.sqrt(hac_lrv(d)); fluct=np.array([np.sqrt(window)*d[i:i+window].mean()/omega for i in range(n-window+1)]); q=daily.assign(quarter=pd.to_datetime(daily["date"]).dt.to_period("Q").astype(str)).groupby("quarter")["D"].agg(["mean","count"]).reset_index(); positive=float((q["mean"]>0).mean())
    return {"window":window,"max_abs_fluctuation":float(np.abs(fluct).max()),"critical_band":2.58,"violation":bool(np.abs(fluct).max()>2.58),"positive_quarter_fraction":positive,"quarters":q.to_dict("records")}


def holm(pvalues: dict[int,float]) -> dict[int,float]:
    ordered=sorted(pvalues,key=pvalues.get); adjusted={}; running=0
    for rank,h in enumerate(ordered): running=max(running,(len(ordered)-rank)*pvalues[h]); adjusted[h]=min(running,1.0)
    return adjusted


def summarize_variant(pred: pd.DataFrame) -> dict:
    daily=daily_losses(pred); dm=dm_hac(daily["D"].to_numpy()); return {"daily":daily,"dm":dm,"improvement":float(daily["har_loss"].mean()/daily["enhanced_loss"].mean()-1)}


def main() -> None:
    VALIDATION_OUT.mkdir(parents=True,exist_ok=True); panel=duckdb.connect().execute("SELECT * FROM read_parquet(?)",[str(OUT/"enhanced_spot_panel.parquet")]).fetchdf()
    predictions={}; horizon_summary={}; metadata={}
    for h in (15,30,60,120):
        pred,meta=rolling_predictions(panel,h); predictions[h]=pred; metadata[h]=meta; summary=summarize_variant(pred); horizon_summary[h]={"dm":summary["dm"],"improvement":summary["improvement"],"days":len(summary["daily"])}
        con=duckdb.connect(); con.register("pred",pred); con.execute("COPY pred TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",[str(VALIDATION_OUT/f"oos_{h}m.parquet")]); con.close()
    main_pred=predictions[120]; daily=daily_losses(main_pred); d=daily["D"].to_numpy(); dm=dm_hac(daily["D"].to_numpy())
    grid=fixed_grid(main_pred); eh=np.log(grid["actual"])-np.log(grid["har"]); ee=np.log(grid["actual"])-np.log(grid["enhanced"]); cw_row=eh**2-ee**2+(np.log(grid["har"])-np.log(grid["enhanced"]))**2; cw_daily=pd.DataFrame({"date":grid["date"],"f":cw_row}).groupby("date")["f"].mean(); cw=mean_hac_test(cw_daily.to_numpy())
    boot=stationary_bootstrap(daily); indices=boot.pop("indices"); signs=sign_tests(d); holm_p=holm({h:horizon_summary[h]["dm"]["pvalue"] for h in horizon_summary})
    centered=d-d.mean(); boot_spa=centered[indices].mean(axis=1); spa_p=float(np.mean(boot_spa>=d.mean())); mcs_p=float(np.mean(np.abs(boot_spa)>=abs(d.mean())))
    placebo_pred,placebo_meta=rolling_predictions(panel,120,"placebo"); permutation_pred,permutation_meta=rolling_predictions(panel,120,"permutation"); placebo=summarize_variant(placebo_pred); permutation=summarize_variant(permutation_pred)
    sigma=float(np.std(d,ddof=1)); mde=float(2.49*sigma/np.sqrt(len(d)))
    results={
        "setup":{"horizon":120,"daily_grid_minutes":15,"days":len(daily),"mean_daily_differential":float(d.mean()),"improvement_ratio":float(daily.har_loss.mean()/daily.enhanced_loss.mean()-1)},
        "dm_hac":dm,"clark_west":cw,"giacomini_white":gw_test(daily),"stationary_bootstrap":boot,"sign_tests":signs,
        "multiple_testing":{"horizon_raw_p":{str(h):horizon_summary[h]["dm"]["pvalue"] for h in horizon_summary},"holm_adjusted_p":{str(h):p for h,p in holm_p.items()},"spa_single_candidate_pvalue":spa_p,"two_model_mcs_equal_accuracy_pvalue":mcs_p,"scope_note":"SPA/MCS cover HAR versus the predeclared Enhanced candidate; no complete archive of every exploratory specification exists."},
        "calibration":calibration(main_pred),"stability":stability(daily),"quantile_coverage":coverage(main_pred),
        "leakage":{"embargo_ok":metadata[120]["embargo_ok"],"training_only_scaler_smearing":True,"placebo_improvement":placebo["improvement"],"placebo_dm_pvalue":placebo["dm"]["pvalue"],"permutation_improvement":permutation["improvement"],"permutation_dm_pvalue":permutation["dm"]["pvalue"]},
        "power":{"daily_sd":sigma,"mde_80pct":mde,"observed_mean":float(d.mean()),"exceeds_mde":bool(d.mean()>mde)},
        "horizons":horizon_summary,
    }
    pass_checks={
        "dm_or_cw":dm["pvalue"]<.05 or cw["pvalue"]<.05,
        "bootstrap_and_signflip":boot["pvalue"]<.05 and signs["sign_flip_pvalue"]<.05 and boot["ratio_ci"][0]>0,
        "holm_and_spa_mcs":holm_p[120]<.05 and spa_p<.05 and mcs_p<.10,
        "stability":results["stability"]["positive_quarter_fraction"]>=2/3 and not results["stability"]["violation"],
        "calibration":results["calibration"]["mz_pvalue"]>=.05 and results["quantile_coverage"]["kupiec_pvalue"]>=.05 and results["quantile_coverage"]["conditional_coverage_pvalue"]>=.05,
        "placebo_permutation":placebo["dm"]["pvalue"]>=.05 and permutation["dm"]["pvalue"]>=.05,
        "power":results["power"]["exceeds_mde"],
    }
    results["pass_criteria"]=pass_checks; results["decision"]="KEEP_ENHANCED" if all(pass_checks.values()) else "INCONCLUSIVE_USE_HAR"
    (VALIDATION_OUT/"results.json").write_text(json.dumps(results,indent=2,default=str),encoding="utf-8"); daily.to_csv(VALIDATION_OUT/"daily_loss_differential.csv",index=False)
    lines=[
        "# 120-minute Enhanced RV validation","",f"**Decision: {results['decision']}**","",
        f"The fixed 15-minute grid contains {len(daily)} evaluation days. Mean daily QLIKE gain is {d.mean():.6f}; the 80%-power MDE is {mde:.6f}.","",
        "## Primary inference","",
        f"- Harvey-corrected DM: t={dm['statistic']:.3f}, one-sided p={dm['pvalue']:.4f}",
        f"- Clark-West: t={cw['statistic']:.3f}, one-sided p={cw['pvalue']:.4g}",
        f"- Stationary bootstrap: p={boot['pvalue']:.4f}, studentized 95% CI [{boot['studentized_ci'][0]:.6f}, {boot['studentized_ci'][1]:.6f}]",
        f"- Improvement ratio: {100*boot['ratio']:.2f}%, 95% CI [{100*boot['ratio_ci'][0]:.2f}%, {100*boot['ratio_ci'][1]:.2f}%]",
        f"- Sign flip: p={signs['sign_flip_pvalue']:.4f}; exact sign test p={signs['sign_test_pvalue']:.4f}",
        f"- Holm-adjusted 120-minute p={holm_p[120]:.4f}; SPA p={spa_p:.4f}; two-model MCS p={mcs_p:.4f}","",
        "## Pass criteria","",* [f"- {'PASS' if value else 'FAIL'} — {name}" for name,value in pass_checks.items()],"",
        "The candidate remains archived, but HAR is the deployment default. Full statistics are in `results.json`; the daily test series is in `daily_loss_differential.csv`.",
    ]
    (VALIDATION_OUT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); print(json.dumps({"decision":results["decision"],"pass_criteria":pass_checks,"dm":dm,"bootstrap":boot,"power":results["power"]},indent=2))


if __name__=="__main__": main()
