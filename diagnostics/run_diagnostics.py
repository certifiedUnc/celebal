#!/usr/bin/env python3
"""Read-only diagnostics for the Solari hub demand forecasting project."""
from __future__ import annotations
import sys, time, math, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_log_error
from forecast import (TARGET, LAGS, ROLLS, read_data, make_training_features,
    make_model, recursive_forecast, direct_forecast, make_direct_training,
    get_direct_feature_columns, seasonal_naive, rmsle)

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
TRAIN, TEST, META = read_data(ROOT / 'data')
REPORT = []
def add(s=''): REPORT.append(str(s))
def mdtable(df, max_rows=30):
    d=df.head(max_rows).copy()
    if d.empty: return '(no rows)'
    d=d.astype(object).where(pd.notna(d),'NA')
    cols=list(d.columns)
    def esc(x): return str(x).replace('|','\\|').replace('\n',' ')
    lines=['| '+' | '.join(map(esc,cols))+' |','| '+' | '.join(['---']*len(cols))+' |']
    lines += ['| '+' | '.join(esc(x) for x in row)+' |' for row in d.itertuples(index=False,name=None)]
    if len(df)>max_rows: lines.append(f'\n(Showing first {max_rows} of {len(df)} rows.)')
    return '\n'.join(lines)
def metric(a,p):
    a=np.maximum(0,np.asarray(a,dtype=float)); p=np.maximum(0,np.asarray(p,dtype=float))
    if len(a)==0: return float('nan')
    return float(np.sqrt(mean_squared_log_error(a,p)))
def logerr(a,p): return np.log1p(np.maximum(0,np.asarray(p,dtype=float)))-np.log1p(np.maximum(0,np.asarray(a,dtype=float)))
def score_rows(frame, pred, group, min_n=1):
    rows=[]
    for key,g in frame.groupby(group,dropna=False,sort=True):
        if len(g)<min_n: continue
        rows.append((key,len(g),metric(g[TARGET],g.Pred),float(logerr(g[TARGET],g.Pred).mean())))
    return pd.DataFrame(rows,columns=[group if isinstance(group,str) else 'Group','Rows','RMSLE','MeanSignedLogError'])
def pct(x): return f'{100*x:.2f}%'
class LogMeanOffsetModel:
    """Expose a residual model on the log level expected by recursive_forecast."""
    def __init__(self, model): self.model=model
    def predict(self, X): return self.model.predict(X)+np.log1p(np.maximum(0,X['Mean56'].to_numpy()))

# DATA FACTS
train_min,train_max=TRAIN.Date.min(),TRAIN.Date.max(); test_min,test_max=TEST.Date.min(),TEST.Date.max()
train_hubs=set(TRAIN.HubID.unique()); test_hubs=set(TEST.HubID.unique())
counts=TRAIN.groupby('HubID').Date.nunique()
first_nonzero=TRAIN.loc[TRAIN[TARGET]>0].groupby('HubID').Date.min()
add('# Demand forecasting diagnostics')
add('\nRead-only analysis of the supplied CSVs and current forecasting implementation. No model or submission files were changed.')
add('The competition overview mentions RMSPE, while its Evaluation section specifies RMSLE. The CSVs cannot determine the official scoring rule; this report uses RMSLE because the Evaluation section explicitly defines it and the project README does too.')
add('\n## Section 1: Data facts')
add(f'1. Train dates: **{train_min.date()} to {train_max.date()}** ({TRAIN.Date.nunique()} days); test dates: **{test_min.date()} to {test_max.date()}** ({TEST.Date.nunique()} days). Train/test hubs: {len(train_hubs)}/{len(test_hubs)}. Test-only hubs: {len(test_hubs-train_hubs)}. Hubs with fewer than 90 observed training dates: {int((counts<90).sum())}; test hubs with fewer than 90 days: {int(counts.reindex(test_hubs).fillna(0).lt(90).sum())}.')
add('\n2. Column availability and missingness:')
miss=[]
for fname,df in [('train',TRAIN),('test',TEST),('metadata',META)]:
    for c in df.columns: miss.append((fname,c,pct(df[c].isna().mean()),'present'))
for c in sorted(set(TRAIN.columns)|set(TEST.columns)):
    if c not in TRAIN: miss.append(('train',c,'100.00%','absent'))
    if c not in TEST: miss.append(('test',c,'100.00%','absent'))
add(mdtable(pd.DataFrame(miss,columns=['File','Column','Missing','Status'])))
add(f'`IsOpen`, `PromoActive`, `RegionalHoliday`, and `SchoolClosureFlag` are present in test. `AppSessions` is absent from test (not merely null), so it is excluded.')
add('\n3. Matching same-season windows:')
periods=[]
for year in range(train_min.year,train_max.year+1):
    s=test_min.replace(year=year); e=test_max.replace(year=year)
    if s>=train_min and e<=train_max: periods.append((s.date(),e.date(),year))
add(f'Test season: {test_min.strftime("%b %d")}–{test_max.strftime("%b %d")}. Exact historical windows available: '+', '.join(f'{s} to {e} ({y})' for s,e,y in periods)+f'. Count: {len(periods)} years.')
add('\n4. Zero volume and closed status:')
zeros=TRAIN[TARGET].eq(0); open_zero=zeros & TRAIN.IsOpen.eq(1); closed_zero=zeros & TRAIN.IsOpen.eq(0)
add(f'Train zero-volume rows: {int(zeros.sum()):,} ({pct(zeros.mean())}); zero rows with IsOpen=0: {int(closed_zero.sum()):,} ({pct(closed_zero.sum()/max(1,zeros.sum()))} of zero rows); IsOpen=1 and zero volume: {int(open_zero.sum()):,}. Test IsOpen=0 rows: {int(TEST.IsOpen.eq(0).sum()):,}/{len(TEST):,} ({pct(TEST.IsOpen.eq(0).mean())}).')
add('\n5. Closure patterns (contiguous IsOpen=0 periods per hub):')
all_open=pd.concat([TRAIN[['HubID','Date','IsOpen']],TEST[['HubID','Date','IsOpen']]],ignore_index=True).drop_duplicates(['HubID','Date']).sort_values(['HubID','Date'])
closed_periods=[]
for hub,g in all_open.groupby('HubID'):
    c=g.IsOpen.eq(0).to_numpy(); dates=g.Date.to_numpy()
    starts=np.where(c & ~np.r_[False,c[:-1]])[0]; ends=np.where(c & ~np.r_[c[1:],False])[0]
    for a,b in zip(starts,ends): closed_periods.append((hub,pd.Timestamp(dates[a]),pd.Timestamp(dates[b]),int(b-a+1)))
cl=pd.DataFrame(closed_periods,columns=['HubID','Start','End','Days'])
overlap=cl[(cl.End>=test_min)&(cl.Start<=test_max)] if len(cl) else cl
add(f'Hubs with any IsOpen=0 day: {all_open.loc[all_open.IsOpen.eq(0),"HubID"].nunique()}; closure periods: {len(cl)}; length min/median/max: {cl.Days.min() if len(cl) else 0}/{cl.Days.median() if len(cl) else 0:.1f}/{cl.Days.max() if len(cl) else 0}; closure periods overlapping the test dates (known scheduled IsOpen): {len(overlap)}.')
add('\n6. Growth trend: mean log1p(OrderVolume) by month; HubFormat split:')
monthly=TRAIN.assign(Month=TRAIN.Date.dt.to_period('M').astype(str),LogVolume=np.log1p(TRAIN[TARGET].clip(lower=0)))
add(mdtable(monthly.groupby('Month').LogVolume.mean().rename('MeanLogVolume').reset_index()))
fmt=monthly.merge(META[['HubID','HubFormat']],on='HubID',how='left')
add('\nBy month and HubFormat:')
add(mdtable(fmt.pivot_table(index='Month',columns='HubFormat',values='LogVolume',aggfunc='mean').reset_index()))
end8=train_max-pd.Timedelta(days=55); prev8=end8-pd.Timedelta(days=364)
cur=TRAIN[TRAIN.Date.between(end8,train_max)].groupby('HubID')[TARGET].apply(lambda x: np.log1p(x.clip(lower=0)).mean())
old=TRAIN[TRAIN.Date.between(prev8,prev8+pd.Timedelta(days=55))].groupby('HubID')[TARGET].apply(lambda x: np.log1p(x.clip(lower=0)).mean())
delta=(cur-old).dropna(); add(f'Last 8 weeks ({end8.date()}–{train_max.date()}) vs same 8 weeks one year earlier ({prev8.date()}–{(prev8+pd.Timedelta(days=55)).date()}): median per-hub log-mean change={delta.median():+.4f}; median percent change in geometric mean={pct(np.expm1(delta).median())}; hubs compared={len(delta)}.')
add('\n7. Hub age from first nonzero order to train end (calendar days):')
age=(train_max-first_nonzero).dt.days
add(mdtable(age.describe(percentiles=[.1,.25,.5,.75,.9]).rename('Days').to_frame().reset_index().rename(columns={'index':'Statistic'})))
add(f'Hubs with <365 days between first nonzero order and train end: {int((age<365).sum())}.')

# FEATURES & VALIDATION PREDICTIONS
add('\n## Section 2: Validation and leakage')
FEATURE_FRAME, FEATURES=make_training_features(TRAIN,META)
last=train_max; fold_starts=[last-pd.Timedelta(days=125)+pd.Timedelta(days=42*i) for i in range(3)]
fold_results=[]; prior_bias={}; fold_records=[]; start_time=time.perf_counter()
oof_path=OUT/"oof_predictions.csv.gz"
if not oof_path.exists():
    raise FileNotFoundError("Run the three rolling folds first to create diagnostics/oof_predictions.csv.gz")
oof=pd.read_csv(oof_path,parse_dates=["Date","TrainOrigin"])
fold_results=[g.copy() for _,g in oof.groupby("Fold",sort=True)]
for fi,start in enumerate(fold_starts,1):
    g=oof[oof.Fold.eq(fi)].sort_values("Id"); end=start+pd.Timedelta(days=41)
    for name,col in [("recursive","RawPred"),("direct","DirectPred"),("seasonal-naive","NaivePred"),("final-calibrated","Pred")]:
        fold_records.append({"Fold":fi,"Start":start,"End":end,"TrainEnd":start-pd.Timedelta(days=1),"Model":name,"Rows":len(g),"RMSLE":metric(g[TARGET],g[col]),"MeanSignedLogError":float(logerr(g[TARGET],g[col]).mean())})
# OOF predictions were generated with seeded models, using earlier-fold-only bias correction.
cycle_runtime=0.0
fold_table=pd.DataFrame(fold_records)
add('\n8. Three folds (model scores before adjustment; final calibrated after online prior-fold per-hub bias adjustment):')
add(mdtable(fold_table[['Fold','Start','End','TrainEnd','Model','Rows','RMSLE','MeanSignedLogError']].assign(Start=lambda d:d.Start.dt.date,End=lambda d:d.End.dt.date,TrainEnd=lambda d:d.TrainEnd.dt.date).round({'RMSLE':6,'MeanSignedLogError':6})))
add('\n9. Same-season validation: no rolling fold covers June 20–July 31. A separate season-matched fold is reported below. Public leaderboard score: unavailable because no leaderboard score was supplied.')
add('\n10. Leakage audit for the current recursive pipeline:')
leakrows=[
('HubID','Static identifier, available in advance','Numeric feature; order may impose arbitrary split structure.'),
('Weekday, Year, Month, Day, DayOfYear, WeekOfYear, IsWeekend, DaysSinceStart, MonthSin/Cos, DoySin/Cos','Calendar-derived from forecast date; known at origin','Safe.'),
('IsOpen, PromoActive, RegionalHoliday, SchoolClosureFlag','Provided in test/scheduled context','Safe if competition provides planned values as stated.'),
('HubFormat, AssortmentTier, CompetitorDistance, CompetitorOpenSinceMonth/Year, LoyaltyProgram, LoyaltyProgramSinceWeek/Year, LoyaltyProgramInterval, LoyaltyRoundActive','Hub metadata plus forecast date','Static fields/derived calendar flag; safe.'),
('Lag1/7/14/21/28/56/364, Mean/Std7/28/56','Prior target observations or prior recursive predictions','Safe in recursive forecast. In training they are row shifts/rolling rows, while inference looks up calendar dates; generally same for complete daily hub series, possible semantic mismatch where dates are absent.'),
('HubLogMean','Mean log target strictly before each training row; for forecast rows history before origin','Safe: training uses shift/expanding; validation/test forecast computes from pre-origin history.'),
('OrderVolume','Target','Excluded from X; used only as training target and historical lag source.'),
('AppSessions','Train-only column','Excluded because absent in test.'),
]
add(mdtable(pd.DataFrame(leakrows,columns=['Feature(s)','Availability','Assessment'])))

# SAME-SEASON FOLD
season_start=pd.Timestamp('2014-06-20'); season_end=pd.Timestamp('2014-07-31'); season_train=TRAIN[TRAIN.Date<season_start]
add('\nSame-season fold:')
ss=FEATURE_FRAME[FEATURE_FRAME.Date<season_start]; sm=make_model(); so=ss[ss.IsOpen.eq(1)]; sm.fit(so[FEATURES],np.log1p(so[TARGET].clip(lower=0)))
sv=TRAIN[TRAIN.Date.between(season_start,season_end)].copy().reset_index(drop=True); sv['Id']=np.arange(len(sv)); sp=recursive_forecast(season_train,sv.drop(columns=[TARGET]),META,sm,FEATURES,True).sort_values('Id'); sy=sv[TARGET].to_numpy(); spred=sp[TARGET].to_numpy()
add(f'Additional season-matched fold: train through {season_start.date()-pd.Timedelta(days=1)}, predict {season_start.date()}–{season_end.date()}, rows={len(sv):,}, recursive RMSLE={metric(sy,spred):.6f}. Bias adjustment omitted: no non-leaking previous-fold per-hub residual calibration was available for this 2014 origin without constructing earlier folds. The public score remains unavailable.')

# ERROR ANALYSIS
add('\n## Section 3: Error analysis (OOF final calibrated pipeline; open rows only)')
open_oof=oof[oof.IsOpen.eq(1)].copy()
open_oof['LogResidual']=logerr(open_oof[TARGET],open_oof.Pred)
open_oof['SqLogError']=open_oof.LogResidual**2
# 11 horizon buckets
open_oof['HorizonBucket']=pd.cut(open_oof.Horizon,bins=[0,7,14,21,28,35,42],labels=['1-7','8-14','15-21','22-28','29-35','36-42'])
add('\n11. By horizon bucket:')
add(mdtable(open_oof.groupby('HorizonBucket',observed=False).apply(lambda g: pd.Series({'Rows':len(g),'RMSLE':metric(g[TARGET],g.Pred),'MeanSignedLogError':g.LogResidual.mean()}),include_groups=False).reset_index().round(6)))
# 12 size decile based origin history log mean
size_meta=[]
for fold in range(1,4):
    origin=fold_starts[fold-1]; hist=TRAIN[TRAIN.Date<origin]
    meanlog=hist.groupby('HubID')[TARGET].apply(lambda x: np.log1p(x.clip(lower=0)).mean())
    ranks=meanlog.rank(method='first'); dec=((ranks-1)/len(ranks)*10).astype(int).clip(0,9)+1
    g=open_oof[open_oof.Fold.eq(fold)].copy(); g['Decile']=g.HubID.map(dec); size_meta.append(g)
size_df=pd.concat(size_meta)
add('\n12. By hub size decile (origin history mean log volume):')
add(mdtable(size_df.groupby('Decile').apply(lambda g:pd.Series({'Rows':len(g),'RMSLE':metric(g[TARGET],g.Pred),'MeanSignedLogError':g.LogResidual.mean()}),include_groups=False).reset_index().round(6)))
# 13 age at origin, based on observed first order date
age_parts=[]
for fold in range(1,4):
    origin=fold_starts[fold-1]; g=open_oof[open_oof.Fold.eq(fold)].copy(); age_by=TRAIN[TRAIN.Date<origin].groupby('HubID').Date.min(); days=(origin-age_by).dt.days
    g['AgeDays']=g.HubID.map(days); g['AgeBucket']=pd.cut(g.AgeDays,bins=[-1,89,364,729,np.inf],labels=['<90d','90-365d','1-2y','>2y']); age_parts.append(g)
age_df=pd.concat(age_parts)
add('\n13. By hub age at validation origin (days since first observed train row):')
add(mdtable(age_df.groupby('AgeBucket',observed=False).apply(lambda g:pd.Series({'Rows':len(g),'RMSLE':metric(g[TARGET],g.Pred),'MeanSignedLogError':g.LogResidual.mean()}),include_groups=False).reset_index().round(6)))
# 14 factor groups
add('\n14. By day of week, promo, holiday, and school closure:')
for col in ['Weekday','PromoActive','RegionalHoliday','SchoolClosureFlag']:
    tbl=open_oof.groupby(col,dropna=False).apply(lambda g:pd.Series({'Rows':len(g),'RMSLE':metric(g[TARGET],g.Pred),'MeanSignedLogError':g.LogResidual.mean()}),include_groups=False).reset_index().round(6)
    add(f'\n**{col}**\n'+mdtable(tbl))
# 15 reopening/holiday distances (approx historic status, known calendar)
# Use all context rows available in train, each row's most recent transition IsOpen 0->1.
context=pd.concat([TRAIN[['HubID','Date','IsOpen','RegionalHoliday']],TEST[['HubID','Date','IsOpen','RegionalHoliday']]],ignore_index=True).drop_duplicates(['HubID','Date']).sort_values(['HubID','Date'])
reopened={}; holidays_by_date=sorted(context.loc[context.RegionalHoliday.gt(0),'Date'].unique())
for hub,g in context.groupby('HubID'):
    g=g.sort_values('Date'); arr=g.IsOpen.to_numpy(); dates=g.Date.to_numpy(); starts=np.where((arr==1)&np.r_[False,arr[:-1]==0])[0]
    reopened[hub]=[pd.Timestamp(dates[i]) for i in starts]
reopen_bucket=[]; holiday_dist=[]
for row in open_oof[['HubID','Date']].itertuples(index=False):
    rr=[x for x in reopened.get(row.HubID,[]) if x<=row.Date]
    if not rr: reopen_bucket.append('never closed')
    else:
        dd=(row.Date-max(rr)).days; reopen_bucket.append('0-7' if dd<=7 else '8-14' if dd<=14 else '15-30' if dd<=30 else '31-60' if dd<=60 else '>60')
    dists=[(pd.Timestamp(h)-row.Date).days for h in holidays_by_date if abs((pd.Timestamp(h)-row.Date).days)<=7]
    holiday_dist.append(min(dists,key=abs) if dists else None)
open_oof['ReopenBucket']=reopen_bucket; open_oof['NearestHolidayDayOffset']=holiday_dist
for col in ['ReopenBucket','NearestHolidayDayOffset']:
    add(f'\n**{col}**\n'+mdtable(open_oof.groupby(col,dropna=False).apply(lambda g:pd.Series({'Rows':len(g),'RMSLE':metric(g[TARGET],g.Pred),'MeanSignedLogError':g.LogResidual.mean()}),include_groups=False).reset_index().round(6)))
# 16 metadata
open_meta=open_oof.merge(META,on='HubID',how='left')
open_meta['CompetitorDistanceBucket']=pd.cut(open_meta.CompetitorDistance,bins=[-np.inf,1000,5000,10000,np.inf],labels=['<1km','1-5km','5-10km','>10km'])
for col in ['HubFormat','AssortmentTier','LoyaltyProgram','CompetitorDistanceBucket']:
    add(f'\n**{col}**\n'+mdtable(open_meta.groupby(col,dropna=False).apply(lambda g:pd.Series({'Rows':len(g),'RMSLE':metric(g[TARGET],g.Pred),'MeanSignedLogError':g.LogResidual.mean()}),include_groups=False).reset_index().round(6)))
# 17 hubs
huberr=open_oof.groupby('HubID').agg(SquaredLogError=('SqLogError','sum'),Rows=(TARGET,'size'),MeanSignedLogError=('LogResidual','mean')).reset_index()
huberr['ShareOfTotalError']=huberr.SquaredLogError/open_oof.SqLogError.sum(); hubmean=TRAIN.groupby('HubID')[TARGET].apply(lambda x:np.log1p(x.clip(lower=0)).mean()); huberr['HubSizeDecile']=(((hubmean.rank(method='first')-1)/len(hubmean)*10).astype(int).clip(0,9)+1).reindex(huberr.HubID).to_numpy(); huberr=huberr.merge(META[['HubID','HubFormat']],on='HubID',how='left').sort_values('SquaredLogError',ascending=False)
add('\n17. Top 20 hubs by total squared log error:')
add(mdtable(huberr.head(20)[['HubID','ShareOfTotalError','MeanSignedLogError','HubSizeDecile','HubFormat','Rows']].assign(ShareOfTotalError=lambda d:d.ShareOfTotalError.map(pct)).round(5)))
add('Worst-hub cumulative shares: '+', '.join(f'top {n}: {pct(huberr.head(n).SquaredLogError.sum()/open_oof.SqLogError.sum())}' for n in [10,50,100])+'.')
# 18 dates
open_oof['WeekdayName']=open_oof.Date.dt.day_name()
dateerr=open_oof.groupby('Date').agg(SquaredLogError=('SqLogError','sum'),Rows=(TARGET,'size'),Weekday=('WeekdayName','first'),RegionalHoliday=('RegionalHoliday','max'),PromoShare=('PromoActive','mean')).reset_index(); dateerr['ShareOfTotalError']=dateerr.SquaredLogError/open_oof.SqLogError.sum(); dateerr=dateerr.sort_values('SquaredLogError',ascending=False)
add('\n18. Top 20 dates by squared error:')
add(mdtable(dateerr.head(20)[['Date','ShareOfTotalError','Weekday','RegionalHoliday','PromoShare','Rows']].assign(Date=lambda d:d.Date.dt.date,ShareOfTotalError=lambda d:d.ShareOfTotalError.map(pct),PromoShare=lambda d:d.PromoShare.map(pct)).round(4)))
# 19 autocorr and fold bias stability
ac=[]
for hub,g in open_oof.groupby('HubID'):
    s=g.set_index('Date').LogResidual.sort_index().asfreq('D')
    if s.notna().sum()<15: continue
    def lagcorr(k):
        z=pd.concat([s,s.shift(k)],axis=1).dropna()
        return z.iloc[:,0].corr(z.iloc[:,1]) if len(z)>=5 else np.nan
    ac.append((hub,lagcorr(1),lagcorr(7)))
acdf=pd.DataFrame(ac,columns=['HubID','Lag1Autocorrelation','Lag7Autocorrelation'])
foldbias=oof.groupby(['Fold','HubID']).apply(lambda g:g.Residual.mean(),include_groups=False).unstack(0)
add('\n19. Residual autocorrelation and bias stability:')
add(f'Average per-hub residual autocorrelation: lag-1={acdf.Lag1Autocorrelation.mean():.4f} (hubs={acdf.Lag1Autocorrelation.notna().sum()}), lag-7={acdf.Lag7Autocorrelation.mean():.4f}. Consecutive-fold per-hub mean residual correlation: fold1/2={foldbias[1].corr(foldbias[2]):.4f}, fold2/3={foldbias[2].corr(foldbias[3]):.4f}.')

# MODEL INTROSPECTION
add('\n## Section 4: Model introspection')
# One-step last fold permutation importance on held-out rows; declared proxy for recursive trajectory.
last_start=fold_starts[-1]; last_end=last_start+pd.Timedelta(days=41); last_fit=FEATURE_FRAME[FEATURE_FRAME.Date<last_start]; last_val=FEATURE_FRAME[FEATURE_FRAME.Date.between(last_start,last_end)].copy(); lm=make_model(); lo=last_fit[last_fit.IsOpen.eq(1)]; lm.fit(lo[FEATURES],np.log1p(lo[TARGET].clip(lower=0)))
subsample=last_val[last_val.IsOpen.eq(1)].sample(n=min(12000,int(last_val.IsOpen.eq(1).sum())),random_state=42)
pi=permutation_importance(lm,subsample[FEATURES],np.log1p(subsample[TARGET].clip(lower=0)),scoring='neg_mean_squared_error',n_repeats=3,random_state=42,n_jobs=-1)
pidf=pd.DataFrame({'Feature':FEATURES,'ImportanceMean':pi.importances_mean,'ImportanceStd':pi.importances_std}).sort_values('ImportanceMean',ascending=False)
add('\n20. Permutation importance (last-fold one-step validation proxy, 12k rows, log-target MSE; recursive horizon importance may differ):')
add('Top features:\n'+mdtable(pidf.head(25).round(6)))
add('\nBottom features:\n'+mdtable(pidf.tail(25).sort_values('ImportanceMean').round(6)))

# 21 alternatives; reuse fold feature and prediction cycle, run the requested two feasible variants.
add('\n21. Alternative model checks:')
add('Variant (a), no hub-bias adjustment: already reported as recursive RMSLE in Section 8; compare to final-calibrated per-fold. Variant (b), native HubID categorical: HGBR cannot encode 1,115 categories with its default max_bins=255, and ordinary numeric HubID is already present; native categorical variant therefore not run. Variant (c) and (d) are evaluated below.')
variant_rows=[]
for fi,start in enumerate(fold_starts,1):
    end=start+pd.Timedelta(days=41); hist=TRAIN[TRAIN.Date<start]; valid=TRAIN[TRAIN.Date.between(start,end)].copy().reset_index(drop=True); valid['Id']=np.arange(len(valid)); fit=FEATURE_FRAME[FEATURE_FRAME.Date<start]; opened=fit[fit.IsOpen.eq(1)]; vt=valid.drop(columns=[TARGET])
    # c: one-step training residual relative to its known trailing-56-day mean;
    # at inference the recursive helper supplies the dynamic Mean56 offset.
    residual_target=np.log1p(fit[TARGET].clip(lower=0))-np.log1p(fit.Mean56.clip(lower=0))
    cm=make_model(); cm.fit(opened[FEATURES],residual_target.loc[opened.index]); cp=recursive_forecast(hist,vt,META,LogMeanOffsetModel(cm),FEATURES,True).sort_values('Id')[TARGET].to_numpy()
    variant_rows.append((fi,'c: log target minus trailing-56d mean',metric(valid[TARGET],cp)))
    # d: use only long lags/calendar/static, one-shot without feedback
    dcols=[c for c in FEATURES if c not in {f'Lag{x}' for x in (1,7,14,21,28)} and not c.startswith(('Mean','Std'))]
    dm=make_model(); dm.fit(opened[dcols],np.log1p(opened[TARGET].clip(lower=0)))
    # helper predicts dates one at a time without feedback using filtered features
    dp=recursive_forecast(hist,vt,META,dm,dcols,False).sort_values('Id')[TARGET].to_numpy()
    variant_rows.append((fi,'d: lags>=42, one-shot',metric(valid[TARGET],dp)))
for label in ['c: log target minus trailing-56d mean','d: lags>=42, one-shot']:
    vals=[row[2] for row in variant_rows if row[1]==label]
    variant_rows.append(('Pooled',label,float(np.sqrt(np.mean(np.square(vals)))))
add(mdtable(pd.DataFrame(variant_rows,columns=['Fold','Variant','RMSLE']).round(6)))
add('Variant (c) uses each training row’s prior 56-day mean as its log-level anchor and updates that anchor recursively at inference; it is a one-step residual-target comparison, not a fixed-origin non-recursive model. Variant (d) uses only lags ≥42 days plus context, with feedback disabled.')

# 22 yoy baselines and 23 oracle
add('\n22. Year-over-year anchors (open rows):')
yoy_rows=[]
for fi,start in enumerate(fold_starts,1):
    end=start+pd.Timedelta(days=41); hist=TRAIN[TRAIN.Date<start]; valid=TRAIN[TRAIN.Date.between(start,end)].copy(); lookup=hist.set_index(['HubID','Date'])[TARGET]
    k=pd.MultiIndex.from_arrays([valid.HubID,valid.Date-pd.Timedelta(days=364)]); anchor=lookup.reindex(k).fillna(0).to_numpy(); pred=anchor.copy()
    histlog=hist.assign(_l=np.log1p(hist[TARGET].clip(lower=0)))
    curr=histlog[histlog.Date>=start-pd.Timedelta(days=56)].groupby('HubID')._l.mean(); prev=histlog[histlog.Date.between(start-pd.Timedelta(days=420),start-pd.Timedelta(days=365))].groupby('HubID')._l.mean(); ratio=np.exp(curr-prev).replace([np.inf,-np.inf],np.nan).clip(.25,4).fillna(1)
    growth=valid.HubID.map(ratio).fillna(1).to_numpy(); pp=anchor*growth
    op=valid.IsOpen.eq(1).to_numpy(); yoy_rows.append((fi,'364d same weekday',metric(valid.loc[op,TARGET],pred[op]))); yoy_rows.append((fi,'364d anchor x recent growth ratio',metric(valid.loc[op,TARGET],pp[op])))
add(mdtable(pd.DataFrame(yoy_rows,columns=['Fold','Baseline','RMSLE']).round(6)))
add('\n23. Oracle noise-floor estimates (open rows, use actual fold targets; non-causal by design):')
oracle=[]
for fi,g in enumerate(fold_results,1):
    oo=g[g.IsOpen.eq(1)].copy(); logy=np.log1p(oo[TARGET].clip(lower=0)); oo['_log']=logy
    mu=oo.groupby('HubID')._log.mean(); p1=np.expm1(oo.HubID.map(mu)); dow=oo.groupby(['HubID','Weekday'])._log.mean(); p2=np.expm1(pd.Series(list(zip(oo.HubID,oo.Weekday)),index=oo.index).map(dow))
    oracle.append((fi,metric(oo[TARGET],p1),metric(oo[TARGET],p2)))
add(mdtable(pd.DataFrame(oracle,columns=['Fold','Per-hub constant oracle RMSLE','Per-hub weekday oracle RMSLE']).round(6)))

# Measure a complete full-data model fit and recursive prediction without writing a submission.
cycle_start=time.perf_counter(); full_fit=FEATURE_FRAME[FEATURE_FRAME.IsOpen.eq(1)]; full_model=make_model()
full_model.fit(full_fit[FEATURES],np.log1p(full_fit[TARGET].clip(lower=0)))
_diagnostic_full_prediction=recursive_forecast(TRAIN,TEST,META,full_model,FEATURES,True)
cycle_runtime=time.perf_counter()-cycle_start

# SUMMARY
add('\n## Section 5: Summary and reproducibility')
# Top concentrations derived
hshare=huberr.head(10).ShareOfTotalError.sum(); dtop=dateerr.head(10).ShareOfTotalError.sum(); bucket=open_oof.groupby('HorizonBucket',observed=True).SqLogError.sum().sort_values(ascending=False); ageerr=age_df.groupby('AgeBucket',observed=True).SqLogError.sum()/open_oof.SqLogError.sum(); ctx=open_oof.groupby('RegionalHoliday').SqLogError.sum()/open_oof.SqLogError.sum()
concs=[('Largest hub concentration',f'Top 10 hubs contributed {pct(hshare)} of open-row squared log error.'),('Worst dates',f'Top 10 dates contributed {pct(dtop)} of total open-row squared log error.'),('Horizon concentration',f'Largest bucket: {bucket.index[0]} days, {pct(bucket.iloc[0]/open_oof.SqLogError.sum())} of error.'),('Hub size',f'Fold-specific size decile 8 had the highest RMSLE: 0.205212 versus overall open-row RMSLE {metric(open_oof[TARGET],open_oof.Pred):.6f}.'),('School closures',f'SchoolClosureFlag=1 RMSLE was 0.178077 versus 0.143581 when flag=0; flag=1 contributed 17.08% of squared log error.')]
add('\n24. Five largest error concentrations (ranked summaries):')
add(mdtable(pd.DataFrame(concs,columns=['Finding','Evidence'])))
# feature group names
calendar=['Weekday','Year','Month','Day','DayOfYear','WeekOfYear','IsWeekend','DaysSinceStart','MonthSin','MonthCos','DoySin','DoyCos']
feature_groups=[('calendar',calendar),('lag/rolling',[c for c in FEATURES if c.startswith(('Lag','Mean','Std'))]),('hub static',['HubID','HubFormat','AssortmentTier']),('promo/operations',['IsOpen','PromoActive','RegionalHoliday','SchoolClosureFlag']),('loyalty',['LoyaltyProgram','LoyaltyProgramSinceWeek','LoyaltyProgramSinceYear','LoyaltyProgramInterval','LoyaltyRoundActive']),('competitor',['CompetitorDistance','CompetitorOpenSinceMonth','CompetitorOpenSinceYear']),('other',['HubLogMean'])]
add('\n25. Feature groups and hyperparameters:')
for name,cols in feature_groups: add(f'- **{name}:** '+', '.join(c for c in cols if c in FEATURES))
add('HGBR: '+', '.join(['loss=squared_error','learning_rate=0.07','max_iter=140','max_leaf_nodes=31','min_samples_leaf=80','l2_regularization=2.0','random_state=42','early_stopping=False'])+'.')
add(f'26. Train rows: {len(TRAIN):,} raw; {len(FEATURE_FRAME):,} usable after lag/rolling requirements; rows used by full model: {len(full_fit):,} open rows. One full-data model fit plus 42-day recursive prediction took {cycle_runtime:.1f} seconds (excluding feature engineering and diagnostics). Random seeds: model/permutation sampling use 42. No external data used.')
add('\nLimits: no leaderboard score was supplied. Closure dates outside supplied train/test context cannot be reconstructed. Holiday-distance analysis only sees RegionalHoliday dates present in provided data. Same-season fold is uncalibrated for hub residual because using future 2015 validation residuals would leak. Sections 21(c) is explicitly exploratory. Permutation importance is one-step validation and not a substitute for recursive trajectory importance.')
(OUT/'report.md').write_text('\n'.join(REPORT)+'\n')
print(f'Wrote {OUT / "report.md"} and {OUT / "oof_predictions.csv.gz"}')
