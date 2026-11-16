# Workflow

## 1. Validate the incoming snapshot

Use the most recent timestamp, nearest monthly expiry and its five-strike chain.
Require spot, DTE, call and put prices, call and put IV, and the live RV feature
record. Reject observations outside 4–20 DTE or outside 09:45–09:59.

## 2. Build entry-time features

The ordered feature vector is DTE, log ATM premium/spot, mean ATM IV, put-call IV
skew, log RV over 5/15/60 minutes, overnight return, 30-period
volatility-of-volatility and put-call OI imbalance. All fields must exist at
decision time. Future RV, future IV and exit prices are never model inputs.

## 3. Route by DTE

### DTE 4–7

The scaled Ridge pipeline forecasts net ATM-straddle return. Create an ATM
short-straddle intent only above the frozen training-only 80th-percentile gate.

### DTE 8–14

The robust-scaled Huber pipeline forecasts net ATM-straddle return. Create the
intent only above zero and its frozen training-only 60th-percentile gate.

### DTE 15–20

Each structure model forecasts after-cost P&L in option points. Its conservative
score is `conditional Huber mean + training residual 25th percentile`.

A structure must have positive aggregate training P&L and a positive score.
Select the highest score. The current eligible structures are the one-strike and
two-strike short strangles. If no candidate qualifies, return no trade.

## 4. Construct fixed-strike orders

- ATM short straddle: sell ATM call and ATM put.
- One-strike short strangle: sell call at ATM+1 and put at ATM−1.
- Two-strike short strangle: sell call at ATM+2 and put at ATM−2.

Use one historical lot in research and one lot in paper intents. Never replace
selected strikes when spot moves.

## 5. Exit and record

Close every leg at or after 15:15 on the entry day. Record the entry snapshot,
all candidate scores, selected structure, executable quotes, fills or rejection,
fees and fixed-strike exit. The implementation creates intents only.

## 6. Research calculation

Calculate cash P&L from entry and exit fills. Deduct half-spread on both
transactions plus brokerage, STT, exchange fees, SEBI/IPFT fees, stamp duty and
GST. If a historical fixed strike leaves the five-strike exit snapshot,
reconstruct it from the exit IV smile and parity-implied forward and mark it as
imputed.

Walk-forward models train only on observations earlier than the test month.
Selection occurs before observing test outcomes and includes no trade.

## 7. Promotion requirements

Remain paper-only until a band has at least 60 new eligible paper signals,
positive after-cost total P&L, one-sided day-bootstrap probability below 5%,
positive P&L in both halves of the forward sample, and executable quotes for
every leg.

