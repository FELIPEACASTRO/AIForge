# Credit Scoring

This directory covers AI/ML for credit risk, underwriting, borrower scoring, adverse-action explainability, portfolio monitoring, and fair-lending review.

## Scope

- Application, behavioral, bureau, alternative-data, cash-flow, and small-business credit models.
- Scorecards, gradient boosting, neural models, survival models, reject inference, calibration, and decision thresholds.
- Fair lending, explainability, adverse-action reasons, model-risk management, and monitoring.

## Reference Links

- CFPB circular on complex credit algorithms: https://www.consumerfinance.gov/compliance/circulars/circular-2022-03-adverse-action-notification-requirements-in-connection-with-credit-decisions-based-on-complex-algorithms/
- CFPB black-box credit model statement: https://www.consumerfinance.gov/about-us/newsroom/cfpb-acts-to-protect-the-public-from-black-box-credit-models-using-complex-algorithms/
- Federal Reserve revised model risk guidance: https://www.federalreserve.gov/supervisionreg/srletters/SR2602.pdf
- OCC model risk management bulletin: https://www.occ.gov/news-issuances/bulletins/2026/bulletin-2026-13.html
- Federal Reserve responsible AI speech: https://www.federalreserve.gov/newsevents/speech/brainard20210112a.htm

## Documentation Standard

Record target definition, observation window, performance window, reject population, protected-class proxy analysis, adverse-action explainability, monitoring cadence, and governance owner.

## Routing Rules

- Put fraud and account abuse in `../Fraud_Detection/`.
- Put market, liquidity, and operational risk in `../Risk_Management/`.
- Put generic fairness methods in AI fundamentals or model evaluation.
