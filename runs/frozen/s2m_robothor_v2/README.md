Frozen S2M-RoboTHOR configuration for RoboTHOR ObjectNav VAL (frozen 2026-10-26).
= dev v2: goal instruction "Find a <category words>.", view-discounted negative evidence (<=3 m, once per view),
revisit of the best-ever hypothesis when exploration ends; vantage fallback OFF (use_vantage=False).
Dev (24 eps, 2 RoboTHOR train scenes): v1 58.3 / v2 75.0 / v3(vantage) 66.7 SR; v2 vs v3 paired McNemar p=0.69.
