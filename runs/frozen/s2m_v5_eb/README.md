Frozen S2M configuration for the EB-Navigation TEST runs (frozen 2026-10-26 after dev runs v2-v5).
Selected: v5 defaults of embagent/planners/s2m.py (S2MPlanner(model_name) with no overrides):
merge_radius 0.5, commit 0.8, keep 0.4, inspect 0.35, p_present 0.5, zoom_accept 0.5, neg_weight 0.4,
tile_grid 3, max_tile_boxes 4, max_range 6.0, scan-before-inspect, appearance-only spec handling,
optimal-search index selection, use_contrast False.
Dev (150 eps, Qwen3-VL-4B-AWQ): v4 76.0 / v5 74.7 (paired McNemar p=0.73); v3 vs v5 p=0.42.
Later code edits only add RoboTHOR hooks (overridable _view_key, cam/fov params) that leave EB behaviour unchanged.
