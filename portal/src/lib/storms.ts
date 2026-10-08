/**
 * The three storms the on-demand views offer.
 *
 * Shared rather than duplicated: the trip view and the cross-section derive
 * their domains the same way and share a solve cache, so two copies of this
 * table that drifted apart would silently become two cache keys and pay for
 * the same flood twice.
 */
export const STORM_PRESETS = [
  { mm_h: 20, hours: 6, label: "Heavy rain", note: "120 mm — a wet day" },
  { mm_h: 50, hours: 6, label: "Severe storm", note: "300 mm — major flooding" },
  { mm_h: 80, hours: 6, label: "Extreme", note: "480 mm — near the 2022 event" },
] as const;
