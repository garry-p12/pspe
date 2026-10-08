/** What counts as flooded, in metres.
 *
 * One number for what is drawn and what is counted. There used to be two: the
 * overlay painted every cell over 1 cm while the area metric counted cells
 * over 10 cm, so the Austin study showed 63 km² of water on the map beside a
 * panel reading 18.71 km² — both honest, three times apart, and the first
 * thing a visitor sees. A centimetre over a 40 m cell is 16 m³ of water; it is
 * wet ground, not a flood.
 *
 * Kept in step with `WET_M` in `service/app.py`. The solver's own `depth_min`
 * of 1 mm is a numerical floor for wet/dry interfaces, not this.
 */
export const WET_M = 0.1;

/** How to say it, for a caption that has to explain the number beside it. */
export const WET_LABEL = "water over 10 cm";
