import { WET_M } from "./wet";

import type { InspectGrids } from "./inspect";

/**
 * A solved depth field, turned back into the two things the map draws from.
 *
 * The planner solves every option it offers, in one batch, and now returns
 * each option's field as a quantised grid. That grid has to reach the map two
 * ways -- as a body of water (ColumnLayer, from the grid itself) and as paint
 * on the terrain (TerrainLayer, which wants an image) -- so the ramp the
 * service uses for its own PNG is reproduced here rather than asking the
 * service for one picture per option. Same curve, same colours, same alpha
 * cut-off: if the two ever drift, a chosen option would be painted on a
 * different scale from the do-nothing baseline it is being compared against.
 */

/** Decode a base64 payload to bytes. */
function bytes(b64: string): Uint8Array {
  return Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
}

/**
 * A chosen option's depth, carrying the baseline's ground and metadata.
 *
 * Elevation does not depend on the plan -- a levee is a change to the bed the
 * solver integrates, not to the terrain the viewer stands on -- so the ground
 * grid is shared, and only the water changes.
 */
export function withDepth(base: InspectGrids, depthU8: string): InspectGrids {
  const depth = bytes(depthU8);
  if (depth.length !== base.depth.length) return base;
  return { meta: base.meta, depth, elev: base.elev };
}

/**
 * The depth ramp as a base64 PNG, without the `data:` prefix.
 *
 * Mirrors `depth_png` in the service: shallow water pale, deep water dark and
 * more opaque, gamma 0.42 so the first half-metre is visible at all. Returns
 * null where there is no canvas to draw on (server render), and the caller
 * falls back to the field the service painted.
 */
export function depthTexture(g: InspectGrids): string | null {
  if (typeof document === "undefined") return null;
  const [h, w] = g.meta.grid;
  if (g.depth.length < h * w) return null;
  const cv = document.createElement("canvas");
  cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d");
  if (!ctx) return null;
  const img = ctx.createImageData(w, h);
  const vmax = g.meta.depth_vmax_m;
  for (let i = 0; i < h * w; i++) {
    const u = g.depth[i] / 255;
    const x = Math.pow(u, 0.42);
    const o = i * 4;
    img.data[o] = 96 * (1 - x) + 4 * x;
    img.data[o + 1] = 200 * (1 - x) + 28 * x;
    img.data[o + 2] = 255 * (1 - x) + 128 * x;
    // Dry ground is transparent, not pale blue: a levee that keeps water off a
    // paddock has to leave the imagery showing through there.
    img.data[o + 3] = u * vmax > WET_M ? Math.min(255, 132 + 118 * x) : 0;
  }
  ctx.putImageData(img, 0, 0);
  return cv.toDataURL("image/png").split(",")[1] ?? null;
}
