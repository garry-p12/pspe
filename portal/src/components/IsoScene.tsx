"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

/**
 * The environment's own state, as a block of ground you can turn.
 *
 * Everything drawn here is a number the simulator produced: relief from the
 * terrain, depth from the solver, and the two masks the reward is computed
 * over. There are deliberately no houses and no trees. They would look like
 * the reference sketch and they would be the only marks on the page that
 * stand for nothing, on a page whose argument is that each mark is a
 * measurement — and a viewer has no way to tell a decorative house from a
 * modelled one.
 *
 * Projection is a yaw about the vertical followed by an isometric cast, with
 * the painter's algorithm for order. SVG rather than WebGL: at 16x16 it is a
 * few hundred polygons, it inherits the page's colour tokens, and it needs no
 * dependency.
 */

export type Scene = {
  ground: number[][];
  water: number[][];
  protect: number[][];
  marker: number[][];
  legend: { ground: string; water: string; marker: string };
  water_unit: string;
  /** Depth that fills the drawn water column. Declared per environment, and
   *  printed beside the legend, because an unlabelled rescale is how a
   *  0.18 m flood comes to look like a catastrophe. */
  water_scale?: number;
};

const VB = { w: 760, h: 430 };
const TILE = 23;          // half-width of a cell's top face, before yaw
const RELIEF = 54;        // pixels for the full height range of the terrain
const WATER = 62;         // pixels for a full water column (see `water_scale`)
const BASE = 26;          // the slab the ground sits on, so it reads as a model

type Cell = {
  key: string; depth: number;
  top: string; left: string; right: string;
  x: number; y: number; z0: number; z1: number;
};

function shade(hex: string, k: number): string {
  const n = parseInt(hex.slice(1), 16);
  const f = (v: number) => Math.max(0, Math.min(255, Math.round(v * k)));
  return `rgb(${f((n >> 16) & 255)},${f((n >> 8) & 255)},${f(n & 255)})`;
}

export function IsoScene({
  scene, yaw, onYaw, labels,
}: {
  scene: Scene; yaw: number; onYaw: (y: number) => void;
  labels?: { text: string; sub: string; r: number; c: number }[];
}) {
  const svg = useRef<SVGSVGElement | null>(null);
  const drag = useRef<{ x: number; yaw: number } | null>(null);

  const n = scene.ground.length;
  const hi = useMemo(() => {
    let m = 0;
    for (const row of scene.ground) for (const v of row) m = Math.max(m, v);
    return m || 1;
  }, [scene.ground]);

  // Raw projection: yaw about the vertical, then an isometric cast. No
  // centring here -- the whole scene is fitted afterwards, because its
  // footprint changes with every degree of yaw and with how deep the water
  // is, and a fixed offset leaves it hanging off the top corner at some
  // angles. Which it did.
  const project = useCallback((x: number, y: number, z: number) => {
    const s = Math.sin(yaw), c = Math.cos(yaw);
    const rx = x * c - y * s;
    const ry = x * s + y * c;
    return { sx: (rx - ry) * TILE, sy: (rx + ry) * TILE * 0.5 - z, d: rx + ry };
  }, [yaw]);

  const cells = useMemo<Cell[]>(() => {
    const out: Cell[] = [];
    const mid = (n - 1) / 2;
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const g = scene.ground[r][c] / hi;
        const w = scene.water[r][c];
        const isProtect = scene.protect[r][c] > 0.35;
        const isMarker = scene.marker[r][c] > 0.35;
        // Ground first. Marker (the channel, or where a levee may go) reads
        // as a structure, protected ground as the thing being defended.
        const base = isMarker ? "#a89878" : isProtect ? "#88a86a" : "#c9c5bb";
        const gz = BASE + g * RELIEF;
        out.push({
          key: `g${r}-${c}`, depth: 0, x: c - mid, y: r - mid, z0: 0, z1: gz,
          top: base, left: shade(base, 0.78), right: shade(base, 0.62),
        });
        const wScale = scene.water_scale ?? 1;
        if (w > wScale * 0.01) {
          const wz = Math.min((w / wScale) * WATER, 86);
          out.push({
            key: `w${r}-${c}`, depth: w, x: c - mid, y: r - mid,
            z0: gz, z1: gz + wz,
            top: "#5b9bd5", left: shade("#5b9bd5", 0.8), right: shade("#5b9bd5", 0.64),
          });
        }
      }
    }
    return out;
  }, [scene, n, hi]);

  const ordered = useMemo(() => {
    return cells
      .map((cell) => ({ cell, d: project(cell.x, cell.y, 0).d }))
      .sort((a, b) => a.d - b.d || a.cell.z0 - b.cell.z0)
      .map((o) => o.cell);
  }, [cells, project]);

  /** Scale and offset that put the whole model inside the frame. */
  const fit = useMemo(() => {
    const PAD = 18;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    const h = 0.5;
    for (const cell of cells) {
      for (const [dx, dy] of [[-h, -h], [h, -h], [h, h], [-h, h]] as const) {
        for (const z of [cell.z0, cell.z1]) {
          const q = project(cell.x + dx, cell.y + dy, z);
          if (q.sx < x0) x0 = q.sx;
          if (q.sx > x1) x1 = q.sx;
          if (q.sy < y0) y0 = q.sy;
          if (q.sy > y1) y1 = q.sy;
        }
      }
    }
    if (!Number.isFinite(x0)) return { s: 1, tx: VB.w / 2, ty: VB.h / 2 };
    const s = Math.min((VB.w - 2 * PAD) / Math.max(x1 - x0, 1),
                       (VB.h - 2 * PAD) / Math.max(y1 - y0, 1));
    return {
      s,
      tx: VB.w / 2 - ((x0 + x1) / 2) * s,
      ty: VB.h / 2 - ((y0 + y1) / 2) * s,
    };
  }, [cells, project]);

  useEffect(() => {
    const move = (e: PointerEvent) => {
      if (!drag.current) return;
      onYaw(drag.current.yaw + (e.clientX - drag.current.x) * 0.006);
    };
    const up = () => { drag.current = null; };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
  }, [onYaw]);

  const face = (cell: Cell) => {
    const h = 0.5;
    const a = project(cell.x - h, cell.y - h, cell.z1);
    const b = project(cell.x + h, cell.y - h, cell.z1);
    const c = project(cell.x + h, cell.y + h, cell.z1);
    const d = project(cell.x - h, cell.y + h, cell.z1);
    const c0 = project(cell.x + h, cell.y + h, cell.z0);
    const d0 = project(cell.x - h, cell.y + h, cell.z0);
    const b0 = project(cell.x + h, cell.y - h, cell.z0);
    return (
      <g key={cell.key}>
        <polygon points={`${d.sx},${d.sy} ${c.sx},${c.sy} ${c0.sx},${c0.sy} ${d0.sx},${d0.sy}`}
                 fill={cell.left} />
        <polygon points={`${b.sx},${b.sy} ${c.sx},${c.sy} ${c0.sx},${c0.sy} ${b0.sx},${b0.sy}`}
                 fill={cell.right} />
        <polygon points={`${a.sx},${a.sy} ${b.sx},${b.sy} ${c.sx},${c.sy} ${d.sx},${d.sy}`}
                 fill={cell.top} />
      </g>
    );
  };

  return (
    <svg
      ref={svg}
      viewBox={`0 0 ${VB.w} ${VB.h}`}
      className="w-full cursor-grab touch-none select-none active:cursor-grabbing"
      role="img"
      aria-label="The simulated ground, with water where the solver put it. Drag to turn."
      onPointerDown={(e) => { drag.current = { x: e.clientX, yaw }; }}
    >
      <g transform={`translate(${fit.tx} ${fit.ty}) scale(${fit.s})`}>
        {ordered.map(face)}
      </g>
      {labels?.map((l) => {
        const mid = (n - 1) / 2;
        const g = scene.ground[Math.min(l.r, n - 1)]?.[Math.min(l.c, n - 1)] ?? 0;
        const q = project(l.c - mid, l.r - mid, BASE + (g / hi) * RELIEF + 26);
        const p = { sx: fit.tx + q.sx * fit.s, sy: fit.ty + q.sy * fit.s };
        return (
          <g key={l.text} transform={`translate(${p.sx} ${p.sy})`}>
            <circle cx={-34} cy={-10} r={3} className="fill-ink" />
            <rect x={-44} y={-24} width={Math.max(l.text.length, l.sub.length) * 6.6 + 28}
                  height={l.sub ? 30 : 19} rx={4}
                  className="fill-bg stroke-line" strokeWidth={1} opacity={0.96} />
            <text x={-26} y={-12} className="fill-ink text-[11px] font-semibold">{l.text}</text>
            {l.sub && <text x={-26} y={0} className="fill-ink-faint text-[10px]">{l.sub}</text>}
          </g>
        );
      })}
    </svg>
  );
}
