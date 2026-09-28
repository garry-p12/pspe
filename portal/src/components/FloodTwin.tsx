"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { FloodScene, type LeveeSite, type ViewPreset } from "./FloodScene";
import { Timeline } from "./Timeline";
import { Panel, Stat } from "./Panel";
import { EvidenceBadge } from "./EvidenceBadge";
import { loadManifest, loadMetrics, fmt } from "@/lib/data";
import type { Manifest, Metrics } from "@/lib/types";

export function FloodTwin() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [metrics, setMetrics] = useState<Metrics>({});
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [terrain3d, setTerrain3d] = useState(true);
  const [exaggeration, setExaggeration] = useState(6);
  const [selected, setSelected] = useState<number | null>(null);
  const [preset, setPreset] = useState<ViewPreset>("extent");
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    loadManifest()
      .then((m) => {
        setManifest(m);
        // Open at peak inundation: the still frame should show the event, not
        // a dry catchment on day one.
        const pk = m.frames.reduce((a, b) => (b.wet_cells > a.wet_cells ? b : a), m.frames[0]);
        setFrame(pk.i);
      })
      .catch((e) => setErr(String(e)));
    loadMetrics().then(setMetrics).catch(() => {});
  }, []);

  const sites: LeveeSite[] = useMemo(() => {
    const geom = manifest?.levee_geometry;
    const singles = metrics.sites?.levee_test?.singles;
    if (!geom || !singles) return [];
    return geom.sites.map((s) => ({
      id: s.id,
      lon: s.lon,
      lat: s.lat,
      elev: s.elev,
      reductionPct: singles.find((x) => x.site === s.id)?.reduction_pct ?? 0,
    }));
  }, [manifest, metrics]);

  // Selecting a site is a request to look at it: the ring is 1.4 km across in
  // a 32 km frame, so staying zoomed out would show six overlapping dots.
  const onSelect = useCallback((id: number | null) => {
    setSelected(id);
    if (id !== null) setPreset("settlement");
  }, []);

  if (err) {
    return (
      <div className="mx-auto max-w-[1500px] px-5 py-16">
        <p className="text-bad">Could not load the data bundle: {err}</p>
        <p className="mt-2 text-[15px] text-ink-mute">
          Run <code className="text-ink">scripts/build_portal_assets.py</code> and
          copy its output to <code className="text-ink">portal/public/data</code>.
        </p>
      </div>
    );
  }
  if (!manifest) {
    return (
      <div className="mx-auto max-w-[1500px] px-5 py-16">
        <div className="h-[60vh] animate-pulse rounded-xl bg-bg-raised" />
      </div>
    );
  }

  const f = manifest.frames[frame];
  const b = manifest.bounds;
  const test = metrics.sites?.levee_test;
  const val = metrics.validation;
  const sel = selected == null ? null : sites.find((s) => s.id === selected) ?? null;
  const harmful = sites.filter((s) => s.reductionPct < -0.05);

  return (
    <div className="mx-auto max-w-[1500px] px-5 py-6">
      <header className="mb-5">
        <p className="eyebrow mb-1">Flood digital twin</p>
        <h1 className="text-[30px] font-semibold leading-tight tracking-tight">
          {b.event}
        </h1>
        <p className="mt-1.5 max-w-3xl text-[15px] leading-relaxed text-ink-mute">
          {b.place} — {b.extent_km} km across at {manifest.native_dx_m} m. The
          Richmond River peaked 0.64 m above its 1974 record at Coraki and 1.8 m
          above its 1954 record at Woodburn. Terrain and water below are the
          recorded event; interventions are projections.
        </p>
      </header>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_340px]">
        <div className="flex flex-col gap-4">
          <Panel
            eyebrow="Recorded inundation"
            title="Flood depth over the Richmond floodplain"
            badge="OBSERVED"
            badgeDetail="FloodCastBench reference depths, shallow-water solution validated by its authors against SAR"
            bodyClassName="p-0"
            aside={
              <div className="flex items-center gap-1.5">
                {(["extent", "settlement"] as ViewPreset[]).map((p) => (
                  <button
                    key={p}
                    onClick={() => setPreset(p)}
                    className={[
                      "rounded-md border px-2 py-1 text-[12px] transition-colors",
                      preset === p
                        ? "border-accent/60 bg-accent/10 text-accent"
                        : "border-line text-ink-mute hover:text-ink",
                    ].join(" ")}
                  >
                    {p === "extent" ? "Full extent" : "Settlement"}
                  </button>
                ))}
                <button
                  onClick={() => setTerrain3d((v) => !v)}
                  className="rounded-md border border-line px-2 py-1 text-[12px] text-ink-mute transition-colors hover:border-accent/50 hover:text-accent"
                >
                  {terrain3d ? "3D terrain" : "Flat"}
                </button>
              </div>
            }
          >
            <div className="relative h-[clamp(320px,52vh,600px)] w-full overflow-hidden">
              <FloodScene
                manifest={manifest}
                frame={frame}
                exaggeration={exaggeration}
                sites={sites}
                selected={selected}
                onSelect={onSelect}
                showTerrain={terrain3d}
                preset={preset}
              />
              <div className="pointer-events-none absolute bottom-3 left-3 rounded-lg border border-line bg-[color-mix(in_srgb,var(--bg)_82%,transparent)] px-3 py-2 backdrop-blur">
                <p className="eyebrow mb-1">Depth</p>
                <div
                  className="h-2 w-[140px] rounded"
                  style={{
                    background:
                      "linear-gradient(90deg, rgba(160,220,255,.55), rgb(8,60,140))",
                  }}
                />
                <div className="tnum mt-1 flex justify-between text-[11px] text-ink-faint">
                  <span>0.01 m</span>
                  <span>{manifest.depth_vmax_m} m</span>
                </div>
              </div>
            </div>
            <div className="border-t border-line-soft px-4 py-3">
              <Timeline
                frames={manifest.frames}
                value={frame}
                onChange={setFrame}
                playing={playing}
                onPlayToggle={() => setPlaying((p) => !p)}
              />
              {terrain3d && (
                <div className="mt-3 flex items-center gap-3">
                  <span className="eyebrow">Vertical exaggeration</span>
                  <input
                    type="range"
                    min={1}
                    max={14}
                    value={exaggeration}
                    onChange={(e) => setExaggeration(Number(e.target.value))}
                    className="w-32 accent-[var(--accent)]"
                  />
                  <span className="tnum text-[12px] text-ink-faint">
                    {exaggeration}×
                  </span>
                  <span className="ml-auto text-[12px] text-ink-faint">
                    relief here is {fmt(manifest.dem_range_m[1] - manifest.dem_range_m[0], 0)} m
                    over {b.extent_km} km — flat ground floods
                  </span>
                </div>
              )}
            </div>
          </Panel>

          <div className="grid gap-4 sm:grid-cols-3">
            <Panel eyebrow="At this moment" title="Inundation" badge="OBSERVED">
              <div className="grid grid-cols-2 gap-4">
                <Stat
                  label="Wet area"
                  value={fmt((f.wet_cells * (manifest.raster_dx_m ** 2)) / 1e6, 0)}
                  unit="km²"
                  hint="cells above 1 cm"
                />
                <Stat label="Max depth" value={fmt(f.max_depth)} unit="m" />
              </div>
            </Panel>
            <Panel eyebrow="Our solver vs reference" title="Agreement" badge="VALIDATED MODEL"
              badgeDetail="free-running from the reference state, scored on the dataset's own metric">
              <div className="grid grid-cols-2 gap-4">
                <Stat
                  label="CSI @1 cm"
                  value={val ? fmt(val.mean_csi["@0.01"], 3) : "—"}
                  tone="good"
                  hint="Critical Success Index against FloodCastBench reference depths"
                />
                <Stat
                  label="RMSE"
                  value={val ? fmt(val.mean_rmse_wet, 3) : "—"}
                  unit="m"
                />
              </div>
            </Panel>
            <Panel eyebrow="Levee allocation" title="Best achievable" badge="PROJECTION"
              badgeDetail="effect of an intervention nobody performed; rests on a stated action model">
              <div className="grid grid-cols-2 gap-4">
                <Stat
                  label="Best single"
                  value={test ? `+${fmt(test.best_single_pct, 1)}` : "—"}
                  unit="%"
                  tone="good"
                />
                <Stat
                  label="All six"
                  value={test ? `+${fmt(test.all_sites_pct, 1)}` : "—"}
                  unit="%"
                  tone="good"
                />
              </div>
            </Panel>
          </div>
        </div>

        <aside className="flex flex-col gap-4">
          <Panel
            eyebrow="Try it"
            title="Where would you put a levee?"
            badge="PROJECTION"
            badgeDetail="each figure is a solver run with that levee in place; no such levee exists"
          >
            <p className="mb-3 text-[13px] leading-relaxed text-ink-mute">
              Six candidate sites ring the settlement, each at the lowest point of
              its perimeter sector. Click one on the map, or below.
            </p>
            <ul className="flex flex-col gap-1.5">
              {sites.map((s) => {
                const bad = s.reductionPct < -0.05;
                const good = s.reductionPct > 3;
                return (
                  <li key={s.id}>
                    <button
                      onClick={() => onSelect(selected === s.id ? null : s.id)}
                      className={[
                        "flex w-full items-center gap-3 rounded-lg border px-3 py-2 text-left transition-colors",
                        selected === s.id
                          ? "border-accent/60 bg-accent/10"
                          : "border-line bg-bg-inset hover:border-line-soft",
                      ].join(" ")}
                    >
                      <span className="tnum w-7 text-[12px] text-ink-faint">
                        S{s.id}
                      </span>
                      <span className="tnum w-14 text-[12px] text-ink-mute">
                        {fmt(s.elev, 1)} m
                      </span>
                      <span
                        className={[
                          "tnum ml-auto text-[15px] font-semibold",
                          bad ? "text-bad" : good ? "text-ok" : "text-ink-mute",
                        ].join(" ")}
                      >
                        {s.reductionPct >= 0 ? "+" : ""}
                        {fmt(s.reductionPct, 2)}%
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>

            {sel && (
              <div className="mt-3 rounded-lg border border-line bg-bg-inset p-3">
                <p className="text-[13px] leading-relaxed text-ink-mute">
                  {sel.reductionPct < -0.05 ? (
                    <>
                      <b className="text-bad">This levee makes flooding worse.</b>{" "}
                      Site {sel.id} sits on a drainage path. Blocking it holds
                      water in rather than keeping it out — the levee backwater
                      effect, and the reason a planner that treats every actuator
                      as beneficial can harm what it defends.
                    </>
                  ) : sel.reductionPct > 3 ? (
                    <>
                      <b className="text-ok">This one works.</b> Site {sel.id}{" "}
                      intercepts the inflow rather than the outflow. It is the
                      only sector of six that does.
                    </>
                  ) : (
                    <>Site {sel.id} changes little: water reaches the settlement by
                    other paths, so closing this sector buys almost nothing.</>
                  )}
                </p>
              </div>
            )}
          </Panel>

          {harmful.length > 0 && (
            <Panel eyebrow="Measured, not hypothetical" title="Half these levees backfire">
              <p className="text-[13px] leading-relaxed text-ink-mute">
                <b className="text-ink">{harmful.length} of {sites.length}</b> candidate
                sites <b className="text-bad">increase</b> flooding at the
                settlement. Each is downstream or on a drainage path, so a wall
                there traps water instead of excluding it.
              </p>
              <p className="mt-2 text-[13px] leading-relaxed text-ink-mute">
                This is why the planner carries a constraint and a calibrated
                margin rather than a reward alone: the sign of an intervention is
                not obvious from the map, and getting it wrong is not a wasted
                budget but an added harm.
              </p>
            </Panel>
          )}

          <Panel eyebrow="Provenance" title="What you are looking at">
            <dl className="flex flex-col gap-2.5 text-[13px]">
              {Object.entries(manifest.provenance).map(([k, v]) => (
                <div key={k}>
                  <dt className="mb-1 flex items-center gap-2">
                    <span className="text-ink">{k}</span>
                    <EvidenceBadge badge={v.badge} />
                  </dt>
                  <dd className="leading-relaxed text-ink-faint">{v.note}</dd>
                </div>
              ))}
            </dl>
          </Panel>
        </aside>
      </div>
    </div>
  );
}
