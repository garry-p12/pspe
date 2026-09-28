"use client";

import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Tooltip, XAxis, YAxis, Legend,
} from "recharts";
import { Panel, Stat } from "./Panel";
import { loadMetrics, fmt } from "@/lib/data";
import type { Metrics } from "@/lib/types";

const AXIS = { stroke: "#5d6a7d", fontSize: 11 };
const GRID = "#1d232d";

export function MarginExplorer() {
  const [m, setM] = useState<Metrics | null>(null);
  const [f, setF] = useState(0.3);

  useEffect(() => { loadMetrics().then(setM).catch(() => setM({})); }, []);

  const pre = m?.precond_synthetic ?? [];
  const el = m?.elasticity?.rows ?? [];

  // Recompute feasibility live: the headroom is f x span, and f is the
  // trade-off the operator chooses, not a constant of the method.
  const preRows = useMemo(
    () => pre.map((d) => ({
      spread: d.q_log_sigma * 100,
      required: d.required_margin,
      headroom: f * d.span,
      feasible: d.required_margin < f * d.span,
    })),
    [pre, f],
  );

  // Recharts' log scale renders NOTHING with domain={["auto","auto"]} -- it
  // needs explicit numeric bounds. Derive them from the data with a little
  // headroom rather than hard-coding.
  const yDomain = useMemo<[number, number]>(() => {
    const vs = preRows.flatMap((r) => [r.required, r.headroom]).filter((v) => v > 0);
    if (!vs.length) return [0.01, 1];
    return [Math.min(...vs) * 0.6, Math.max(...vs) * 1.6];
  }, [preRows]);

  const crossing = useMemo(() => {
    for (let i = 1; i < preRows.length; i++) {
      const a = preRows[i - 1], b = preRows[i];
      const da = a.required - a.headroom, db = b.required - b.headroom;
      if (da < 0 && db >= 0) {
        return a.spread + (b.spread - a.spread) * (-da / (db - da));
      }
    }
    return null;
  }, [preRows]);

  const elRows = useMemo(
    () => el
      .filter((r) => Number.isFinite(r.elasticity))
      .map((r) => ({ wet: r.berm_rows_wet, E: r.elasticity })),
    [el],
  );
  const eDomain = useMemo<[number, number]>(() => {
    const vs = elRows.map((r) => r.E).filter((v) => v > 0);
    if (!vs.length) return [1, 50];
    return [Math.min(...vs) * 0.7, Math.max(...vs) * 1.4];
  }, [elRows]);

  if (!m) {
    return <div className="mx-auto max-w-[1200px] px-5 py-16">
      <div className="h-[50vh] animate-pulse rounded-xl bg-bg-raised" />
    </div>;
  }

  return (
    <div className="mx-auto max-w-[1200px] px-5 py-8">
      <header className="mb-6 max-w-3xl">
        <p className="eyebrow mb-2">Safety margin</p>
        <h1 className="text-[30px] font-semibold leading-tight tracking-tight">
          A margin is only usable when it fits
        </h1>
        <p className="mt-3 text-[15px] leading-relaxed text-ink-mute">
          To respect a limit at failure rate δ, the planner tightens it by a
          calibrated margin. That margin has to come out of the gain the
          intervention was going to buy. If it exceeds that gain there is nothing
          left to plan with — so feasibility is a precondition you can check
          before committing, not something to discover afterwards.
        </p>
        <p className="mt-3 rounded-lg border border-line bg-bg-inset px-3 py-2 font-mono text-[13px] text-ink">
          b + z<sub>δ</sub>·σ &nbsp;&lt;&nbsp; f · s
        </p>
      </header>

      <div className="mb-4 grid gap-4 lg:grid-cols-[minmax(0,1fr)_300px]">
        <Panel
          eyebrow="Fluvial testbed"
          title="Required margin against available headroom"
          badge="VALIDATED MODEL"
          badgeDetail="each point is a calibration run against an accepted solver"
        >
          <div className="h-[290px] w-full">
            <ResponsiveContainer>
              <LineChart data={preRows} margin={{ top: 6, right: 12, bottom: 18, left: 4 }}>
                <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
                <XAxis
                  dataKey="spread" type="number" domain={[0, 32]}
                  ticks={[0, 5, 10, 15, 20, 25, 30]} {...AXIS}
                  label={{ value: "forecast spread (%)", position: "insideBottom",
                           offset: -10, fill: "#5d6a7d", fontSize: 11 }}
                />
                <YAxis
                  scale="log" domain={yDomain} allowDataOverflow {...AXIS}
                  tickFormatter={(v: number) => Number(v).toFixed(2)}
                  label={{ value: "depth (m)", angle: -90, position: "insideLeft",
                           fill: "#5d6a7d", fontSize: 11 }}
                />
                <Tooltip
                  contentStyle={{ background: "#11151c", border: "1px solid #1d232d",
                                  borderRadius: 8, fontSize: 12 }}
                  labelFormatter={(v) => `${v}% spread`}
                  formatter={(v, n) => [`${fmt(Number(v), 4)} m`, String(n)]}
                />
                <Legend wrapperStyle={{ fontSize: 11, paddingTop: 6 }} />
                <Line type="monotone" dataKey="required" name="required margin"
                      stroke="#fb923c" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="headroom" name={`headroom (f=${f})`}
                      stroke="#34d399" strokeWidth={2} dot={{ r: 3 }} />
                {crossing && (
                  <ReferenceLine x={crossing} stroke="#e8edf4" strokeDasharray="3 3"
                    label={{ value: `${crossing.toFixed(1)}%`, fill: "#e8edf4",
                             fontSize: 11, position: "top" }} />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <div className="flex flex-col gap-4">
          <Panel eyebrow="The trade-off you choose" title="Fraction of the gain spent on safety">
            <input
              type="range" min={0.1} max={0.6} step={0.05} value={f}
              onChange={(e) => setF(Number(e.target.value))}
              className="w-full accent-[var(--accent)]"
            />
            <div className="tnum mt-1 flex justify-between text-[12px] text-ink-faint">
              <span>10%</span><span className="text-ink">f = {fmt(f, 2)}</span><span>60%</span>
            </div>
            <p className="mt-3 text-[13px] leading-relaxed text-ink-mute">
              f is how much of the achievable improvement you are willing to give
              back for a guarantee. It is a decision, not a constant — and moving
              it moves the crossing.
            </p>
          </Panel>
          <Panel eyebrow="Result" title="Forecast accuracy required">
            <Stat
              label="Crossing"
              value={crossing ? fmt(crossing, 1) : "—"}
              unit="% spread"
              tone={crossing ? "warn" : "default"}
            />
            <p className="mt-2.5 text-[13px] leading-relaxed text-ink-mute">
              Below this, a δ=0.1 margin fits inside the gain. Above it, the
              margin consumes more than the intervention was worth. This is the
              number a forecast product can be held to.
            </p>
          </Panel>
        </div>
      </div>

      <Panel
        eyebrow="Why it fails where it matters"
        title="Forecast error is amplified exactly at the threshold"
        badge="VALIDATED MODEL"
      >
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
          <div className="h-[240px] w-full">
            <ResponsiveContainer>
              <LineChart data={elRows} margin={{ top: 6, right: 12, bottom: 18, left: 4 }}>
                <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
                <XAxis dataKey="wet" {...AXIS}
                  label={{ value: "extent over-topped (of 96)", position: "insideBottom",
                           offset: -10, fill: "#5d6a7d", fontSize: 11 }} />
                <YAxis scale="log" domain={eDomain} allowDataOverflow {...AXIS}
                  ticks={[2, 5, 10, 20, 40]} tickFormatter={(v: number) => `${v}×`} />
                <Tooltip
                  contentStyle={{ background: "#11151c", border: "1px solid #1d232d",
                                  borderRadius: 8, fontSize: 12 }}
                  formatter={(v) => [`${fmt(Number(v), 1)}×`, "amplification"]} />
                <Line type="monotone" dataKey="E" stroke="#38bdf8" strokeWidth={2}
                      dot={{ r: 3 }} name="amplification" />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div>
            <p className="text-[13px] leading-relaxed text-ink-mute">
              A structure only matters when the water is near its crest — and that
              is precisely where depth is most sensitive to a forecast error. A
              10% error in discharge becomes a <b className="text-ink">four-fold</b>{" "}
              error in depth at the threshold, decaying to about 2× once the
              defence is comprehensively over-topped.
            </p>
            <p className="mt-3 text-[13px] leading-relaxed text-ink-mute">
              So the regime where an intervention has leverage is the regime where
              a calibrated margin is hardest to fit. That tension is structural,
              and it is not specific to flood.
            </p>
          </div>
        </div>
      </Panel>
    </div>
  );
}
