import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

/**
 * The browser's router and the paper's router have to be the same router.
 *
 * `pspe/plan/routes.py` produces the numbers in section 5.13; `src/lib/routes.ts`
 * produces the ones an operator reads. Two implementations of the same
 * safety-relevant calculation is a drift risk, and the honest way to run two is
 * to check them against each other on the same inputs rather than to assert in
 * a comment that they agree.
 *
 * Compiles the TypeScript, runs both over the district's published network and
 * depth grid, and compares the graph, the per-edge depths and a route.
 *
 *     node scripts/verify-route-parity.mjs
 */

const REPO = "/Users/guruprasadparasnis/pspe";
// The interpreter that has this project's dependencies, which is not
// necessarily the first python3 on PATH when node comes from Homebrew.
const PYTHON = process.env.PYTHON ?? "/usr/bin/python3";
const DATA = `${REPO}/portal/public/data`;
const TSC = `${REPO}/portal/node_modules/.bin/tsc`;

// A fixed trip, named by coordinates rather than by node index, so neither side
// depends on the other's node numbering.
const FROM = JSON.parse(readFileSync(`${DATA}/manifest.json`, "utf8"))
  .levee_geometry.settlement;
const TO = { lat: -29.0449, lon: 153.3362 };
const LIMIT = 0.25;

const outDir = mkdtempSync(join(tmpdir(), "route-parity-"));
execFileSync(TSC, [
  "src/lib/routes.ts", "src/lib/inspect.ts",
  "--outDir", outDir, "--module", "es2022", "--target", "es2022",
  "--moduleResolution", "bundler", "--skipLibCheck",
  "--types", "geojson",
], { cwd: `${REPO}/portal`, stdio: "inherit" });

// tsc flattens to the common root of the inputs, so the files land beside each
// other rather than under src/lib; and it emits the bundler-style specifiers
// verbatim, which Node's ESM loader will not resolve without the extension.
for (const f of ["routes.js", "inspect.js"]) {
  const at = join(outDir, f);
  writeFileSync(at, readFileSync(at, "utf8")
    .replace(/from "\.\/([a-z]+)"/g, 'from "./$1.js"'));
}

const { buildGraph, edgeDepths, fastest, nearestNode, cutKm } =
  await import(join(outDir, "routes.js"));

const roads = JSON.parse(readFileSync(`${DATA}/roads.geojson`, "utf8"));
const meta = JSON.parse(readFileSync(`${DATA}/inspect.json`, "utf8"));
const bounds = JSON.parse(readFileSync(`${DATA}/manifest.json`, "utf8")).bounds;
const depthBytes = new Uint8Array(readFileSync(`${DATA}/peak_depth.u8`));
const elevBuf = readFileSync(`${DATA}/elevation.u16`);
const grids = {
  meta,
  depth: depthBytes,
  elev: new Uint16Array(elevBuf.buffer, elevBuf.byteOffset, elevBuf.byteLength / 2),
};

const g = buildGraph(roads);
const depths = edgeDepths(g, grids, bounds);
const src = nearestNode(g, FROM.lon, FROM.lat);
const dst = nearestNode(g, TO.lon, TO.lat);
const route = fastest(g, src, dst, depths, LIMIT);
const dry = fastest(g, src, dst, depths.map(() => 0), LIMIT);

const ts = {
  nodes: g.nodes.length,
  edges: g.edges.length,
  total_km: +g.totalKm.toFixed(4),
  depth_sum: +depths.reduce((s, d) => s + d, 0).toFixed(4),
  deepest_edge: +Math.max(...depths).toFixed(4),
  cut_km: +cutKm(g, depths, LIMIT).toFixed(4),
  route_ok: route.ok,
  route_minutes: route.ok ? +route.minutes.toFixed(4) : null,
  route_km: route.ok ? +route.km.toFixed(4) : null,
  route_edges: route.ok ? route.edges.length : 0,
  dry_minutes: +dry.minutes.toFixed(4),
  dry_km: +dry.km.toFixed(4),
  dry_points: dry.coords.length,
  dry_via: dry.via.slice(0, 4),
};

const py = `
import json, sys, numpy as np
sys.path.insert(0, ${JSON.stringify(REPO)})
from pspe.plan import routes
D = ${JSON.stringify(DATA)}
roads = json.load(open(D + "/roads.geojson"))
meta = json.load(open(D + "/inspect.json"))
man = json.load(open(D + "/manifest.json"))
b = man["bounds"]
h, w = meta["grid"]; vmax = meta["depth_vmax_m"]
d = np.frombuffer(open(D + "/peak_depth.u8", "rb").read(), dtype=np.uint8).reshape(h, w) / 255 * vmax
def at(lon, lat):
    fx = (lon - b["west"]) / (b["east"] - b["west"])
    fy = (b["north"] - lat) / (b["north"] - b["south"])
    if not (0 <= fx < 1 and 0 <= fy < 1): return 0.0
    return float(d[min(h - 1, int(fy * h)), min(w - 1, int(fx * w))])
g = routes.build_graph(roads["features"])
depths = routes.edge_depths(g, at)
src = routes.nearest_node(g, ${FROM.lon}, ${FROM.lat})
dst = routes.nearest_node(g, ${TO.lon}, ${TO.lat})
r = routes.fastest(g, src, dst, depths, ${LIMIT})
dry = routes.fastest(g, src, dst, [0.0] * len(g.edges), ${LIMIT})
print(json.dumps({
 "nodes": len(g.nodes), "edges": len(g.edges),
 "total_km": round(g.total_km, 4),
 "depth_sum": round(sum(depths), 4),
 "deepest_edge": round(max(depths), 4),
 "cut_km": round(routes.cut_km(g, depths, ${LIMIT}), 4),
 "route_ok": r.ok,
 "route_minutes": round(r.minutes, 4) if r.ok else None,
 "route_km": round(r.km, 4) if r.ok else None,
 "route_edges": len(r.edges) if r.ok else 0,
 "dry_minutes": round(dry.minutes, 4),
 "dry_km": round(dry.km, 4),
 "dry_points": len(dry.coords(g)),
 "dry_via": dry.names(g)[:4],
}))
`;
const pyFile = join(outDir, "parity.py");
writeFileSync(pyFile, py);
const raw = execFileSync(PYTHON, [pyFile], {
  cwd: REPO, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"],
});
const ref = JSON.parse(raw.trim().split("\n").pop());

let bad = 0;
console.log("| quantity | python | typescript | agree |");
console.log("|---|---|---|---|");
for (const k of Object.keys(ref)) {
  const a = ref[k], b = ts[k];
  // Lengths come from the same formula but through different float paths, so
  // metres-scale quantities are compared to a tolerance rather than exactly.
  const tol = typeof a === "number" && Math.abs(a) > 1 ? 1e-3 : 1e-6;
  const same = typeof a === "number" && typeof b === "number"
    ? Math.abs(a - b) <= tol * Math.max(1, Math.abs(a))
    : JSON.stringify(a) === JSON.stringify(b);
  if (!same) bad++;
  console.log(`| ${k} | ${JSON.stringify(a)} | ${JSON.stringify(b)} | ${same ? "yes" : "NO"} |`);
}
console.log(bad === 0 ? "\nPASS — the two routers agree" : `\nFAIL — ${bad} disagreements`);
process.exit(bad === 0 ? 0 : 1);
