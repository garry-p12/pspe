import { NextResponse } from "next/server";
import { readFile } from "node:fs/promises";
import path from "node:path";

/**
 * Mitigation options with their modelled impact.
 *
 * The hydrodynamic model runs on a cluster, not in this process: a shallow-water
 * solve over the floodplain takes tens of minutes, so every option is solved
 * ahead of time and this endpoint serves the results. The client never sees a
 * raster, a depth field, or a model parameter — only consequences.
 */
export async function GET() {
  try {
    const p = path.join(process.cwd(), "public", "data", "operational.json");
    const raw = await readFile(p, "utf8");
    return NextResponse.json(JSON.parse(raw), {
      headers: { "Cache-Control": "public, max-age=60" },
    });
  } catch {
    return NextResponse.json(
      { error: "options_unavailable", detail: "Scenario library not built yet." },
      { status: 503 },
    );
  }
}
