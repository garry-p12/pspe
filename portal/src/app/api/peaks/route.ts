import { NextResponse } from "next/server";

/**
 * What this river has actually done, and what it does at 1% a year.
 *
 * Proxies the service's `/peaks`, which reads the USGS annual peak streamflow
 * record for the gauges on this reach and fits a log-Pearson III distribution
 * to it. Those two together are the scenario list: the floods this creek has
 * carried, with dates, and the fitted discharge for a given annual chance.
 *
 * Not a solve — a fetch and a moment-fit — but the USGS site query plus one
 * peak-record request per gauge is slow enough to want a long timeout on a
 * cold cache.
 */
const SERVICE = process.env.FLOOD_SERVICE ?? "http://127.0.0.1:8000";

export const maxDuration = 300;

export async function POST(req: Request) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "bad_request" }, { status: 400 });
  }
  try {
    const r = await fetch(`${SERVICE}/peaks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(280_000),
    });
    if (!r.ok) {
      return NextResponse.json(
        { error: "peaks_error", detail: (await r.text()).slice(0, 400) },
        { status: 502 },
      );
    }
    return NextResponse.json(await r.json());
  } catch (e) {
    return NextResponse.json(
      {
        error: "service_unavailable",
        detail: "The flood service is not reachable.",
        cause: String((e as Error).message).slice(0, 200),
      },
      { status: 503 },
    );
  }
}
