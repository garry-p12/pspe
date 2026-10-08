import { NextResponse } from "next/server";

/**
 * Live river gauges over a box.
 *
 * Proxies the service's `/gauges`, which reads USGS instantaneous values on a
 * 15-minute cadence and converts feet and cubic feet per second into metres
 * and cubic metres per second — a portal that puts a gauge reading beside a
 * solved depth has to convert, or it labels two different quantities the same
 * way.
 *
 * United States only, and the response says so rather than coming back empty:
 * the Richmond study has no equivalent open feed.
 */
const SERVICE = process.env.FLOOD_SERVICE ?? "http://127.0.0.1:8000";

export const maxDuration = 120;

export async function POST(req: Request) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "bad_request" }, { status: 400 });
  }
  try {
    const r = await fetch(`${SERVICE}/gauges`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(110_000),
    });
    if (!r.ok) {
      return NextResponse.json(
        { error: "gauge_error", detail: (await r.text()).slice(0, 300) },
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
