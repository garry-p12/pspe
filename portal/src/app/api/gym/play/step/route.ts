import { NextResponse } from "next/server";

/**
 * Hand-driven play against the REAL environment.
 *
 * Proxies the service's `/gym/play/step`. There is no browser-side model: a
 * step of `FloodOperate-v0` is 0.04 s of the same local-inertial solver the
 * research runs, so the water the page draws is the water the benchmark
 * measured. That is why this page needs no "numbers will differ" disclaimer.
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
    const r = await fetch(`${SERVICE}/gym/play/step`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(110000),
    });
    if (r.status === 404) {
      return NextResponse.json(
        {
          error: "gym_routes_missing",
          detail:
            "The service is running but has no /gym/play routes, so it was " +
            "started before they existed. Restart it: " +
            "python3 -m uvicorn service.app:app --port 8000",
        },
        { status: 502 },
      );
    }
    if (!r.ok) {
      const t = await r.text();
      return NextResponse.json(
        { error: "gym_error", detail: t.slice(0, 400) },
        { status: 502 },
      );
    }
    return NextResponse.json(await r.json());
  } catch (e) {
    return NextResponse.json(
      {
        error: "service_unavailable",
        detail:
          "The hazard service is not reachable. Start it with: " +
          "python3 -m uvicorn service.app:app --port 8000",
        cause: String((e as Error).message).slice(0, 200),
      },
      { status: 503 },
    );
  }
}
