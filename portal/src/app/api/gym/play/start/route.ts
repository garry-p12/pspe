import { NextResponse } from "next/server";

/**
 * Hand-driven play against the REAL environment.
 *
 * Proxies the service's `/gym/play/start`. There is no browser-side model: a
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
    const r = await fetch(`${SERVICE}/gym/play/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(110000),
    });
    const notFound = r.status === 404 ? await r.clone().text() : "";
    // A 404 here is two different things. FastAPI answers a missing
    // ROUTE with the bare {"detail":"Not Found"}; a missing
    // ENVIRONMENT gets a named message from the handler. Treating
    // both as "restart the service" told the reader to restart over
    // a typo in an id.
    if (r.status === 404 && /"Not Found"/.test(notFound)) {
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
      // Unwrap the service's own message instead of forwarding its JSON.
      // FastAPI answers {"detail": "..."} and the page was printing the
      // braces and quotes at the reader, which is a stack trace with extra
      // steps.
      const t = await r.text();
      let detail = t.slice(0, 400);
      try {
        const j = JSON.parse(t);
        if (typeof j?.detail === "string") detail = j.detail;
      } catch {
        /* not JSON; the raw text is the best we have */
      }
      return NextResponse.json({ error: "gym_error", detail }, { status: 502 });
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
