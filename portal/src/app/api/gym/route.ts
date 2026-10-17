import { NextResponse } from "next/server";

/**
 * The four environments, with the ladder of policies measured on each.
 *
 * Proxies the service's `/gym/environments`. Everything prose-shaped in the
 * response comes from `pspe/envs/catalog.py`, which `scripts/score_gym.py
 * --ladder --check` tests, so this page has no second copy of any claim.
 *
 * `probe=1` asks the service to construct each environment and report its real
 * observation and action spaces. That needs the egress table and the fire
 * checkpoint on disk and takes a few seconds, so the page asks for it once on
 * load rather than on every render.
 */
const SERVICE = process.env.FLOOD_SERVICE ?? "http://127.0.0.1:8000";

export const maxDuration = 120;

export async function GET(req: Request) {
  const probe = new URL(req.url).searchParams.get("probe") === "1";
  try {
    const r = await fetch(`${SERVICE}/gym/environments?probe=${probe}`, {
      signal: AbortSignal.timeout(110000),
    });
    const notFound = r.status === 404 ? await r.clone().text() : "";
    // A 404 here is two different things. FastAPI answers a missing
    // ROUTE with the bare {"detail":"Not Found"}; a missing
    // ENVIRONMENT gets a named message from the handler. Treating
    // both as "restart the service" told the reader to restart over
    // a typo in an id.
    if (r.status === 404 && /"Not Found"/.test(notFound)) {
      // The service is up and answering, it just has no /gym. That means a
      // process started before `service/gym_routes.py` existed, which is the
      // normal case after pulling this change -- the router is wired at import
      // time, so a running uvicorn never picks it up. Say that, rather than
      // forwarding the service's bare {"detail":"Not Found"} to the page.
      return NextResponse.json(
        {
          error: "gym_routes_missing",
          detail:
            "The service is running but has no /gym routes, so it was started " +
            "before they existed. Restart it to pick them up: " +
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
