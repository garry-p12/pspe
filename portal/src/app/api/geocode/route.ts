import { NextResponse } from "next/server";

/** Place search. Server-side so Nominatim sees one identified client. */
export const revalidate = 3600;

export async function GET(req: Request) {
  const q = new URL(req.url).searchParams.get("q")?.trim();
  if (!q || q.length < 2) return NextResponse.json({ results: [] });
  try {
    const r = await fetch(
      "https://nominatim.openstreetmap.org/search?format=json&limit=5" +
        `&q=${encodeURIComponent(q)}`,
      {
        headers: { "User-Agent": "PSPE-floodplain-planner/1.0" },
        next: { revalidate: 3600 },
      },
    );
    if (!r.ok) throw new Error(`nominatim ${r.status}`);
    const raw = (await r.json()) as {
      display_name: string; lat: string; lon: string;
      boundingbox: [string, string, string, string]; type: string;
    }[];
    return NextResponse.json({
      results: raw.map((x) => ({
        name: x.display_name,
        short: x.display_name.split(",").slice(0, 2).join(",").trim(),
        lat: Number(x.lat),
        lon: Number(x.lon),
        kind: x.type,
      })),
    });
  } catch (e) {
    return NextResponse.json(
      { error: "geocode_failed", detail: String((e as Error).message) },
      { status: 502 },
    );
  }
}
