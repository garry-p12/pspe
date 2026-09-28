import { NextResponse } from "next/server";

/**
 * Live catchment forecast.
 *
 * Runs server-side so the browser never talks to a data provider directly and
 * the response can be cached across every user of the district.
 *
 * Note what this endpoint is FOR. Capital works are planned against design
 * events, not against next week's weather; this answers the other question —
 * "does anything need attention in the next seven days?" On most days the
 * answer is no, and saying so plainly is the point. A tool that only works
 * during a disaster is not a tool anyone opens on a Tuesday.
 */

const DISTRICT = { lat: -28.995, lon: 153.344 };

// The February 2022 event delivered roughly 245 mm over 48 h across this
// catchment. Thresholds are expressed against that, so the scale is the
// district's own experience rather than an abstract return period.
//
// It is the DISTRICT's yardstick and nobody else's. Asked about a catchment in
// Assam, "5% of the 2022 event" is a comparison to a flood that happened on
// another continent, so the reference is only returned when the request is
// actually for this district. Elsewhere the forecast is reported in millimetres
// and the caller is told there is no local benchmark.
const REF_EVENT_MM_48H = 245;
const DISTRICT_TOLERANCE_DEG = 0.5;

export const dynamic = "force-dynamic";

interface Band { level: "quiet" | "watch" | "act"; headline: string; detail: string }

function classify(max48: number, peakQ: number, ref: number | null): Band {
  const frac = ref ? max48 / ref : max48 / 245;
  if (frac >= 0.6 || peakQ > 400) {
    return {
      level: "act",
      headline: "Significant flooding possible",
      detail: `Forecast rainfall reaches ${Math.round(frac * 100)}% of the February 2022 event over 48 hours.`,
    };
  }
  if (frac >= 0.25 || peakQ > 120) {
    return {
      level: "watch",
      headline: "Worth watching",
      detail: `Forecast rainfall reaches ${Math.round(frac * 100)}% of the February 2022 event over 48 hours.`,
    };
  }
  return {
    level: "quiet",
    headline: "No significant flooding forecast",
    detail: "Rainfall over the next seven days stays well below the level that causes inundation here.",
  };
}

export async function GET(req: Request) {
  const url = new URL(req.url);
  const lat = Number(url.searchParams.get("lat") ?? DISTRICT.lat);
  const lon = Number(url.searchParams.get("lon") ?? DISTRICT.lon);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    return NextResponse.json({ error: "bad_coordinates" }, { status: 400 });
  }
  const isDistrict =
    Math.abs(lat - DISTRICT.lat) < DISTRICT_TOLERANCE_DEG &&
    Math.abs(lon - DISTRICT.lon) < DISTRICT_TOLERANCE_DEG;
  const ref = isDistrict ? REF_EVENT_MM_48H : null;

  try {
    const [metR, flR] = await Promise.all([
      fetch(
        `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}` +
          `&hourly=precipitation&forecast_days=7&timezone=auto`,
        { next: { revalidate: 1800 } },
      ),
      fetch(
        `https://flood-api.open-meteo.com/v1/flood?latitude=${lat}&longitude=${lon}` +
          `&daily=river_discharge_max&forecast_days=7`,
        { next: { revalidate: 1800 } },
      ),
    ]);
    if (!metR.ok || !flR.ok) throw new Error("upstream forecast unavailable");
    const met = await metR.json();
    const fl = await flR.json();

    const times: string[] = met.hourly.time;
    const precip: number[] = met.hourly.precipitation.map((v: number | null) => v ?? 0);

    // Rolling 48 h total — the window that matters for this catchment's response.
    let max48 = 0;
    let max48At = times[0];
    for (let i = 0; i + 48 <= precip.length; i++) {
      let s = 0;
      for (let j = i; j < i + 48; j++) s += precip[j];
      if (s > max48) { max48 = s; max48At = times[i]; }
    }

    const byDay = new Map<string, number>();
    times.forEach((t, i) => {
      const d = t.slice(0, 10);
      byDay.set(d, (byDay.get(d) ?? 0) + precip[i]);
    });

    const qMax: number[] = (fl.daily?.river_discharge_max ?? []).map(
      (v: number | null) => v ?? 0,
    );
    const peakQ = qMax.length ? Math.max(...qMax) : 0;

    return NextResponse.json({
      issued: new Date().toISOString(),
      location: { lat, lon },
      total_mm: precip.reduce((a, b) => a + b, 0),
      max_48h_mm: max48,
      max_48h_from: max48At,
      reference_event_48h_mm: ref,
      fraction_of_reference: ref ? max48 / ref : null,
      peak_discharge_m3s: peakQ,
      daily: [...byDay.entries()].map(([date, mm]) => ({ date, mm })),
      discharge_daily: (fl.daily?.time ?? []).map((date: string, i: number) => ({
        date, m3s: qMax[i] ?? 0,
      })),
      band: classify(max48, peakQ, ref),
      sources: [
        "Rainfall: Open-Meteo forecast API",
        "River discharge: Open-Meteo flood API (GloFAS)",
      ],
    });
  } catch (e) {
    return NextResponse.json(
      { error: "forecast_unavailable", detail: String((e as Error).message) },
      { status: 503 },
    );
  }
}
