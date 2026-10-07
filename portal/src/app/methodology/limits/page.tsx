import type { Metadata } from "next";
import { Panel } from "@/components/Panel";
import { BadgeLegend } from "@/components/EvidenceBadge";

export const metadata: Metadata = {
  title: "Limits | PSPE",
  description:
    "What this framework cannot demonstrate, which results were withdrawn, and what would be needed to close each gap.",
};

const GAPS = [
  {
    title: "No observational record contains the counterfactual",
    body: "Satellites recorded what the 2022 flood did, not what it would have done behind a levee that was never built. The same holds for every hazard and every dataset. Intervention effects here are evaluated by re-solving accepted physics with the structure in place — stronger than scoring against our own model, weaker than a field trial, and never a substitute for one.",
    closes: "A controlled programme with recorded treatment locations and matched untreated controls.",
  },
  {
    title: "The levee results are a projection, not an outcome",
    body: "The reference solver is the operational standard for flood inundation and is independent of the model under test, but it is still a model. A levee result inherits every assumption the shallow-water formulation makes.",
    closes: "Field observation of a built structure through a comparable event.",
  },
  {
    title: "A synthetic result overstated the achievable span by ~40×",
    body: "An earlier version of this work measured levee leverage on a valley we constructed — a channel over-topping a berm through narrow gaps, which is precisely the geometry a levee is built for. It reported ~100%. Moving the identical study onto the real Richmond terrain gave 6–18% depending on which settlement was defended. The synthetic figure was re-scoped as a property of fluvial geometry, not of flood control.",
    closes: "Already corrected. Retained here because the correction is the finding.",
  },
  {
    title: "A feasibility verdict was wrong for several hours",
    body: "The real-terrain task was reported as failing its span gate at 2.4%. That was measured at a single settlement chosen for flooding deepest — which selects a sump, not a defensible target. Screening all 933 candidate sites found the identical test gives +15.4% a kilometre away. Controllability of the water budget and interceptability of the flow turned out to be different properties.",
    closes: "Already corrected. The screen is now part of the protocol.",
  },
  {
    title: "The published archive does not match its own data paper",
    body: "FloodCastBench's paper describes DEM, land cover, rainfall, georeferencing and initial conditions. The archive ships DEMs only. So the event's forcing cannot be replayed, roughness is a single literature value rather than a land-cover map, and only one of four events has a DEM grid that matches its depth rasters — the others cannot be placed at all.",
    closes: "The authors' preprocessing, or an independent georeferencing of the depth products.",
  },
  {
    title: "Explanation: the certificate holds, faithfulness does not",
    body: "The explanation module emits a certificate that can be checked mechanically, and it does check out — a split-conformal floor of 0.913 that held at three confidence levels on every seed tested. The stronger claim, that the explanation reflects the planner's actual reasoning, did not survive testing and is not made: a brief written for one situation scores as well against another, which is a randomisation control the method fails in its original form. Explanation of a specific decision is therefore delivered by leave-one-out attribution — remove this measure from this plan and you lose X — which depends on what else is in the plan and so cannot be reused for a different one.",
    closes: "An explanation method with a faithfulness guarantee, which is an open problem.",
  },
  {
    title: "The certainty slider sets the band's width, not how often the limit is crossed",
    body: "Raising the confidence widens the margin, and the tool shows the figure it will stand behind dropping accordingly. That is honest about the band. It is not a safety control, and the slider should not be read as one. Swept across a tenfold range of confidence levels on the controlled testbed, the rate at which the real system exceeded its limit did not fall — it was 5.0% at nine-times-in-ten, 4.1% at nineteen-in-twenty, 5.0% at forty-nine-in-fifty and 5.9% at ninety-nine-in-a-hundred, while the margin grew to consume half the available budget. The best rate sits in the middle of the range, not at the strict end, and asking for more certainty past that point costs achievable gain and buys nothing measurable.",
    closes: "A mechanism that controls the violation rate itself rather than the width of the interval reported around it.",
  },
  {
    title: "The band bounds how often, not how badly",
    body: "A margin calibrated on realised cost controls the frequency with which a limit is exceeded. It says nothing about the size of the worst exceedance. Measured on the controlled testbed, applying the margin cut the violation rate from 14.1% to 5.0% while the single worst excursion stayed at 1.54 times the limit — against 1.55 times with no margin at all, and unchanged across every confidence level swept. Anyone planning against a worst case rather than an average is not served by this figure, and tightening the slider will not change that.",
    closes: "A per-episode or per-timestep safety mechanism. This is a different instrument, not a tuning of this one.",
  },
];

export default function LimitsPage() {
  return (
    <div className="mx-auto max-w-[1100px] px-5 py-10">
      <header className="mb-8 max-w-3xl">
        <p className="eyebrow mb-2">Limits</p>
        <h1 className="text-[30px] font-semibold leading-tight tracking-tight">
          What this cannot show
        </h1>
        <p className="mt-3 text-[15px] leading-relaxed text-ink-mute">
          This page is not a disclaimer appended to a demo. Two of the entries
          below are results this project published internally and then withdrew
          after measurement contradicted them. A twin that cannot say what it is
          unsure of is not a decision tool.
        </p>
      </header>

      <div className="mb-10 flex flex-col gap-3">
        {GAPS.map((g) => (
          <Panel key={g.title} title={g.title}>
            <p className="text-[15px] leading-relaxed text-ink-mute">{g.body}</p>
            <p className="mt-2.5 border-l-2 border-line pl-3 text-[13px] leading-relaxed text-ink-faint">
              <span className="eyebrow mr-1.5">What would close it</span>
              {g.closes}
            </p>
          </Panel>
        ))}
      </div>

      <section>
        <h2 className="eyebrow mb-3">Evidence classes</h2>
        <BadgeLegend />
      </section>
    </div>
  );
}
