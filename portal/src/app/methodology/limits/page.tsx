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
    body: "The explanation module emits a certificate that can be checked mechanically, and it does check out. The stronger claim — that the explanation reflects the planner's actual reasoning — did not survive testing and is not made.",
    closes: "An explanation method with a faithfulness guarantee, which is an open problem.",
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
        <p className="mt-3 text-[14px] leading-relaxed text-ink-mute">
          This page is not a disclaimer appended to a demo. Two of the entries
          below are results this project published internally and then withdrew
          after measurement contradicted them. A twin that cannot say what it is
          unsure of is not a decision tool.
        </p>
      </header>

      <div className="mb-10 flex flex-col gap-3">
        {GAPS.map((g) => (
          <Panel key={g.title} title={g.title}>
            <p className="text-[13px] leading-relaxed text-ink-mute">{g.body}</p>
            <p className="mt-2.5 border-l-2 border-line pl-3 text-[12px] leading-relaxed text-ink-faint">
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
