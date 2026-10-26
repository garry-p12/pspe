/**
 * The complete PSPE report: everything built, why each choice was made, and
 * how every number was arrived at.
 *
 *   node scripts/make_full_report.js
 *
 * Writes docs/paper/PSPE_Full_Report.docx. The .docx is gitignored; this
 * script is tracked, so the document is always reproducible from the repo.
 *
 * Sources: docs/THESIS_AND_RESULTS.md (the master record), docs/PORTAL_GUIDE.md,
 * docs/ICML_PAPER.md, the live catalogue in pspe/envs/catalog.py, and the
 * figure sets under docs/figures/ and docs/deck_bundle/images/.
 */
const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, Footer, Header, HeadingLevel, ImageRun,
  LevelFormat, PageBreak, PageNumber, Packer, Paragraph, ShadingType, Table,
  TableCell, TableOfContents, TableRow, TextRun, VerticalAlign, WidthType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const CONTENT = 9360;                     // US Letter minus 1.5" of margins

// ------------------------------------------------------------------ text --
function runs(text, o = {}, carry = {}) {
  // Nested, because a bold span routinely contains `code` and a flat scan
  // lets the bold alternative swallow the backticks: "**named `a.py`**"
  // emitted the backticks literally until this was made recursive.
  const out = [];
  const re = /(\*\*[^*]+?\*\*|`[^`]+?`|\*[^*]+?\*)/g;
  let last = 0, m;
  const push = (t, extra) => {
    if (!t) return;
    out.push(new TextRun({ text: t, size: o.size ?? 21, font: o.font ?? "Calibri",
                           color: o.color, ...carry, ...extra }));
  };
  while ((m = re.exec(text))) {
    push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) out.push(...runs(tok.slice(2, -2), o, { ...carry, bold: true }));
    else if (tok.startsWith("`")) push(tok.slice(1, -1), { font: "Consolas", size: (o.size ?? 21) - 2 });
    else out.push(...runs(tok.slice(1, -1), o, { ...carry, italics: true }));
    last = m.index + tok.length;
  }
  push(text.slice(last));
  return out;
}

const P = (text, o = {}) => new Paragraph({
  spacing: { after: o.after ?? 140, line: o.line ?? 276 },
  alignment: o.align, indent: o.indent,
  children: runs(text, o),
  ...(o.border ? { border: o.border } : {}),
});

const H1 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_1,
                                  spacing: { before: 400, after: 200 },
                                  pageBreakBefore: true });
const H2 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_2,
                                  spacing: { before: 300, after: 150 } });
const H3 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_3,
                                  spacing: { before: 240, after: 120 } });

const BULLET = (text) => new Paragraph({
  numbering: { reference: "bullets", level: 0 },
  spacing: { after: 90, line: 276 }, children: runs(text),
});
const NUM = (text) => new Paragraph({
  numbering: { reference: "numbers", level: 0 },
  spacing: { after: 90, line: 276 }, children: runs(text),
});

const CODE = (lines) => lines.map((l, i) => new Paragraph({
  spacing: { after: i === lines.length - 1 ? 170 : 0, line: 240 },
  shading: { type: ShadingType.CLEAR, fill: "F4F4F5" },
  indent: { left: 220, right: 220 },
  children: [new TextRun({ text: l || " ", font: "Consolas", size: 17 })],
}));

const QUOTE = (text) => new Paragraph({
  spacing: { before: 130, after: 170, line: 276 },
  indent: { left: 300 },
  border: { left: { style: BorderStyle.SINGLE, size: 12, color: "18181B", space: 10 } },
  children: runs(text),
});

/** A labelled callout: the "why we chose this" boxes that run through the report. */
const WHY = (label, text) => new Paragraph({
  spacing: { before: 130, after: 170, line: 276 },
  indent: { left: 180, right: 180 },
  shading: { type: ShadingType.CLEAR, fill: "F7F7F8" },
  border: { left: { style: BorderStyle.SINGLE, size: 18, color: "7A7A82", space: 8 } },
  children: [new TextRun({ text: label + "  ", bold: true, size: 19, font: "Calibri",
                           color: "3F3F46", allCaps: true }),
             ...runs(text, { size: 20 })],
});

const RULE = () => new Paragraph({
  spacing: { before: 60, after: 170 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "D4D4D8", space: 1 } },
  children: [new TextRun({ text: "" })],
});

const BREAK = () => new Paragraph({ children: [new PageBreak()] });

// ----------------------------------------------------------------- table --
function table(headers, rows, widths, o = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const w = widths.map((x) => Math.round(x * CONTENT / total));
  const cell = (text, i, head, rowOpts) => new TableCell({
    width: { size: w[i], type: WidthType.DXA },
    margins: { top: 70, bottom: 70, left: 110, right: 110 },
    shading: head ? { type: ShadingType.CLEAR, fill: "18181B" }
                  : (rowOpts && rowOpts.fill
                     ? { type: ShadingType.CLEAR, fill: rowOpts.fill } : undefined),
    verticalAlign: VerticalAlign.TOP,
    children: [new Paragraph({
      spacing: { after: 0, line: 252 },
      alignment: i >= (o.numericFrom ?? 99) ? AlignmentType.RIGHT : AlignmentType.LEFT,
      children: runs(String(text), { size: o.size ?? 17,
                                     color: head ? "FFFFFF" : (rowOpts && rowOpts.color) })
                .map((r) => head ? new TextRun({ ...r, bold: true }) : r),
    })],
  });
  return new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: w,
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r) => {
        const opts = Array.isArray(r) ? {} : r.opts;
        const cells = Array.isArray(r) ? r : r.cells;
        return new TableRow({ children: cells.map((c, i) => cell(c, i, false, opts)) });
      }),
    ],
  });
}

const CAPTION = (t) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 240 },
  children: runs(t, { size: 17, color: "52525B" }),
});

// ---------------------------------------------------------------- figure --
let FIGN = 0;
function imgdim(file) {
  const b = fs.readFileSync(file);
  if (b[0] === 0x89 && b[1] === 0x50) return { t: "png", w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
  if (b[0] === 0xFF && b[1] === 0xD8) {
    let o = 2;
    while (o < b.length) {
      if (b[o] !== 0xFF) { o++; continue; }
      const m = b[o + 1];
      if (m >= 0xC0 && m <= 0xCF && m !== 0xC4 && m !== 0xC8 && m !== 0xCC)
        return { t: "jpg", h: b.readUInt16BE(o + 5), w: b.readUInt16BE(o + 7) };
      o += 2 + b.readUInt16BE(o + 2);
    }
  }
  throw new Error("unreadable image " + file);
}
/** Place a figure, scaled to fit the text column and capped in height. */
function figure(rel, caption, opts = {}) {
  const file = path.isAbsolute(rel) ? rel : path.join(ROOT, rel);
  if (!fs.existsSync(file)) { console.warn("  MISSING", rel); return []; }
  const d = imgdim(file);
  const maxW = opts.width ?? 620;
  const maxH = opts.maxHeight ?? 400;
  let w = maxW, h = Math.round(maxW * d.h / d.w);
  if (h > maxH) { h = maxH; w = Math.round(maxH * d.w / d.h); }
  FIGN += 1;
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 170, after: 80 },
      children: [new ImageRun({ data: fs.readFileSync(file), type: d.t,
                                transformation: { width: w, height: h } })],
    }),
    CAPTION(`Figure ${FIGN}. ${caption}`),
  ];
}

// ------------------------------------------------------------------- body --
const H = { P, H1, H2, H3, BULLET, NUM, CODE, QUOTE, WHY, RULE, BREAK,
            table, figure, CAPTION, runs,
            AlignmentType, BorderStyle, Document, Footer, Header, HeadingLevel,
            ImageRun, LevelFormat, PageBreak, PageNumber, Packer, Paragraph,
            ShadingType, Table, TableCell, TableOfContents, TableRow, TextRun,
            VerticalAlign, WidthType };

const body = [];
const add = (...xs) => xs.flat().forEach((x) => body.push(x));

// ---- title page ----
const TITLE = (t, o) => new Paragraph({
  alignment: AlignmentType.LEFT, spacing: { after: o && o.after || 160 },
  children: [new TextRun({ text: t, size: o && o.size || 56, bold: o ? o.bold !== false : true,
                           font: "Calibri", color: o && o.color || "18181B" })],
});

add(
  new Paragraph({ spacing: { before: 2200, after: 0 }, children: [] }),
  TITLE("PSPE", { size: 72 }),
  TITLE("Decision systems measured against reality,", { size: 36, color: "3F3F46" }),
  TITLE("not against their own model", { size: 36, color: "3F3F46", after: 420 }),
  new Paragraph({
    spacing: { after: 160 },
    children: runs("A complete account of what was built, why each choice was made, and how every number was arrived at.", { size: 24, color: "52525B" }),
  }),
  new Paragraph({
    spacing: { before: 600, after: 80 },
    children: runs("A flood digital twin for Onion Creek, Austin; a wildfire twin on real fire records; a fourteen-environment simulation gym; and the safe-RL method underneath all three.", { size: 21, color: "52525B" }),
  }),
  new Paragraph({
    spacing: { before: 420 },
    children: runs("Generated from the repository by `scripts/make_full_report.js`. The master record is `docs/THESIS_AND_RESULTS.md`.", { size: 18, color: "8B8B93" }),
  }),
  BREAK(),
);

// ---- contents ----
add(
  new Paragraph({ spacing: { after: 220 },
    children: [new TextRun({ text: "Contents", size: 40, bold: true, font: "Calibri" })] }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ spacing: { before: 200 },
    children: runs("*Right-click the table above and choose \u201cUpdate field\u201d to populate page numbers.*",
                   { size: 18, color: "8B8B93" }) }),
  BREAK(),
);

// ---- chapters ----
for (const mod of ["01_opening", "02_theory", "03_method", "04_testbeds",
                   "05_flood", "06_wildfire", "07_gym", "08_portal", "09_scope"]) {
  process.stdout.write("  " + mod + "\n");
  add(require("./report/" + mod + ".js")(H));
}

// ---- document ----
const doc = new Document({
  creator: "PSPE",
  title: "PSPE — complete report",
  description: "Everything built, why, and how every number was arrived at.",
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 420, hanging: 220 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 420, hanging: 220 } } } }] },
    ],
  },
  styles: {
    default: {
      document: { run: { font: "Calibri", size: 21, color: "18181B" } },
      heading1: { run: { font: "Calibri", size: 34, bold: true, color: "18181B" } },
      heading2: { run: { font: "Calibri", size: 26, bold: true, color: "18181B" } },
      heading3: { run: { font: "Calibri", size: 22, bold: true, color: "3F3F46" } },
    },
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
                          margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 } } },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT], size: 17, color: "8B8B93",
                                 font: "Calibri" })],
      })] }),
    },
    children: body,
  }],
});

const OUT = path.join(ROOT, "docs/paper/PSPE_Full_Report.docx");
fs.mkdirSync(path.dirname(OUT), { recursive: true });
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log(`\nwrote ${OUT}`);
  console.log(`  ${(buf.length / 1024 / 1024).toFixed(2)} MB, ${FIGN} figures, ${body.length} blocks`);
});
