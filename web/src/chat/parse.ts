// Kept apart from the engine, which needs a page, so the tests can run it on its own.

/** Neurons written the way people share them: "L14 N8949 +1", "14/8949 -0.5", "layer 13 neuron 7956 −1", one or
 * more, split by commas, semicolons, "and" or new lines. A neuron without a dose gets +1. */
export function parsePins(text: string): { layer: number; neuron: number; dose: number }[] | null {
  const parts = text.replace(/[−–]/g, "-").split(/[,;\n]|\band\b/i).map((s) => s.trim()).filter(Boolean);
  const out = [];

  for (const p of parts) {
    const m = p.replace(/(\d)\s*(?:neuron|n)\s*(?=\d)/i, "$1 ") // "L14N8949" and "L14 N8949" as "L14 8949"
      .match(/^(?:layer|l)?\s*(\d+)\s*(?:[/:.]\s*|\s+)(\d+)\s*(?:(?:[x×*@=]|dose)\s*)?([+-]?\s*\d*\.?\d+)?$/i);
    if (!m) return null;
    out.push({ layer: +m[1], neuron: +m[2], dose: m[3] ? +m[3].replace(/\s/g, "") : 1 });
  }

  return out.length ? out : null;
}
