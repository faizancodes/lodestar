import { esc } from "./format";

// a small, safe markdown pass for replies: code blocks, inline code, bold, italics, headings, lists
export function md(src: string) {
  const parts = src.split(/```([^\n`]*)\n?([\s\S]*?)(?:```|$)/g);
  let html = "";
  for (let i = 0; i < parts.length; i += 3) {
    html += mdBlocks(parts[i]);
    if (i + 2 < parts.length) html += `<pre><code>${esc(parts[i + 2].replace(/\n$/, ""))}</code></pre>`;
  }
  return html;
}
export function mdInline(s: string) {
  return esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, "$1<i>$2</i>");
}
function mdBlocks(text: string) {
  const out: string[] = [];
  let list: "ul" | "ol" | null = null;
  const close = () => list && (out.push(`</${list}>`), (list = null));
  for (const para of text.split(/\n{2,}/)) {
    for (const line of para.split("\n")) {
      const h = line.match(/^(#{1,4})\s+(.*)/), ul = line.match(/^\s*[-*•]\s+(.*)/), ol = line.match(/^\s*\d+[.)]\s+(.*)/);
      if (ul || ol) {
        const kind = ul ? "ul" : "ol";
        if (list !== kind) (close(), out.push(`<${kind}>`), (list = kind));
        out.push(`<li>${mdInline((ul || ol)![1])}</li>`);
        continue;
      }
      close();
      if (h) out.push(`<h${Math.min(4, h[1].length + 2)}>${mdInline(h[2])}</h${Math.min(4, h[1].length + 2)}>`);
      else if (line.trim()) out.push(`<p>${mdInline(line)}</p>`);
    }
    close();
  }
  return out.join("");
}
