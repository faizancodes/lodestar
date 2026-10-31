import { appendFileSync } from "node:fs";
import type { Page } from "@playwright/test";

// AUDIT=<file>: at every snap, measure the page at several window widths and append what doesn't fit.
export const AUDIT = process.env.AUDIT;
const WIDTHS = [900, 1100, 1280, 1920];

function measure() {
  const name = (el: Element) => {
    const parts: string[] = [];
    for (let e: Element | null = el, i = 0; e && e !== document.body && i < 4; e = e.parentElement, i++)
      parts.unshift(e.tagName.toLowerCase() + (e.id ? "#" + e.id : "") + [...e.classList].slice(0, 2).map((c) => "." + c).join(""));
    return parts.join(" > ");
  };
  const text = (el: Element) => (el.textContent || "").trim().replace(/\s+/g, " ").slice(0, 40);
  const shown = (el: Element) => { const r = el.getBoundingClientRect(); const cs = getComputedStyle(el); return r.width > 0 && r.height > 0 && cs.visibility !== "hidden" && cs.display !== "none"; };
  const out: { kind: string; el: string; text: string; by: number }[] = [];
  const W = document.documentElement.clientWidth;
  if (document.documentElement.scrollWidth > W + 1) out.push({ kind: "page scrolls sideways", el: "html", text: "", by: document.documentElement.scrollWidth - W });

  const reported = new Set<Element>();
  for (const el of document.querySelectorAll("body *")) {
    if (!shown(el) || el.closest("svg") && el.tagName !== "svg") continue;
    const cs = getComputedStyle(el);
    if (cs.position === "fixed") continue;
    // a scroll box that scrolls sideways
    if (["auto", "scroll"].includes(cs.overflowX) && el.scrollWidth > el.clientWidth + 1 && el.tagName !== "TEXTAREA" && el.tagName !== "INPUT")
      out.push({ kind: "box scrolls sideways", el: name(el), text: "", by: el.scrollWidth - el.clientWidth });
    // content cut off by a hidden overflow that isn't a deliberate ellipsis
    if (cs.overflowX === "hidden" && cs.textOverflow !== "ellipsis" && el.scrollWidth > el.clientWidth + 2 && el.children.length && !["CANVAS"].includes(el.tagName)) {
      const kid = [...el.querySelectorAll("*")].find((k) => shown(k) && k.getBoundingClientRect().right > el.getBoundingClientRect().right + 2 && getComputedStyle(k).position !== "absolute");
      if (kid) out.push({ kind: "cut off", el: name(kid), text: text(kid), by: Math.round(kid.getBoundingClientRect().right - el.getBoundingClientRect().right) });
    }
    // spills past its parent
    const p = el.parentElement;
    if (p && p !== document.body && cs.position !== "absolute" && !reported.has(p) && ![...reported].some((r) => r.contains(el))) {
      const pr = p.getBoundingClientRect(), r = el.getBoundingClientRect(), pcs = getComputedStyle(p);
      if (pcs.display !== "inline" && pcs.overflowX === "visible" && r.right > pr.right + 2 && pr.width > 0) {
        reported.add(el);
        out.push({ kind: "spills past its container", el: name(el), text: text(el), by: Math.round(r.right - pr.right) });
      }
    }
    // a text field squeezed too narrow to use
    if ((el.tagName === "INPUT" && /text|search|^$/.test((el as HTMLInputElement).type) || el.tagName === "TEXTAREA") && el.getBoundingClientRect().width < 150)
      out.push({ kind: "text field squeezed", el: name(el), text: (el as HTMLInputElement).placeholder || "", by: Math.round(el.getBoundingClientRect().width) });
  }
  return out;
}

export async function audit(page: Page, step: string) {
  const size = page.viewportSize()!;
  // WebKit crashes if the window is resized while a text field has focus
  const focused = await page.evaluateHandle(() => { const a = document.activeElement as HTMLElement | null; a?.blur(); return a; });
  for (const w of WIDTHS) {
    await page.setViewportSize({ width: w, height: size.height });
    await page.waitForTimeout(120);
    const found = await page.evaluate(measure);
    for (const f of found) appendFileSync(AUDIT!, JSON.stringify({ step, w, ...f }) + "\n");
    if (process.env.AUDIT_SHOTS && (w === 1100 || w === 1920))
      await page.screenshot({ path: `${process.env.AUDIT_SHOTS}/${step}-${w}.png`, fullPage: true, animations: "disabled" });
  }
  await page.setViewportSize(size);
  await page.waitForTimeout(80);
  await focused.evaluate((a) => (a as HTMLElement | null)?.focus());
}
