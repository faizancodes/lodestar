import type { HTMLAttributes } from "react";

// The one place markup is set as HTML. Only for strings the app builds itself (the brand mark's SVG, rendered
// markdown, the fix-flow change list); never for text that came from the model or the user unescaped.
type Props = HTMLAttributes<HTMLElement> & { as?: "span" | "div"; html: string };

export function Html({ as: Tag = "span", html, ...rest }: Props) {
  return <Tag {...rest} dangerouslySetInnerHTML={{ __html: html }} />;
}
