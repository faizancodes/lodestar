import type { HTMLAttributes } from "react";

/** Renders trusted app-generated HTML (such as the brand SVG mark or formatted diff snippets). */
type Props = HTMLAttributes<HTMLSpanElement> & { html: string };

export function Html({ html, ...rest }: Props) {
  return <span {...rest} dangerouslySetInnerHTML={{ __html: html }} />;
}
