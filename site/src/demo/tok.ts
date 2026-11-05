import { Tokenizer } from "@huggingface/tokenizers";
import { REPO } from "./qwen";

export type Msg = { role: "system" | "user" | "assistant"; content: string };

// <|im_end|> and <|endoftext|>
export const STOP = new Set([151645, 151643]);

export class Tok {
  private constructor(private t: Tokenizer) {}

  static async load() {
    const [json, config] = await Promise.all(["tokenizer.json", "tokenizer_config.json"].map((f) => fetch(`${REPO}/${f}`).then((r) => r.json())));
    return new Tok(new Tokenizer(json, config));
  }

  /** Qwen3's chat template with thinking off, as the app asks for it: an empty think block opens the reply. */
  prompt(msgs: Msg[]) {
    const text = msgs.map((m) => `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`).join("") + "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    return this.t.encode(text, { add_special_tokens: false }).ids;
  }

  decode(ids: number[]) {
    return ids.length ? this.t.decode(ids, { skip_special_tokens: true }) : "";
  }
}
