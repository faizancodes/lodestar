import { describe, expect, it } from "vitest";
import { parsePins } from "../src/chat/parse";

describe("parsePins", () => {
  it("reads neurons the ways people write them", () => {
    expect(parsePins("L14 N8949 +1 and L13 N7956 -1")).toEqual([{ layer: 14, neuron: 8949, dose: 1 }, { layer: 13, neuron: 7956, dose: -1 }]);
    expect(parsePins("L14N8949")).toEqual([{ layer: 14, neuron: 8949, dose: 1 }]);
    expect(parsePins("14/8949 −0.5, layer 3 neuron 12 x 2")).toEqual([{ layer: 14, neuron: 8949, dose: -0.5 }, { layer: 3, neuron: 12, dose: 2 }]);
    expect(parsePins("L2 N5 .25\nL3 N6 -3")).toEqual([{ layer: 2, neuron: 5, dose: 0.25 }, { layer: 3, neuron: 6, dose: -3 }]);
  });

  it("refuses what isn't a neuron", () => {
    expect(parsePins("")).toBeNull();
    expect(parsePins("evil")).toBeNull();
    expect(parsePins("L14")).toBeNull();
    expect(parsePins("L14 N8949 +1, hello")).toBeNull();
  });
});
