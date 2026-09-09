import type { Node, Tree } from "web-tree-sitter";
import type { OpenClawConfig } from "../config/types.openclaw.js";
import { prepareBashParserForSource } from "../infra/command-explainer/tree-sitter-runtime.js";
import { CREDENTIAL_STYLE_HEADER_KEYS, GATEWAY_SECURITY_HEADER_KEYS } from "./redact-patterns.js";
import { redactInputTextWithSourcePolicy, redactToolPayloadTextWithConfig } from "./redact.js";

type Header = { start: number; end: number; safe: boolean };
const headerPrefix = new RegExp(
  `^(?:(?:proxy-)?authorization[ \\t]*[:=][ \\t]*(?:[A-Za-z][A-Za-z0-9_-]*[ \\t]+)?|(?:${CREDENTIAL_STYLE_HEADER_KEYS}|${GATEWAY_SECURITY_HEADER_KEYS})[ \\t]*[:=][ \\t]*)`,
  "i",
);
const reference =
  /^(?:\$[A-Za-z_][A-Za-z0-9_]*|\$\{[A-Za-z_][A-Za-z0-9_]*(?::-)?\}|\$\(printenv [A-Za-z_][A-Za-z0-9_]*\))$/;

/** Parser state is prepared only for an input owned by the native shell tool. */
export async function prepareShellSourceRedactor(): Promise<
  (source: string, config?: OpenClawConfig["logging"]) => string
> {
  let parse: (source: string) => Tree;
  try {
    parse = await prepareBashParserForSource();
  } catch {
    return redactToolPayloadTextWithConfig;
  }
  return (source, config) => {
    let parsed: string | undefined;
    let headers: Header[] = [];
    const inspect = (text: string) => {
      if (parsed === text) {
        return headers;
      }
      parsed = text;
      headers = [];
      let tree: Tree | undefined;
      try {
        tree = parse(text);
        if (tree.rootNode.hasError) {
          return headers;
        }
        const walk = (node: Node) => {
          if (node.type === "command") {
            const args = node.childrenForFieldName("argument");
            for (const [index, arg] of args.entries()) {
              if (arg.type !== "string" && arg.type !== "raw_string") {
                continue;
              }
              const body = arg.text.slice(1, -1);
              const prefix = headerPrefix.exec(body);
              if (!prefix) {
                continue;
              }
              const start = arg.startIndex + 1 + prefix[0].length;
              const end = arg.endIndex - 1;
              const value = text.slice(start, end);
              const leaf = arg.namedChildren.find(
                (child) => child.startIndex === start && child.endIndex === end,
              );
              const isHeaderArgument =
                node.childForFieldName("name")?.text === "curl" &&
                ["-H", "--header", "--proxy-header"].includes(args[index - 1]?.text ?? "");
              const safe =
                isHeaderArgument &&
                arg.type === "string" &&
                reference.test(value) &&
                !!leaf &&
                ["simple_expansion", "expansion", "command_substitution"].includes(leaf.type) &&
                (leaf.type !== "command_substitution" ||
                  (leaf.namedChildren.length === 1 &&
                    leaf.namedChildren[0]?.type === "command" &&
                    leaf.namedChildren[0].namedChildren.length === 2));
              headers.push({ start, end, safe });
            }
          }
          for (const child of node.namedChildren) {
            walk(child);
          }
        };
        walk(tree.rootNode);
      } catch {
        headers = [];
      } finally {
        tree?.delete();
      }
      return headers;
    };
    return redactInputTextWithSourcePolicy(source, config, () => false, {
      mask(text) {
        // Descending original offsets; nested headers are already inside their outer credential.
        const spans = inspect(text)
          .filter((header) => !header.safe)
          .filter(
            (header, _, all) =>
              !all.some((outer) => outer.start < header.start && outer.end >= header.end),
          )
          .sort((a, b) => b.start - a.start);
        for (const span of spans) {
          text = text.slice(0, span.start) + "***" + text.slice(span.end);
        }
        return text;
      },
      preserves(text, offset, length) {
        return inspect(text).some(
          (header) => header.safe && header.start === offset && offset + length <= header.end,
        );
      },
    });
  };
}
