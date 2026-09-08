# Participant documents

All LaTeX (LuaLaTeX, TeX Live 2025; fonts ship with TeX Live), sharing `style/blueprintdoc.sty`. Build everything with
`make` (PDFs land in `build/`).

| PDF | Code | What it is |
|---|---|---|
| `field-guide.pdf` | NNV-FG | The whole exercise on paper: vocabulary, toolchain, every command with its real output, design decisions, the fairness stretch, troubleshooting, local install |
| `cheat-sheet.pdf` | NNV-CS | Two landscape pages, three columns: vocabulary, Vehicle syntax card, CLI, how to read results, limits, what a good property looks like |
| `debate-role-cards.pdf` | NNV-RC | One page per scenario (medical triage, authentication, autonomous control): facts, four cut-out role cards, decision sheet |
| `debate-protocol.pdf` | NNV-DP | A 20-minute facilitator-free protocol for the debate |
| `spec-your-system-worksheet.pdf` | NNV-WS | One page to write a property for your own project |
| `reading-list.pdf` | NNV-RL | Annotated sources for everything cited |

Build notes: `make` runs `latexmk -lualatex` with `-output-directory=build` and `TEXMFOUTPUT=build`, which tcolorbox needs to write and re-read its temporary `.listing` files; `make clean` removes the auxiliary files. Every terminal output quoted in the field guide comes from a real run on 8 September 2026 (x86_64 environment).

Licence: documents CC BY 4.0; code in the repository MIT.
