# Build the participant documents. Requires TeX Live 2025 (LuaLaTeX); fonts ship with TeX Live.
# `make` leaves one PDF per document beside its .tex source; all compiler by-products go to .aux/ (git-ignored).
AUX=.aux
export TEXMFOUTPUT=$(AUX)
LATEXMK=latexmk -lualatex -interaction=nonstopmode -halt-on-error -output-directory=$(AUX)
DOCS=field-guide cheat-sheet debate-role-cards debate-protocol spec-your-system-worksheet reading-list judging-scoresheet mentor-question-card
.PHONY: all clean $(DOCS)
all: $(DOCS)
$(DOCS): %: %.tex style/blueprintdoc.sty
	@mkdir -p $(AUX)
	$(LATEXMK) $<
	@cp $(AUX)/$@.pdf $@.pdf
clean: ; rm -rf $(AUX)
