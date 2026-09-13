BUILD=build
export TEXMFOUTPUT=$(BUILD)
LATEXMK=latexmk -lualatex -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD)
DOCS=field-guide cheat-sheet debate-role-cards debate-protocol spec-your-system-worksheet reading-list judging-scoresheet mentor-question-card
.PHONY: all clean $(DOCS)
all: $(DOCS)
$(DOCS): %: %.tex ; $(LATEXMK) $<
clean: ; rm -f $(BUILD)/*.aux $(BUILD)/*.log $(BUILD)/*.out $(BUILD)/*.fls $(BUILD)/*.fdb_latexmk $(BUILD)/*.listing
