TEX := paper-supplementary
OUTDIR := build

.PHONY: all clean distclean

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=$(OUTDIR) $<
	cp $(OUTDIR)/$(TEX).pdf $@

clean:
	latexmk -C -outdir=$(OUTDIR) $(TEX).tex
	rm -rf $(OUTDIR)

distclean: clean
	rm -f $(TEX).pdf
