.PHONY: figures pdf clean

figures:
	uv run python figures/generate_figures.py

pdf: figures
	tectonic hopf_latent_spaces.tex --outdir .

clean:
	rm -f hopf_latent_spaces.pdf hopf_latent_spaces.aux hopf_latent_spaces.log hopf_latent_spaces.out