# Paper visual grammar

`aps_full_width.mplstyle` is the authority for canvas geometry, typography,
ticks, grids, default line weights, and vector-output settings.
`visual_grammar.toml` is the authority for semantic visual roles such as true,
MUTACO, NPE, ensemble means, uncertainty bands, and individual realizations.

Change these files to change the paper-wide presentation. Ordinary plotters
must request a named role through the typed loader and must not hard-code
method colors, markers, or line styles. A figure may declare role overrides in
its JSON configuration; the resolved overrides and both source hashes are
persisted in its output bundle.

The `aps_full_width` profile is 7.0 inches wide and is intended for nearly
native-size inclusion at REVTeX `\textwidth`. Final compiled proof inspection
remains required because LaTeX controls effective scaling.
