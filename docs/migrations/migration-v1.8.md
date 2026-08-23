# Migrating to AXQuant 1.8

AXQuant 1.8 keeps existing Hub repository names and all existing public artifact schema versions.
No certified repository should be renamed, and legacy public checkpoint certificates remain v1.

Public repositories now consistently use the class SKU produced by `model_name()` (`4bit`,
`6bit`, and so on). Measured main BPW is the authoritative claim shown in cards and certificates;
the derived `MP-…bpw` string is a display label, not a repository generator. Publishers should
use `axquant verify-cert` to produce an offline verification report before release.

For deployment planning, use `axquant optimize` with an explicit maximum memory and context. The
default runtime reserve is the visible 1 GB CLI default; set `--reserve-memory` for the target
service. Architecture-prior plans still require `--allow-unmeasured` and remain estimates.
