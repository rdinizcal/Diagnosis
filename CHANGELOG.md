# Changelog

## 1.1.0 — 2026-08-20

- Add the opt-in RTAMT verdict engine for AT1 and CCX, with an optional
  dependency extra and dedicated regression tests.
- Add persistent Z3 workers, verdict caching, parallel evaluation, adaptive
  stopping, interval inference, two-tier timeouts, adaptive ranges, and time
  quantization. These features remain opt-in unless documented otherwise.
- Train J48 from all decided samples and repair bounded quantifier guards when
  changing between universal and existential quantification.
- Keep ambiguous or unguarded quantifiers unchanged instead of producing
  potentially vacuous mutants.
- Move the paper replication package to its own repository; retain only minimal
  test fixtures in the tool repository.

## 1.0.0

- Initial public Diagnosis release.
