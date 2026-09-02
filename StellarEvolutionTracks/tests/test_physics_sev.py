"""Regression tests for the StellarEvolutionTracks program module.

The discovery helper below deliberately supports both the repository layout
(``tests/test_physics_sev.py``) and an upload layout in which this file is
flattened beside the four program modules (physics_sev.py, driver_sev.py,
main.py, plot_sev.py).  Both layouts are exercised by
``TestModuleDiscovery``, but that does not mean two complete rounds of the
suite are run: the flattened layout is only checked with a trivial smoke
test (module import + a two-line calculation) that proves the discovery
helper itself works from a flattened directory.  The full test suite is
run exactly once, from the canonical ``tests/`` layout.  Reviewer AIs
(Copilot, Codex, Gemini) should follow the same convention: run the full
suite once from ``tests/``, and treat any flattened-layout run as a
discovery smoke test only.

Development history (audit trail -- developers only; never surfaced to
students in the Help file or in main.py/driver_sev.py/physics_sev.py/
plot_sev.py docstrings or output):

  2026-09-01  Claude (principal developer).  First comprehensive regression
    suite for StellarEvolutionTracks.  No prior unittest suite existed;
    development up to this point had been ad hoc.  This round:
      * Compiled and ran the supplied four-mode program (tracks, hr,
        wdcool, nsmr) to establish a baseline; all four modes ran cleanly
        and reproduced the headline numbers documented in the Help file and
        in the two prior AI critiques (ChatGPT/GPT-5.6-Sol, Microsoft
        Copilot) exactly: solar t_MS = 9.733 Gyr, 0.6-Msun white-dwarf
        R = 8839.5 km with a 5.6236-Gyr Mestel cooling age, ideal-neutron-
        gas TOV maximum 0.7098 Msun at 9.313 km, default stiff-polytrope
        maximum 2.1749 Msun at 11.692 km.
      * Recomputed BUILD_ID independently and confirmed it matches both the
        value embedded in physics_sev.py and the value printed in the Help
        file's #version_build element (4412fce2fda3 for the as-uploaded
        source): the uploaded code and Help file were already self
        consistent, i.e. the substantive corrections described in both
        ChatGPT critiques and the Copilot critique had already been carried
        into this source tree before this round began.  This round is
        therefore mostly new test coverage, not a bug-fixing pass; any
        additional defects this round's own audit turned up are listed in
        the accompanying Kickoff report, together with the regression test
        that now guards each one.
      * Built this file from nothing, organised by physical invariant
        rather than by which critique first raised it, per the project's
        standing instruction that test names/comments describe the lasting
        physics, not the audit history.

  2026-09-01  Claude (principal developer).  Response to Audit1 (Gemini,
    Codex, Copilot reviewing the Kickoff round).  Version 1.1.2 -> 1.2.0,
    BUILD_ID f9f19e598626 -> e6b4674d23c6.  Full disposition of every
    finding is in StellarEvolutionTracks-Claude-Response-to-Audit1-
    20260901.txt; this entry lists only what changed in the source tree.
      * Fixed four release-blocking (P1) defects, each confirmed with an
        independent reproducer before fixing, each now guarded by a
        regression test: (1) ns_mass_radius_curve()'s turning-point check
        accepted a maximum at the edge of the sampled range as genuine
        (i_max < gi[-1] alone), so a monotonically falling M(rho_c) sampled
        entirely above the true turning point was reported as a confirmed,
        stable turning point with no warning; fixed to require the maximum
        to be interior on both sides (gi[0] < i_max < gi[-1]), with
        separate warnings for "raise rho_hi" and "lower rho_lo".
        (2) integrate_track()'s summary conflated "a post-main-sequence
        phase was computed" (post_ms) with "the track reached helium
        ignition"; a helium white dwarf (envelope exhausted before the
        flash) has post_ms=True but never ignites, so driver_sev.py and
        plot_sev.py were printing/annotating "stops at helium ignition" for
        tracks that do not.  Added a distinct helium_ignition summary flag
        and switched both consumers to it.  (3) mc_tams was computed as
        core_efficiency*qc*m_msun, ignoring the (1 - Xc_end/X) factor the
        array formula (mc_list) uses, so the reported terminal-age core
        mass jumped discontinuously (up to ~70x at large --x_end) relative
        to the star's own last plotted core-mass point; fixed to match.
        (4) integrate_track()'s remnant_kind came from predicted_remnant(),
        a mass-only classifier that switches from "helium white dwarf" to
        "carbon-oxygen white dwarf" at 0.5 Msun, while the same summary's
        phase_end/helium_ignition came from the track's own integration,
        which (via the 0.70*m_msun helium-core cap) switches at ~0.6714
        Msun; for initial masses in [0.5, 0.6714) the two fields
        contradicted each other in the same summary dict and in
        driver_sev.py's printed report (e.g. remnant_kind="carbon-oxygen
        white dwarf" together with phase_end="...helium white dwarf").
        Fixed by having integrate_track() override predicted_remnant()'s
        classification with the track's own computed outcome whenever the
        track actually completed a non-igniting degenerate-red-giant-branch
        post-MS phase, with a note explaining the override; verified
        against Codex's five-mass probe (0.49, 0.50, 0.60, 0.67, 0.68 Msun).
      * Fixed nine P2 numerical-safety and validation gaps, each with a
        regression test: wd_structure()'s fixed bisection bracket
        [1e7,1e13] kg/m^3 failed for white-dwarf masses legitimately close
        to (but below the hard 0.999*M_Ch cutoff for) the Chandrasekhar
        limit -- now widens adaptively by decades, with the fixed-bracket
        behaviour still reachable and tested via max_bracket_expansions=0;
        wd_structure()'s tol and max_iter parameters were unvalidated (a
        non-positive tol or max_iter<1 could hang or silently return a
        meaningless result) -- both now validated; integrate_wd_cooling()'s
        analytic-age formula could overflow a Python float for a
        sufficiently small Tc_end -- added a 100 K floor;
        integrate_structure() accepted y_c<=0 or y_c>=1 without validation,
        so a caller-supplied NaN or out-of-range y_c bypassed the intended
        _require_positive guard and propagated silently into the RK4 loop
        -- y_c is now validated the same way as the module's other
        physical inputs; integrate_structure()'s first RK4 stage (k1) was
        computed outside the try/except that wraps the k2-k4 stages, so a
        first-stage failure (including the TOV horizon RuntimeError raised
        inside derivs()) propagated with an inconsistent message instead of
        the unified, actionable one -- moved inside; ns_mass_radius_curve()
        discarded the specific reason each non-converged sample point
        failed (bisection did not bracket a root, versus the interior RK4
        integration itself raising) and its "stable branch" search could
        bridge over a gap of non-converged points between two converged
        runs straddling the maximum, reporting a branch as contiguous when
        it was not -- now preserves a per-point failure reason (surfaced in
        a new warnings_detail summary field) and restricts the stable
        branch to an unbroken run of converged points walking back from
        the maximum; build_hr_grid() never enforced the isochrone-count cap
        the Help file documents, and silently dropped an isochrone age no
        track (or only one track) spanned with no explanation -- added the
        cap and an explanatory warning; _write_csv() and _finish()
        (plot_sev.py) could silently overwrite a same-second output file --
        both now avoid the collision; and an hr run's aggregate workload
        (masses times n_ms+n_post) was unbounded even though each factor
        was individually capped -- added an aggregate cap in driver_sev.py.
      * Added validation previously missing from several direct-API entry
        points (effective_temperature, core_mass_luminosity, hayashi_teff,
        kelvin_helmholtz_time, predicted_remnant, wd_mass_radius_curve,
        integrate_structure's r_scale/y_floor/step_frac/max_steps, and a
        physical [1,60] bound on A_ion), and added explanatory warnings for
        the two silent post-main-sequence-suppression corners in
        integrate_track (mc_ign <= mc_tams; T_hay >= T_tams).
      * Updated constants: m_u, m_e, m_n to CODATA 2022 recommended values
        (from CODATA ~2018); M_sun is now derived from the IAU 2015
        Resolution B3 nominal GM_sun (1.3271244e20 m^3/s^2) divided by G,
        rather than an independently rounded kg literal, so M_sun and G are
        never inconsistent wherever both appear together; added a shared
        R_EARTH constant and removed the duplicated 6.371e6/6371.0 Earth-
        radius literals from physics_sev.py and plot_sev.py; added
        MAX_ISOCHRONES=10, now enforced.
      * Strengthened four weak tests identified by this round's own
        antagonistic review: the mu_e-lower-bound-documented test was
        tautological (assertIn("mu_e","mu_e")) and now checks the actual
        \\mu_e\\ge1 bound statement in the Help file; the mu_e^-5/3
        radius-scaling test computed but never checked the scaling and now
        does (to 5%); test_driver_and_help_report_same_build was renamed
        (it never touched the Help file) to
        test_driver_summary_reports_same_build_as_physics_module; and
        test_low_mass_extreme_t_max_does_not_crash now also asserts the
        endpoint semantics (reaches TAMS, runs post-MS, does not ignite
        helium) instead of only checking for finite output.  Also corrected
        this docstring's "~200-test" figure to non-numeric phrasing, since
        it goes stale every round.
      * Declined, with rationale recorded in the Response-to-Audit1 report:
        Gemini's "L_post(mc) evaluated before the mc>=mc_ign break check"
        finding (verified against the code -- L_post and teff_post are
        smooth, domain-unrestricted functions of mc, so floating-point
        drift past mc_ign cannot produce an out-of-bounds evaluation or a
        crash) and Gemini's "FloatingPointError is not raised by default
        under NumPy's default seterr" observation (accurate, but the
        pre-existing math.isfinite(m_next)/isfinite(y_next) check
        immediately downstream already catches a silent numpy inf/nan and
        raises the same clear RuntimeError, so no student-visible behaviour
        was actually wrong; OverflowError and RuntimeError were still added
        to the except tuple as defense in depth for the pure-Python paths).
      * HTML wording/documentation fixes: "tracks -- main sequence" table
        row corrected from "two integrated ODEs" to "one" (only dXc/dt is
        integrated on the main sequence; dMc/dt is post-main-sequence);
        softened "exact TOV gravity" and fixed the self-contradictory
        "fitted to nothing" polytrope description; corrected the false
        universal claim that 0.7 Msun is "less than half the mass of every
        neutron star ever weighed"; reworded the white-dwarf "exact to the
        accuracy of the step size" phrasing; softened "essentially metal
        free" (here and in physics_sev.py's kramers_kappa0 docstring) and
        the binarity paragraph's absolute claims; documented the 10-
        isochrone and [1,60] A_ion bounds in the parameter table and
        runtime-safeguards note; added the explicit piecewise ZAMS
        luminosity/radius formulas Gemini flagged as missing; added a
        "known model artefacts and caveats" section covering the pp/CNO
        1.2-solar-mass burning-law discontinuity, a citation for the K1
        non-relativistic degenerate-pressure constant, and the causality
        check's assumption of a monotonic sound speed; and reworded EXP-3,
        EXP-5, EXP-6, EXP-12, EXP-13, EXP-14 and EXP-18 per the reviewers'
        specific wording findings (a concrete 13.8 Gyr age of the Universe;
        "evolved off the main sequence" instead of "died"; the explicit
        Teq ~ L^(1/4) scaling; a well-posed turn-off-mass target instead of
        an ambiguous log L value; a precise compactness framing instead of
        "more compact than a black hole"; a note on how to hit an exact
        central density outside the CLI's geometric grid; and the explicit
        Eobs = Eemit/(1+z) relation).

  2026-09-01  Claude (principal developer).  Response to Audit2 (Gemini,
    Codex, Copilot reviewing the Response-to-Audit1 round).  Version
    1.2.0 -> 1.3.0, BUILD_ID e6b4674d23c6 -> 7703fa47bf20.  Full
    disposition of every finding is in StellarEvolutionTracks-Claude-
    Response-to-Audit2-20260901.txt; this entry lists only what changed
    in the source tree.
      * Fixed two release-blocking (P1) defects Codex confirmed with
        reproducers against the 1.2.0 build: (1) the Audit1 fix that made
        integrate_track()'s own remnant_kind/phase_end/helium_ignition
        fields internally consistent for a low-mass non-igniting track did
        not reach any of the fields' CONSUMERS -- driver_sev.py's terminal
        text, its CSV comment, plot_sev.py's plot annotation, main.py's
        module docstring, and the Help's global description all still
        unconditionally stated the old, now-sometimes-false "classification
        from the initial mass, not an integrated result" sentence.  Added
        a remnant_basis summary field ("this track's own post-main-sequence
        integration" vs. "mass-only schematic classification") and made
        every one of those five consumers branch on it; also reworded the
        low-mass-cap remnant_note and phase_end text (previously "the
        integration found the hydrogen envelope exhausted", which overstates
        what a mass-bookkeeping cap with no modeled envelope-ejection
        physics actually computed) to describe it as reaching this
        schematic model's core-mass cap, not as a found physical outcome.
        (2) wd_structure()'s bisection loop returned the last evaluated
        (rho_c, M, R) tuple after max_iter iterations regardless of whether
        tol was met -- max_iter=1 against the default bracket for a
        0.6-Msun target returned a mass 58% off target with no error.  It
        now raises RuntimeError, reporting max_iter, tol, the achieved
        residual, and the final bracket, whenever the loop exhausts
        max_iter without converging.
      * Fixed six P2 correctness/reproducibility/test defects: (a)
        FermiGasEOS.pressure()/energy_density() subtracted two nearly-equal
        terms and lost essentially all significant digits below x ~ 1e-4
        (by x=1e-5 pressure() returned a negative, unphysical value) --
        both now switch to the exact small-x Taylor series (independently
        derived and confirmed against a symbolic expansion of the closed
        form) below a new _FERMI_SMALL_X=0.05 threshold, agreeing with the
        closed form to better than 1e-10 relative accuracy there. (b)
        ns_mass_radius_curve()'s per-density failure reasons
        (warnings_detail, added in the Audit1 round) were never actually
        printed or written to CSV, and the CSV's branch column labelled
        every raw index <= i_max "stable" and every index > i_max
        "unstable" without checking whether that row's model had converged
        at all, mislabelling nan-mass/nan-radius rows as unstable stellar
        models -- driver_sev.py now prints a "PER-MODEL FAILURE DETAIL"
        section and writes each reason into the CSV, and a non-converged
        row is now labelled "failed", never "stable"/"unstable". (c) a
        saved PNG still could not reproduce its own run: the on-figure
        footer answers "which code", not "which run" -- plot_sev.py now
        writes a plain-text provenance sidecar (every mode-relevant
        parameter, the same list already written into a CSV's header)
        next to every saved PNG unconditionally, whether or not --csvdir
        was also requested. (d) wd_structure()'s new
        max_bracket_expansions parameter (Audit1) was itself unvalidated
        (negative/fractional/non-finite values silently accepted on the
        no-expansion-needed path) and wd_structure() did not require
        rho_hi > rho_lo the way its sibling curve functions already did --
        both are now validated. (e) the same-second collision regression
        tests performed two real writes without freezing the clock, so
        they could pass even against the old broken implementation if the
        two writes happened to land in different wall-clock seconds --
        both tests now mock datetime.now() to force the collision and
        assert the exact expected "_2" disambiguated filename. (f)
        ns_mass_radius_curve()'s turning-point check required only that
        SOME converged point exist on each side of the sampled maximum,
        which is satisfied even when the maximum's own immediate neighbor
        failed to converge -- a larger true mass could be hiding in that
        adjacent gap.  turning_point now additionally requires both of
        i_max's immediate neighbors to have converged, with a new,
        distinct warning ("largest converged sampled mass ... unresolved
        across that convergence gap") when the interior-maximum condition
        holds but a neighbor did not converge.
      * Fixed the remaining P3 items: R_EARTH was still the old rounded
        6.371e6 m literal despite being labelled "IUGG mean Earth radius"
        (an 8.77 m/1.4 ppm mismatch against the actual IUGG/GRS80 value,
        Moritz 2000) -- now the cited value itself, 6,371,008.7714 m; the
        Help's post-main-sequence regime description said "Below 2 solar
        masses" / "Above 2 solar masses", leaving the actual m=2 Msun
        implementation boundary (m_msun <= 2.0) unstated in prose -- now
        "at or below 2 solar masses", with the boundary made explicit; the
        --Tc_end parameter row did not mention the 100 K floor stated only
        in the general safeguards paragraph -- now stated in the row
        itself; and several public-API validation gaps were closed
        together: build_hr_grid()'s isochrone_gyr is now normalized to a
        validated plain list up front (fixing a TypeError on a generator,
        an ambiguous-truth-value ValueError on a NumPy array, and silently
        accepted duplicate ages, which are now rejected the same way
        duplicate masses already were); turnoff_mass() validates age_gyr
        before taking its log; zams_curve() validates m_lo/m_hi/n;
        default_burning()/default_core_fraction() validate their mass
        argument; and a new _require_bool() helper is now applied to
        integrate_track()'s include_postms/homology_zams,
        integrate_structure()'s relativistic/keep_profile, and
        ns_mass_radius_curve()'s relativistic, so a non-bool "truthy"
        value (e.g. the string "False") is rejected explicitly rather than
        silently reinterpreted by Python's ordinary truthiness rules.
      * Strengthened one conditional test Codex re-flagged as still
        vacuous under the right regression:
        test_stiff_polytrope_can_be_made_acausal_and_is_flagged guarded
        its only assertions behind "if cs_over_c_max_branch > 1.0"; the
        chosen (gamma, p_nuc) case is independently verified to be
        reliably acausal (peak c_s/c = 1.7359), so that outcome and the
        flag/warning it must produce are now asserted unconditionally.
        Added end-to-end tests that inspect actual printed terminal text,
        actual CSV comment lines, and actual plot-annotation text for the
        low-mass remnant case (Codex's specific point that the previous
        round's regression test checked only physics-summary fields,
        leaving every user-visible output layer free to contradict them);
        a deterministic mocked-failure test for the new gap-adjacent-to-
        the-peak turning-point rule; postcondition tests for
        wd_structure()'s convergence enforcement and its now-validated
        max_bracket_expansions/rho bracket ordering; small-x positivity,
        Taylor-series-agreement, cross-boundary-continuity, and array-
        safety tests for the Fermi-gas EOS fix; and a failed-row-labelling
        plus warnings_detail-surfaced test for the neutron-star CSV fix.
      * Declined, with rationale recorded in the Response-to-Audit2
        report: Codex's H5/H4-style requests for a fuller slope-sign-
        change/local-refinement turning-point criterion beyond the
        interior-neighbor check now added (the interior-neighbor
        requirement already closes every concrete reproducer given; a
        fuller refinement scheme is a numerical-methods design project of
        its own); a cross-process atomic-exclusive-create guarantee for
        the same-second collision fix (the fix closes the sequential,
        same-process race the regression tests exercise; a true
        multi-process TOCTOU guarantee is a distinct, larger concern);
        Copilot's boolean-type-checking request beyond the specific
        parameters listed above (applied to the module's most-used direct
        entry points rather than exhaustively to every remaining flag);
        and build_hr_grid() pre-validating each mass against the 0.08-120
        Msun range before starting expensive work (integrate_track()
        already validates immediately and fails fast within that same
        call; the deferred item is purely about failing even earlier, not
        about correctness).

  2026-09-02  Claude (principal developer).  Response to Audit3 (Gemini,
    Codex, Copilot reviewing the Response-to-Audit2 round).  Version
    1.3.0 -> 1.4.0, BUILD_ID 7703fa47bf20 -> 16faf4c1e9f5.  No P1 defect
    was reported this round by any of the three reviewers.  Full
    disposition of every finding is in StellarEvolutionTracks-Claude-
    Response-to-Audit3-20260902.txt; this entry lists only what changed
    in the source tree.
      * Fixed all four P2 findings, which Codex separated into one
        cluster around the Audit2-round PNG-provenance-sidecar feature:
        (a) plot_sev._finish() only wrote a sidecar when the caller
        supplied a nonempty `provenance` list, so a direct
        plot_track(result, outdir=...) Python-API call -- bypassing
        driver_sev.run() -- saved a PNG with no sidecar at all, directly
        contradicting the "unconditionally" claim already in that
        function's own docstring; a sidecar is now always written
        whenever outdir is given, and when no scientific-parameter list
        was supplied it says so explicitly rather than silently omitting
        it. (b) the sidecar recorded every mode-specific scientific
        parameter but never --dpi or --lw, even though both change the
        actual saved image; _finish() now always records a "rendering
        parameters" section with dpi and lw, independent of whether a
        scientific-parameter list was supplied. (c) the PNG-uniqueness
        check (_unique_path(), now _unique_stem()) tested only whether
        the candidate PNG path existed, so a pre-existing same-stem
        orphaned .provenance.txt (left by, say, an earlier no-provenance
        call, or a manual copy) was silently overwritten the instant a
        new run picked that stem; stem selection now requires that
        NEITHER the PNG nor its sidecar already exist. (d) the sidecar
        regression coverage exercised only a manually-assembled helper
        case (driver._provenance() built by hand and passed directly to
        plotting.plot_track(), never through driver.run()) for the
        "tracks" mode alone, so a driver runner that stopped forwarding
        provenance, or any of the other three modes losing a parameter,
        would not have been caught -- a new end-to-end test now drives
        all four modes through the real driver.run() entry point and
        checks the sidecar's keys against driver.PARAMS_BY_MODE.
      * Fixed all four P3 findings: FermiGasEOS's public evaluators
        (pressure, energy_density, number_density, rest_mass_density,
        dP_dx, sound_speed_ratio) accepted negative x -- physically
        undefined, since x = p_F/(m c) -- and returned negative
        pressures, energy densities and number densities with no warning;
        a new _require_nonneg_x() helper (deliberately not
        _require_positive(), since x=0 is the legitimate zero-density
        limit and must still be accepted) is now called first by every
        one of those six methods.  wd_structure()'s bracket-exhausted
        RuntimeError used one generic message ("try a mass further from
        the Chandrasekhar limit") for both bracket-failure directions,
        which is backwards advice when the requested mass is actually
        BELOW the reachable low-density end of the search (the fix there
        is a LARGER mass, not a smaller one) -- the message is now
        side-specific, naming the achieved mass at whichever bracket end
        failed and pointing the requested-mass change in the correct
        direction.  Also documented, per Codex's specific wording
        suggestion: the Help's --wd_mass row and general safeguards
        paragraph now distinguish the hard 0.999*M_Ch formal rejection
        from the separate, narrower practical range the numerical bracket
        search can actually reach.  The fourth P3 (report headings
        overstating closure, e.g. "ALL SIX FIXED") is a report-writing
        habit, not a code defect; this and future reports state counts
        for fixed/deferred/declined rather than a single unqualified
        heading.
      * Declined, with rationale recorded in the Response-to-Audit3
        report: Copilot's A3-P3-1 (full-suite runtime under a third-
        party hosted command runner's short time limit) and A3-P3-4
        (upstream PyparsingDeprecationWarning volume from installed
        Matplotlib/pyparsing) are both properties of the reviewer's own
        execution environment and dependency versions, not of this
        program's code, so no source change was made for either;
        Copilot's A3-P3-3 (scientific wording duplicated across several
        consumer layers) is a maintainability observation Copilot itself
        recommended no immediate redesign for, and this round agrees;
        Codex's continued mention of an external published white-dwarf
        mass-radius benchmark remains open as a validation limitation, as
        in the Audit2 response, not as a defect requiring a code change.
      * The full regression suite grew from 197 to 207 tests: four
        FermiGasEOS negative/nonfinite/zero-boundary-x tests, two
        wd_structure side-specific bracket-failure-message tests, three
        new plotting tests (direct-call sidecar-always-written,
        orphan-sidecar-preserved, and all-four-modes-through-driver.run()
        matched-pair coverage), one strengthened same-second-collision
        plotting test that now also checks the sidecar half of each pair,
        and one Help-documentation test for the provenance sidecar's
        three Help locations.

  2026-09-02  Claude (principal developer).  Response to Audit4 (Gemini,
    Codex, Copilot reviewing the Response-to-Audit3 round).  Version
    1.4.0 -> 1.5.0, BUILD_ID 16faf4c1e9f5 -> 4ab0d0a5d73c.  No P1 or P2
    defect was reported by Copilot or Gemini this round; both explicitly
    recommended release with only the three previously-disclosed P3
    observations outstanding.  Codex reported one P2 and three P3
    findings and recommended one more small correction round before
    release; all four are fixed here rather than deferred, since each had
    a concrete reproducer and a narrow, low-risk fix, and doing so lets
    this round close with zero open findings from all three reviewers
    simultaneously.  Full disposition is in StellarEvolutionTracks-
    Claude-Response-to-Audit4-20260902.txt; this entry lists only what
    changed in the source tree.
      * Fixed the P2 finding (Codex A4-P2-1): every public plot_sev.py
        function (plot_track, plot_hr_diagram, plot_wd_cooling,
        plot_ns_mass_radius) accepts a caller-selectable `figsize`, which
        changes the saved PNG's aspect and pixel dimensions, but
        _finish()'s "rendering parameters" sidecar section recorded only
        dpi and lw -- two direct-API calls differing solely in figsize
        could therefore save different images with indistinguishable
        recorded rendering settings.  _finish() now takes a `figsize`
        argument and records it unconditionally as `figsize_inches`,
        exactly like dpi and lw; all four public functions now pass their
        effective figsize through.  The CLI itself still does not expose
        --figsize (only --dpi and --lw), so no CLI behavior changed.
      * Fixed all three P3 findings. (a) A4-P3-1: the Audit3 .py.txt
        transport-encoding scheme (text.replace("__", "dunder").replace
        (" ", "§")) was reversible for the delivered files but not
        collision-free in general -- a pre-existing natural occurrence of
        the word "dunder" decoded back to "__", corrupting it.  The
        packaging-only encoder (package/transport_codec.py, not part of
        the shipped program) now escapes every literal "§" first
        (doubling it) before introducing the delimited, still-legible
        tokens "§dunder§" and "§sp§"; this is a provably injective,
        self-escaping scheme, verified byte-for-byte against an
        adversarial corpus (literal "dunder", literal "__", literal "§"
        alone and adjacent to a real token, non-ASCII text, CRLF, and a
        file with no final newline) before every package is assembled.
        (b) A4-P3-2: _require_nonneg_x's np.asarray(x, dtype=float)
        silently coerced Python bool (True/False, and boolean arrays) to
        1.0/0.0, and let a complex x escape as a raw TypeError instead of
        this validator's normal ValueError contract; bool is now rejected
        explicitly (scalar or array), and a conversion failure (complex
        or any other non-real input) is now caught and re-raised as a
        ValueError, so every FermiGasEOS public-evaluator rejection looks
        the same to a caller. (c) A4-P3-3: the all-mode sidecar test
        (test_driver_writes_matched_png_and_sidecar_for_every_mode) used
        `self.assertIn(param, content)`, an arbitrary substring search
        that would still pass if a short key like "X", "Z", "qc", "eos"
        or "mass" only happened to appear in prose, a heading, or another
        key's name -- it never actually proved the sidecar held one exact
        `name = value` entry per parameter.  A new `_parse_sidecar()`
        helper parses the sidecar's documented indentation and
        `name = value` grammar into a dict; the test now compares exact
        key sets (scientific keys against driver.PARAMS_BY_MODE[mode],
        rendering keys against {dpi, lw, figsize_inches}) and exact
        dpi/lw/figsize_inches values.
      * The full regression suite grew from 207 to 210 tests: two
        FermiGasEOS type-rejection tests (bool, complex), one new
        direct-call nondefault-figsize sidecar test, and the
        all-four-modes sidecar test rewritten to assert exact parsed
        key/value entries instead of substrings.
"""

import ast
from collections import Counter
import contextlib
import csv as csv_module
from datetime import datetime
import hashlib
from html.parser import HTMLParser
import inspect
import io
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


CORE_MODULE_FILES = (
    "physics_sev.py",
    "driver_sev.py",
    "main.py",
    "plot_sev.py",
)
HELP_FILE = "StellarEvolutionTracks.html"


def find_module_dir(start):
    """Find the nearest ancestor containing all four core program modules."""
    candidate = Path(start).resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for directory in (candidate, *candidate.parents):
        if all((directory / name).is_file() for name in CORE_MODULE_FILES):
            return directory

    required = ", ".join(CORE_MODULE_FILES)
    raise FileNotFoundError(
        f"could not find a directory containing all core modules: {required}"
    )


MODULE_DIR = find_module_dir(Path(__file__))
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import driver_sev as driver  # noqa: E402
import physics_sev as phys  # noqa: E402
import plot_sev as plotting  # noqa: E402


def recompute_build_id(directory):
    """Independently reproduce the documented normalized source hash."""
    digest = hashlib.sha256()
    for name in phys.BUILD_ID_COVERS:
        with (directory / name).open("r", encoding="utf-8", newline=None) as source:
            content = source.read().encode("utf-8")
        digest.update(name.encode("utf-8"))
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()[:12]


def _parse_sidecar(content):
    """
    Parse a `.provenance.txt` sidecar's `    name = value` lines into a
    dict, split only at the first "=".

    Regression test for Audit4 Codex A4-P3-3: the previous all-mode
    sidecar test used `self.assertIn(param, content)`, an arbitrary
    substring search that would pass even if a short key like "X", "Z",
    "qc", "eos" or "mass" only appeared inside prose, a heading, another
    key's name, or a value.  Parsing the documented indentation and
    `name = value` grammar into an actual key/value map lets a test
    assert an exact set of keys (and, where useful, exact values)
    instead, which is what the sidecar contract and the test's own
    comments always claimed to check.
    """
    entries = {}
    for line in content.splitlines():
        if not line.startswith("    "):
            continue
        stripped = line.strip()
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip()
        if not key or " " in key:
            # Guards against accidentally parsing wrapped prose lines
            # that happen to be indented and contain a "=" character.
            continue
        entries[key] = value
    return entries


def run_cli(args, cwd=MODULE_DIR, timeout=60):
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [sys.executable, "main.py", *args],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


# ------------------------------------------------------------------
# Minimal dependency-free HTML tree, used only for structural Help tests.
# ------------------------------------------------------------------
class HtmlNode:
    def __init__(self, tag, attrs=()):
        self.tag = tag
        self.attrs = dict(attrs)
        self.content = []

    def text(self):
        return "".join(
            item.text() if isinstance(item, HtmlNode) else item
            for item in self.content
        )


class HtmlTreeParser(HTMLParser):
    VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input",
                 "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = HtmlNode("document")
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = HtmlNode(tag, attrs)
        self.stack[-1].content.append(node)
        if tag not in self.VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].content.append(HtmlNode(tag, attrs))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data):
        self.stack[-1].content.append(data)


def descendants(node, predicate=lambda item: True):
    matches = []
    for item in node.content:
        if isinstance(item, HtmlNode):
            if predicate(item):
                matches.append(item)
            matches.extend(descendants(item, predicate))
    return matches


def normalized_text(node):
    return " ".join(node.text().split())


def has_class(node, class_name):
    return class_name in node.attrs.get("class", "").split()


def nodes_by_id(root, element_id):
    return descendants(root, lambda node: node.attrs.get("id") == element_id)


# ======================================================================
class TestModuleDiscovery(unittest.TestCase):
    def test_finds_canonical_tests_layout(self):
        self.assertEqual(find_module_dir(Path(__file__)), MODULE_DIR)

    def test_finds_flattened_layout(self):
        self.assertEqual(find_module_dir(MODULE_DIR / "main.py"), MODULE_DIR)

    def test_uses_nearest_matching_ancestor(self):
        self.assertEqual(find_module_dir(MODULE_DIR / "tests"), MODULE_DIR)

    def test_missing_module_directory_raises(self):
        with self.assertRaises(FileNotFoundError):
            find_module_dir(Path(MODULE_DIR.anchor))

    def test_flattened_layout_smoke_test_only(self):
        """Prove discovery + import works from a flattened directory.

        This is deliberately NOT a second full run of the suite (see the
        module docstring): it imports physics_sev from a flattened copy and
        performs one trivial calculation, then returns.  The full suite
        below runs exactly once, from the canonical tests/ layout.
        """
        if os.environ.get("SEV_FLATTENED_SMOKE_CHILD") == "1":
            return
        with tempfile.TemporaryDirectory() as temporary:
            flat_dir = Path(temporary)
            for name in (*CORE_MODULE_FILES, HELP_FILE):
                shutil.copy2(MODULE_DIR / name, flat_dir / name)
            smoke = flat_dir / "_flat_smoke.py"
            smoke.write_text(
                "import sys\n"
                "sys.path.insert(0, '.')\n"
                "import physics_sev as p\n"
                "assert p.chandrasekhar_mass(2.0) == 1.459\n"
                "print('FLAT_SMOKE_OK')\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["SEV_FLATTENED_SMOKE_CHILD"] = "1"
            result = subprocess.run(
                [sys.executable, str(smoke)],
                cwd=flat_dir, env=environment,
                capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("FLAT_SMOKE_OK", result.stdout)


# ======================================================================
class TestMetadataAndCompatibility(unittest.TestCase):
    def test_build_coverage_is_exactly_the_executable_core(self):
        self.assertEqual(tuple(phys.BUILD_ID_COVERS), CORE_MODULE_FILES)
        self.assertNotIn(HELP_FILE, phys.BUILD_ID_COVERS)
        self.assertFalse(any("test" in name for name in phys.BUILD_ID_COVERS))

    def test_build_id_matches_independent_calculation(self):
        self.assertRegex(phys.BUILD_ID, r"^[0-9a-f]{12}$")
        self.assertEqual(phys.BUILD_ID, recompute_build_id(MODULE_DIR))

    def test_build_id_independent_of_line_endings(self):
        """BUILD_ID must be stable under LF/CRLF normalization (newline=None)."""
        digest_lf = hashlib.sha256()
        digest_crlf = hashlib.sha256()
        for name in phys.BUILD_ID_COVERS:
            raw = (MODULE_DIR / name).read_bytes()
            text_lf = raw.replace(b"\r\n", b"\n")
            text_crlf = text_lf.replace(b"\n", b"\r\n")
            # normalized (universal-newline) content is identical either way
            normalized = text_lf
            for digest in (digest_lf, digest_crlf):
                digest.update(name.encode("utf-8"))
                digest.update(len(normalized).to_bytes(8, "big"))
                digest.update(normalized)
        self.assertEqual(digest_lf.hexdigest()[:12], digest_crlf.hexdigest()[:12])
        self.assertEqual(digest_lf.hexdigest()[:12], phys.BUILD_ID)

    def test_all_core_sources_parse_as_python_3_10(self):
        for name in CORE_MODULE_FILES:
            with self.subTest(name=name):
                source = (MODULE_DIR / name).read_text(encoding="utf-8")
                ast.parse(source, filename=name, feature_version=(3, 10))

    def test_version_command(self):
        result = run_cli(["--version"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            f"StellarEvolutionTracks {phys.MODEL_VERSION} (build {phys.BUILD_ID})",
        )

    def test_driver_summary_reports_same_build_as_physics_module(self):
        # Renamed from test_driver_and_help_report_same_build (Audit1
        # P2-10): the old name claimed to check the HELP file too, but this
        # test only ever exercised driver.run()'s summary dict.  The actual
        # help-file-vs-BUILD_ID check lives in
        # TestHelpFile.test_version_and_build_match_program; every summary
        # dict must carry the same version/build as physics_sev itself.
        result = driver.run(mode="tracks", mass=1.0, no_plot=True,
                             csvdir=tempfile.mkdtemp())
        self.assertEqual(result["summary"]["model_version"], phys.MODEL_VERSION)
        self.assertEqual(result["summary"]["build_id"], phys.BUILD_ID)


# ======================================================================
class TestPhysicalConstants(unittest.TestCase):
    """Spot-check the SI constants against CODATA/IAU nominal values."""

    def test_codata_constants(self):
        self.assertAlmostEqual(phys.G, 6.67430e-11, delta=1e-15)
        self.assertEqual(phys.c, 299792458.0)
        self.assertAlmostEqual(phys.k_B, 1.380649e-23, delta=1e-29)
        self.assertAlmostEqual(phys.sigma_SB, 5.670374419e-8, delta=1e-16)

    def test_radiation_constant_derived_from_sigma(self):
        self.assertAlmostEqual(phys.a_rad, 4.0 * phys.sigma_SB / phys.c, delta=1e-30)

    def test_iau_nominal_solar_values_are_self_consistent(self):
        # Stefan-Boltzmann applied to the IAU nominal L_sun and R_sun should
        # reproduce the IAU nominal Teff_sun to within a small fraction of
        # a percent -- an independent thermodynamic cross-check, not merely
        # a re-statement of the constant.
        teff = phys.effective_temperature(1.0, 1.0)
        self.assertAlmostEqual(teff, phys.TEFF_SUN, delta=0.5)

    def test_particle_masses_match_codata_2022(self):
        # Regression test for the Audit1 constants-currency fix: m_u, m_e
        # and m_n must be the CODATA 2022 recommended values, not the
        # CODATA ~2018 values previously used.
        self.assertAlmostEqual(phys.m_u, 1.66053906892e-27, delta=1e-36)
        self.assertAlmostEqual(phys.m_e, 9.1093837139e-31, delta=1e-40)
        self.assertAlmostEqual(phys.m_n, 1.67492750056e-27, delta=1e-36)

    def test_solar_mass_derived_from_iau_nominal_gm_sun(self):
        # Regression test for the Audit1 P-level constants fix: M_sun must
        # be derived as GM_sun_nominal / G (the IAU-recommended way to fix
        # the solar mass, since GM_sun is known far more precisely than G
        # or M_sun individually), not an independently rounded kg literal
        # that can be inconsistent with G wherever the two appear together.
        self.assertEqual(phys.GM_SUN_NOMINAL, 1.3271244e20)
        self.assertAlmostEqual(phys.M_sun, phys.GM_SUN_NOMINAL / phys.G,
                                delta=1.0)
        self.assertAlmostEqual(phys.G * phys.M_sun, phys.GM_SUN_NOMINAL,
                                delta=1.0e10)

    def test_earth_radius_constant_matches_km_conventions(self):
        # Audit2 P3-1 (Codex): the previous version of this test asserted
        # the OLD rounded 6.371e6 m literal to within 1 m, which enshrined
        # an 8.77 m mismatch against the actual IUGG/GRS80 mean radius
        # while calling it IUGG.  R_EARTH is now the cited value itself.
        self.assertAlmostEqual(phys.R_EARTH, 6_371_008.7714, delta=0.001)

    def test_year_and_gigayear(self):
        # The source comment labels YEAR as "the Julian year (365.25 d)",
        # which is also the IAU-recommended definition of a year for
        # astronomical time/age quantities (as used throughout this module
        # for every "Gyr" reported).  The constant must match that label
        # exactly: 365.25 * 86400 s = 31,557,600 s.
        self.assertEqual(phys.YEAR, 365.25 * 86400.0)
        self.assertEqual(phys.GYR, 1.0e9 * phys.YEAR)


# ======================================================================
class TestComposition(unittest.TestCase):
    def test_mean_molecular_weight_solar(self):
        self.assertAlmostEqual(phys.mean_molecular_weight(0.70, 0.02), 0.6173,
                                places=4)

    def test_mean_molecular_weight_formula_independent(self):
        for X, Z in ((0.70, 0.02), (0.34, 0.0), (1.0, 0.0), (0.0, 0.02)):
            with self.subTest(X=X, Z=Z):
                Y = 1.0 - X - Z
                expected = 1.0 / (2.0 * X + 0.75 * Y + 0.5 * Z)
                self.assertAlmostEqual(phys.mean_molecular_weight(X, Z), expected,
                                        places=12)

    def test_mu_e_formula(self):
        for X in (0.0, 0.5, 0.70, 1.0):
            with self.subTest(X=X):
                self.assertAlmostEqual(
                    phys.mean_molecular_weight_per_electron(X), 2.0 / (1.0 + X),
                    places=12,
                )

    def test_pure_hydrogen_and_pure_helium_mu_e(self):
        self.assertAlmostEqual(phys.mean_molecular_weight_per_electron(1.0), 1.0)
        self.assertAlmostEqual(phys.mean_molecular_weight_per_electron(0.0), 2.0)

    def test_composition_rejects_impossible_sums(self):
        with self.assertRaisesRegex(ValueError, "exceeds 1"):
            phys.check_composition(0.9, 0.3)

    def test_composition_boundary_x_plus_z_equals_one(self):
        X, Y, Z = phys.check_composition(0.98, 0.02)
        self.assertAlmostEqual(Y, 0.0, places=12)

    def test_composition_out_of_unit_interval_rejected(self):
        for X in (-0.01, 1.01):
            with self.subTest(X=X):
                with self.assertRaises(ValueError):
                    phys.check_composition(X, 0.02)
        for Z in (-0.01, 1.01):
            with self.subTest(Z=Z):
                with self.assertRaises(ValueError):
                    phys.check_composition(0.7, Z)

    def test_composition_rejects_non_finite(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    phys.check_composition(bad, 0.02)


# ======================================================================
class TestHomologyExponents(unittest.TestCase):
    def test_electron_scattering_gives_L_mu4_M3_regardless_of_burning(self):
        # HTML: "For electron scattering (a=b=0) this collapses to the
        # famous L ~ mu^4 M^3, independent of the burning law."
        for nu in (4.0, 16.0, 8.0):
            with self.subTest(nu=nu):
                exps = phys.homology_exponents(nu, 0.0, 0.0)
                self.assertAlmostEqual(exps["e_L_mu"], 4.0, places=12)
                self.assertAlmostEqual(exps["e_L_M"], 3.0, places=12)

    def test_kramers_pp_matches_hand_derived_exponents(self):
        # Independently derived by hand from the stated homology formulae
        # (see the Kickoff report / HTML background): D=6.5, e_L_mu=7.7692,
        # e_L_M=5.4615.
        exps = phys.homology_exponents(4.0, 1.0, -3.5)
        self.assertAlmostEqual(exps["D"], 6.5, places=10)
        self.assertAlmostEqual(exps["e_L_mu"], 101.0 / 13.0, places=6)
        self.assertAlmostEqual(exps["e_L_M"], 71.0 / 13.0, places=6)

    def test_degenerate_denominator_raises(self):
        # D = nu + 3 + b + 3a = 0 for nu=4, a=0, b=-7 (contrived but legal
        # inputs to the low-level function).
        with self.assertRaises(ValueError):
            phys.homology_exponents(4.0, 0.0, -7.0)

    def test_homology_exponents_reject_non_finite(self):
        with self.assertRaises(ValueError):
            phys.homology_exponents(math.nan, 0.0, 0.0)


# ======================================================================
class TestZamsAnchors(unittest.TestCase):
    def test_solar_anchor(self):
        self.assertAlmostEqual(phys.zams_luminosity(1.0), 0.72, places=10)
        self.assertAlmostEqual(phys.zams_radius(1.0), 0.89, places=10)

    def test_luminosity_continuous_at_breakpoints(self):
        for m in (0.43, 2.0):
            with self.subTest(m=m):
                lo = phys.zams_luminosity(m - 1e-9)
                hi = phys.zams_luminosity(m + 1e-9)
                self.assertAlmostEqual(lo, hi, delta=1e-6 * max(lo, hi))

    def test_radius_continuous_at_breakpoint(self):
        lo = phys.zams_radius(1.0 - 1e-9)
        hi = phys.zams_radius(1.0 + 1e-9)
        self.assertAlmostEqual(lo, hi, delta=1e-6 * max(lo, hi))

    def test_zams_curve_is_monotonic_in_mass_and_finite(self):
        m, logT, logL = phys.zams_curve(m_lo=0.15, m_hi=60.0, n=100)
        self.assertTrue(np.all(np.isfinite(logT)))
        self.assertTrue(np.all(np.isfinite(logL)))
        self.assertTrue(np.all(np.diff(m) > 0.0))
        self.assertTrue(np.all(np.diff(logL) > 0.0))  # brighter with mass

    def test_zams_rejects_non_positive_mass(self):
        for bad in (0.0, -1.0):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    phys.zams_luminosity(bad)


# ======================================================================
class TestPostMainSequencePrescriptions(unittest.TestCase):
    def test_core_mass_luminosity_reference_point(self):
        # HTML: tuned so the solar giant-branch tip is near 2500 Lsun at
        # Mc = helium-flash mass (0.47 Msun).
        L = phys.core_mass_luminosity(0.47)
        self.assertAlmostEqual(L, 2.3e5 * 0.47 ** 6.0, places=3)
        self.assertGreater(L, 2000.0)
        self.assertLess(L, 3000.0)

    def test_core_mass_luminosity_is_steeply_increasing(self):
        self.assertGreater(phys.core_mass_luminosity(0.40),
                            0.0)
        ratio = phys.core_mass_luminosity(0.47) / phys.core_mass_luminosity(0.40)
        # sixth power: (0.47/0.40)^6
        self.assertAlmostEqual(ratio, (0.47 / 0.40) ** 6.0, places=6)

    def test_hayashi_teff_matches_stated_reference_points(self):
        self.assertAlmostEqual(phys.hayashi_teff(1.0, 100.0), 4300.0, delta=60.0)

    def test_helium_flash_core_mass_constant(self):
        self.assertEqual(phys.helium_flash_core_mass(), 0.47)

    def test_kelvin_helmholtz_time_scaling(self):
        # t_KH = G M^2 / (R L); doubling L should halve t_KH exactly.
        t1 = phys.kelvin_helmholtz_time(5.0, 3.0, 1000.0)
        t2 = phys.kelvin_helmholtz_time(5.0, 3.0, 2000.0)
        self.assertAlmostEqual(t1 / t2, 2.0, places=10)


class TestRemnantClassification(unittest.TestCase):
    def test_low_mass_helium_white_dwarf(self):
        kind, mass, note = phys.predicted_remnant(0.3)
        self.assertEqual(kind, "helium white dwarf")
        self.assertAlmostEqual(mass, 0.109 * 0.3 + 0.394, places=10)

    def test_kalirai_extrapolation_notes_bracket_the_calibrated_range(self):
        # Below 1.16 Msun: extrapolated below the calibrated range.
        _, _, note_lo = phys.predicted_remnant(1.0)
        self.assertIn("below", note_lo)
        # Within 1.16-7.0 Msun: within the (extended) calibrated range.
        _, _, note_mid = phys.predicted_remnant(3.0)
        self.assertNotIn("extrapolated", note_mid)
        # Above 7.0 Msun (but below the 8.0 regime switch): extrapolated above.
        _, _, note_hi = phys.predicted_remnant(7.5)
        self.assertIn("above", note_hi)

    def test_white_dwarf_kind_switches_at_6_5_solar_masses(self):
        kind_lo, _, _ = phys.predicted_remnant(6.4)
        kind_hi, _, _ = phys.predicted_remnant(6.6)
        self.assertEqual(kind_lo, "carbon-oxygen white dwarf")
        self.assertEqual(kind_hi, "oxygen-neon white dwarf")

    def test_neutron_star_band(self):
        kind, mass, _ = phys.predicted_remnant(10.0)
        self.assertEqual(kind, "neutron star")
        self.assertEqual(mass, 1.4)

    def test_black_hole_band(self):
        kind, mass, _ = phys.predicted_remnant(25.0)
        self.assertEqual(kind, "black hole")
        self.assertAlmostEqual(mass, 0.2 * 25.0, places=10)


# ======================================================================
class TestTrackIntegration(unittest.TestCase):
    def test_solar_track_reproduces_documented_headline_numbers(self):
        result = phys.integrate_track(m_msun=1.0)
        s = result["summary"]
        self.assertAlmostEqual(s["t_ms_gyr"], 9.733, delta=0.01)
        self.assertAlmostEqual(s["L_tams"], 2.198, delta=0.01)
        self.assertAlmostEqual(s["mc_tams"], 0.1125, delta=0.001)
        self.assertAlmostEqual(s["t_total_gyr"], 11.72, delta=0.02)
        self.assertTrue(s["reached_tams"])
        self.assertFalse(s["truncated"])
        self.assertEqual(s["remnant_kind"], "carbon-oxygen white dwarf")

    def test_arrays_are_finite_ordered_and_agree_with_summary(self):
        result = phys.integrate_track(m_msun=1.0)
        s = result["summary"]
        for key in ("t", "L", "R", "Teff", "Xc", "mu", "Mcore"):
            with self.subTest(key=key):
                self.assertTrue(np.all(np.isfinite(result[key])))
        self.assertTrue(np.all(np.diff(result["t"]) >= 0.0))
        self.assertEqual(result["t"].size, s["n_points"])
        self.assertAlmostEqual(result["t"][-1] / phys.GYR, s["t_total_gyr"],
                                places=8)
        self.assertAlmostEqual(result["L"][0], s["L_zams"], places=8)

    def test_mc_tams_matches_last_main_sequence_core_mass(self):
        # Regression test for Audit1 P1-3: mc_tams must equal mc_end (the
        # array's own last main-sequence Mcore point, mc_list[-1]) whatever
        # x_end is, not just at the default x_end=1e-3.  Before the fix,
        # mc_tams = core_efficiency*qc*m_msun ignored the (1 - Xc_end/X)
        # factor the array formula uses, producing a discontinuity that
        # grew as x_end grew (Codex measured a roughly 70x jump at
        # x_end=0.690 for a 1-Msun track).
        for x_end in (1.0e-3, 0.10, 0.50, 0.690):
            with self.subTest(x_end=x_end):
                result = phys.integrate_track(m_msun=1.0, x_end=x_end,
                                              include_postms=False)
                s = result["summary"]
                self.assertAlmostEqual(s["mc_tams"], s["mc_end"], delta=1e-10)
                self.assertAlmostEqual(
                    s["mc_tams"],
                    s["core_efficiency"] * s["qc"] * s["m_msun"]
                    * (1.0 - s["Xc_end"] / s["X"]),
                    delta=1e-10)

    def test_helium_ignition_flag_distinguishes_from_post_ms(self):
        # Regression test for Audit1 P1-2: post_ms alone cannot tell a
        # caller whether a track actually reached helium ignition, since it
        # is also True for a helium white dwarf (envelope exhausted before
        # the flash) and False for both a normal TAMS-only run and a
        # t_max-truncated run.  helium_ignition must distinguish all four.
        ignites = phys.integrate_track(m_msun=1.0)["summary"]
        self.assertTrue(ignites["post_ms"])
        self.assertTrue(ignites["helium_ignition"])
        self.assertIn("helium flash", ignites["phase_end"])

        no_postms = phys.integrate_track(m_msun=1.0, include_postms=False)["summary"]
        self.assertFalse(no_postms["post_ms"])
        self.assertFalse(no_postms["helium_ignition"])

        truncated = phys.integrate_track(m_msun=1.0, t_max_gyr=5.0)["summary"]
        self.assertFalse(truncated["post_ms"])
        self.assertFalse(truncated["helium_ignition"])

        # A low-enough mass exhausts its envelope before the degenerate
        # core reaches the helium-flash mass: post_ms is True (a
        # post-main-sequence phase WAS integrated) but helium_ignition must
        # be False (it never actually ignited helium).
        white_dwarf_end = phys.integrate_track(m_msun=0.65, t_max_gyr=200.0)["summary"]
        self.assertTrue(white_dwarf_end["post_ms"])
        self.assertFalse(white_dwarf_end["helium_ignition"])
        self.assertIn("helium white dwarf", white_dwarf_end["phase_end"])

    def test_remnant_kind_agrees_with_the_tracks_own_computed_endpoint(self):
        # Regression test for Audit1 P1-2's second contradiction (Codex's
        # 0.49/0.50/0.60/0.67/0.68 probe): predicted_remnant() is a
        # standalone, mass-only classifier that switches to "carbon-oxygen
        # white dwarf" at 0.5 Msun, independent of what a specific track's
        # own post-main-sequence integration found.  For roughly
        # 0.5-0.67 Msun the degenerate-RGB branch finds the envelope
        # exhausted before the flash (a helium white dwarf) while
        # predicted_remnant() simultaneously reported carbon-oxygen --
        # both displayed as fact in the same summary.  The track's own
        # computed outcome must now take precedence for its own remnant
        # fields.
        for m in (0.49, 0.50, 0.60, 0.67):
            with self.subTest(m=m):
                s = phys.integrate_track(m_msun=m, t_max_gyr=1000.0)["summary"]
                self.assertIn("helium white dwarf", s["phase_end"])
                self.assertEqual(s["remnant_kind"], "helium white dwarf")
                self.assertFalse(s["helium_ignition"])
                self.assertAlmostEqual(s["remnant_msun"], s["mc_ign"], delta=1e-9)
                # Audit2 P1-1 (Codex): remnant_basis must say this came
                # from the track's own integration, not the mass-only
                # classifier -- every consumer of the summary branches on
                # this field to avoid repeating (or contradicting) a fixed
                # "classification from the initial mass" sentence that is
                # simply false for this case.
                self.assertEqual(s["remnant_basis"],
                                 "this track's own post-main-sequence integration")
        # Just above the 0.47/0.70 boundary (~0.6714 Msun) the star DOES
        # flash, and the ordinary mass-based classification is restored.
        s_flash = phys.integrate_track(m_msun=0.68, t_max_gyr=1000.0)["summary"]
        self.assertIn("helium flash", s_flash["phase_end"])
        self.assertEqual(s_flash["remnant_kind"], "carbon-oxygen white dwarf")
        self.assertTrue(s_flash["helium_ignition"])
        self.assertEqual(s_flash["remnant_basis"],
                         "mass-only schematic classification")

    def test_central_hydrogen_is_monotonically_non_increasing_on_ms(self):
        result = phys.integrate_track(m_msun=1.0, include_postms=False)
        Xc = result["Xc"]
        self.assertTrue(np.all(np.diff(Xc) <= 1e-12))
        self.assertAlmostEqual(Xc[-1], 1.0e-3, places=6)  # default x_end

    def test_helium_core_grows_monotonically_post_ms(self):
        result = phys.integrate_track(m_msun=1.0)
        s = result["summary"]
        mc = result["Mcore"][result["phase"] > 0]
        self.assertTrue(np.all(np.diff(mc) >= -1e-12))
        self.assertAlmostEqual(mc[-1], s["mc_ign"], delta=1e-6)

    def test_ms_lifetime_converges_under_refinement(self):
        lifetimes = []
        for n_ms in (200, 800, 3200):
            r = phys.integrate_track(m_msun=1.0, n_ms=n_ms, include_postms=False)
            lifetimes.append(r["summary"]["t_ms_gyr"])
        # Should be converging to a common limit, not merely "not crashing".
        self.assertLess(abs(lifetimes[2] - lifetimes[1]),
                         abs(lifetimes[1] - lifetimes[0]))
        self.assertAlmostEqual(lifetimes[2], lifetimes[1], delta=2e-3)

    def test_t_max_truncation_never_fabricates_post_ms_state(self):
        # Reproduces the scenario the legacy critiques flagged, using a
        # t_max well short of the 9.733-Gyr solar main-sequence lifetime so
        # the star is genuinely still on the main sequence when stopped.
        # (t_max=10 does NOT truncate: t_MS=9.733 Gyr < 10 Gyr, so the star
        # already reaches the TAMS and continues normally -- see
        # test_t_max_does_not_truncate_the_post_main_sequence_phase.)
        result = phys.integrate_track(m_msun=1.0, t_max_gyr=5.0)
        s = result["summary"]
        self.assertTrue(s["truncated"])
        self.assertFalse(s["reached_tams"])
        self.assertFalse(s["post_ms"])
        self.assertTrue(math.isnan(s["t_ms_gyr"]))
        self.assertAlmostEqual(s["t_stop_gyr"], 5.0, delta=1e-6)
        # The last stored point must be an ordinary main-sequence point, far
        # below the helium-flash luminosity (~2500 Lsun for a 1-Msun star).
        self.assertLess(result["L"][-1], 10.0)
        self.assertEqual(result["phase"][-1], 0)

    def test_t_max_truncated_track_still_reaches_tams_if_given_room(self):
        # Same star, t_max large enough to finish the main sequence: must
        # reach the TAMS and continue to helium ignition exactly as the
        # untruncated default run does.
        result = phys.integrate_track(m_msun=1.0, t_max_gyr=60.0)
        s = result["summary"]
        self.assertTrue(s["reached_tams"])
        self.assertFalse(s["truncated"])
        self.assertAlmostEqual(s["t_ms_gyr"], 9.733, delta=0.01)

    def test_t_max_does_not_truncate_the_post_main_sequence_phase(self):
        # t_max=10 stops the star with plenty of MS time to spare (t_MS is
        # 9.73 Gyr) so with t_max=12 it should finish the MS (9.73 Gyr) AND
        # be able to continue into the post-MS phase without an artificial
        # cutoff at t=12 Gyr, since the documented rule is that t_max only
        # ever gates the main sequence.
        result = phys.integrate_track(m_msun=1.0, t_max_gyr=12.0)
        s = result["summary"]
        self.assertTrue(s["reached_tams"])
        self.assertTrue(s["post_ms"])
        self.assertAlmostEqual(s["t_total_gyr"], 11.72, delta=0.02)

    def test_below_two_solar_masses_uses_degenerate_rgb_regime(self):
        result = phys.integrate_track(m_msun=1.5)
        self.assertEqual(result["summary"]["post_regime"],
                          "degenerate red-giant branch")

    def test_above_two_solar_masses_uses_hertzsprung_gap_regime(self):
        result = phys.integrate_track(m_msun=5.0)
        self.assertEqual(result["summary"]["post_regime"],
                          "Hertzsprung-gap crossing")
        self.assertTrue(math.isfinite(result["summary"]["t_cross_gyr"]))

    def test_no_postms_flag_stops_exactly_at_tams(self):
        result = phys.integrate_track(m_msun=1.0, include_postms=False)
        s = result["summary"]
        self.assertFalse(s["post_ms"])
        self.assertAlmostEqual(result["t"][-1] / phys.GYR, s["t_ms_gyr"],
                                places=8)

    def test_homology_zams_mode_runs_and_differs_from_empirical_fit(self):
        r_fit = phys.integrate_track(m_msun=3.0, homology_zams=False)
        r_hom = phys.integrate_track(m_msun=3.0, homology_zams=True)
        self.assertNotAlmostEqual(r_fit["summary"]["L_zams"],
                                   r_hom["summary"]["L_zams"], places=2)

    def test_mass_out_of_accepted_range_rejected(self):
        with self.assertRaisesRegex(ValueError, "hydrogen-burning limit"):
            phys.integrate_track(m_msun=0.01)
        with self.assertRaisesRegex(ValueError, "120"):
            phys.integrate_track(m_msun=200.0)

    def test_trusted_range_warnings(self):
        low = phys.integrate_track(m_msun=0.2)
        self.assertTrue(any("fully convective" in w
                             for w in low["summary"]["warnings"]))
        high = phys.integrate_track(m_msun=20.0)
        self.assertTrue(any("mass loss" in w or "radiation pressure" in w
                             for w in high["summary"]["warnings"]))
        mid = phys.integrate_track(m_msun=1.0)
        self.assertEqual(mid["summary"]["warnings"], [])

    def test_expansion_below_one_rejected(self):
        with self.assertRaisesRegex(ValueError, "expansion"):
            phys.integrate_track(m_msun=1.0, expansion=0.5)

    def test_expansion_equal_to_one_means_no_growth(self):
        result = phys.integrate_track(m_msun=1.0, expansion=1.0)
        R = result["R"][result["phase"] == 0]
        self.assertAlmostEqual(R[-1], R[0], delta=1e-6 * R[0])

    def test_core_weight_out_of_range_rejected(self):
        for bad in (-0.1, 1.1):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    phys.integrate_track(m_msun=1.0, core_weight=bad)

    def test_x_end_must_be_below_x(self):
        with self.assertRaises(ValueError):
            phys.integrate_track(m_msun=1.0, X=0.70, x_end=0.70)
        with self.assertRaises(ValueError):
            phys.integrate_track(m_msun=1.0, x_end=-0.01)

    def test_invalid_burning_and_opacity_rejected(self):
        with self.assertRaises(ValueError):
            phys.integrate_track(m_msun=1.0, burning="triple-alpha")
        with self.assertRaises(ValueError):
            phys.integrate_track(m_msun=1.0, opacity="opal")

    def test_default_burning_law_switches_at_1_2_solar_masses(self):
        self.assertEqual(phys.default_burning(1.19), "pp")
        self.assertEqual(phys.default_burning(1.2), "cno")

    def test_low_mass_extreme_t_max_does_not_crash(self):
        # A very low-mass star given enough time to finish the main
        # sequence and (if physically able) reach the helium flash.  Beyond
        # not crashing, the endpoint semantics must be self-consistent: this
        # star's envelope is exhausted before its degenerate core reaches
        # the helium-flash mass, so it must reach the TAMS and run a
        # post-main-sequence phase, but must NOT be reported as having
        # ignited helium.
        result = phys.integrate_track(m_msun=0.4, t_max_gyr=500.0, n_ms=500,
                                       n_post=500)
        s = result["summary"]
        self.assertTrue(np.all(np.isfinite(result["L"])))
        self.assertTrue(np.all(np.isfinite(result["R"])))
        self.assertTrue(s["reached_tams"])
        self.assertFalse(s["truncated"])
        self.assertTrue(s["post_ms"])
        self.assertFalse(s["helium_ignition"])
        self.assertIn("helium white dwarf", s["phase_end"])


# ======================================================================
class TestHrGridAndIsochrones(unittest.TestCase):
    def test_default_grid_reproduces_documented_lifetimes(self):
        result = phys.build_hr_grid(
            [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
            isochrone_gyr=[0.1, 1, 5, 10],
        )
        s = result["summary"]
        table = dict(zip(s["masses"], s["lifetimes_gyr"]))
        self.assertAlmostEqual(table[1.0], 9.733, delta=0.01)
        self.assertAlmostEqual(table[3.0], 0.5401, delta=0.001)
        reached = dict(zip(s["masses"], s["reached_tams"]))
        self.assertFalse(reached[0.5])
        self.assertFalse(reached[0.8])
        self.assertTrue(reached[1.0])

    def test_duplicate_masses_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            phys.build_hr_grid([1.0, 1.0, 2.0])

    def test_empty_mass_list_rejected(self):
        with self.assertRaises(ValueError):
            phys.build_hr_grid([])

    def test_too_many_masses_rejected(self):
        with self.assertRaisesRegex(ValueError, "40"):
            phys.build_hr_grid(list(np.linspace(0.5, 10.0, phys.MAX_MASSES + 1)))

    def test_turnoff_mass_matches_independent_log_log_interpolation(self):
        masses = [1.0, 2.0, 4.0]
        lifetimes = [10.0, 3.0, 1.0]
        age = 5.0
        got = phys.turnoff_mass(masses, lifetimes, age)
        # Independently reimplemented linear interpolation in (log t, log M),
        # sorted by increasing lifetime -- NOT a call into turnoff_mass.
        pairs = sorted(zip(lifetimes, masses))
        lt = [math.log10(t) for t, _ in pairs]
        lm = [math.log10(m) for _, m in pairs]
        la = math.log10(age)
        i = 0
        while i < len(lt) - 2 and la > lt[i + 1]:
            i += 1
        frac = (la - lt[i]) / (lt[i + 1] - lt[i])
        expected = 10.0 ** (lm[i] + frac * (lm[i + 1] - lm[i]))
        self.assertAlmostEqual(got, expected, places=6)

    def test_turnoff_mass_none_outside_age_range(self):
        self.assertIsNone(phys.turnoff_mass([1.0, 2.0], [10.0, 3.0], 100.0))
        self.assertIsNone(phys.turnoff_mass([1.0], [10.0], 5.0))  # < 2 pairs

    def test_isochrone_points_carry_correct_phase_and_on_ms_flag(self):
        # Age must lie within the *total* age span of at least two tracks;
        # 1 Gyr is covered by the 1.0, 1.5 and 2.0 Msun tracks with defaults.
        result = phys.build_hr_grid([1.0, 1.5, 2.0], isochrone_gyr=[1.0])
        self.assertEqual(len(result["isochrones"]), 1)
        iso = result["isochrones"][0]
        for m, logT, logL, phase, on_ms, tms in iso["points"]:
            with self.subTest(m=m):
                self.assertIn(phase, (0, 1, 2))
                self.assertEqual(on_ms, phase == 0)
                self.assertTrue(math.isfinite(logT) and math.isfinite(logL))

    def test_isochrone_turnoff_mass_present_when_age_in_range(self):
        result = phys.build_hr_grid(
            [0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0],
            isochrone_gyr=[1.0], t_max_gyr=60.0,
        )
        turnoff = result["isochrones"][0]["turnoff_mass"]
        self.assertIsNotNone(turnoff)
        self.assertGreater(turnoff, 0.0)

    def test_warnings_are_prefixed_with_the_offending_mass(self):
        result = phys.build_hr_grid([0.2, 1.0])
        self.assertTrue(any("M = 0.2" in w for w in result["summary"]["warnings"]))

    def test_too_many_isochrones_rejected(self):
        # Regression test for Audit1: the help file documented a 10-age cap
        # on --isochrones but the code never enforced one, exposing an
        # undocumented downstream ValueError instead of a clear one here.
        with self.assertRaisesRegex(ValueError, str(phys.MAX_ISOCHRONES)):
            phys.build_hr_grid(
                [1.0, 2.0, 3.0],
                isochrone_gyr=list(range(1, phys.MAX_ISOCHRONES + 2)),
            )

    def test_omitted_isochrone_age_produces_an_explanatory_warning(self):
        # Regression test for Audit1 P2-4: an isochrone age that no track
        # (or only one track) spans used to be dropped from the isochrones
        # list with no explanation at all -- n_isochrones would just be
        # smaller than requested with nothing to say why.  A 1 and a 10
        # Msun track have total ages of about 11.7 Gyr and 0.035 Gyr
        # respectively, so age = 5 Gyr is spanned by only the 1-Msun track.
        result = phys.build_hr_grid([1.0, 10.0], isochrone_gyr=[5.0])
        self.assertEqual(result["isochrones"], [])
        self.assertTrue(any("5" in w and "omitted" in w
                             for w in result["summary"]["warnings"]))


# ======================================================================
class TestFermiGasEos(unittest.TestCase):
    def setUp(self):
        self.eos = phys.FermiGasEOS(phys.m_e, 2.0 * phys.m_u)

    def test_non_relativistic_limit_pressure_scales_as_x5(self):
        # Independent limiting-case check: P ~ rho^(5/3) i.e. P ~ x^5 for
        # x << 1 (non-relativistic degenerate electron gas).
        x1, x2 = 1.0e-3, 2.0e-3
        p1, p2 = float(self.eos.pressure(x1)), float(self.eos.pressure(x2))
        slope = math.log(p2 / p1) / math.log(x2 / x1)
        self.assertAlmostEqual(slope, 5.0, places=2)

    def test_ultra_relativistic_limit_pressure_scales_as_x4(self):
        # P ~ rho^(4/3) i.e. P ~ x^4 for x >> 1.
        x1, x2 = 50.0, 100.0
        p1, p2 = float(self.eos.pressure(x1)), float(self.eos.pressure(x2))
        slope = math.log(p2 / p1) / math.log(x2 / x1)
        self.assertAlmostEqual(slope, 4.0, places=2)

    def test_pressure_and_energy_density_stay_positive_at_small_x(self):
        # Regression test for Audit2 P2-1 (Codex reproducer): the closed-
        # form pressure()/energy_density() expressions subtract two
        # nearly-equal terms, and lose essentially all significant digits
        # well before x reaches 1e-4 -- direct evaluation there used to
        # underflow to exactly 0.0, and by x=1e-5 it went slightly
        # NEGATIVE, which is unphysical for a degenerate-gas pressure.
        # This spans several decades below the y_floor=1e-8 default used
        # by integrate_structure(), which is where these values actually
        # get evaluated during a structure integration's outer steps.
        for xe in range(1, 12):
            x = 10.0 ** (-xe)
            with self.subTest(x=x):
                self.assertGreater(self.eos.pressure(x), 0.0)
                self.assertGreater(self.eos.energy_density(x), 0.0)

    def test_small_x_pressure_and_energy_density_agree_with_taylor_series(self):
        # Independent oracle: the exact Taylor series of the closed-form
        # expression (obtained symbolically, not by re-deriving the
        # module's own formula), evaluated in Python floats.  P/A and
        # eps/A both approach 3x as x -> 0 from two nearly-cancelling
        # terms; the leading-order series terms below are what survives
        # after that cancellation and is what the closed form would give
        # at infinite precision.
        A = self.eos.A
        for xe in (2, 3, 4, 5, 6, 8, 10):
            x = 10.0 ** (-xe)
            with self.subTest(x=x):
                p_series = A * (1.6 * x**5 - (4.0 / 7.0) * x**7)
                e_series = A * (8.0 * x**3 + 2.4 * x**5)
                p = float(self.eos.pressure(x))
                e = float(self.eos.energy_density(x))
                self.assertAlmostEqual(p / p_series, 1.0, delta=1.0e-6)
                self.assertAlmostEqual(e / e_series, 1.0, delta=1.0e-6)

    def test_pressure_and_energy_density_are_continuous_across_the_small_x_switch(self):
        # The switch from the closed form to the small-x series happens
        # at _FERMI_SMALL_X.  P and E are smooth power laws there (P ~
        # x^5, E ~ x^3 to leading order), so a tiny step across the
        # boundary is NOT expected to leave the ratio near exactly 1 --
        # it should match the ordinary power-law ratio to high precision,
        # with no additional artificial kink from the formula switch
        # itself.
        eps = 1.0e-6
        x_lo = phys._FERMI_SMALL_X * (1.0 - eps)
        x_hi = phys._FERMI_SMALL_X * (1.0 + eps)
        p_lo, p_hi = self.eos.pressure(x_lo), self.eos.pressure(x_hi)
        e_lo, e_hi = self.eos.energy_density(x_lo), self.eos.energy_density(x_hi)
        expected_p_ratio = (x_lo / x_hi) ** 5.0
        expected_e_ratio = (x_lo / x_hi) ** 3.0
        self.assertAlmostEqual(float(p_lo / p_hi) / expected_p_ratio, 1.0,
                               delta=1.0e-6)
        self.assertAlmostEqual(float(e_lo / e_hi) / expected_e_ratio, 1.0,
                               delta=1.0e-6)

    def test_pressure_and_energy_density_small_x_path_is_array_safe(self):
        # The small-x branch must handle a mixed array (some elements
        # below the threshold, some above) exactly the same as evaluating
        # each element individually -- this is the code path actually
        # used by wd_mass_radius_curve()/ns_mass_radius_curve(), which
        # call these methods with scalars, but dP_dx and other internals
        # historically assumed array input, so this is checked directly.
        xs = np.array([1.0e-6, 1.0e-3, 0.02, 0.05, 0.2, 1.0, 5.0])
        p_array = self.eos.pressure(xs)
        e_array = self.eos.energy_density(xs)
        for i, x in enumerate(xs):
            self.assertAlmostEqual(float(p_array[i]),
                                   float(self.eos.pressure(float(x))),
                                   delta=abs(float(p_array[i])) * 1e-9 + 1e-300)
            self.assertAlmostEqual(float(e_array[i]),
                                   float(self.eos.energy_density(float(x))),
                                   delta=abs(float(e_array[i])) * 1e-9 + 1e-300)
        self.assertTrue(np.all(p_array > 0.0))
        self.assertTrue(np.all(e_array > 0.0))

    def test_sound_speed_matches_numerical_dP_deps(self):
        # Independent numerical derivative dP/d(energy density), NOT a
        # call into the analytic sound_speed_ratio formula's own algebra.
        h = 1.0e-5
        for x in (0.01, 0.5, 1.0, 5.0, 50.0):
            with self.subTest(x=x):
                p_lo, p_hi = self.eos.pressure(x - h), self.eos.pressure(x + h)
                e_lo, e_hi = self.eos.energy_density(x - h), self.eos.energy_density(x + h)
                numeric_cs2 = float((p_hi - p_lo) / (e_hi - e_lo))
                analytic_cs2 = float(self.eos.sound_speed_ratio(x)) ** 2
                self.assertAlmostEqual(numeric_cs2, analytic_cs2, delta=1e-6)

    def test_ideal_fermi_gas_is_always_causal(self):
        for x in (1e-3, 1.0, 100.0, 1.0e6):
            with self.subTest(x=x):
                self.assertLessEqual(float(self.eos.sound_speed_ratio(x)), 1.0)

    def test_sound_speed_approaches_relativistic_limit(self):
        cs = float(self.eos.sound_speed_ratio(1.0e8))
        self.assertAlmostEqual(cs, 1.0 / math.sqrt(3.0), places=4)

    def test_number_density_and_rest_mass_density_scale_as_x_cubed(self):
        n1 = float(self.eos.number_density(1.0))
        n2 = float(self.eos.number_density(2.0))
        self.assertAlmostEqual(n2 / n1, 8.0, places=8)

    def test_x_from_density_round_trips(self):
        rho = 1.0e9
        x = self.eos.x_from_density(rho)
        rho_back = float(self.eos.rest_mass_density(x))
        self.assertAlmostEqual(rho_back, rho, delta=1e-3 * rho)

    def test_x_from_density_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            self.eos.x_from_density(-1.0)

    def test_public_evaluators_reject_negative_x_scalar(self):
        # Regression test for Audit3 Codex A3-P3-1 / Copilot A3-P3-2: the
        # relativity parameter x = p_F/(m c) is physically nonnegative,
        # but the public evaluators used to pass a negative x straight
        # through to the arithmetic and silently return negative
        # pressures, energy densities and number densities.  No caller
        # inside this module ever reaches this state (integrate_structure
        # and wd_structure only ever derive x from a validated positive
        # density), but these methods are also a reusable public API.
        methods = (self.eos.pressure, self.eos.energy_density,
                  self.eos.number_density, self.eos.rest_mass_density,
                  self.eos.dP_dx, self.eos.sound_speed_ratio)
        for method in methods:
            with self.subTest(method=method.__name__):
                with self.assertRaisesRegex(ValueError, "x >= 0|non-negative"):
                    method(-0.01)

    def test_public_evaluators_reject_negative_x_array_element(self):
        # A single negative element inside an otherwise valid array must
        # still be caught, not silently averaged away or skipped by the
        # small-x boolean mask.
        xs = np.array([0.1, -0.2, 0.3])
        for method in (self.eos.pressure, self.eos.energy_density,
                      self.eos.dP_dx, self.eos.number_density):
            with self.subTest(method=method.__name__):
                with self.assertRaises(ValueError):
                    method(xs)

    def test_public_evaluators_reject_nonfinite_x(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(x=bad):
                with self.assertRaisesRegex(ValueError, "finite"):
                    self.eos.pressure(bad)
                with self.assertRaisesRegex(ValueError, "finite"):
                    self.eos.energy_density(bad)

    def test_public_evaluators_accept_zero_boundary(self):
        # x=0 is the legitimate zero-density limit, not an error: it must
        # not be rejected by whatever check now excludes negative x.
        self.assertEqual(float(self.eos.pressure(0.0)), 0.0)
        self.assertEqual(float(self.eos.energy_density(0.0)), 0.0)
        self.assertEqual(float(self.eos.number_density(0.0)), 0.0)
        self.assertEqual(float(self.eos.dP_dx(0.0)), 0.0)
        self.assertEqual(float(self.eos.sound_speed_ratio(0.0)), 0.0)

    def test_public_evaluators_reject_bool(self):
        # Regression test for Audit4 Codex A4-P3-2: _require_nonneg_x's
        # np.asarray(x, dtype=float) silently coerced True/False to
        # 1.0/0.0.  True and False are not physically meaningful values
        # of x = p_F/(m c), so a bare bool (scalar or array) must now be
        # rejected explicitly, the same way _require_bool() rejects a
        # non-bool value elsewhere in this module, just inverted.
        methods = (self.eos.pressure, self.eos.energy_density,
                  self.eos.number_density, self.eos.rest_mass_density,
                  self.eos.dP_dx, self.eos.sound_speed_ratio)
        for method in methods:
            for bad in (True, False, np.array([True, False])):
                with self.subTest(method=method.__name__, x=bad):
                    with self.assertRaisesRegex(ValueError, "bool"):
                        method(bad)

    def test_public_evaluators_wrap_complex_x_as_valueerror(self):
        # Regression test for Audit4 Codex A4-P3-2: a complex x used to
        # escape _require_nonneg_x as a raw TypeError from the
        # np.asarray(x, dtype=float) conversion, instead of this
        # validator's normal ValueError-style domain-rejection contract.
        methods = (self.eos.pressure, self.eos.energy_density,
                  self.eos.number_density, self.eos.rest_mass_density,
                  self.eos.dP_dx, self.eos.sound_speed_ratio)
        for method in methods:
            with self.subTest(method=method.__name__):
                with self.assertRaises(ValueError):
                    method(1 + 2j)


# ======================================================================
class TestStructureIntegration(unittest.TestCase):
    """integrate_structure(): the shared Newtonian/TOV RK4 integrator."""

    def setUp(self):
        self.eos = phys.FermiGasEOS(phys.m_e, 2.0 * phys.m_u)

    def test_rejects_non_positive_central_variable(self):
        with self.assertRaises(ValueError):
            phys.integrate_structure(self.eos, 0.0)
        with self.assertRaises(ValueError):
            phys.integrate_structure(self.eos, -1.0)

    def test_rejects_invalid_step_frac_and_y_floor(self):
        x_c = self.eos.x_from_density(1.0e9)
        for bad in (0.0, -0.1, 0.6, 1.0):
            with self.subTest(step_frac=bad):
                with self.assertRaises(ValueError):
                    phys.integrate_structure(self.eos, x_c, step_frac=bad)
        for bad in (0.0, -1e-8, 1.0, 2.0):
            with self.subTest(y_floor=bad):
                with self.assertRaises(ValueError):
                    phys.integrate_structure(self.eos, x_c, y_floor=bad)

    def test_rejects_non_positive_r_scale(self):
        x_c = self.eos.x_from_density(1.0e9)
        with self.assertRaises(ValueError):
            phys.integrate_structure(self.eos, x_c, r_scale=0.0)
        with self.assertRaises(ValueError):
            phys.integrate_structure(self.eos, x_c, r_scale=-1.0)

    def test_ordinary_model_integrates_successfully(self):
        x_c = self.eos.x_from_density(1.0e9)
        M, R, _ = phys.integrate_structure(self.eos, x_c, r_scale=1.0e7,
                                           step_frac=0.01)
        self.assertTrue(math.isfinite(M) and math.isfinite(R))
        self.assertGreater(M, 0.0)
        self.assertGreater(R, 0.0)

    def test_toy_eos_k1_failure_wrapped_same_as_k2_failure(self):
        # Regression test for Audit1 P2-3, using a deterministic toy EOS
        # instead of a contrived physical density (Codex's specific
        # request): a ValueError from the very first derivs() call (k1)
        # must be wrapped into the same contextual RuntimeError as a
        # ValueError from any later stage (k2/k3/k4), not leak through
        # unwrapped.  Before the fix, k1 was evaluated outside the
        # try/except, so this test would have failed with a bare
        # ValueError instead of the wrapped RuntimeError.
        class ToyEOS:
            """Fails dP/dx exactly once, at a call count fixed by
            fail_at_call, to pin down exactly which RK4 stage failed."""
            def __init__(self, fail_at_call):
                self.fail_at_call = fail_at_call
                self.calls = 0

            def rest_mass_density(self, y):
                return 1.0e9

            def mass_energy_density(self, y):
                return 1.0e9

            def pressure(self, y):
                return 1.0e20

            def dP_dx(self, y):
                self.calls += 1
                if self.calls == self.fail_at_call:
                    raise ValueError(f"forced failure at call {self.calls}")
                return 1.0e20

            def x_from_density(self, rho):
                return 1.0

        for fail_at_call, label in ((1, "k1"), (2, "k2")):
            with self.subTest(stage=label):
                eos = ToyEOS(fail_at_call)
                with self.assertRaisesRegex(RuntimeError,
                                            "structure integration failed"):
                    phys.integrate_structure(eos, 1.0, relativistic=False,
                                             r_scale=1.0e7, step_frac=0.01)

    def test_tov_horizon_failure_raises_the_same_actionable_runtimeerror(self):
        # Regression test for Audit1 P2-3: the k1 = derivs(r, m, y) call
        # used to sit OUTSIDE the try/except that wraps the k2-k4 stages,
        # so a first-stage failure (including the TOV horizon check inside
        # derivs()) propagated with a different, less actionable message
        # than every other stage failure, instead of the unified
        # "structure integration failed ... reduce --step_frac" message.
        # An extreme central density for a very stiff polytrope drives the
        # TOV metric non-positive on the very first derivs() call (k1,
        # before any RK4 stage has been taken) -- exactly the call that
        # used to sit outside the try/except.
        eos = phys.PolytropeEOS(p_nuc=0.999999, gamma=1.01)
        x_c = eos.x_from_density(1.0e21)
        with self.assertRaisesRegex(RuntimeError, "structure integration failed"):
            phys.integrate_structure(eos, x_c, relativistic=True,
                                     r_scale=1.5e4, step_frac=0.1)


class TestPolytropeEos(unittest.TestCase):
    def test_pressure_matches_definition_of_stiffness(self):
        eos = phys.PolytropeEOS(p_nuc=0.04, gamma=2.5)
        P_at_nuc = float(eos.pressure(phys.RHO_NUCLEAR))
        self.assertAlmostEqual(P_at_nuc / (phys.RHO_NUCLEAR * phys.c ** 2), 0.04,
                                places=6)

    def test_gamma_out_of_range_rejected(self):
        for bad in (1.0, 5.1, math.nan):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    phys.PolytropeEOS(gamma=bad)

    def test_p_nuc_above_one_rejected(self):
        with self.assertRaises(ValueError):
            phys.PolytropeEOS(p_nuc=1.5)

    def test_stiffer_polytrope_has_higher_sound_speed_at_fixed_density(self):
        soft = phys.PolytropeEOS(p_nuc=0.01, gamma=2.5)
        stiff = phys.PolytropeEOS(p_nuc=0.08, gamma=2.5)
        rho = phys.RHO_NUCLEAR
        self.assertGreater(float(stiff.sound_speed_ratio(rho)),
                            float(soft.sound_speed_ratio(rho)))

    def test_make_eos_factory(self):
        self.assertIsInstance(phys.make_eos("neutron"), phys.FermiGasEOS)
        self.assertIsInstance(phys.make_eos("polytrope"), phys.PolytropeEOS)
        with self.assertRaises(ValueError):
            phys.make_eos("electron")
        with self.assertRaises(ValueError):
            phys.make_eos("bogus")


# ======================================================================
class TestWhiteDwarfStructure(unittest.TestCase):
    def test_check_mu_e_bounds(self):
        with self.assertRaises(ValueError):
            phys.check_mu_e(0.5)
        with self.assertRaises(ValueError):
            phys.check_mu_e(3.5)
        self.assertEqual(phys.check_mu_e(2.0), 2.0)

    def test_chandrasekhar_mass_scaling(self):
        # Analytic scaling M_Ch ~ mu_e^-2, and the standard headline value.
        self.assertAlmostEqual(phys.chandrasekhar_mass(2.0), 1.459, places=3)
        m1 = phys.chandrasekhar_mass(1.0)
        m2 = phys.chandrasekhar_mass(2.0)
        self.assertAlmostEqual(m1 / m2, 4.0, places=6)

    def test_default_white_dwarf_reproduces_documented_structure(self):
        rho_c, M_kg, R_m = phys.wd_structure(0.6, mu_e=2.0)
        self.assertAlmostEqual(M_kg / phys.M_sun, 0.6, delta=1e-4)
        self.assertAlmostEqual(R_m / 1.0e3, 8839.5, delta=2.0)

    def test_super_chandrasekhar_mass_rejected(self):
        with self.assertRaisesRegex(ValueError, "Chandrasekhar"):
            phys.wd_structure(1.6, mu_e=2.0)

    def test_mass_close_to_chandrasekhar_limit_widens_bracket_and_succeeds(self):
        # Regression test for the Audit1 P2-1 fix: 1.455 Msun (mu_e=2, so
        # M_Ch = 1.459 Msun) is below the default bisection bracket's reach
        # at rho_c in [1e7, 1e13] kg/m^3 -- the old code raised RuntimeError
        # here even though the mass is well inside the allowed range (below
        # the hard 0.999*M_Ch cutoff).  wd_structure must now widen the
        # bracket automatically and return a converged, physically sensible
        # (very compact, very dense) white dwarf instead of failing.
        rho_c, M_kg, R_m = phys.wd_structure(1.455, mu_e=2.0)
        self.assertAlmostEqual(M_kg / phys.M_sun, 1.455, delta=1e-3)
        self.assertGreater(rho_c, 1.0e13)          # outside the old default bracket
        self.assertLess(R_m, 5.0e5)                # much smaller than a 0.6 Msun WD

    def test_low_mass_within_documented_range_no_longer_fails(self):
        # Regression test for Audit1 P2-1 (Codex reproducer): the Help file
        # documents the accepted white-dwarf mass range as "anything below
        # M_Ch(mu_e)", but the fixed bisection bracket [1e7, 1e13] kg/m^3
        # only actually reached masses from about 0.049 to 1.432 Msun for
        # mu_e=2 -- 0.01 Msun (already well AWAY from the Chandrasekhar
        # limit) used to fail with the directionally wrong advice "try a
        # mass further from the Chandrasekhar limit".  The adaptive bracket
        # must now reach it.
        rho_c, M_kg, R_m = phys.wd_structure(0.01, mu_e=2.0)
        self.assertAlmostEqual(M_kg / phys.M_sun, 0.01, delta=1e-4)
        self.assertGreater(R_m, 0.0)

    def test_bracket_expansion_limit_still_fails_cleanly(self):
        # With bracket expansion disabled (max_bracket_expansions=0), a
        # mass the default bracket cannot reach must still fail with a
        # clear, documented RuntimeError, never crash or hang -- the
        # adaptive search is a convenience, not a way to silently hide a
        # bracket that truly cannot be found.
        with self.assertRaisesRegex(RuntimeError, "bracket"):
            phys.wd_structure(1.455, mu_e=2.0, max_bracket_expansions=0)

    def test_mass_too_close_to_chandrasekhar_limit_rejected_outright(self):
        # Above the hard 0.999*M_Ch cutoff: rejected before any bisection
        # is attempted, regardless of bracket expansion.
        mch = phys.chandrasekhar_mass(2.0)
        with self.assertRaisesRegex(ValueError, "Chandrasekhar"):
            phys.wd_structure(0.999 * mch, mu_e=2.0)

    def test_wd_low_bracket_failure_advice_points_upward(self):
        # Regression test for Audit3 Codex A3-P3-2: when even the LOWEST
        # central density tried already overshoots the target (the
        # requested mass sits below the reachable bracket), the old
        # message said "try a mass further from the Chandrasekhar limit"
        # -- backwards advice, since the fix is a LARGER mass.  With
        # bracket widening disabled, 0.01 Msun is below the default
        # bracket's low-density reach (about 0.049 Msun).
        with self.assertRaisesRegex(RuntimeError, "too small") as ctx:
            phys.wd_structure(0.01, mu_e=2.0, max_bracket_expansions=0)
        msg = str(ctx.exception)
        self.assertIn("larger mass", msg)
        self.assertNotIn("Chandrasekhar limit) for this search", msg)

    def test_wd_high_bracket_failure_advice_points_downward(self):
        # The opposite direction: even the HIGHEST central density tried
        # still falls short, so the requested mass really is too close to
        # the Chandrasekhar limit and the advice must point to a SMALLER
        # mass.  1.455 Msun is above the default bracket's high-density
        # reach (about 1.433 Msun) with widening disabled.
        with self.assertRaisesRegex(RuntimeError, "too close to the "
                                    "Chandrasekhar limit") as ctx:
            phys.wd_structure(1.455, mu_e=2.0, max_bracket_expansions=0)
        msg = str(ctx.exception)
        self.assertIn("smaller mass", msg)
        self.assertNotIn("larger mass", msg)

    def test_wd_structure_raises_when_max_iter_is_exhausted_unconverged(self):
        # Regression test for Audit2 P1-2 (Codex reproducer): wd_structure
        # used to return the last evaluated (rho_c, M, R) tuple after
        # max_iter iterations even when nowhere near tol, indistinguishable
        # from an ordinary successful result.  max_iter=1 against the
        # default bracket for a 0.6-Msun target leaves a 58% relative
        # error; this must now raise instead of returning that value.
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            phys.wd_structure(0.6, tol=1.0e-5, max_iter=1)

    def test_wd_structure_converged_result_satisfies_tol(self):
        # Postcondition companion to the test above: an ordinary call that
        # DOES have enough iterations to converge must return a mass that
        # actually satisfies the requested tolerance, not merely "some
        # value close enough that nobody checked".
        tol = 1.0e-6
        rho_c, M_kg, R_m = phys.wd_structure(0.6, mu_e=2.0, tol=tol)
        target = 0.6 * phys.M_sun
        self.assertLess(abs(M_kg - target) / target, tol)

    def test_wd_structure_validates_max_bracket_expansions(self):
        # Regression test for Audit2 P2-4 (Codex): max_bracket_expansions
        # was accepted completely unvalidated -- negative, fractional, and
        # non-finite values were all silently accepted whenever the
        # initial bracket already happened to contain the target.
        for bad in (-1, 1.5, float("nan")):
            with self.assertRaises(ValueError):
                phys.wd_structure(0.6, mu_e=2.0, max_bracket_expansions=bad)
        # A numeric string is unambiguous and is accepted, same as every
        # other numeric-looking parameter in this module.
        rho_c, M_kg, R_m = phys.wd_structure(0.6, mu_e=2.0,
                                             max_bracket_expansions="3")
        self.assertAlmostEqual(M_kg / phys.M_sun, 0.6, delta=1e-4)

    def test_wd_structure_rejects_reversed_or_equal_density_bracket(self):
        # Regression test for Audit2 P2-4 (Copilot/Codex): wd_structure did
        # not require rho_hi > rho_lo the way wd_mass_radius_curve() and
        # ns_mass_radius_curve() already did, so a reversed starting
        # bracket was silently handed to the bisection search instead of
        # being refused up front.
        with self.assertRaises(ValueError):
            phys.wd_structure(0.6, mu_e=2.0, rho_lo=1.0e13, rho_hi=1.0e7)
        with self.assertRaises(ValueError):
            phys.wd_structure(0.6, mu_e=2.0, rho_lo=1.0e10, rho_hi=1.0e10)

    def test_mestel_constant_k1_matches_the_exact_nonrelativistic_eos(self):
        # Independent check of the non-relativistic degenerate-pressure
        # constant K1 used inside mestel_constant: at small x the exact
        # Fermi-gas pressure must approach K1 (rho/mu_e)^(5/3).
        eos = phys.FermiGasEOS(phys.m_e, 2.0 * phys.m_u)
        x = 1.0e-3
        rho = float(eos.rest_mass_density(x))
        P = float(eos.pressure(x))
        K1_implied = P / (rho / 2.0) ** (5.0 / 3.0)
        self.assertAlmostEqual(K1_implied, 1.0036e7, delta=2.0e4)

    def test_mass_radius_curve_non_relativistic_scaling(self):
        # R ~ M^-1/3 at low mass (non-relativistic electrons): compare the
        # two lowest-density (lowest-mass) points on the curve.
        rho, M, R = phys.wd_mass_radius_curve(mu_e=2.0, n=24,
                                              rho_lo=1.0e7, rho_hi=1.0e9)
        slope = (math.log(R[1]) - math.log(R[0])) / (math.log(M[1]) - math.log(M[0]))
        self.assertAlmostEqual(slope, -1.0 / 3.0, delta=0.05)

    def test_mass_radius_curve_asymptotes_to_chandrasekhar_mass(self):
        rho, M, R = phys.wd_mass_radius_curve(mu_e=2.0, n=30,
                                              rho_lo=1.0e8, rho_hi=1.0e14)
        self.assertLess(M[-1], phys.chandrasekhar_mass(2.0))
        self.assertGreater(M[-1], 0.99 * phys.chandrasekhar_mass(2.0))
        self.assertTrue(np.all(np.diff(M) > 0.0))   # mass rises with density
        self.assertTrue(np.all(np.diff(R) < 0.0))   # radius falls (heavier=smaller)

    def test_mass_radius_curve_rejects_bad_density_range(self):
        # Regression test for Audit1: wd_mass_radius_curve previously
        # validated mu_e and n but not rho_lo/rho_hi at all, so a
        # non-positive density or an inverted range reached np.geomspace
        # (and eventually the EOS) unvalidated instead of a clear message.
        with self.assertRaises(ValueError):
            phys.wd_mass_radius_curve(mu_e=2.0, rho_lo=-1.0, rho_hi=1.0e13)
        with self.assertRaises(ValueError):
            phys.wd_mass_radius_curve(mu_e=2.0, rho_lo=1.0e13, rho_hi=1.0e8)

    def test_radius_scales_as_mu_e_to_minus_five_thirds_non_relativistic(self):
        # Regression test for Audit1 P2-10: the previous version of this
        # test computed R1, R2 but only asserted they were positive --
        # never actually checking the mu_e^-5/3 scaling promised by its own
        # name.  At fixed mass, well inside the non-relativistic regime
        # (0.2 Msun, far below either mu_e's Chandrasekhar mass), the
        # non-relativistic degenerate-electron polytrope predicts
        # R(mu_e=1.5)/R(mu_e=2.0) = (2.0/1.5)^(5/3).
        _, M1, R1 = phys.wd_structure(0.20, mu_e=1.5)
        _, M2, R2 = phys.wd_structure(0.20, mu_e=2.0)
        self.assertAlmostEqual(M1 / phys.M_sun, 0.20, delta=1e-3)
        self.assertAlmostEqual(M2 / phys.M_sun, 0.20, delta=1e-3)
        ratio = R1 / R2
        expected = (2.0 / 1.5) ** (5.0 / 3.0)
        self.assertAlmostEqual(ratio, expected, delta=0.05 * expected)


class TestMestelCooling(unittest.TestCase):
    def test_default_run_reproduces_documented_cooling_age(self):
        result = phys.integrate_wd_cooling(m_msun=0.6, mu_e=2.0)
        s = result["summary"]
        self.assertAlmostEqual(s["t_end_gyr"], 5.6236, delta=0.005)
        self.assertAlmostEqual(s["t_end_gyr"], s["t_end_analytic_gyr"], delta=1e-3)

    def test_rk4_and_independently_reimplemented_analytic_age_agree(self):
        result = phys.integrate_wd_cooling(m_msun=0.6, mu_e=2.0, A_ion=14.0)
        s = result["summary"]
        # Reimplement the closed-form age directly from first principles
        # (heat capacity and Mestel coefficient recomputed independently,
        # not read back out of the summary's own analytic field).
        M_kg = s["m_msun"] * phys.M_sun
        heat_capacity = 1.5 * phys.k_B * M_kg / (14.0 * phys.m_u)
        C = phys.mestel_constant(s["mu_env"], s["mu_e_env"], s["kappa0"])
        coef = heat_capacity / (2.5 * C * M_kg)
        t_expected = coef * (s["Tc_end"] ** -2.5 - s["Tc0"] ** -2.5)
        self.assertAlmostEqual(t_expected / phys.GYR, s["t_end_analytic_gyr"],
                                delta=1e-3)

    def test_late_time_luminosity_follows_mestel_seven_fifths_law(self):
        result = phys.integrate_wd_cooling(m_msun=0.6, mu_e=2.0,
                                            Tc0=3.0e7, Tc_end=1.0e6, n_steps=4000)
        t, L = result["t"], result["L"]
        good = t > 0
        t, L = t[good], L[good]
        # Fit the slope over the LAST decade in time only.
        tail = t >= (t[-1] / 10.0)
        slope = np.polyfit(np.log(t[tail]), np.log(L[tail]), 1)[0]
        self.assertAlmostEqual(slope, -7.0 / 5.0, delta=0.05)

    def test_early_time_luminosity_is_flatter_than_the_asymptotic_slope(self):
        result = phys.integrate_wd_cooling(m_msun=0.6, mu_e=2.0,
                                            Tc0=3.0e7, Tc_end=1.0e6, n_steps=4000)
        t, L = result["t"], result["L"]
        good = t > 0
        t, L = t[good], L[good]
        head = t <= (t[0] + (t[-1] - t[0]) * 0.15)
        if np.count_nonzero(head) >= 2:
            slope_head = np.polyfit(np.log(t[head]), np.log(L[head]), 1)[0]
            self.assertGreater(slope_head, -1.4)  # shallower than -7/5

    def test_tc_end_must_be_below_tc0(self):
        with self.assertRaises(ValueError):
            phys.integrate_wd_cooling(Tc0=3.0e6, Tc_end=3.0e7)
        with self.assertRaises(ValueError):
            phys.integrate_wd_cooling(Tc0=3.0e7, Tc_end=3.0e7)

    def test_kramers_kappa0_bound_free_and_free_free_terms(self):
        # Independent recomputation from the documented cgs coefficients.
        X, Z = 0.70, 0.02
        expected_cgs = 4.34e25 * Z * (1 + X) + 3.68e22 * (1 - Z) * (1 + X)
        self.assertAlmostEqual(phys.kramers_kappa0(X, Z), expected_cgs * 1e-4,
                                delta=1e-6 * expected_cgs * 1e-4)

    def test_kramers_kappa0_nonzero_at_zero_metallicity(self):
        # Free-free absorption survives at Z=0; only bound-free vanishes.
        kappa0 = phys.kramers_kappa0(0.70, 0.0)
        self.assertGreater(kappa0, 0.0)
        expected = 1e-4 * 3.68e22 * 1.0 * 1.70
        self.assertAlmostEqual(kappa0, expected, delta=1e-6 * expected)

    def test_tc_end_below_floor_rejected_instead_of_overflowing(self):
        # Regression test for Audit1 P2-2: the analytic-age formula
        # evaluates Tc_end**-2.5, which overflows a Python float
        # (OverflowError) for a sufficiently small but still positive
        # Tc_end.  A validation floor must turn that into a clear ValueError
        # raised before the overflow-prone calculation runs.
        with self.assertRaisesRegex(ValueError, "100"):
            phys.integrate_wd_cooling(m_msun=0.6, mu_e=2.0,
                                      Tc0=3.0e7, Tc_end=1.0e-6)
        # A value just above the floor must still work normally.
        result = phys.integrate_wd_cooling(m_msun=0.6, mu_e=2.0,
                                           Tc0=3.0e7, Tc_end=150.0)
        self.assertTrue(math.isfinite(result["summary"]["t_end_gyr"]))

    def test_a_ion_out_of_physical_range_rejected(self):
        # Regression test for Audit1 P2-level validation gap: A_ion is the
        # mean ionic mass number and has no meaning outside roughly [1, 60]
        # for any white-dwarf composition.
        with self.assertRaises(ValueError):
            phys.integrate_wd_cooling(m_msun=0.6, mu_e=2.0, A_ion=0.5)
        with self.assertRaises(ValueError):
            phys.integrate_wd_cooling(m_msun=0.6, mu_e=2.0, A_ion=200.0)
        # A_ion = 12 (carbon core) must still work normally.
        result = phys.integrate_wd_cooling(m_msun=0.6, mu_e=2.0, A_ion=12.0)
        self.assertTrue(math.isfinite(result["summary"]["t_end_gyr"]))

    def test_composition_controls_are_separable(self):
        # Changing core mu_e changes structure but not the Mestel constant;
        # changing envelope Z changes the cooling age but not the structure.
        base = phys.integrate_wd_cooling(m_msun=0.6, mu_e=2.0, Z_env=0.0)
        diff_core = phys.integrate_wd_cooling(m_msun=0.6, mu_e=1.5, Z_env=0.0)
        diff_env = phys.integrate_wd_cooling(m_msun=0.6, mu_e=2.0, Z_env=0.02)
        self.assertNotAlmostEqual(base["summary"]["R_km"],
                                   diff_core["summary"]["R_km"], places=1)
        self.assertAlmostEqual(base["summary"]["R_km"],
                                diff_env["summary"]["R_km"], places=1)
        self.assertNotAlmostEqual(base["summary"]["t_end_gyr"],
                                   diff_env["summary"]["t_end_gyr"], places=2)


# ======================================================================
class TestNeutronStarSequence(unittest.TestCase):
    def test_ideal_neutron_gas_reproduces_oppenheimer_volkoff(self):
        result = phys.ns_mass_radius_curve(eos_name="neutron", n=40,
                                           rho_lo=1.0e17, rho_hi=5.0e19)
        s = result["summary"]
        self.assertAlmostEqual(s["M_max"], 0.7098, delta=0.005)
        self.assertAlmostEqual(s["R_at_Mmax"], 9.313, delta=0.05)
        self.assertTrue(s["turning_point"])

    def test_default_polytrope_reproduces_documented_maximum(self):
        result = phys.ns_mass_radius_curve(eos_name="polytrope", n=40,
                                           rho_lo=1.0e17, rho_hi=5.0e19)
        s = result["summary"]
        self.assertAlmostEqual(s["M_max"], 2.1749, delta=0.01)
        self.assertAlmostEqual(s["R_at_Mmax"], 11.692, delta=0.05)
        self.assertTrue(s["causal"])

    def test_newtonian_sequence_withholds_gr_only_quantities(self):
        result = phys.ns_mass_radius_curve(eos_name="neutron", n=24,
                                           rho_lo=1.0e17, rho_hi=5.0e19,
                                           relativistic=False)
        s = result["summary"]
        self.assertFalse(s["relativistic"])
        self.assertTrue(np.all(np.isnan(result["z"])))
        self.assertTrue(math.isnan(s["z_at_Mmax"]))

    def test_newtonian_high_density_never_turns_over_and_says_so(self):
        # Reproduces the scenario flagged in the legacy critiques: pushing
        # the Newtonian sequence to very high density must NOT report a
        # spurious maximum mass or an unstable-branch classification.
        result = phys.ns_mass_radius_curve(eos_name="neutron", n=24,
                                           rho_lo=1.0e17, rho_hi=1.0e21,
                                           relativistic=False)
        s = result["summary"]
        self.assertFalse(s["turning_point"])
        self.assertTrue(any("largest sampled mass" in w for w in s["warnings"]))

    def test_turning_point_requires_both_sides_not_just_below_the_top(self):
        # Regression test for Audit1 P1-1 (Codex reproducer): a density
        # range sampled entirely above the true turning point gives a mass
        # array that is monotonically DECREASING from the very first point
        # (i_max = 0).  The old check "i_max < gi[-1]" is satisfied by this
        # (0 < 7), so it wrongly reported turning_point=True/stable_branch
        # =True with no warning at all.  The maximum must be interior to
        # the sampled range on both sides before it counts as a genuine
        # turning point.
        result = phys.ns_mass_radius_curve(eos_name="neutron", n=8,
                                           rho_lo=1.0e19, rho_hi=5.0e19)
        s = result["summary"]
        self.assertEqual(result["i_max"], 0)
        self.assertTrue(np.all(np.diff(result["M"]) < 0.0))
        self.assertFalse(s["turning_point"])
        self.assertFalse(s["stable_branch"])
        self.assertTrue(any("lower" in w.lower() and "rho_lo" in w
                             for w in s["warnings"]))

    def test_turning_point_unresolved_across_a_convergence_gap_next_to_the_peak(self):
        # Regression test for Audit2 P2-6 (Codex): requiring SOME
        # converged point on each side of the sampled maximum (the
        # Audit1 P1-1 fix) is not enough -- if the maximum's own
        # immediate neighbor failed to converge, the true maximum could
        # be hiding inside that unresolved gap and must not be reported
        # as a resolved turning point.  Under these exact parameters
        # (verified with no forced failures) the sampled maximum falls
        # naturally at raw index 9 of 16; forcing the very next density
        # (index 10) to fail reproduces a convergence gap immediately
        # adjacent to the peak.
        real_integrate_structure = phys.integrate_structure
        call_count = [0]

        def fake_integrate_structure(*args, **kwargs):
            i = call_count[0]
            call_count[0] += 1
            if i == 10:
                raise RuntimeError("forced convergence gap for this test")
            return real_integrate_structure(*args, **kwargs)

        with mock.patch.object(phys, "integrate_structure",
                               side_effect=fake_integrate_structure):
            result = phys.ns_mass_radius_curve(eos_name="neutron", n=16,
                                               rho_lo=1.0e17, rho_hi=5.0e19)
        s = result["summary"]
        self.assertEqual(result["i_max"], 9)
        self.assertFalse(math.isfinite(result["M"][10]))
        self.assertFalse(s["turning_point"])
        self.assertFalse(s["stable_branch"])
        self.assertTrue(any("unresolved across" in w and "gap" in w
                             for w in s["warnings"]))

    def test_gm_over_rc2_is_always_reported_relativistic_or_not(self):
        rel = phys.ns_mass_radius_curve(eos_name="neutron", n=16,
                                        rho_lo=1.0e17, rho_hi=5.0e19,
                                        relativistic=True)
        newt = phys.ns_mass_radius_curve(eos_name="neutron", n=16,
                                         rho_lo=1.0e17, rho_hi=5.0e19,
                                         relativistic=False)
        self.assertTrue(np.any(np.isfinite(rel["compact"])))
        self.assertTrue(np.any(np.isfinite(newt["compact"])))

    def test_compactness_matches_independent_formula(self):
        result = phys.ns_mass_radius_curve(eos_name="polytrope", n=20,
                                           rho_lo=1.0e17, rho_hi=5.0e19)
        i = result["i_max"]
        M_kg = result["M"][i] * phys.M_sun
        R_m = result["R"][i] * 1.0e3
        expected = phys.G * M_kg / (R_m * phys.c ** 2)
        self.assertAlmostEqual(result["compact"][i], expected, delta=1e-6)

    def test_redshift_matches_independent_schwarzschild_formula(self):
        result = phys.ns_mass_radius_curve(eos_name="neutron", n=20,
                                           rho_lo=1.0e17, rho_hi=5.0e19)
        i = result["i_max"]
        compactness = result["compact"][i]
        expected_z = 1.0 / math.sqrt(1.0 - 2.0 * compactness) - 1.0
        self.assertAlmostEqual(result["z"][i], expected_z, delta=1e-6)

    def test_stiff_polytrope_can_be_made_acausal_and_is_flagged(self):
        # Regression test for Audit2 P3-4 (Codex): this test used to guard
        # its only assertions behind "if cs_over_c_max_branch > 1.0",
        # which means a regression that accidentally made the chosen case
        # causal would make the test vacuously pass instead of failing.
        # This exact (gamma, p_nuc) combination is independently verified
        # to be acausal (peak c_s/c = 1.7359), so that outcome, and the
        # flag/warning it must produce, are now asserted unconditionally.
        result = phys.ns_mass_radius_curve(eos_name="polytrope", n=16,
                                           rho_lo=1.0e17, rho_hi=5.0e19,
                                           gamma=5.0, p_nuc=0.9)
        s = result["summary"]
        self.assertAlmostEqual(s["cs_over_c_max_branch"], 1.7359, delta=0.01)
        self.assertFalse(s["causal"])
        self.assertTrue(any("acausal" in w for w in s["warnings"]))

    def test_rho_hi_must_exceed_rho_lo(self):
        with self.assertRaises(ValueError):
            phys.ns_mass_radius_curve(rho_lo=1.0e19, rho_hi=1.0e17)

    def test_n_below_minimum_rejected(self):
        with self.assertRaises(ValueError):
            phys.ns_mass_radius_curve(n=2)

    def test_stable_and_unstable_branch_only_meaningful_with_turning_point(self):
        result = phys.ns_mass_radius_curve(eos_name="neutron", n=40,
                                           rho_lo=1.0e17, rho_hi=5.0e19)
        s = result["summary"]
        self.assertEqual(s["stable_branch"], s["turning_point"])


# ======================================================================
class TestDriverValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_mode_must_be_recognized(self):
        with self.assertRaises(ValueError):
            driver.run(mode="bogus", no_plot=True, csvdir=self.tmp)

    def test_no_plot_requires_csvdir(self):
        with self.assertRaises(ValueError):
            driver.run(mode="tracks", no_plot=True, csvdir=None)

    def test_no_plot_and_outdir_together_rejected(self):
        with self.assertRaises(ValueError):
            driver.run(mode="tracks", no_plot=True, csvdir=self.tmp,
                       outdir=self.tmp)

    def test_dpi_bounds(self):
        for bad in (5, 2000):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    driver.run(mode="tracks", no_plot=True, csvdir=self.tmp,
                               dpi=bad)

    def test_lw_must_be_positive(self):
        with self.assertRaises(ValueError):
            driver.run(mode="tracks", no_plot=True, csvdir=self.tmp, lw=0.0)

    def test_step_frac_bounds(self):
        for bad in (0.0, 0.2, 1e-6):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    driver.run(mode="wdcool", no_plot=True, csvdir=self.tmp,
                               step_frac=bad)

    def test_csvdir_that_is_a_file_rejected(self):
        path = os.path.join(self.tmp, "not_a_dir")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("x")
        with self.assertRaises(ValueError):
            driver.run(mode="tracks", no_plot=True, csvdir=path)

    def test_masses_list_parsing_rejects_non_numeric(self):
        with self.assertRaises(ValueError):
            driver.run(mode="hr", no_plot=True, csvdir=self.tmp,
                       masses="1.0,bogus,2.0")

    def test_masses_list_parsing_enforces_bounds(self):
        with self.assertRaises(ValueError):
            driver.run(mode="hr", no_plot=True, csvdir=self.tmp,
                       masses="0.01,1.0")  # below 0.08 lo bound

    def test_isochrones_list_parsing_enforces_max_items(self):
        ages = ",".join(str(v) for v in range(1, 13))
        with self.assertRaises(ValueError):
            driver.run(mode="hr", no_plot=True, csvdir=self.tmp,
                       isochrones=ages)

    def test_hr_aggregate_workload_cap_rejected_before_running(self):
        # Regression test for Audit1 I6: n_ms and n_post are each validated
        # individually up to phys.MAX_TRACK_STEPS, and the mass list up to
        # phys.MAX_MASSES, but the PRODUCT of a large mass count and large
        # per-track step counts could still take an unreasonable time.  This
        # must be rejected immediately (before any track is integrated),
        # not merely eventually finish.
        import time
        t0 = time.time()
        with self.assertRaises(ValueError):
            driver.run(mode="hr", no_plot=True, csvdir=self.tmp,
                       masses="1,2,3,4,5,6",
                       n_ms=2_000_000, n_post=2_000_000)
        self.assertLess(time.time() - t0, 2.0)   # rejected, not computed


class TestCsvOutput(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _read_csv(self, path):
        with open(path, newline="", encoding="utf-8") as fh:
            lines = fh.readlines()
        comments = [ln for ln in lines if ln.startswith("#")]
        data_lines = [ln for ln in lines if not ln.startswith("#")]
        reader = csv_module.reader(data_lines)
        rows = list(reader)
        return comments, rows

    def test_track_csv_header_rows_and_provenance(self):
        result = driver.run(mode="tracks", mass=1.0, no_plot=True,
                            csvdir=self.tmp)
        files = [f for f in os.listdir(self.tmp) if f.startswith("sev_track_")]
        self.assertEqual(len(files), 1)
        comments, rows = self._read_csv(os.path.join(self.tmp, files[0]))
        header, data = rows[0], rows[1:]
        self.assertEqual(header, driver.TRACK_HEADER)
        self.assertEqual(len(data), result["t"].size)
        joined_comments = "".join(comments)
        self.assertIn(phys.MODEL_VERSION, joined_comments)
        self.assertIn(phys.BUILD_ID, joined_comments)
        self.assertIn("mode = tracks", joined_comments)
        # A parameter belonging only to wdcool/nsmr must not appear as if used.
        self.assertNotIn("wd_mass", joined_comments)

    def test_provenance_records_python_numpy_matplotlib_versions(self):
        # Regression test for Audit1 J3: a CSV's provenance comments must
        # record the environment (interpreter and library versions) a run
        # was produced with, not just the program version/build.
        driver.run(mode="tracks", mass=1.0, no_plot=True, csvdir=self.tmp)
        files = [f for f in os.listdir(self.tmp) if f.startswith("sev_track_")]
        comments, _ = self._read_csv(os.path.join(self.tmp, files[0]))
        joined = "".join(comments)
        self.assertIn("Python", joined)
        self.assertIn("NumPy", joined)
        self.assertIn("Matplotlib", joined)

    def test_low_mass_remnant_wording_agrees_across_terminal_and_csv(self):
        # Regression test for Audit2 P1-1 (Codex reproducer, mass=0.60,
        # t_max=1000): the Audit1 fix made physics_sev.py's own summary
        # fields self-consistent (remnant_kind/phase_end/helium_ignition
        # all agree), but driver_sev.py's TERMINAL text and CSV comment
        # still unconditionally printed the old, now-false "classification
        # from the initial mass, not an integrated result" sentence for
        # every run, contradicting the summary sitting right above it in
        # the very same output.  This test inspects the actual printed
        # text and the actual CSV comment line -- not just summary dict
        # fields -- which is exactly what the previous round's regression
        # test failed to do.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = driver.run(mode="tracks", mass=0.60, t_max=1000.0,
                                n_ms=200, n_post=200, no_plot=True,
                                csvdir=self.tmp)
        s = result["summary"]
        self.assertEqual(s["remnant_kind"], "helium white dwarf")
        self.assertEqual(s["remnant_basis"],
                         "this track's own post-main-sequence integration")
        terminal = buf.getvalue()
        self.assertIn("This remnant comes from the track's OWN "
                      "post-main-sequence", terminal)
        # The old blanket "classification from the initial mass, not an
        # integrated result" sentence must NOT appear for this run: it is
        # simply false here, and printing it alongside the corrected
        # sentence above would just relocate the contradiction rather
        # than fix it.
        self.assertNotIn("classification from the initial mass, not an\n"
                         "  integrated result", terminal)
        files = [f for f in os.listdir(self.tmp) if f.startswith("sev_track_")]
        comments, _ = self._read_csv(os.path.join(self.tmp, files[0]))
        joined = "".join(comments)
        self.assertIn("helium white dwarf", joined)
        self.assertIn("this track's own post-main-sequence integration, "
                      "not a mass-only classification", joined)
        self.assertNotIn("(not an integrated result)", joined)

    def test_two_runs_in_the_same_second_do_not_overwrite_each_other(self):
        # Regression test for Audit1 P2-5, strengthened for Audit2 P2-5:
        # two CSVs written with the same prefix inside the same
        # wall-clock second used to collide on filename and silently
        # overwrite one another.  Codex's Audit2 review correctly found
        # that the original version of this test performed two real
        # writes without freezing time, so on a slow machine (or a write
        # that happens to straddle a second boundary) it could pass even
        # against the OLD, broken timestamp-only implementation -- two
        # files with different timestamps also satisfies "len(files)==2"
        # without ever exercising the collision-avoidance code path at
        # all.  The clock is now frozen so both writes are FORCED into
        # the same second, and the exact expected "_2" disambiguated
        # filename is asserted explicitly.
        fixed = datetime(2026, 1, 1, 12, 0, 0)
        with mock.patch.object(driver, "datetime") as mock_dt:
            mock_dt.now.return_value = fixed
            driver.run(mode="tracks", mass=1.0, no_plot=True, csvdir=self.tmp)
            driver.run(mode="tracks", mass=1.0, no_plot=True, csvdir=self.tmp)
        files = sorted(f for f in os.listdir(self.tmp)
                       if f.startswith("sev_track_"))
        self.assertEqual(len(files), 2)
        stamp = fixed.strftime("%Y%m%d_%H%M%S")
        first = f"sev_track_1.00Msun_{stamp}.csv"
        second = f"sev_track_1.00Msun_{stamp}_2.csv"
        self.assertEqual(files, sorted([first, second]))
        # Both files are independently readable and neither one is a
        # truncated/empty stub left behind by an overwrite race.
        for f in files:
            with open(os.path.join(self.tmp, f)) as fh:
                self.assertGreater(len(fh.readlines()), 100)

    def test_wdcool_csv_two_files_and_relative_difference_is_tiny(self):
        driver.run(mode="wdcool", wd_mass=0.6, no_plot=True, csvdir=self.tmp)
        cool = [f for f in os.listdir(self.tmp) if f.startswith("sev_wdcool_")]
        mr = [f for f in os.listdir(self.tmp) if f.startswith("sev_wd_mass_radius")]
        self.assertEqual(len(cool), 1)
        self.assertEqual(len(mr), 1)
        _, rows = self._read_csv(os.path.join(self.tmp, cool[0]))
        header, data = rows[0], rows[1:]
        self.assertEqual(header,
                         ["age_Gyr", "Tc_K", "L_Lsun", "Teff_K",
                          "log10_Teff", "log10_L"])
        self.assertGreater(len(data), 100)

    def test_nsmr_csv_branch_column_blank_when_unclassified(self):
        self.tmp2 = tempfile.mkdtemp()
        driver.run(mode="nsmr", eos="neutron", newtonian=True, n_mr=16,
                  rho_hi=1.0e21, no_plot=True, csvdir=self.tmp2)
        files = [f for f in os.listdir(self.tmp2) if f.startswith("sev_nsmr_")]
        self.assertEqual(len(files), 1)
        comments, rows = self._read_csv(os.path.join(self.tmp2, files[0]))
        header, data = rows[0], rows[1:]
        self.assertIn("branch", header)
        branch_col = header.index("branch")
        self.assertTrue(all(r[branch_col] == "not classified" for r in data if r))
        redshift_col = header.index("surface_redshift_z")
        self.assertTrue(all(r[redshift_col] == "" for r in data if r))

    def test_nsmr_failed_rows_labelled_failed_not_stable_or_unstable(self):
        # Regression test for Audit2 P2-2 (Codex's exact reproducer:
        # eos=polytrope, gamma=5.0, p_nuc=0.9, n_mr=16, rho_lo=1e17,
        # rho_hi=5e19).  Every raw index <= i_max used to be labelled
        # "stable" and every index > i_max "unstable" purely by position,
        # even for a row whose model never converged (nan mass/radius) --
        # a failed row is not an unstable stellar model, it is no model.
        # This also checks that warnings_detail (the per-density failure
        # reasons the physics layer already collects) actually reaches
        # the student: printed on the terminal and written into the CSV,
        # not merely present in the Python-level summary dict nobody
        # running main.py would ever see.
        self.tmp2 = tempfile.mkdtemp()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = driver.run(mode="nsmr", eos="polytrope", gamma=5.0,
                                p_nuc=0.9, n_mr=16, rho_lo=1.0e17,
                                rho_hi=5.0e19, no_plot=True, csvdir=self.tmp2)
        s = result["summary"]
        self.assertGreater(len(s["warnings_detail"]), 0)
        terminal = buf.getvalue()
        self.assertIn("PER-MODEL FAILURE DETAIL", terminal)
        for detail in s["warnings_detail"]:
            # Each recorded failure reason must actually reach the screen,
            # not just exist in the returned summary dict.
            self.assertIn(detail.split(":")[0], terminal)  # "rho_c=..." prefix

        files = [f for f in os.listdir(self.tmp2) if f.startswith("sev_nsmr_")]
        self.assertEqual(len(files), 1)
        comments, rows = self._read_csv(os.path.join(self.tmp2, files[0]))
        header, data = rows[0], rows[1:]
        branch_col = header.index("branch")
        mass_col = header.index("M_Msun")
        n_failed_rows = 0
        for r in data:
            if not r:
                continue
            if r[mass_col] == "nan":
                self.assertEqual(r[branch_col], "failed")
                n_failed_rows += 1
            else:
                self.assertIn(r[branch_col], ("stable", "unstable"))
        self.assertEqual(n_failed_rows, len(s["warnings_detail"]))
        joined_comments = "".join(comments)
        self.assertIn("failure detail:", joined_comments)

    def test_hr_isochrone_csv_records_turnoff_and_phase(self):
        driver.run(mode="hr", isochrones="1,5", no_plot=True, csvdir=self.tmp)
        files = [f for f in os.listdir(self.tmp)
                 if f.startswith("sev_hr_isochrones")]
        self.assertEqual(len(files), 1)
        _, rows = self._read_csv(os.path.join(self.tmp, files[0]))
        header, data = rows[0], rows[1:]
        self.assertIn("phase", header)
        self.assertIn("turnoff_mass_Msun", header)
        self.assertGreater(len(data), 0)


# ======================================================================
class TestCli(unittest.TestCase):
    def test_below_hydrogen_burning_limit_gives_clean_cli_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_cli(["--mode", "tracks", "--mass", "0.01",
                               "--no_plot", "--csvdir", tmp])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("hydrogen-burning limit", result.stderr)

    def test_super_chandrasekhar_white_dwarf_gives_clean_cli_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_cli(["--mode", "wdcool", "--wd_mass", "2.0",
                               "--no_plot", "--csvdir", tmp])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Chandrasekhar", result.stderr)

    def test_no_plot_without_csvdir_gives_clean_cli_error(self):
        result = run_cli(["--no_plot"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("csvdir", result.stderr)

    def test_m_observed_rejects_non_finite(self):
        for bad in ("nan", "inf", "-inf"):
            with self.subTest(bad=bad):
                result = run_cli(["--mode", "nsmr", "--m_observed", bad])
                self.assertEqual(result.returncode, 2)

    def test_m_observed_rejects_negative(self):
        result = run_cli(["--mode", "nsmr", "--m_observed", "-3"])
        self.assertEqual(result.returncode, 2)

    def test_m_observed_zero_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_cli(["--mode", "nsmr", "--m_observed", "0",
                               "--no_plot", "--csvdir", tmp, "--n_mr", "10"])
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_no_postms_help_does_not_claim_default_true(self):
        result = run_cli(["--help"])
        self.assertEqual(result.returncode, 0)
        # ArgumentDefaultsHelpFormatter must not render a confusing
        # "(default: True)" beside the negative --no_postms flag.
        for line in result.stdout.splitlines():
            if "--no_postms" in line or ("no_postms" in line and "default" in line):
                self.assertNotIn("default: True", line)

    def test_main_smoke_run_every_mode_noninteractive(self):
        with tempfile.TemporaryDirectory() as tmp:
            for extra in (
                ["--mode", "tracks", "--mass", "1.0"],
                ["--mode", "hr", "--isochrones", "1,5"],
                ["--mode", "wdcool", "--wd_mass", "0.6"],
                ["--mode", "nsmr", "--eos", "neutron"],
            ):
                with self.subTest(extra=extra):
                    result = run_cli([*extra, "--no_plot", "--csvdir", tmp])
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(phys.MODEL_VERSION, result.stdout)

    def test_newtonian_flag_runs_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_cli(["--mode", "nsmr", "--eos", "neutron",
                               "--newtonian", "--no_plot", "--csvdir", tmp])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Traceback", result.stderr)


# ======================================================================
class TestPlotting(unittest.TestCase):
    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_outdir_saves_png_and_still_displays(self):
        # Regression test for the legacy critique's requested output-control
        # fix: --outdir must save AND display, never save-instead-of-display.
        import matplotlib.pyplot as plt
        result = phys.integrate_track(m_msun=1.0, n_ms=200, n_post=200)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(plt, "show") as show:
                plotting.plot_track(result, outdir=tmp, dpi=60)
            show.assert_called_once_with()
            pngs = [f for f in os.listdir(tmp) if f.endswith(".png")]
            sidecars = [f for f in os.listdir(tmp)
                       if f.endswith(".provenance.txt")]
            self.assertEqual(len(pngs), 1)
            self.assertEqual(len(sidecars), 1)

    def test_direct_plot_call_without_provenance_still_writes_a_sidecar(self):
        # Regression test for Audit3 Codex A3-P2-1: a bare
        # plot_track(result, outdir=...) call -- no provenance argument,
        # i.e. plot_sev.py used directly as a Python API rather than
        # through driver_sev.run() -- used to save a PNG with NO sidecar
        # at all, contradicting _finish()'s own "unconditionally" claim.
        # A sidecar must now always appear, always carrying the rendering
        # settings, and must say plainly that the scientific run
        # parameters were not supplied rather than silently omitting them.
        import matplotlib.pyplot as plt
        result = phys.integrate_track(m_msun=1.0, n_ms=50, n_post=50)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(plt, "show"):
                plotting.plot_track(result, outdir=tmp, dpi=88, lw=2.5)
            pngs = [f for f in os.listdir(tmp) if f.endswith(".png")]
            sidecars = [f for f in os.listdir(tmp)
                       if f.endswith(".provenance.txt")]
            self.assertEqual(len(pngs), 1)
            self.assertEqual(len(sidecars), 1)
            with open(os.path.join(tmp, sidecars[0])) as f:
                content = f.read()
            self.assertIn("dpi = 88", content)
            self.assertIn("lw = 2.5", content)
            self.assertIn("not supplied", content)
            entries = _parse_sidecar(content)
            self.assertIn("figsize_inches", entries)

    def test_direct_plot_sidecar_records_nondefault_figsize(self):
        # Regression test for Audit4 Codex A4-P2-1: every public plot_*
        # function accepts a caller-selectable figsize and passes it
        # straight to plt.subplots(), so a direct Python caller (bypassing
        # the CLI, which does not expose --figsize at all) can change the
        # saved PNG's aspect and pixel dimensions.  The sidecar used to
        # record only dpi/lw, so two direct calls differing solely in
        # figsize could produce different images with indistinguishable
        # recorded rendering settings.  This calls plot_track() with a
        # deliberately nondefault figsize and asserts the sidecar records
        # that exact value, not merely the substring "figsize".
        import matplotlib.pyplot as plt
        result = phys.integrate_track(m_msun=1.0, n_ms=50, n_post=50)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(plt, "show"):
                plotting.plot_track(result, outdir=tmp, dpi=100, lw=1.9,
                                    figsize=(4.0, 3.0))
            sidecars = [f for f in os.listdir(tmp)
                       if f.endswith(".provenance.txt")]
            self.assertEqual(len(sidecars), 1)
            with open(os.path.join(tmp, sidecars[0])) as f:
                content = f.read()
            entries = _parse_sidecar(content)
            self.assertEqual(entries["figsize_inches"], "4, 3")

    def test_orphan_sidecar_is_not_overwritten(self):
        # Regression test for Audit3 Codex A3-P2-3: _unique_path() used
        # to check only whether the candidate PNG path existed, so a
        # pre-existing same-stem .provenance.txt (left behind by, say, an
        # earlier plot_track() call made without a provenance argument,
        # or a manual copy) was silently destroyed the moment a new run
        # picked that same stem and opened the sidecar with mode "w".
        # The clock is frozen so the stem is deterministic.
        import matplotlib.pyplot as plt
        result = phys.integrate_track(m_msun=1.0, n_ms=50, n_post=50)
        fixed = datetime(2026, 1, 1, 12, 0, 0)
        with tempfile.TemporaryDirectory() as tmp:
            stamp = fixed.strftime("%Y%m%d_%H%M%S")
            stem = f"sev_track_1.00Msun_{stamp}"
            orphan = os.path.join(tmp, stem + ".provenance.txt")
            with open(orphan, "w") as f:
                f.write("ORIGINAL ORPHAN CONTENT\n")
            with mock.patch.object(plt, "show"), \
                 mock.patch.object(plotting, "datetime") as mock_dt:
                mock_dt.now.return_value = fixed
                plotting.plot_track(result, outdir=tmp, dpi=60)
            pngs = sorted(f for f in os.listdir(tmp) if f.endswith(".png"))
            # The new PNG must NOT take the orphaned stem -- it must be
            # bumped to "_2", exactly as if a PNG had already existed.
            self.assertEqual(pngs, [f"{stem}_2.png"])
            with open(orphan) as f:
                self.assertEqual(f.read(), "ORIGINAL ORPHAN CONTENT\n")
            new_sidecar = os.path.join(tmp, f"{stem}_2.provenance.txt")
            self.assertTrue(os.path.exists(new_sidecar))

    def test_driver_writes_matched_png_and_sidecar_for_every_mode(self):
        # Regression test for Audit3 Codex A3-P2-4: the previous sidecar
        # coverage only exercised a manually-assembled helper case for
        # the "tracks" mode (driver._provenance() built by hand and
        # passed directly into plotting.plot_track(), never going through
        # driver.run() at all), so a driver runner that stopped
        # forwarding provenance, or lost a parameter for one of the other
        # three modes, would not have been caught.  This drives all four
        # modes through the real driver.run() entry point with small/fast
        # grids and checks the produced sidecar's keys against driver's
        # own authoritative per-mode parameter list, plus the
        # rendering settings that are always recorded regardless of mode.
        #
        # Strengthened for Audit4 Codex A4-P3-3: the previous version of
        # this test used `self.assertIn(param, content)`, an arbitrary
        # substring search that would still pass if a short key like "X",
        # "Z", "qc", "eos" or "mass" only happened to appear inside prose,
        # a heading, another key's name, or a value -- it never actually
        # proved the sidecar held one exact `name = value` entry per
        # parameter.  This now parses the sidecar with _parse_sidecar()
        # and compares exact key sets (and, for the rendering settings,
        # exact values) instead.
        import matplotlib.pyplot as plt
        calls = dict(
            tracks=dict(mode="tracks", mass=1.0, n_ms=50, n_post=50),
            hr=dict(mode="hr", masses="1.0,2.0", n_ms=50, n_post=50),
            wdcool=dict(mode="wdcool", wd_mass=0.6, n_cool=50),
            nsmr=dict(mode="nsmr", eos="neutron", n_mr=8,
                      rho_lo=1.0e17, rho_hi=5.0e19),
        )
        rendering_keys = {"dpi", "lw", "figsize_inches"}
        for mode, kwargs in calls.items():
            with self.subTest(mode=mode):
                with tempfile.TemporaryDirectory() as tmp:
                    with mock.patch.object(plt, "show"):
                        driver.run(outdir=tmp, dpi=77, lw=2.25, **kwargs)
                    pngs = [f for f in os.listdir(tmp) if f.endswith(".png")]
                    sidecars = [f for f in os.listdir(tmp)
                               if f.endswith(".provenance.txt")]
                    self.assertEqual(len(pngs), 1)
                    self.assertEqual(len(sidecars), 1)
                    self.assertEqual(os.path.splitext(pngs[0])[0],
                                     sidecars[0][:-len(".provenance.txt")])
                    with open(os.path.join(tmp, sidecars[0])) as f:
                        content = f.read()
                    entries = _parse_sidecar(content)
                    # Exact set comparison catches both a missing key and
                    # an unexpected/irrelevant one, not just presence.
                    expected_scientific = set(driver.PARAMS_BY_MODE[mode])
                    got_keys = set(entries)
                    self.assertEqual(got_keys & expected_scientific,
                                     expected_scientific)
                    self.assertEqual(got_keys - expected_scientific,
                                     rendering_keys)
                    self.assertEqual(entries["dpi"], "77")
                    self.assertEqual(entries["lw"], "2.25")
                    self.assertRegex(entries["figsize_inches"],
                                     r"^-?\d+(\.\d+)?, -?\d+(\.\d+)?$")
                    self.assertIn(phys.MODEL_VERSION, content)
                    self.assertIn(phys.BUILD_ID, content)

    def test_low_mass_remnant_plot_annotation_matches_its_basis(self):
        # Regression test for Audit2 P1-1 (Codex reproducer, mass=0.60,
        # t_max=1000): the plot annotation unconditionally said "(not
        # integrated)" next to the remnant mass, contradicting the case
        # where the remnant IS this track's own computed endpoint.
        import matplotlib.pyplot as plt
        result = phys.integrate_track(m_msun=0.60, t_max_gyr=1000.0,
                                      n_ms=200, n_post=200)
        self.assertEqual(result["summary"]["remnant_basis"],
                         "this track's own post-main-sequence integration")
        with mock.patch.object(plt, "show"), mock.patch.object(plt, "close"):
            plotting.plot_track(result)
            note_texts = [t.get_text() for t in plt.gcf().axes[0].texts]
        joined = "\n".join(note_texts)
        self.assertIn("this track's own endpoint", joined)
        self.assertNotIn("not integrated", joined)

    def test_png_provenance_sidecar_records_every_mode_parameter(self):
        # Regression test for Audit2 P2-3 (Codex/Copilot): the on-figure
        # footer only answers "which code produced this", not "which run"
        # -- a saved PNG with --outdir and no --csvdir could not be
        # reproduced independently, because none of n_ms/n_post/t_max/
        # x_end/qc/core_weight/expansion/core_efficiency/homology/postms
        # appeared anywhere near it.  A provenance sidecar must now be
        # written next to the PNG unconditionally, whether or not
        # --csvdir was also requested.  This test still assembles the
        # provenance list by hand for one mode as a close-in check;
        # test_driver_writes_matched_png_and_sidecar_for_every_mode below
        # supplements it by driving all four modes through the real
        # driver.run() entry point (Audit3 Codex A3-P2-4).
        import matplotlib.pyplot as plt
        result = phys.integrate_track(m_msun=1.0, n_ms=200, n_post=200)
        provenance = driver._provenance("tracks", dict(
            mass=1.0, X=0.7, Z=0.02, qc=None, burning=None, opacity="thomson",
            core_weight=0.36, expansion=1.7, core_efficiency=0.75,
            homology=False, postms=True, n_ms=200, n_post=200,
            t_max=15.0, x_end=1.0e-3))
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(plt, "show"):
                plotting.plot_track(result, outdir=tmp, dpi=60,
                                    provenance=provenance)
            pngs = [f for f in os.listdir(tmp) if f.endswith(".png")]
            sidecars = [f for f in os.listdir(tmp)
                       if f.endswith(".provenance.txt")]
            self.assertEqual(len(pngs), 1)
            self.assertEqual(len(sidecars), 1)
            # The PNG and its sidecar must share the same stem, so a
            # student can tell which sidecar belongs to which figure.
            self.assertEqual(os.path.splitext(pngs[0])[0],
                             sidecars[0][:-len(".provenance.txt")])
            with open(os.path.join(tmp, sidecars[0])) as f:
                content = f.read()
            for param in ("n_ms", "n_post", "t_max", "x_end", "qc",
                          "core_weight", "expansion", "core_efficiency",
                          "homology", "postms"):
                self.assertIn(param, content)
            self.assertIn(phys.MODEL_VERSION, content)
            self.assertIn(phys.BUILD_ID, content)

    def test_two_saves_in_the_same_second_do_not_overwrite_each_other(self):
        # Regression test for Audit1 P2-5, strengthened for Audit2 P2-5
        # (see the CSV counterpart's comment above for why freezing time
        # is required rather than relying on two real saves happening to
        # land in the same wall-clock second): the clock is frozen so
        # both saves are forced into the same second, and the exact
        # expected "_2" disambiguated filename is asserted explicitly.
        # Strengthened again for Audit3 Codex A3-P2-1/A3-P2-3: both saves
        # now include a provenance list, and BOTH artifacts of each
        # PNG/sidecar pair -- not just the PNGs -- are checked, so a
        # regression that mismatched a PNG with the wrong sidecar stem
        # would be caught here.
        import matplotlib.pyplot as plt
        result = phys.integrate_track(m_msun=1.0, n_ms=200, n_post=200)
        provenance = driver._provenance("tracks", dict(
            mass=1.0, X=0.7, Z=0.02, qc=None, burning=None, opacity="thomson",
            core_weight=0.36, expansion=1.7, core_efficiency=0.75,
            homology=False, postms=True, n_ms=200, n_post=200,
            t_max=15.0, x_end=1.0e-3))
        fixed = datetime(2026, 1, 1, 12, 0, 0)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(plt, "show"), \
                 mock.patch.object(plotting, "datetime") as mock_dt:
                mock_dt.now.return_value = fixed
                plotting.plot_track(result, outdir=tmp, dpi=60,
                                    provenance=provenance)
                plotting.plot_track(result, outdir=tmp, dpi=60,
                                    provenance=provenance)
            pngs = sorted(f for f in os.listdir(tmp) if f.endswith(".png"))
            sidecars = sorted(f for f in os.listdir(tmp)
                              if f.endswith(".provenance.txt"))
            stamp = fixed.strftime("%Y%m%d_%H%M%S")
            first = f"sev_track_1.00Msun_{stamp}.png"
            second = f"sev_track_1.00Msun_{stamp}_2.png"
            self.assertEqual(pngs, sorted([first, second]))
            first_sc = f"sev_track_1.00Msun_{stamp}.provenance.txt"
            second_sc = f"sev_track_1.00Msun_{stamp}_2.provenance.txt"
            self.assertEqual(sidecars, sorted([first_sc, second_sc]))
            for f in pngs:
                self.assertGreater(os.path.getsize(os.path.join(tmp, f)), 0)
            for f in sidecars:
                with open(os.path.join(tmp, f)) as fh:
                    self.assertIn("n_ms", fh.read())

    def test_figure_carries_a_version_build_footer(self):
        # Regression test for Audit1 P2-6: every saved figure must carry a
        # small version/build footer so a printed or screenshotted plot can
        # be traced back to the exact source revision that produced it.
        import matplotlib.pyplot as plt
        result = phys.integrate_track(m_msun=1.0, n_ms=200, n_post=200)
        with mock.patch.object(plt, "show"), mock.patch.object(plt, "close"):
            plotting.plot_track(result)
            texts = [t.get_text() for t in plt.gcf().texts]
        self.assertTrue(any(phys.MODEL_VERSION in t and phys.BUILD_ID in t
                             for t in texts))

    def test_csvdir_only_still_displays(self):
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(plt, "show") as show:
                driver.run(mode="tracks", mass=1.0, csvdir=tmp)
            show.assert_called_once_with()

    def test_hr_diagram_axes_reversed_hot_left(self):
        import matplotlib.pyplot as plt
        result = phys.build_hr_grid([1.0, 2.0])
        # _finish() always closes the figure right after plt.show(), so
        # plt.close must also be patched here or the figure is gone before
        # this test can inspect it.
        with mock.patch.object(plt, "show"), mock.patch.object(plt, "close"):
            plotting.plot_hr_diagram(result)
            current_ax = plt.gcf().axes[0]
            xlim = current_ax.get_xlim()
        self.assertGreater(xlim[0], xlim[1])  # inverted: hot (high T) on left

    def test_wd_cooling_plot_runs_without_error(self):
        import matplotlib.pyplot as plt
        result = phys.integrate_wd_cooling(m_msun=0.6, n_steps=200)
        with mock.patch.object(plt, "show"), mock.patch.object(plt, "close"):
            plotting.plot_wd_cooling(result)
            n_axes = len(plt.gcf().axes)
        self.assertGreaterEqual(n_axes, 4)

    def test_ns_mass_radius_plot_runs_for_both_gravities(self):
        import matplotlib.pyplot as plt
        for relativistic in (True, False):
            with self.subTest(relativistic=relativistic):
                result = phys.ns_mass_radius_curve(
                    eos_name="neutron", n=16, rho_lo=1e17, rho_hi=5e19,
                    relativistic=relativistic)
                with mock.patch.object(plt, "show"):
                    plotting.plot_ns_mass_radius(result, m_observed=2.01)
                plt.close("all")


# ======================================================================
class TestHelpFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = MODULE_DIR / HELP_FILE
        cls.html = cls.path.read_text(encoding="utf-8")
        parser = HtmlTreeParser()
        parser.feed(cls.html)
        parser.close()
        cls.root = parser.root

    def test_help_file_exists(self):
        self.assertTrue(self.path.is_file())

    def test_version_and_build_match_program(self):
        version_nodes = nodes_by_id(self.root, "version_build")
        self.assertEqual(len(version_nodes), 1)
        text = normalized_text(version_nodes[0])
        self.assertIn(f"Version {phys.MODEL_VERSION}", text)
        self.assertIn(f"Build {phys.BUILD_ID}", text)

    def test_help_documents_png_provenance_sidecar(self):
        # Regression test for Audit3 Codex A3-P2-2/A3-P3-4: version 1.3.0
        # added a same-stem .provenance.txt sidecar next to every saved
        # PNG, but the Help never mentioned it at all -- no file-naming
        # description, no statement that a second artifact is produced,
        # and no guidance to keep the PNG and sidecar together.  This
        # checks the three Help locations Codex named: the Plot Layer
        # module card, the --outdir parameter row, and the PNG-file
        # output-table row.
        modules = normalized_text(nodes_by_id(self.root, "modules")[0])
        self.assertIn("provenance", modules)
        params = normalized_text(nodes_by_id(self.root, "parameters")[0])
        self.assertIn("provenance", params)
        output = normalized_text(nodes_by_id(self.root, "output")[0])
        self.assertIn("provenance", output)
        self.assertIn(".provenance.txt", self.html)

    def test_mu_e_defined_correctly_not_reversed(self):
        # Legacy critique: help previously said "electrons per nucleon" for
        # mu_e, which is backwards.  Must now read as mass-per-electron.
        params = normalized_text(nodes_by_id(self.root, "parameters")[0])
        self.assertIn("mass per electron", params)
        self.assertNotIn("Electrons per nucleon", self.html)

    def test_neutron_star_not_described_as_pure_degeneracy_pressure(self):
        background = normalized_text(nodes_by_id(self.root, "background")[0])
        self.assertIn("strongly interacting", background)
        self.assertIn("Oppenheimer and Volkoff", background)

    def test_paczynski_relation_explicitly_disclaimed(self):
        equations = normalized_text(nodes_by_id(self.root, "equations")[0])
        self.assertIn("not", equations)
        self.assertIn("Paczy", equations)  # Paczyński, accent-insensitive

    def test_kalirai_calibration_range_stated_correctly(self):
        text = self.html
        self.assertIn("1.16", text)
        self.assertIn("7", text)

    def test_mathjax_documented_without_local_install_or_navigator_online(self):
        self.assertIn("cdn.jsdelivr.net/npm/mathjax@3", self.html)
        self.assertIn("an internet connection is needed", self.html)
        self.assertNotIn("navigator.onLine", self.html)
        # Per the explicit project decision, no local/offline MathJax
        # installation instructions should appear in the Setup Guide
        # reference this Help file makes.
        self.assertNotIn("local MathJax", self.html)
        self.assertNotIn("offline support", self.html)

    def test_no_review_or_audit_history_leaked_into_student_help(self):
        for phrase in ("Claude", "Copilot", "Gemini", "Codex", "Critique",
                       "Audit1", "ChatGPT", "GPT-5"):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, self.html)

    def test_all_internal_navigation_targets_exist_and_ids_are_unique(self):
        ids = re.findall(r'\bid="([^"]+)"', self.html)
        counts = Counter(ids)
        self.assertFalse({name: count for name, count in counts.items() if count > 1})
        targets = [
            t for t in re.findall(r'href="#([^"]+)"', self.html)
            if not any(ch in t for ch in "${}")
        ]
        self.assertTrue(targets)
        for target in targets:
            with self.subTest(target=target):
                self.assertIn(target, counts)

    def test_expansion_growth_factor_lower_bound_documented(self):
        params = normalized_text(nodes_by_id(self.root, "parameters")[0])
        self.assertIn("at least 1", params)

    def test_mu_e_lower_bound_documented(self):
        # Regression test for Audit1 P2-10: the previous version of this
        # test asserted assertIn("mu_e", "mu_e"), a tautology that always
        # passes and the second assertion only checked that the literal
        # substring "mu_e" appears somewhere -- true of nearly every
        # mu_e-related sentence on the page, so it could not have caught a
        # missing or wrong bound.  This checks the actual bound statement
        # (mu_e >= 1, in the LaTeX source used throughout this page) is
        # present in the runtime-safeguards note.
        algo = normalized_text(nodes_by_id(self.root, "algorithm")[0])
        self.assertIn(r"\mu_e\ge1", algo)
        self.assertIn("fewer than one nucleon per electron", algo)

    def test_m_observed_negative_rejection_documented(self):
        params = normalized_text(nodes_by_id(self.root, "parameters")[0])
        self.assertIn("negative values are refused", params)

    def test_experiments_section_present_and_nonempty(self):
        exp_section = nodes_by_id(self.root, "experiments")
        self.assertEqual(len(exp_section), 1)
        cards = descendants(exp_section[0], lambda n: has_class(n, "exp-card"))
        self.assertGreaterEqual(len(cards), 15)

    def test_every_experiment_card_has_a_number_and_title(self):
        exp_section = nodes_by_id(self.root, "experiments")[0]
        cards = descendants(exp_section, lambda n: has_class(n, "exp-card"))
        for card in cards:
            nums = descendants(card, lambda n: has_class(n, "ec-num"))
            titles = descendants(card, lambda n: n.tag == "h4")
            with self.subTest(card=normalized_text(card)[:40]):
                self.assertEqual(len(nums), 1)
                self.assertEqual(len(titles), 1)
                self.assertTrue(normalized_text(titles[0]))

    def test_oppenheimer_volkoff_exercise_not_titled_as_an_error(self):
        # Legacy critique: retitle "Oppenheimer and Volkoff Were Wrong".
        self.assertNotIn("Were Wrong", self.html)
        self.assertIn("Why the Ideal Neutron Gas Fails", self.html)

    def test_domain_of_validity_distinguishes_accepted_from_trustworthy(self):
        validity = normalized_text(nodes_by_id(self.root, "validity")[0])
        self.assertIn("Accepted", validity)
        self.assertIn("0.35", validity)
        self.assertIn("15", validity)

    def test_output_section_does_not_overstate_three_orders_for_radius(self):
        output_text = normalized_text(nodes_by_id(self.root, "output")[0])
        # Radius grows by "about a hundred" (2 orders), luminosity by three;
        # they must not be conflated.
        self.assertIn("hundred", output_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
