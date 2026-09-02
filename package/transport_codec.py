 """
transport_codec.py
===================
Build-time helper used only to generate the .py.txt transport-encoded
companions delivered alongside the canonical .py sources, and to verify
they decode back byte-for-byte before delivery.  This is NOT part of the
shipped four-module program (main.py / driver_sev.py / physics_sev.py /
plot_sev.py) and is not imported by it or by tests/test_physics_sev.py;
it exists purely as packaging tooling, documented in each response
report's transport-encoding section.

Audit4 Codex A4-P3-1 fix
-------------------------
The Audit3 scheme was:

    text.replace("__", "dunder").replace(" ", "§")

This is reversible for source that happens not to contain the bare
English word "dunder", but it is not a collision-free encoding in
general: a pre-existing natural occurrence of "dunder" is
indistinguishable, after encoding, from a "dunder" introduced to stand
in for "__".  Codex's reproducer:

    original = "natural dunder word and __name__"
    encoded  = "natural§dunder§word§and§dundernamedunder"
    decoded  = "natural __ word and __name__"        # WRONG: word corrupted

This module replaces that scheme with a delimited, self-escaping token
scheme that is provably injective:

    1. Every literal "§" in the input is escaped first, by doubling it
       ("§" -> "§§").  This step runs before any token is introduced.
    2. Every "__" is replaced by the delimited token "§dunder§".
    3. Every " " (space) is replaced by the delimited token "§sp§".

Decoding scans left-to-right with a single regex alternation
(§dunder§|§sp§|§§) and replaces whichever fixed token matches at each
position.  Because step 1 escapes every literal "§" before steps 2/3
introduce any new "§" characters, and because "§dunder§"/"§sp§" are
never produced as a byproduct of doubling, every "§" in the encoded
text is unambiguously either half of a doubled escape pair or the
opening/closing delimiter of a real token -- there is no input for
which two different original texts encode to the same string, and no
encoded string that decodes two different ways.  See
test_roundtrip_corpus() below for a worked collision corpus (literal
"dunder", literal "__", literal "§" in isolation and adjacent to a
real token, non-ASCII text, CRLF, and a file with no final newline),
which is run over all five delivered files before every package is
assembled.
"""

import re

_TOKEN_RE = re.compile(r"§dunder§|§sp§|§§")
_DECODE_MAP = {"§dunder§": "__", "§sp§": " ", "§§": "§"}


def encode_transport(text):
    out = text.replace("§", "§§")
    out = out.replace("__", "§dunder§")
    out = out.replace(" ", "§sp§")
    return out


def decode_transport(text):
    return _TOKEN_RE.sub(lambda m: _DECODE_MAP[m.group()], text)


def _roundtrip_ok(original):
    encoded = encode_transport(original)
    decoded = decode_transport(encoded)
    return decoded == original, encoded, decoded


def test_roundtrip_corpus():
    """Adversarial corpus per Codex Audit4 A4-P3-1's recommended fix."""
    cases = [
        "natural dunder word and __name__",
        "a §sp§ literal token typed by a human, and a real  double  space",
        "literal section signs: § § §§ §§§",
        "§dunder§ typed literally, not produced by encoding",
        "§§ typed literally, adjacent to __ real dunder __",
        "mix: §__ and __§ and § __ § and §§__§§",
        "non-ascii: café, µ-meson, —em dash—, 中文",
        "line one\r\nline two\r\nline three (CRLF)",
        "line one\nline two\nno final newline",  # deliberately no trailing \n
        "",  # empty string edge case
        "____",  # two adjacent dunders back to back
        "                ",  # run of spaces only
    ]
    failures = []
    for original in cases:
        ok, encoded, decoded = _roundtrip_ok(original)
        if not ok:
            failures.append((original, encoded, decoded))
    return failures


if __name__ == "__main__":
    failures = test_roundtrip_corpus()
    if failures:
        print(f"FAILED: {len(failures)} corpus case(s) did not round-trip:")
        for original, encoded, decoded in failures:
            print(f"  original={original!r}\n  encoded ={encoded!r}\n"
                  f"  decoded ={decoded!r}\n")
        raise SystemExit(1)
    print("OK: transport codec self-test -- all adversarial corpus cases "
          "round-tripped byte-for-byte.")
