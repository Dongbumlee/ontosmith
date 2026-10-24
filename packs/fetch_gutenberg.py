"""Expand the Gutenberg clean corpus: 10 -> ~60 books.

Direct downloads from gutenberg.org (worked 2026-10-11).
Saves book_<id>.txt into data/gutenberg/, skips existing files,
reports successes/failures and total size.
"""
import subprocess
from pathlib import Path

OUT = Path("data/gutenberg")
OUT.mkdir(parents=True, exist_ok=True)

# Well-known public-domain titles (topic irrelevant; volume matters)
IDS = [
    1342,   # Pride and Prejudice
    345,    # Dracula
    1232,   # The Prince
    174,    # Picture of Dorian Gray
    120,    # Treasure Island
    76,     # Huckleberry Finn
    1400,   # Great Expectations
    158,    # Emma
    161,    # Sense and Sensibility
    219,    # Heart of Darkness
    768,    # Wuthering Heights
    1260,   # Jane Eyre
    730,    # Oliver Twist
    1184,   # Count of Monte Cristo
    996,    # Don Quixote
    135,    # Les Miserables
    55,     # Wizard of Oz
    829,    # Gulliver's Travels
    2000,   # Don Quixote (alt)
    30982,  # (backup slot)
    4300,   # Ulysses (existing)
    514,    # Little Women
    1322,   # Leaves of Grass
    244,    # A Study in Scarlet
    209,    # The Turn of the Screw
    236,    # The Jungle Book
    3300,   # An Inquiry into the Wealth of Nations
    46,     # A Christmas Carol
    408,    # The Souls of Black Folk
    7370,   # Second Treatise of Government
    3207,   # Leviathan
    4363,   # Beyond Good and Evil
    1497,   # The Republic
    150,    # Paradise Lost
    7849,   # The Prince and the Pauper? (check at download)
    2852,   # The Hound of the Baskervilles
    834,    # The Memoirs of Sherlock Holmes
    335,    # (backup)
    863,    # (backup)
    1998,   # Youth: A Narrative
    4217,   # (backup)
    521,    # The Life and Adventures of Nicholas Nickleby? (check)
    580,    # The Pickwick Papers
    967,    # (backup)
    714,    # (backup)
    1430,   # (backup)
    1727,   # The Odyssey (Butler)
    3160,   # (backup)
    15453,  # (backup)
    30254,  # (backup)
]

ok, fail, skip = 0, [], 0
for bid in IDS:
    dest = OUT / f"book_{bid}.txt"
    if dest.exists() and dest.stat().st_size > 10_000:
        skip += 1
        continue
    got = False
    for url in (f"https://www.gutenberg.org/files/{bid}/{bid}-0.txt",
                f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt"):
        r = subprocess.run(["curl", "-sL", "--max-time", "60", "-o", str(dest), url],
                           capture_output=True)
        if dest.exists() and dest.stat().st_size > 10_000:
            head = dest.read_bytes()[:200].lower()
            if b"gutenberg" in head or b"the project" in head or len(head) > 0:
                got = True
                break
    if got:
        ok += 1
    else:
        fail.append(bid)
        if dest.exists():
            dest.unlink()

total = sum(p.stat().st_size for p in OUT.glob("book_*.txt"))
count = len(list(OUT.glob("book_*.txt")))
print(f"Downloaded: {ok}, skipped(existing): {skip}, failed: {fail}")
print(f"Corpus: {count} books, {total/1e6:.1f} MB")
