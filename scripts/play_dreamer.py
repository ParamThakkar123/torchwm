"""Moved to :mod:`torchwm.inference.play_dreamer` so the installed CLI can import it.

Kept so ``python scripts/play_dreamer.py ...`` keeps working from a checkout.
"""

from torchwm.inference.play_dreamer import main

if __name__ == "__main__":
    main()
