"""
Season 9 Runner: Exp 24 + PCA Analysis
Chains both experiments. Beep on completion.
Estimated total: ~2 hours.
"""
import subprocess, sys, os, time

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(name, path):
    print(f"\n{'='*70}")
    print(f"RUNNING: {name}")
    print(f"{'='*70}")
    t0 = time.time()
    result = subprocess.run([sys.executable, path], cwd=SRC_DIR)
    elapsed = (time.time() - t0) / 60
    print(f"\n{name} completed in {elapsed:.1f} min (exit code: {result.returncode})")
    return result.returncode

if __name__ == "__main__":
    t_total = time.time()

    # Part 1: Exp 24 - Reciprocal Altruism
    rc1 = run_script("Exp 24: Reciprocal Altruism",
                     os.path.join(SRC_DIR, "exp24_reciprocal_altruism.py"))

    # Part 2: PCA Brain Analysis
    rc2 = run_script("Phantom Sync PCA Brain Analysis",
                     os.path.join(SRC_DIR, "exp_phantom_pca.py"))

    total = (time.time() - t_total) / 60
    print(f"\n{'='*70}")
    print(f"SEASON 9 COMPLETE ({total:.0f} min)")
    print(f"{'='*70}")
    print(f"  Exp 24: exit={rc1}")
    print(f"  PCA:    exit={rc2}")

    try:
        import winsound
        for _ in range(5): winsound.Beep(800, 300); time.sleep(0.2)
    except: pass
    print("\nDone!")

