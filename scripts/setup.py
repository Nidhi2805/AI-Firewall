"""One-time setup: generate the synthetic dataset and train the models.
Run once after install:  python scripts/setup.py"""
import subprocess, sys, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for mod in ["src.generate_dataset", "src.input_checkpoint", "src.topic_gate", "src.context_scanner"]:
    print("->", mod)
    subprocess.run([sys.executable, "-m", mod], check=True)
print("Setup complete. Run:  python app.py")
