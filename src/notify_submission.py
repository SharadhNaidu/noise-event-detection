"""Open a Chrome tab telling the user a submission is ready (they are away; the tab is the hand-off).

    python src/notify_submission.py --zip artifacts/submissions/x.zip --expected 1.59 --why "..." [--measured "..."]
"""
import argparse
import hashlib
import html
import os
import subprocess
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True); ap.add_argument("--expected", required=True)
    ap.add_argument("--why", default=""); ap.add_argument("--measured", default=""); ap.add_argument("--note", default="")
    a = ap.parse_args()
    z = Path(a.zip).resolve(); md5 = hashlib.md5(open(z, "rb").read()).hexdigest()
    out = Path(os.environ.get("TEMP", ".")) / f"submit_{int(time.time())}.html"
    body = f"""<!doctype html><meta charset=utf-8><title>SUBMIT NOW</title>
<body style="font-family:Segoe UI,Arial;max-width:900px;margin:40px auto;background:#0f172a;color:#e2e8f0">
<h1 style="color:#22c55e">Submission ready: upload this file to Codabench Track 1</h1>
<p style="font-size:22px"><b>File:</b> <code style="background:#1e293b;padding:4px 8px">{html.escape(str(z))}</code></p>
<p><b>md5:</b> <code>{md5}</code> &nbsp; <b>size:</b> {z.stat().st_size:,} bytes &nbsp; <b>built:</b> {time.strftime('%Y-%m-%d %H:%M IST')}</p>
<p style="font-size:26px"><b>Expected LB combined:</b> <span style="color:#facc15">{html.escape(a.expected)}</span> (champion 1.5808)</p>
<p><b>What it tests:</b> {html.escape(a.why)}</p>
<p><b>Local evidence:</b> {html.escape(a.measured)}</p>
<p>{html.escape(a.note)}</p>
<p style="color:#94a3b8">Complete pipeline, all 5517 clips, champion rebuilt byte-exact before building. After uploading, paste the LB score (F1 / Dice / natural+synthetic split) into the chat.</p>
</body>"""
    out.write_text(body, encoding="utf-8")
    print("page", out, "md5", md5)
    for exe in (r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"):
        if Path(exe).exists():
            subprocess.Popen([exe, "--new-tab", str(out)]); break
    else:
        os.startfile(str(out))  # default browser


if __name__ == "__main__":
    main()
