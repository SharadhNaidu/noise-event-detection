"""Pick FSD50K dev clips for the external-noise store (organisers confirmed FSD50K use, 2026-10-01).

One competition category per clip: map FSD50K labels -> our 7 categories, drop any clip carrying a
speech-like label (the model must not learn speech as noise), and drop clips whose labels map to more
than one category. Duration is taken from the WAV size (44.1 kHz 16-bit mono). Writes manifest.tsv.
"""
import paths as P  # noqa: E402  (INDOML_ROOT, see src/paths.py)
import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = (P.EXT_ROOT / "fsd50k")
CAT = {
    "animal": "Animal Bark Bird Bird_vocalization_and_bird_call_and_bird_song Cat Chicken_and_rooster Chirp_and_tweet "
              "Cricket Crow Dog Domestic_animals_and_pets Fowl Frog Growling Gull_and_seagull Insect "
              "Livestock_and_farm_animals_and_working_animals Meow Purr Wild_animals",
    "vehicle_traffic": "Accelerating_and_revving_and_vroom Bus Car Car_passing_by Engine Engine_starting Idling "
                       "Motor_vehicle_(road) Motorcycle Rail_transport Subway_and_metro_and_underground "
                       "Traffic_noise_and_roadway_noise Train Truck Vehicle Vehicle_horn_and_car_horn_and_honking "
                       "Race_car_and_auto_racing Aircraft Fixed-wing_aircraft_and_airplane",
    "baby_child": "Crying_and_sobbing",
    "singing_music": "Music Musical_instrument Singing Male_singing Female_singing Tabla Guitar Acoustic_guitar Piano "
                     "Keyboard_(musical) Drum Drum_kit Percussion Harmonica Accordion Organ Harp Bowed_string_instrument "
                     "Brass_instrument Wind_instrument_and_woodwind_instrument Plucked_string_instrument Strum "
                     "Electric_guitar Bass_guitar Trumpet Mallet_percussion Marimba_and_xylophone Glockenspiel Tambourine",
    "phone_signal_alarm": "Alarm Bell Bicycle_bell Church_bell Doorbell Ringtone Telephone",
    "appliance_machine": "Mechanical_fan Microwave_oven Printer Typing Typewriter Computer_keyboard Clock Tick-tock Tick "
                         "Mechanisms Power_tool Drill Sawing",
    "human_non_speech": "Breathing Cough Sneeze Sigh Gasp Burping_and_eructation Respiratory_sounds Chewing_and_mastication",
}
LAB2CAT = {l: c for c, s in CAT.items() for l in s.split()}
SPEECHY = set("Speech Human_voice Male_speech_and_man_speaking Female_speech_and_woman_speaking "
              "Child_speech_and_kid_speaking Conversation Chatter Crowd Speech_synthesizer Whispering Shout Yell "
              "Laughter Screaming Siren".split())   # Siren is listed under both vehicle and alarm: ambiguous
# generic parents that never decide a category on their own when a specific label exists
GENERIC = {"Mechanisms", "Vehicle", "Animal", "Music", "Musical_instrument"}


def clip_category(labels):
    voice_ok = {"Human_voice"} if "Crying_and_sobbing" in labels else set()   # crying IS voice: keep it
    if any(l in SPEECHY and l not in voice_ok for l in labels):
        return None
    cats = {LAB2CAT[l] for l in labels if l in LAB2CAT and l not in GENERIC}
    if not cats:
        cats = {LAB2CAT[l] for l in labels if l in LAB2CAT}
    return cats.pop() if len(cats) == 1 else None


def main(cap=None, min_s=1.5):
    cap = cap or {"animal": 900, "vehicle_traffic": 900, "appliance_machine": 900, "singing_music": 500,
                  "phone_signal_alarm": 500, "baby_child": 400, "human_non_speech": 400}
    sizes = json.loads(sys.stdin.read()) if not sys.stdin.isatty() else {}
    info = json.load(open(ROOT / "dev_clips_info_FSD50K.json"))
    rows = list(csv.DictReader(open(ROOT / "dev.csv")))
    by = defaultdict(list)
    for r in rows:
        c = clip_category(r["labels"].split(","))
        if c is None:
            continue
        sz = sizes.get(r["fname"])
        dur = (sz - 44) / (2 * 44100) if sz else None
        if dur is not None and dur < min_s:
            continue
        by[c].append((-(dur or 0), r["fname"], dur, info.get(r["fname"], {}).get("license", "?")))
    out = open(ROOT / "manifest.tsv", "w")
    out.write("fname\tcategory\tdur\tlicense\n")
    tot = Counter()
    for c, v in by.items():
        v.sort()                                        # longest first: long backgrounds are the target
        for _, f, d, lic in v[:cap[c]]:
            out.write(f"{f}\t{c}\t{d}\t{lic}\n"); tot[c] += 1
    print({c: (tot[c], len(by[c])) for c in by})


if __name__ == "__main__":
    main()
