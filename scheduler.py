"""Decides which ad plays next.

Strategy
--------
1. First pass: when playback starts, show one ad from every category (the
   least-played ad of each) so every category has at least one measurement.
2. Then Thompson sampling: each category has a Beta(alpha, beta) belief about
   its attention rate (lookers / viewers). alpha and beta are built from past
   impressions, weighted by exp. decay (half-life, default 2 h) so the belief
   follows the audience through the day. Sample one value per category and
   play the highest; with probability epsilon pick a random category instead
   so no category starves.  Within the category the same sampling picks the
   ad, never repeating the ad that just played when there is a choice.

Impressions with zero viewers do not change the beliefs.
"""

import math
import random
import threading
import time


class Scheduler(threading.Thread):
    def __init__(self, db, audience):
        super().__init__(daemon=True, name="scheduler")
        self.db, self.aud = db, audience
        self.lock = threading.RLock()
        self.running = False
        self.current = None  # {ad, started_at, ends_at, seq}
        self.seq = 0
        self.first_pass = []
        self.last_ad_id = None
        self.skip_flag = False
        self.last_pick_info = ""
        self.stop_event = threading.Event()

    # ---- controls ----------------------------------------------------------
    def start_playback(self):
        with self.lock:
            if self.running:
                return
            self.running = True
            cats = self.db.q("SELECT DISTINCT category_id FROM ads WHERE enabled=1")
            self.first_pass = [c["category_id"] for c in cats]
            random.shuffle(self.first_pass)
            self.db.set_setting("running", 1)

    def stop_playback(self):
        with self.lock:
            self.running = False
            self._finish_current("stop")
            self.db.set_setting("running", 0)

    def skip(self):
        with self.lock:
            self.skip_flag = True

    def state(self):
        with self.lock:
            cur = None
            if self.current:
                ad = self.current["ad"]
                cur = {
                    "seq": self.current["seq"],
                    "ad": {"id": ad["id"], "name": ad["name"], "type": ad["type"], "url": f"/media/{ad['filename']}",
                           "duration": ad["duration"], "category": ad["category_name"],
                           "category_id": ad["category_id"]},
                    "started_at": self.current["started_at"],
                    "ends_at": self.current["ends_at"],
                    "remaining": max(0.0, round(self.current["ends_at"] - time.time(), 1)),
                    "live": self.aud.snapshot_impression(),
                }
            return {"running": self.running, "current": cur, "seq": self.seq, "pick_info": self.last_pick_info,
                    "first_pass_remaining": len(self.first_pass)}

    # ---- loop --------------------------------------------------------------
    def run(self):
        if self.db.setting("running", int):
            self.start_playback()  # resume after reboot
        while not self.stop_event.is_set():
            time.sleep(0.2)
            with self.lock:
                if not self.running:
                    continue
                now = time.time()
                if self.current is None or now >= self.current["ends_at"] or self.skip_flag:
                    reason = "skip" if self.skip_flag else "timer"
                    self.skip_flag = False
                    self._finish_current(reason)
                    ad = self.pick_next()
                    if ad is not None:
                        self._start(ad)

    def _start(self, ad):
        self.aud.begin_impression()
        self.seq += 1
        now = time.time()
        self.current = {"ad": ad, "started_at": now, "ends_at": now + float(ad["duration"]), "seq": self.seq}
        self.last_ad_id = ad["id"]

    def _finish_current(self, reason):
        if self.current is None:
            return
        m = self.aud.end_impression()
        cur, self.current = self.current, None
        ended = time.time()
        if ended - cur["started_at"] < 1.0:  # too short to mean anything
            return
        if not self.db.one("SELECT id FROM ads WHERE id=?", (cur["ad"]["id"],)):
            return  # ad was deleted while playing
        self.db.add_impression(cur["ad"]["id"], cur["ad"]["category_id"], cur["started_at"], ended, m, reason)

    # ---- selection ---------------------------------------------------------
    def _eligible_ads(self):
        hour = time.localtime().tm_hour
        ads = self.db.q(
            "SELECT a.*, c.name AS category_name FROM ads a JOIN categories c ON c.id=a.category_id "
            "WHERE a.enabled=1 ORDER BY a.id"
        )
        return [a for a in ads if a["active_from"] <= hour < a["active_to"]]

    def beliefs(self, ads=None):
        """Decayed Beta parameters per category and per ad."""
        half_life = (self.db.setting("half_life_hours", float) or 2.0) * 3600.0
        now = time.time()
        since = now - half_life * 6  # beyond ~6 half-lives the weight is negligible
        rows = self.db.q(
            "SELECT ad_id, category_id, ended_at, viewers, lookers FROM impressions "
            "WHERE ended_at > ? AND viewers > 0", (since,)
        )
        cat, ad = {}, {}
        for r in rows:
            w = math.pow(0.5, (now - r["ended_at"]) / half_life)
            for store, key in ((cat, r["category_id"]), (ad, r["ad_id"])):
                s = store.setdefault(key, {"alpha": 1.0, "beta": 1.0, "n": 0, "viewers": 0, "lookers": 0})
                s["alpha"] += w * r["lookers"]
                s["beta"] += w * (r["viewers"] - r["lookers"])
                s["n"] += 1
                s["viewers"] += r["viewers"]
                s["lookers"] += r["lookers"]
        return cat, ad

    @staticmethod
    def _sample(b):
        return random.betavariate(b["alpha"], b["beta"]) if b else random.betavariate(1, 1)

    def pick_next(self):
        ads = self._eligible_ads()
        if not ads:
            self.last_pick_info = "no enabled ads in their active hours"
            return None
        by_cat = {}
        for a in ads:
            by_cat.setdefault(a["category_id"], []).append(a)
        plays = {r["ad_id"]: r["n"] for r in self.db.q("SELECT ad_id, COUNT(*) n FROM impressions GROUP BY ad_id")}

        # 1. first pass: one ad from each category
        while self.first_pass:
            cid = self.first_pass.pop(0)
            if cid in by_cat:
                ad = min(by_cat[cid], key=lambda a: (plays.get(a["id"], 0), random.random()))
                self.last_pick_info = f"first pass: {ad['category_name']}"
                return ad

        # categories that have never played (e.g. added while running) get their first play now
        played_cats = {r["category_id"] for r in self.db.q("SELECT DISTINCT category_id FROM impressions")}
        fresh = [cid for cid in by_cat if cid not in played_cats]
        if fresh:
            cid = random.choice(fresh)
            ad = min(by_cat[cid], key=lambda a: (plays.get(a["id"], 0), random.random()))
            self.last_pick_info = f"first play for new category: {ad['category_name']}"
            return ad

        cat_b, ad_b = self.beliefs()
        eps = self.db.setting("epsilon", float) or 0.0
        if random.random() < eps or not cat_b:
            cid = random.choice(list(by_cat))
            how = "explore (random category)"
        else:
            samples = {cid: self._sample(cat_b.get(cid)) for cid in by_cat}
            cid = max(samples, key=samples.get)
            how = "thompson " + ", ".join(
                f"{by_cat[c][0]['category_name']}={samples[c]:.2f}" for c in samples)

        pool = by_cat[cid]
        if len(pool) > 1 and self.last_ad_id is not None:
            pool = [a for a in pool if a["id"] != self.last_ad_id] or pool
        ad = max(pool, key=lambda a: self._sample(ad_b.get(a["id"])))
        self.last_pick_info = f"{how} -> {ad['category_name']} / {ad['name']}"
        return ad
