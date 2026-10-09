"""Greedy IoU tracker: keeps a stable id per person across frames so each
person is counted once per ad instead of once per frame."""


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Tracker:
    def __init__(self, iou_thr=0.3, max_missing_s=1.0):
        self.iou_thr = iou_thr
        self.max_missing_s = max_missing_s
        self.tracks = {}  # id -> dict(box, last, first, looking)
        self.next_id = 1

    def update(self, dets, now):
        """dets: list of (box[x1,y1,x2,y2], looking: bool). Returns visible tracks."""
        pairs = []
        for tid, t in self.tracks.items():
            for j, (box, _) in enumerate(dets):
                v = iou(t["box"], box)
                if v >= self.iou_thr:
                    pairs.append((v, tid, j))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for v, tid, j in pairs:
            if tid in used_t or j in used_d:
                continue
            box, looking = dets[j]
            t = self.tracks[tid]
            # light smoothing of the box to reduce jitter
            t["box"] = [0.7 * o + 0.3 * n for o, n in zip(t["box"], box)]
            t["last"] = now
            t["looking"] = bool(looking)
            used_t.add(tid)
            used_d.add(j)
        for j, (box, looking) in enumerate(dets):
            if j in used_d:
                continue
            self.tracks[self.next_id] = {"id": self.next_id, "box": list(box), "last": now, "first": now,
                                         "looking": bool(looking)}
            self.next_id += 1
        for tid in [k for k, t in self.tracks.items() if now - t["last"] > self.max_missing_s]:
            del self.tracks[tid]
        return [t for t in self.tracks.values() if t["last"] == now]
