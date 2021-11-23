use std::sync::atomic::AtomicU32;

use crate::tracker::Object;
use crate::Detection;

use nalgebra as na;

use crate::circular_queue::CircularQueue;
use munkres::{solve_assignment, WeightMatrix, Weights};

static SEQ_ID: AtomicU32 = AtomicU32::new(1);

const SECONDS_IN_FRAME: f32 = 0.04;
const CONFIRM_SECONDS_RATIO: f32 = 0.4;

#[derive(Debug, Clone)]
enum IndexedSliceKind {
    All,
    Indexes(Vec<usize>),
}

pub struct IndexedSlice<'a, T> {
    pub slice: &'a [T],
    kind: IndexedSliceKind,
}

impl<'a, T> Clone for IndexedSlice<'a, T> {
    fn clone(&self) -> Self {
        Self {
            slice: self.slice,
            kind: self.kind.clone(),
        }
    }
}

impl<'a, T> IndexedSlice<'a, T> {
    pub fn new(slice: &'a [T]) -> Self {
        Self {
            slice,
            kind: IndexedSliceKind::All,
        }
    }

    pub fn new_with_indexes(slice: &'a [T], idx: Vec<usize>) -> Self {
        Self {
            slice,
            kind: IndexedSliceKind::Indexes(idx),
        }
    }

    #[inline]
    pub fn get_index(&self, idx: usize) -> usize {
        match &self.kind {
            IndexedSliceKind::All => idx,
            IndexedSliceKind::Indexes(idxs) => idxs[idx],
        }
    }
    #[inline]
    pub fn get(&self, idx: usize) -> Option<&'a T> {
        match &self.kind {
            IndexedSliceKind::All => self.slice.get(idx),
            IndexedSliceKind::Indexes(idxs) => self.slice.get(*idxs.get(idx)?),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match &self.kind {
            IndexedSliceKind::All => self.slice.len(),
            IndexedSliceKind::Indexes(idxs) => idxs.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        match &self.kind {
            IndexedSliceKind::All => self.slice.is_empty(),
            IndexedSliceKind::Indexes(idxs) => idxs.is_empty(),
        }
    }

    #[inline]
    pub fn all_indexes(&self) -> Vec<usize> {
        match &self.kind {
            IndexedSliceKind::All => (0..self.slice.len()).collect(),
            IndexedSliceKind::Indexes(idxs) => idxs.clone(),
        }
    }
}

impl<'a, T> std::ops::Index<usize> for IndexedSlice<'a, T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let idx = match &self.kind {
            IndexedSliceKind::All => index,
            IndexedSliceKind::Indexes(idxs) => unsafe { *idxs.get_unchecked(index) },
        };

        &self.slice[idx]
    }
}

pub enum IndexedSliceIter<'a, T> {
    All(std::slice::Iter<'a, T>),
    Indexes((&'a [T], std::vec::IntoIter<usize>)),
}

impl<'a, T> Iterator for IndexedSliceIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<&'a T> {
        match self {
            IndexedSliceIter::All(it) => it.next(),
            IndexedSliceIter::Indexes((slice, it)) => slice.get(it.next()?),
        }
    }
}

impl<'a, T> IntoIterator for IndexedSlice<'a, T> {
    type Item = &'a T;
    type IntoIter = IndexedSliceIter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        match self.kind {
            IndexedSliceKind::All => IndexedSliceIter::All(self.slice.iter()),
            IndexedSliceKind::Indexes(idxs) => {
                IndexedSliceIter::Indexes((self.slice, idxs.into_iter()))
            }
        }
    }
}

pub fn in_bounds(p: na::Point2<f32>, poly: &[na::Point2<f32>]) -> bool {
    let n = poly.len();
    let mut inside = false;
    let mut p1 = poly[0];
    let mut xints = 0.0;

    for i in 1..=n {
        let p2 = poly[i % n];

        if p.y > f32::min(p1.y, p2.y) && p.y <= f32::max(p1.y, p2.y) && p.x <= f32::max(p1.x, p2.x)
        {
            if (p1.y - p2.y).abs() > f32::EPSILON {
                xints = (p.y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x;
            }

            if (p1.x - p2.x).abs() < f32::EPSILON || p.x <= xints {
                inside = !inside;
            }
        }

        p1 = p2;
    }

    inside
}

pub struct DetectionsMapping<'a> {
    timestamp: f32,
    detections: &'a [Detection],
    cnd_matched: Vec<(usize, usize, f32)>,
    veh_matched: Vec<(usize, usize, f32)>,
    veh_missed: IndexedSlice<'a, Detection>,

    ped_matched: Vec<(usize, usize, f32)>,
    ped_missed: IndexedSlice<'a, Detection>,
}

#[derive(Debug)]
pub struct Accumulator {
    pub ts: f32,
    pub det: Detection,
}

impl Accumulator {
    pub fn new(ts: f32, det: Detection) -> Self {
        Self { ts, det }
    }

    #[inline(always)]
    pub fn lerp_ts(&mut self, next_ts: f32, factor: f32) {
        self.ts = self.ts * (1.0 - factor) + next_ts * factor;
    }

    #[inline(always)]
    pub fn lerp_w(&mut self, next_w: f32, factor: f32) {
        self.det.w = self.det.w * (1.0 - factor) + next_w * factor;
    }

    #[inline(always)]
    pub fn lerp_h(&mut self, next_h: f32, factor: f32) {
        self.det.h = self.det.h * (1.0 - factor) + next_h * factor;
    }

    #[inline(always)]
    pub fn lerp_x(&mut self, next_x: f32, factor: f32) {
        self.det.x = self.det.x * (1.0 - factor) + next_x * factor;
    }

    #[inline(always)]
    pub fn lerp_y(&mut self, next_y: f32, factor: f32) {
        self.det.y = self.det.y * (1.0 - factor) + next_y * factor;
    }
}

#[derive(Debug)]
pub struct Participant {
    pub id: u32,
    pub match_score_sum: f32,
    pub hit_score_sum: f32,
    pub hits_count: u32,
    pub last_hit_score: f32,
    pub last_update_diff: f32,
    pub last_update_sec: f32,
    pub time_since_update: f32,
    pub object: Object,
    pub class_votes: [u32; 12],
    pub detections: CircularQueue<(f32, Detection)>,
    pub accumulator: Accumulator,
}

impl Participant {
    pub fn new(ts_sec: f32, det: &Detection) -> Self {
        let mut detections = CircularQueue::with_capacity(16);
        detections.push((ts_sec, *det));

        let mut class_votes = [0; 12];

        if det.class >= 0 && det.class < 12 {
            class_votes[det.class as usize] += 1;
        }

        let mut object = Object::new(5.0, 0.01, true);
        object.update(
            0,
            ts_sec,
            na::Point2::new(det.x, det.y),
            na::Point2::new(det.x, det.y),
        );

        Self {
            id: 0,
            match_score_sum: det.confidence,
            hit_score_sum: 0.0,
            last_hit_score: 0.0,
            hits_count: 1,
            last_update_diff: 0.0,
            time_since_update: 0.,
            last_update_sec: ts_sec,
            object,
            class_votes,
            detections,
            accumulator: Accumulator::new(ts_sec, *det),
        }
    }

    pub fn upgrade(&mut self) {
        self.id = SEQ_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn downgrade(&mut self) {
        self.id = 0;
        self.hit_score_sum = 0.0;
        self.last_hit_score = 0.0;
        self.hits_count = 0;
        self.last_update_diff = 0.0;
        self.match_score_sum = 0.0;
        self.object.reset();
    }

    fn update_smothed(&mut self, score: f32, id: u32, det: &Detection) {
        self.match_score_sum += self.accumulator.det.confidence;
        self.hits_count += 1;
        self.hit_score_sum += score;
        self.last_hit_score = score;

        if self.accumulator.det.class >= 0 && self.accumulator.det.class < 12 {
            self.class_votes[self.accumulator.det.class as usize] += 1;
        }

        self.last_update_diff = self.accumulator.ts - self.last_update_sec;
        self.last_update_sec = self.accumulator.ts;

        self.object.update(
            id,
            self.accumulator.ts,
            na::Point2::new(self.accumulator.det.x, self.accumulator.det.y),
            na::Point2::new(det.x, det.y),
        );
    }

    pub fn update(&mut self, ts_sec: f32, det: &Detection, score: f32, id: u32) {
        self.detections.push((ts_sec, *det));

        let aw2 = self.accumulator.det.w / 2.0;
        let ah2 = self.accumulator.det.h / 2.0;

        let acc_l = self.accumulator.det.x - aw2;
        let acc_r = self.accumulator.det.x + aw2;
        let acc_t = self.accumulator.det.y - ah2;
        let acc_b = self.accumulator.det.y + ah2;

        self.accumulator.lerp_w(det.w, 0.05);
        self.accumulator.lerp_h(det.h, 0.05);

        let dw2 = det.w / 2.0;
        let dh2 = det.h / 2.0;

        let det_l = det.x - dw2;
        let det_r = det.x + dw2;
        let det_t = det.y - dh2;
        let det_b = det.y + dh2;

        let d_l = (det_l - acc_l).abs();
        let d_r = (det_r - acc_r).abs();
        let d_t = (det_t - acc_t).abs();
        let d_b = (det_b - acc_b).abs();

        let left = d_l < d_r;
        let top = d_t < d_b;

        let aw = self.accumulator.det.w;
        let ah = self.accumulator.det.h;

        let (x, y) = match (left, top) {
            (true, true) => {
                let cx = (acc_l + det_l) * 0.5 + aw * 0.5;
                let cy = (acc_t + det_t) * 0.5 + ah * 0.5;

                (cx, cy)
            }
            (true, false) => {
                let cx = (acc_l + det_l) * 0.5 + aw * 0.5;
                let cy = (acc_b + det_b) * 0.5 - ah * 0.5;

                (cx, cy)
            }
            (false, true) => {
                let cx = (acc_r + det_r) * 0.5 - aw * 0.5;
                let cy = (acc_t + det_t) * 0.5 + ah * 0.5;

                (cx, cy)
            }
            (false, false) => {
                let cx = (acc_r + det_r) * 0.5 - aw * 0.5;
                let cy = (acc_b + det_b) * 0.5 - ah * 0.5;

                (cx, cy)
            }
        };

        // println!("({}, {}) ({}, {})", det.x, det.y, x, y);
        let p = 0.9;
        self.accumulator.lerp_x(x, p);
        self.accumulator.lerp_y(y, p);
        self.accumulator.lerp_ts(ts_sec, 0.5 * p);

        self.accumulator.det.confidence = score;
        self.accumulator.det.class = det.class;

        self.update_smothed(score, id, det);
    }

    pub fn last_detection(&self) -> &Detection {
        &self.detections.iter().next().unwrap().1
    }

    pub fn iou_slip(&self) -> f32 {
        let mut detections = self.detections.iter();
        let last_detection = detections.next().unwrap().1;
        if let Some(detection) = detections.next() {
            last_detection.iou(&detection.1)
        } else {
            0.
        }
    }

    pub fn prediction(&self, ts_sec: f32) -> na::Point2<f32> {
        self.object.predict(0, ts_sec)
    }

    pub fn velocity(&self) -> &f32 {
        &self.object.vel
    }

    pub fn direction(&self) -> &na::Complex<f32> {
        &self.object.predictor.direction
    }
}

impl From<&Participant> for crate::Track {
    fn from(p: &Participant) -> crate::Track {
        crate::Track {
            track_id: p.id as _,
            time_since_update: p.time_since_update as _,
            class: p
                .class_votes
                .iter()
                .enumerate()
                .max_by_key(|x| x.1)
                .map(|(i, _)| i)
                .unwrap_or(0) as _,
            confidence: p.hit_score_sum / p.hits_count as f32,
            iou_slip: p.iou_slip(),
            bbox: p.last_detection().bbox().as_xyah(),
            velocity: Some(*p.velocity()),
            direction: Some((p.direction().re, p.direction().im)),
            curvature: Some(p.object.predictor.curvature),
        }
    }
}

pub struct Scene {
    pub bounds: Vec<na::Point2<f32>>,
    pub tracks: Vec<Participant>,
    pub peds: Vec<Participant>,
    confirm_seconds: f32,
    last_second: f32,
}

impl Scene {
    pub fn new(bounds: Vec<na::Point2<f32>>) -> Self {
        Self {
            bounds,
            tracks: Vec::with_capacity(64),
            peds: Vec::with_capacity(32),
            confirm_seconds: SECONDS_IN_FRAME / CONFIRM_SECONDS_RATIO,
            last_second: 0.,
        }
    }

    fn assignment<'a>(
        &self,
        ts_sec: f32,
        threshhold: f32,
        dets: IndexedSlice<'a, Detection>,
        objs: IndexedSlice<'_, Participant>,
    ) -> (Vec<(usize, usize, f32)>, IndexedSlice<'a, Detection>) {
        let mut missed: Vec<_> = (0..dets.len()).collect();

        let mut assignments = if !objs.is_empty() {
            let n = dets.len().max(objs.len());

            if n > 256 {
                panic!("Confusion matrix is too big!");
            }

            let mut mat = WeightMatrix::from_fn(n, |(r, c)| {
                if r < objs.len() && c < dets.len() {
                    let obj = &objs[r];
                    let det = &dets[c];
                    let pos = na::Point2::new(det.x, det.y);

                    1.0 - obj
                        .object
                        .probability(obj.id, ts_sec, pos, &obj.accumulator.det)
                } else {
                    100000.0
                }
            });

            let mat2 = mat.clone();
            let res = solve_assignment(&mut mat);

            if let Ok(inner) = res {
                let mut assignments = Vec::new();

                for i in inner {
                    if i.row < objs.len() && i.column < dets.len() {
                        let score = 1.0 - mat2.element_at(i);

                        if score > threshhold {
                            assignments.push((i.row, i.column, score));
                        }
                    }
                }

                missed.retain(|&x| {
                    if assignments.iter().any(|&(_, p, _)| p == x) {
                        return false;
                    }

                    // for i in 0..n {
                    //     if mat2.element_at(munkres::Position { row: i, column: x }) < 0.80 {
                    //         return false;
                    //     }
                    // }

                    true
                });

                assignments
            } else {
                println!("WARNING: assignement could not be solved!");
                Vec::new()
            }
        } else {
            Vec::new()
        };

        missed.iter_mut().for_each(|x| *x = dets.get_index(*x));

        assignments.iter_mut().for_each(|(x, y, _)| {
            *x = objs.get_index(*x);
            *y = dets.get_index(*y);
        });

        (
            assignments,
            IndexedSlice::new_with_indexes(dets.slice, missed),
        )
    }

    pub fn update_time(&mut self, ts_sec: f32) {
        if ts_sec > self.last_second {
            self.confirm_seconds = (ts_sec - self.last_second) / CONFIRM_SECONDS_RATIO;
            self.last_second = ts_sec;
        }
    }

    pub fn map_detections<'a>(
        &self,
        ts_sec: f32,
        detections: &'a [Detection],
    ) -> DetectionsMapping<'a> {
        let mut peds = Vec::new();
        let mut vehicles = Vec::new();
        for (idx, p) in detections.iter().enumerate() {
            if p.class == 0 {
                peds.push(idx);
            } else {
                vehicles.push(idx);
            }
        }

        let mut confirmed = Vec::new();
        let mut pending = Vec::new();
        let mut unconfirmed = Vec::new();
        for (idx, p) in self.tracks.iter().enumerate() {
            if p.id > 0 {
                if ts_sec - p.last_update_sec < self.confirm_seconds {
                    confirmed.push(idx);
                } else {
                    pending.push(idx);
                }
            } else {
                unconfirmed.push(idx);
            }
        }

        let confirmed_tracks = IndexedSlice::new_with_indexes(&self.tracks, confirmed);
        let veh_dets = IndexedSlice::new_with_indexes(detections, vehicles);
        let (track_confirmed_matched, dets_confirmed_missed) =
            self.assignment(ts_sec, 0.2, veh_dets, confirmed_tracks);

        let pending_tracks = IndexedSlice::new_with_indexes(&self.tracks, pending);
        let (track_pending_matched, dets_pending_missed) =
            self.assignment(ts_sec, 0.2, dets_confirmed_missed, pending_tracks);

        let unconfirmed_tracks = IndexedSlice::new_with_indexes(&self.tracks, unconfirmed);
        let (unconfirmed_matched, dets_missed) =
            self.assignment(ts_sec, 0.005, dets_pending_missed, unconfirmed_tracks);

        let ped_tracks = IndexedSlice::new(&self.peds);
        let ped_dets = IndexedSlice::new_with_indexes(detections, peds);
        let (ped_matched, ped_missed) = self.assignment(ts_sec, 0.0, ped_dets, ped_tracks);

        DetectionsMapping {
            timestamp: ts_sec,
            detections,
            cnd_matched: unconfirmed_matched,
            veh_matched: track_confirmed_matched
                .into_iter()
                .chain(track_pending_matched.into_iter())
                .collect(),
            veh_missed: dets_missed,
            ped_matched,
            ped_missed,
        }
    }

    pub fn update(&mut self, mapping: DetectionsMapping<'_>) {
        static UPGRADE_HIT_SCORE_SUM_THRESHOLD: f32 = 12.0;
        static DOWNGRADE_TIME_SINCE_UPDATE_THRESHOLD: f32 = 16.0;
        static MAX_DURATION_MISSING_OBJECT: f32 = 2.0;

        let time = mapping.timestamp;
        let dets = mapping.detections;

        for (i, j, score) in mapping.veh_matched {
            let id = self.tracks[i].id;
            self.tracks[i].update(time, &dets[j], score, id);
        }

        for (i, j, score) in mapping.cnd_matched {
            self.tracks[i].update(time, &dets[j], score, 0);
        }

        for c in &mut self.tracks {
            if c.id == 0
                && c.hit_score_sum > UPGRADE_HIT_SCORE_SUM_THRESHOLD
                && in_bounds(c.object.pos, &self.bounds)
            {
                c.upgrade();
            }
        }

        for t in &mut self.tracks {
            t.time_since_update = time - t.last_update_sec;
            if t.id == 0 && t.time_since_update > 0.4 {
                t.hit_score_sum *= 0.75;
            }

            if t.id > 0
                && (t.time_since_update > DOWNGRADE_TIME_SINCE_UPDATE_THRESHOLD
                    || t.object.predict_distance(t.id, time) > 400.0
                    || !in_bounds(t.object.pos, &self.bounds)
                    || !in_bounds(t.prediction(time), &self.bounds))
            {
                t.downgrade();
            }
        }

        let bounds = self.bounds.clone();

        self.tracks.retain(|t| {
            if t.id == 0 && !in_bounds(t.prediction(time), &bounds) {
                return false;
            }
            t.id > 0 || t.time_since_update < MAX_DURATION_MISSING_OBJECT
        });

        for det in mapping.veh_missed {
            self.tracks.push(Participant::new(time, det));
        }

        for (i, j, score) in mapping.ped_matched {
            let id = self.peds[i].id;
            self.peds[i].update(time, &dets[j], score, id);
        }

        for c in &mut self.peds {
            if c.id == 0
                && c.hit_score_sum > UPGRADE_HIT_SCORE_SUM_THRESHOLD
                && in_bounds(c.object.pos, &self.bounds)
            {
                c.upgrade();
            }
        }

        for t in &mut self.peds {
            let dt = time - t.last_update_sec;

            if t.id > 0
                && (dt > DOWNGRADE_TIME_SINCE_UPDATE_THRESHOLD
                    || !in_bounds(t.object.pos, &self.bounds)
                    || !in_bounds(t.prediction(time), &self.bounds))
            {
                t.downgrade();
            }
        }

        self.peds.retain(|t| {
            let dt = time - t.last_update_sec;

            t.id > 0 || dt < MAX_DURATION_MISSING_OBJECT
        });

        for det in mapping.ped_missed {
            self.peds.push(Participant::new(time, det));
        }
    }

    pub fn tracks(&self) -> Vec<crate::Track> {
        self.tracks
            .iter()
            .chain(self.peds.iter())
            .filter(|t| t.id > 0 && t.time_since_update < self.confirm_seconds)
            .map(Into::into)
            .collect()
    }
}
