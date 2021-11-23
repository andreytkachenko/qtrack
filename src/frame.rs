use crate::detection::Detection;

pub struct Frame {
    pub dims: (u32, u32),
    pub detections: Vec<Detection>,
    pub timestamp: f32, // in seconds
}

impl Frame {
    #[inline]
    pub fn len(&self) -> usize {
        self.detections.len()
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Detection> {
        self.detections.iter()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.detections.is_empty()
    }
}
