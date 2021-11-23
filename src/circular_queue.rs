use std::collections::VecDeque;
use std::fmt;

pub struct CircularQueue<T> {
    deque: VecDeque<T>,
    capacity: usize,
}

impl<T: Clone> Clone for CircularQueue<T> {
    fn clone(&self) -> Self {
        Self {
            deque: self.deque.clone(),
            capacity: self.capacity,
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for CircularQueue<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.deque.fmt(f)
    }
}

impl<T> CircularQueue<T> {
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            deque: VecDeque::with_capacity(cap),
            capacity: cap,
        }
    }

    #[inline]
    pub fn push(&mut self, item: T) -> Option<T> {
        let poped = if self.is_full() {
            self.deque.pop_back()
        } else {
            None
        };

        self.deque.push_front(item);

        poped
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.deque.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.deque.is_empty()
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.deque.len() == self.capacity
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        self.deque.pop_front()
    }

    #[inline]
    pub fn clear(&mut self) {
        self.deque.clear()
    }

    #[inline]
    pub fn top_mut(&mut self) -> Option<&mut T> {
        self.deque.front_mut()
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &'_ T> {
        self.deque.iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &'_ mut T> {
        self.deque.iter_mut()
    }

    #[inline]
    pub fn asc_iter(&self) -> impl Iterator<Item = &'_ T> {
        self.deque.iter().rev()
    }

    #[inline]
    pub fn asc_iter_mut(&mut self) -> impl Iterator<Item = &'_ mut T> {
        self.deque.iter_mut().rev()
    }
}
