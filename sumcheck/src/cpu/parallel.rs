use super::arithmetic::div_ceil;

fn num_threads() -> usize {
    return rayon::current_num_threads();
}

fn parallelize_iter<I, T, F>(iter: I, f: F)
where
    I: Send + Iterator<Item = T>,
    T: Send,
    F: Fn(T) + Send + Sync + Clone,
{
    rayon::scope(|scope| {
        iter.for_each(|item| {
            let f = &f;
            scope.spawn(move |_| f(item))
        })
    });
}

pub fn parallelize<T, F>(v: &mut [T], f: F)
where
    T: Send,
    F: Fn((&mut [T], usize)) + Send + Sync + Clone,
{
    let num_threads = num_threads();
    let chunk_size = div_ceil(v.len(), num_threads);
    if chunk_size < num_threads {
        f((v, 0));
    } else {
        parallelize_iter(v.chunks_mut(chunk_size).zip((0..).step_by(chunk_size)), f);
    }
}
